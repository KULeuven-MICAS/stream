"""What a constraint family builds its part of the allocation model from."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from stream.opt.allocation.constraint_optimization.diagnosis import ConstraintTag
from stream.opt.allocation.constraint_optimization.utils import resource_key
from stream.opt.solver import SolverModel, SolverVar, SolverVarType

if TYPE_CHECKING:
    from stream.hardware.architecture.core import Core
    from stream.mapping.mapping import Resource
    from stream.opt.allocation.constraint_optimization.diagnosis import ResourceKind, StructuralRule
    from stream.opt.allocation.constraint_optimization.quantities import QuantityRegistry
    from stream.opt.allocation.constraint_optimization.space import Choice, DecisionSpace, Placement
    from stream.workload.workload import Tensor


@dataclass(frozen=True)
class DecisionVariables:
    """The core decision variables, which exist before any family builds: ``x`` places a tensor, ``y`` routes a
    transfer, ``z_stop`` stops a tensor's reuse at a level, ``z_single`` holds its window there in one buffer, and
    ``slot_latency`` is each slot's length."""

    x: dict[tuple[Tensor, Placement], SolverVar]
    y: dict[Choice, SolverVar]
    z_stop: dict[tuple[Tensor, int], SolverVar]
    z_single: dict[tuple[Tensor, int], SolverVar]
    slot_latency: dict[int, SolverVar]


@dataclass
class ResourceLedger:
    """What the constraints of one model stand for, for the infeasibility diagnosis and the capacity reports: each
    named constraint's tag; per (limit, core) its bound, its demand terms and the indicators of its solved load; per
    memory the residency terms of what it holds and the handover bits it holds."""

    tags: dict[str, ConstraintTag] = field(default_factory=dict)
    bounds: dict[tuple[ResourceKind, int], float] = field(default_factory=dict)
    terms: dict[tuple[ResourceKind, int], dict[str, Any]] = field(default_factory=lambda: defaultdict(dict))
    loads: dict[tuple[ResourceKind, int], list[tuple[SolverVar, int]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    memory: dict[int, list[tuple[SolverVar, int, str]]] = field(default_factory=lambda: defaultdict(list))
    handover_bits: dict[int, int] = field(default_factory=dict)


class FormulationContext:
    """What a family builds from: ``space``, the read-only problem and the choices derived from it; ``vars``, the
    core decision variables; ``model``; ``quantities``; ``ledger``, what its constraints stand for; and the modelling
    helpers every family shares. One context serves one model build."""

    def __init__(
        self,
        space: DecisionSpace,
        variables: DecisionVariables,
        model: SolverModel,
        quantities: QuantityRegistry,
        ledger: ResourceLedger,
    ) -> None:
        self.space = space
        self.vars = variables
        self.model = model
        self.quantities = quantities
        self.ledger = ledger
        self._names: dict[str, int] = {}
        self._on_core: dict[tuple[Tensor, Core], SolverVar] = {}

    def add_constr(
        self,
        expr: Any,
        *,
        name: str,
        resource: Resource | None = None,
        kind: ResourceKind | None = None,
        subject: str | None = None,
        rule: StructuralRule | None = None,
        bound: float | None = None,
    ) -> None:
        """Add a constraint with what it stands for (see :class:`ConstraintTag`); ``bound`` is the value of the
        limit ``kind`` on ``resource`` it states."""
        self.model.add_constr(expr, name=name)
        self.ledger.tags[name] = ConstraintTag(resource, kind, subject, rule)
        if bound is not None:
            assert kind is not None and resource is not None
            self.ledger.bounds[(kind, resource.id)] = bound

    def _add_tagged(self, expr: Any, name: str, tag: ConstraintTag | None) -> None:
        self.model.add_constr(expr, name=name)
        if tag is not None:
            self.ledger.tags[name] = tag

    def unique_name(self, name: str) -> str:
        """``name`` with whitespace and colons replaced, suffixed with a count when this build used it before
        (MathOpt rejects duplicate names; Gurobi accepts them)."""
        sanitized = str(name).replace(" ", "_").replace(":", "_")
        count = self._names.get(sanitized, 0)
        self._names[sanitized] = count + 1
        return sanitized if count == 0 else f"{sanitized}_{count}"

    def tensor_on_core_expr(self, t: Tensor, core: Core) -> Any:
        """Whether ``t`` sits on ``core``: a constant for a fixed tensor, else the sum of its placements there."""
        return self.tensor_on_cores_expr(t, (core,))

    def tensor_on_cores_expr(self, t: Tensor, cores: Sequence[Core]) -> Any:
        """Whether ``t`` sits on any of ``cores``: a constant for a fixed tensor, else the sum of its placements on
        them."""
        space = self.space
        if space.is_fixed(t):
            return int(any(c in space.fixed_choice(t) for c in cores))
        x = self.vars.x
        return self.model.quicksum(
            x[(t, choice)]._raw for choice in space.tensor_choices[t] if any(c in choice for c in cores)
        )

    def tensor_uses_core_var(self, t: Tensor, core: Core) -> SolverVar:
        """A binary equal to whether ``t`` sits on ``core``, one per (tensor, core) for the whole model."""
        key = (t, core)
        if (u := self._on_core.get(key)) is not None:
            return u
        u = self.model.add_var(vtype=SolverVarType.BINARY, name=f"u_{t.name}_{resource_key(core)}")
        self._on_core[key] = u
        self.add_constr(
            u == self.tensor_on_core_expr(t, core),
            name=f"u_eq_{t.name}_{resource_key(core)}",
            resource=core,
            subject=t.name,
        )
        return u

    def tensor_in_memory_var(self, t: Tensor, cores: list[Core]) -> SolverVar:
        """Whether ``t`` sits on any of ``cores``, the cores sharing one memory."""
        if len(cores) == 1:
            return self.tensor_uses_core_var(t, cores[0])
        key = "__".join(resource_key(c) for c in cores)
        v = self.model.add_var(vtype=SolverVarType.BINARY, name=f"u_{t.name}_{key}")
        self.add_constr(
            v == self.tensor_on_cores_expr(t, cores), name=f"u_eq_{t.name}_{key}", resource=cores[0], subject=t.name
        )
        return v

    def binary_product(
        self, *, a: SolverVar, b: SolverVar, base_name: str, tag: ConstraintTag | None = None
    ) -> SolverVar:
        """A binary equal to ``a * b`` for binaries ``a`` and ``b``, its constraints tagged with ``tag``."""
        n = self.unique_name(base_name)
        w = self.model.add_var(vtype=SolverVarType.BINARY, name=f"{n}__and")
        self._add_tagged(w <= a, f"{n}__ub1", tag)
        self._add_tagged(w <= b, f"{n}__ub2", tag)
        self._add_tagged(w >= a + b - 1, f"{n}__lb", tag)
        return w

    def binary_scaled_continuous(
        self,
        *,
        binary_var: SolverVar,
        continuous_var: SolverVar,
        continuous_ub: float,
        base_name: str,
        result_lb: float = 0.0,
        tag: ConstraintTag | None = None,
    ) -> SolverVar:
        """The exact linearization of ``binary_var * continuous_var`` for ``0 <= continuous_var <= continuous_ub``,
        its constraints tagged with ``tag``."""
        assert continuous_ub >= 0.0, "continuous_ub must be nonnegative"
        n = self.unique_name(base_name)
        z = self.model.add_var(vtype=SolverVarType.CONTINUOUS, lb=result_lb, ub=continuous_ub, name=f"{n}__prod")
        self._add_tagged(z <= continuous_var, f"{n}__prod_ub1", tag)
        self._add_tagged(z <= continuous_ub * binary_var, f"{n}__prod_ub2", tag)
        self._add_tagged(z >= continuous_var - continuous_ub * (1 - binary_var), f"{n}__prod_lb1", tag)
        self._add_tagged(z >= 0.0, f"{n}__prod_lb2", tag)
        return z

    def binary_times_const_over_linexpr(
        self,
        *,
        binary_var: SolverVar,
        numerator: float,
        denominator_expr: Any,
        denominator_lb: float,
        base_name: str,
        denominator_ub: float | None = None,
        selectors: list[tuple[SolverVar, float]] | None = None,
    ) -> SolverVar:
        """``binary_var * numerator / denominator_expr``: a non-linear division where the backend has one, else
        the sum over the ``selectors`` (one-hot binary, denominator value) of ``numerator / value``."""
        assert denominator_lb > 0.0
        result_ub = float(numerator) / denominator_lb
        if self.model.supports_nonlinear:
            ratio_var = self._const_over_linexpr(
                numerator=numerator,
                denominator_expr=denominator_expr,
                base_name=base_name,
                denominator_lb=denominator_lb,
                denominator_ub=denominator_ub,
                result_ub=result_ub,
            )
        else:
            assert selectors is not None, "selectors required for linear-only backends"
            ratio_var = self._const_over_discrete_denominators(
                numerator=numerator, selectors=selectors, base_name=base_name
            )
        return self.binary_scaled_continuous(
            binary_var=binary_var, continuous_var=ratio_var, continuous_ub=result_ub, base_name=f"{base_name}__gated"
        )

    def _const_over_linexpr(
        self,
        *,
        numerator: float,
        denominator_expr: Any,
        base_name: str,
        denominator_lb: float,
        denominator_ub: float | None,
        result_ub: float,
    ) -> SolverVar:
        n = self.unique_name(base_name)
        den = self.model.add_var(
            vtype=SolverVarType.CONTINUOUS,
            lb=denominator_lb,
            ub=denominator_ub if denominator_ub is not None else self.model.INFINITY,
            name=f"{n}__den",
        )
        res = self.model.add_var(vtype=SolverVarType.CONTINUOUS, lb=0.0, ub=result_ub, name=f"{n}__val")
        self.model.add_constr(den == denominator_expr, name=f"{n}__def_den")
        self.model.add_genconstr_nl(res, float(numerator) / den._raw, name=f"{n}__def_div")
        return res

    def _const_over_discrete_denominators(
        self, *, numerator: float, selectors: list[tuple[SolverVar, float]], base_name: str
    ) -> SolverVar:
        n = self.unique_name(base_name)
        result_ub = numerator / min(d for _, d in selectors)
        result = self.model.add_var(vtype=SolverVarType.CONTINUOUS, lb=0.0, ub=result_ub, name=f"{n}__val")
        self.model.add_constr(
            result._raw == self.model.quicksum(z._raw * (numerator / d) for z, d in selectors)._raw,
            name=f"{n}__def_div",
        )
        return result
