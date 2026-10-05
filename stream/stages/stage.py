import functools
import logging
from abc import ABCMeta, abstractmethod
from collections.abc import Callable, Container, Iterable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any, ClassVar, Protocol, runtime_checkable

from stream.profiling import span
from stream.stages.context import RUNNING, StageContext, StageContractError

logger = logging.getLogger(__name__)

CONTRACT = ("reads", "optional_reads", "writes", "result_reads", "result_writes")


class _EveryField:
    """The fields a stage that declares no contract may touch: all of them."""

    def __contains__(self, field: object) -> bool:
        return True


class Stage(metaclass=ABCMeta):
    """A step of a pipeline, which runs the stages after it and yields what they yield. Its contract names the fields
    it touches: it needs its ``reads``, may use its ``optional_reads``, sets its ``writes`` before the stages after it
    run, and reads its ``result_reads`` and sets its ``result_writes`` on the context those stages yield."""

    reads: ClassVar[tuple[str, ...]] = ()
    optional_reads: ClassVar[tuple[str, ...]] = ()
    writes: ClassVar[tuple[str, ...]] = ()
    result_reads: ClassVar[tuple[str, ...]] = ()
    result_writes: ClassVar[tuple[str, ...]] = ()
    declares_contract: ClassVar[bool] = False
    readable: ClassVar[Container[str]] = _EveryField()
    writable: ClassVar[Container[str]] = _EveryField()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls.declares_contract = any(
            field in base.__dict__ for base in cls.__mro__ if base not in (Stage, object) for field in CONTRACT
        )
        if cls.declares_contract:
            cls.writable = frozenset((*cls.writes, *cls.result_writes))
            cls.readable = frozenset((*cls.reads, *cls.optional_reads, *cls.result_reads)) | cls.writable
        if "__init__" in cls.__dict__:
            cls.__init__ = _in_stage(cls.__init__)
        if "run" in cls.__dict__:
            cls.run = _run_in_stage(cls.run)

    def __init__(self, list_of_callables: list["StageCallable"], ctx: StageContext):
        """
        @param list_of_callables: a list of callables, that must have a signature compatible with this __init__ function
        and return a Stage instance. This is used to flexibly build iterators upon other iterators.
        @param ctx: shared stage context containing all inputs and outputs for the pipeline
        """
        self.ctx = ctx
        self.list_of_callables = list_of_callables
        if self.is_leaf() and list_of_callables not in ([], tuple(), set(), None):
            raise ValueError("Leaf runnable received a non empty list_of_callables")

        if list_of_callables in ([], tuple(), set(), None) and not self.is_leaf():
            raise ValueError(
                "List of callables empty on a non leaf runnable, so nothing can be generated. "
                "Final callable in list_of_callables must return Stage instances that have is_leaf() == True"
            )
        if self.reads and (missing := [f for f in self.reads if ctx.data.get(f) is None]):
            raise StageContractError(f"{type(self).__name__} reads {missing}, which the context does not hold")

    def __iter__(self):
        return self.run()

    def is_leaf(self) -> bool:
        """Returns true if the runnable is a leaf runnable, meaning that it does not use (or thus need)
        any substages to be able to yield a result. Final element in list_of_callables must always have
        is_leaf() == True, except for that final element that has an empty list_of_callables
        """
        return False

    @abstractmethod
    def run(self) -> Iterator[StageContext]: ...


@contextmanager
def _entered(stage: Stage) -> Iterator[None]:
    """Run the enclosed code of ``stage`` against its contract and in a span named after it, once however deep its
    own methods call each other."""
    if RUNNING.get() is stage:
        yield
        return
    token = RUNNING.set(stage)
    try:
        with span(type(stage).__name__):
            yield
    finally:
        RUNNING.reset(token)


def _in_stage(init: Callable[..., None]) -> Callable[..., None]:
    @functools.wraps(init)
    def entered(self: Stage, *args: Any, **kwargs: Any) -> None:
        with _entered(self):
            init(self, *args, **kwargs)

    return entered


def _run_in_stage(run: Callable[[Stage], Iterable[StageContext]]) -> Callable[[Stage], Iterator[StageContext]]:
    @functools.wraps(run)
    def entered(self: Stage) -> Iterator[StageContext]:
        with _entered(self):
            results = iter(run(self))
        while True:
            with _entered(self):
                try:
                    result = next(results)
                except StopIteration:
                    return
            yield result

    return entered


def check_contracts(stages: Sequence["StageCallable"], fields: Iterable[str]) -> set[str] | None:
    """The fields a context holding ``fields`` holds after ``stages`` run, each stage running the ones after it, or
    None from a stage that declares no contract on, which runs unchecked; raises a :class:`StageContractError` naming
    the stage that reads a field neither the context nor a stage before it writes."""
    available = set(fields)
    if not stages:
        return available
    stage = stages[0]
    if not (isinstance(stage, type) and issubclass(stage, Stage) and stage.declares_contract):
        _warn_unchecked(stage)
        return None
    _require(stage, stage.reads, available)
    after = check_contracts(stages[1:], available | set(stage.writes))
    if after is None:
        return None
    _require(stage, stage.result_reads, after)
    return after | set(stage.result_writes)


@functools.cache
def _warn_unchecked(stage: "StageCallable") -> None:
    logger.warning(
        "%s declares no contract (reads, writes, ...), so the pipeline cannot check the fields it and the stages "
        "after it touch",
        getattr(stage, "__name__", stage),
    )


def _require(stage: type[Stage], reads: tuple[str, ...], available: set[str]) -> None:
    if missing := [field for field in reads if field not in available]:
        raise StageContractError(f"{stage.__name__} reads {missing}, which neither the context nor a stage writes")


@runtime_checkable
class StageCallable(Protocol):
    def __call__(self, list_of_callables: list["StageCallable"], ctx: StageContext) -> Stage: ...


class MainStage:
    """! Not actually a Stage, as running it does return (not yields!) a list of results instead of a generator
    Can be used as the main entry point
    """

    def __init__(self, list_of_callables: list[StageCallable], ctx: StageContext):
        check_contracts(list_of_callables, ctx.data)
        self.ctx = ctx
        self.list_of_callables = list_of_callables

    def run(self):
        answers: list[StageContext] = []
        for ctx in self.list_of_callables[0](self.list_of_callables[1:], self.ctx).run():
            answers.append(ctx)
        return answers


class LeafStage(Stage):
    """Leaf stage class that doesn't do anything besides return the ctx; it touches no field."""

    reads: ClassVar[tuple[str, ...]] = ()

    def __init__(self, list_of_callables: list[StageCallable], ctx: StageContext):
        assert not list_of_callables, "LeafStage must have an empty list_of_callables"
        super().__init__(list_of_callables, ctx)

    def is_leaf(self) -> bool:
        return True

    def run(self):
        yield self.ctx
