"""Memories apart from cores: several cores compute out of one memory, which none of them owns."""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from xdsl.dialects.builtin import bf16
from xdsl.ir.affine import AffineMap

from stream.allocation.lowering import reader_positions
from stream.api import evaluate_mapping
from stream.cost_model.communication_manager import DemandItem
from stream.hardware.architecture.memory import Memory
from stream.hardware.ports import OUTPUT, READ, WRITE, declared_ports, input_role
from stream.inputs.testing.workload.make_2_conv import TwoConvWorkloadConfig, make_2_conv_workload
from stream.mapping.mapping import role_of
from stream.opt.allocation.constraint_optimization.families import available_families
from stream.opt.allocation.constraint_optimization.families.memory import (
    OBJECT_FIFO_DEPTH,
    BufferDescriptors,
    ObjectFifoDepth,
)
from stream.opt.allocation.constraint_optimization.formulation import (
    DecisionVariables,
    FormulationContext,
    ResourceLedger,
)
from stream.opt.allocation.constraint_optimization.quantities import QuantityRegistry
from stream.opt.allocation.constraint_optimization.space import DecisionSpace
from stream.opt.solver import SolverBackend, SolverParams, SolverVarType, create_solver
from stream.parser.accelerator_validator import AcceleratorValidator
from stream.stages.parsing.accelerator_parser import parse_accelerator
from stream.workload.node import ComputationNode, TransferNode
from stream.workload.tensor import Tensor

L3 = """
  l3:
    offchip: true
    views: []
    capacity_bits: 10000000000
    ports:
      rw_port_1: {bandwidth: 64, serves: {any: [read, write]}, read_energy: 15.625, write_energy: 15.625}
"""
TWO_L1S = (
    """
name: two-l1s
cores:
  0: shared_testing_core1.yaml
  1: shared_testing_core2.yaml
  2: shared_testing_core1.yaml
  3: shared_testing_core2.yaml
unit_energy_cost: 0
memories:
  l1_a:
    offchip: false
    views: [0.sram_64KB, 1.sram_64KB]
    ports:
      r_port_1: {from: r_port_1, cores: [0, 1]}
      w_port_1: {from: w_port_1, cores: [0, 1]}
  l1_b:
    offchip: false
    views: [2.sram_64KB, 3.sram_64KB]
    ports:
      r_port_1: {from: r_port_1, cores: [2, 3]}
      w_port_1: {from: w_port_1, cores: [2, 3]}"""
    + L3
    + """connections:
  - {type: link, ends: {l1_a: [r_port_1, w_port_1], l3: [rw_port_1]}, bandwidth: 64}
  - {type: link, ends: {l1_b: [r_port_1, w_port_1], l3: [rw_port_1]}, bandwidth: 64}
  - {type: link, ends: {l1_a: [r_port_1, w_port_1], l1_b: [r_port_1, w_port_1]}, bandwidth: 32}
"""
)
"""Two L1s of two cores each, joined to each other and to L3."""


def _l1(*views: str) -> dict:
    """An on-chip memory of shared testing cores' ``views``, each port one all of them reach."""
    cores = [int(view.split(".")[0]) for view in views]
    ports = {name: {"from": name, "cores": cores} for name in ("r_port_1", "w_port_1")}
    return {"offchip": False, "views": list(views), "ports": ports}


def _memories(**views: list[str]) -> dict:
    """The memories ``views`` names, each an on-chip memory of them, and L3."""
    return {name: _l1(*entries) for name, entries in views.items()} | yaml.safe_load(L3)


WHOLE = ((0, 15), (0, 15))


def _accelerator(tmp_path, text: str = TWO_L1S, **overrides):
    data = yaml.safe_load(text) | overrides
    path = tmp_path / "hw.yaml"
    path.write_text(yaml.safe_dump(data))
    return parse_accelerator(str(path))


def _errors(tmp_path, **overrides) -> list[str]:
    data = yaml.safe_load(TWO_L1S) | overrides
    (tmp_path / "hw.yaml").write_text(yaml.safe_dump(data))
    validator = AcceleratorValidator(data, str(tmp_path / "hw.yaml"))
    assert not validator.validate()
    return validator.errors


def test_the_cores_of_a_memory_share_it_and_none_owns_it(tmp_path):
    accelerator = _accelerator(tmp_path)
    a = accelerator.memories["l1_a"]
    assert [(core.id, instance) for core, instance in a.views] == [(0, "sram_64KB"), (1, "sram_64KB")]
    for role in (OUTPUT, input_role(1), input_role(2)):
        assert accelerator.memory_of(accelerator.get_core(0), role) is a
        assert accelerator.memory_of(accelerator.get_core(1), role) is a
    assert accelerator.offchip_memory is accelerator.memories["l3"] and accelerator.offchip_memory.offchip
    assert set(accelerator.network.nodes()) == set(accelerator.memories.values())


def test_any_core_of_a_memory_sends_over_its_links(tmp_path):
    """Either core of one L1 reaches either core of the other over the link between the two L1s."""
    accelerator = _accelerator(tmp_path)
    for source, target in ((0, 2), (1, 3), (1, 2)):
        plan = accelerator.communication_manager.route(
            (accelerator.memory_of(accelerator.get_core(source), OUTPUT),),
            (accelerator.memory_of(accelerator.get_core(target), input_role(1)),),
            (DemandItem(0, (0,), WHOLE),),
            16,
            lambda one, other: False,
        )
        assert [str(link) for link in plan.links_used] == ["CL(l1_a, l1_b, bw=32)"]


def test_a_bandwidth_model_is_keyed_by_its_memory(tmp_path):
    accelerator = _accelerator(tmp_path, bandwidth={"l1_a": {"ceiling": 64, "contiguous": 64}})
    assert list(accelerator.bandwidth) == [accelerator.memories["l1_a"]]


@pytest.mark.parametrize("key", [0, "core0", "0.rw_port_1", "l1_a.sram_64KB.rw_port_1"])
def test_a_bandwidth_key_naming_no_memory_is_rejected(tmp_path, key):
    assert _errors(tmp_path, bandwidth={key: {"ceiling": 64, "contiguous": 64}})


def test_a_port_key_of_a_measured_memory_is_rejected(tmp_path):
    model = {"ceiling": 64, "contiguous": 64}
    errors = _errors(tmp_path, bandwidth={"l1_a": model, "l1_a.rw_port_1": model})
    assert any("already has a measured bandwidth" in error for error in errors)


def test_an_instance_in_two_memories_is_rejected(tmp_path):
    views = {"l1_a": ["0.sram_64KB", "1.sram_64KB"], "l1_b": ["1.sram_64KB", "2.sram_64KB", "3.sram_64KB"]}
    errors = _errors(tmp_path, memories=_memories(**views))
    assert any("`1.sram_64KB` is a view of both l1_a and l1_b" in error for error in errors)


def test_a_core_in_no_memory_is_rejected(tmp_path):
    views = {"l1_a": ["0.sram_64KB", "1.sram_64KB"], "l1_b": ["2.sram_64KB"]}
    errors = _errors(tmp_path, memories=_memories(**views))
    assert any("core 3 is a view of no memory" in error for error in errors)


def test_a_memory_no_connection_reaches_is_rejected(tmp_path):
    views = {"l1_a": ["0.sram_64KB", "1.sram_64KB"], "l1_b": ["2.sram_64KB"], "spare": ["3.sram_64KB"]}
    errors = _errors(tmp_path, memories=_memories(**views))
    assert any("spare" in error and "no connection" in error for error in errors)


def test_a_connection_through_a_port_its_memory_does_not_declare_is_rejected(tmp_path):
    links = [{"type": "link", "ends": {"l1_a": ["rw_port_9"], "l3": ["rw_port_1"]}, "bandwidth": 64}]
    errors = _errors(tmp_path, connections=links)
    assert any("ends.l1_a` names ports ['rw_port_9'] it does not declare" in error for error in errors)


def test_a_connection_naming_its_memories_without_their_ports_is_rejected(tmp_path):
    links = [{"type": "link", "memories": ["l1_a", "l3"], "bandwidth": 64}]
    assert _errors(tmp_path, connections=links)


def test_a_connection_naming_cores_is_rejected(tmp_path):
    links = [{"type": "link", "cores": [0, 2], "bandwidth": 64}]
    assert _errors(tmp_path, connections=links)


def test_a_view_of_an_instance_its_core_does_not_declare_is_rejected(tmp_path):
    views = {"l1_a": ["0.sram_64KB", "1.l1"], "l1_b": ["2.sram_64KB", "3.sram_64KB"]}
    assert any("core 1 declares no memory `l1`" in error for error in _errors(tmp_path, memories=_memories(**views)))


def test_a_view_of_an_inner_level_is_rejected(tmp_path):
    """A view binds a core's top-level instance: an inner level is the core's own, not where transfers land."""
    text = (
        """
name: inner
cores: {0: testing_core1.yaml}
memories:
  l1: {offchip: false, views: [0.sram_64KB], ports: {}}"""
        + L3
        + """connections: [{type: link, ends: {l1: [], l3: [rw_port_1]}, bandwidth: 64}]
"""
    )
    with pytest.raises(ValueError, match="is no top-level memory"):
        _accelerator(tmp_path, text)


def test_a_link_joins_exactly_two_memories(tmp_path):
    links = [{"type": "link", "ends": {"l1_a": [], "l1_b": [], "l3": []}, "bandwidth": 64}]
    assert any("joins exactly two memories" in error for error in _errors(tmp_path, connections=links))


def test_parallel_links_between_two_memories_are_distinct_resources(tmp_path):
    """Two links declared between the same memories are two resources each way, told apart by their index."""
    data = yaml.safe_load(TWO_L1S)
    data["connections"].append({"type": "link", "ends": {"l1_a": ["r_port_1"], "l3": ["rw_port_1"]}, "bandwidth": 64})
    accelerator = _accelerator(tmp_path, yaml.safe_dump(data))
    a, l3 = accelerator.memories["l1_a"], accelerator.memories["l3"]
    links = [d["cl"] for u, v, d in accelerator.network.edges(data=True) if (u, v) == (a, l3)]
    assert sorted(map(str, links)) == ["CL#1(l1_a, l3, bw=64)", "CL(l1_a, l3, bw=64)"]
    assert links[0] != links[1]


def test_views_sharing_a_port_must_agree_on_it(tmp_path):
    core = yaml.safe_load(open("stream/inputs/testing/hardware/cores/shared_testing_core2.yaml"))
    core["memories"]["sram_64KB"]["ports"][0]["bandwidth_max"] = 256
    data = yaml.safe_load(TWO_L1S)
    data["cores"][1] = core
    with pytest.raises(ValueError, match="declare r_port_1 differently"):
        _ = _accelerator(tmp_path, yaml.safe_dump(data)).ports


def test_the_ir_lists_each_memory_with_its_views(tmp_path):
    memories = _accelerator(tmp_path).get_ir()["memories"]
    assert memories == {
        "l1_a": {"offchip": False, "views": ["0.sram_64KB", "1.sram_64KB"]},
        "l1_b": {"offchip": False, "views": ["2.sram_64KB", "3.sram_64KB"]},
        "l3": {"offchip": True, "views": []},
    }


def test_a_measured_on_chip_memory_leaves_the_off_chip_traffic_charged(tmp_path):
    """Only a measured off-chip memory times what crosses the off-chip boundary."""
    family = available_families()["offchip_traffic"]()
    model = {"ceiling": 64, "contiguous": 64}
    for measured, charged in (("l1_a", True), ("l3", False)):
        accelerator = _accelerator(tmp_path, bandwidth={measured: model})
        space = SimpleNamespace(
            accelerator=accelerator, shared_bandwidth=accelerator.bandwidth, iterations=4, offchip_bandwidth=lambda: 64
        )
        quantities = QuantityRegistry()
        family.build(SimpleNamespace(space=space, quantities=quantities))
        assert ("offchip_traffic_weight" in quantities) is charged


META_PROTOTYPE = "stream/inputs/examples/hardware/meta_prototype_dual_core_simd_offchip.yaml"


def test_a_core_reads_each_operand_in_the_memory_its_top_level_instance_for_it_is_a_view_of():
    """A meta_prototype core keeps its activations and outputs in one SRAM and its weights in another."""
    accelerator = parse_accelerator(META_PROTOTYPE)
    core = accelerator.get_core(0)
    assert accelerator.memory_of(core, OUTPUT).name == accelerator.memory_of(core, input_role(1)).name
    assert accelerator.memory_of(core, input_role(1)).name == "meta_prototype_0"
    assert accelerator.memory_of(core, input_role(2)).name == "meta_prototype_0_weights"
    assert [m.name for m in accelerator.memories_of(core)] == ["meta_prototype_0", "meta_prototype_0_weights"]


def test_a_top_level_instance_no_memory_binds_is_rejected(tmp_path):
    """Each top-level instance holds operands transfers land in, so each must be a view of a memory."""
    data = yaml.safe_load(open(META_PROTOTYPE))
    del data["memories"]["meta_prototype_0_weights"]
    bus = next(c for c in data["connections"] if c["type"] == "bus")
    del bus["ends"]["meta_prototype_0_weights"]
    with pytest.raises(ValueError, match=r"top-level memories \['sram_1MB_W'\] are views of no memory"):
        _accelerator(tmp_path, yaml.safe_dump(data))


def test_weights_land_in_the_weights_memory(tmp_path):
    """A convolution's weights are its second input: their copy sits in the weights SRAM of the cores reading them,
    and only there."""
    workload = make_2_conv_workload(
        TwoConvWorkloadConfig(
            batch_size=1,
            in_channels=8,
            height=32,
            width=32,
            out_channels_1=16,
            out_channels_2=32,
            kernel_size=3,
            in_dtype="bf16",
            weight_dtype="bf16",
        )
    )
    allocation = evaluate_mapping(META_PROTOTYPE, workload, str(tmp_path)).context.get("allocation")
    workload = allocation.problem.workload
    weights = [
        copy
        for node in workload.get_computation_nodes()
        for copy in node.inputs[1:2]
        if any(isinstance(p, TransferNode) and copy in p.outputs for p in workload.predecessors(node))
    ]
    assert weights
    for copy in weights:
        placed = allocation.solution.tensor_placements[copy]
        assert placed and all(memory.name.endswith("_weights") for memory in placed), (copy.name, placed)


def test_the_off_chip_memory_is_the_one_declared_so(tmp_path):
    """A memory says whether it is off-chip; an accelerator has at most one."""
    accelerator = _accelerator(tmp_path)
    assert [memory.name for memory in accelerator.memories.values() if memory.offchip] == ["l3"]
    memories = _memories(l1_a=["0.sram_64KB", "1.sram_64KB"], l1_b=["2.sram_64KB", "3.sram_64KB"])
    memories["l1_a"]["offchip"] = True
    assert any("at most one off-chip memory" in error for error in _errors(tmp_path, memories=memories))


def test_a_memory_declares_its_capacity_only_where_no_core_views_it(tmp_path):
    memories = _memories(l1_a=["0.sram_64KB", "1.sram_64KB"], l1_b=["2.sram_64KB", "3.sram_64KB"])
    memories["l1_a"]["capacity_bits"] = 8
    del memories["l3"]["capacity_bits"]
    errors = _errors(tmp_path, memories=memories)
    assert any("`memories.l1_a`: its views declare its capacity" in error for error in errors)
    assert any("`memories.l3`: no core views it" in error for error in errors)


def test_a_port_naming_a_core_that_is_no_view_is_rejected(tmp_path):
    memories = _memories(l1_a=["0.sram_64KB", "1.sram_64KB"], l1_b=["2.sram_64KB", "3.sram_64KB"])
    memories["l1_a"]["ports"]["r_port_1"]["cores"] = [0, 1, 2]
    assert any("names cores [2], no views of it" in error for error in _errors(tmp_path, memories=memories))


@pytest.mark.parametrize(
    ("ports", "missing"),
    [
        ({"w_port_1": {"from": "w_port_1", "cores": [0, 1]}}, "core 0's port r_port_1 is in ports []"),
        (
            {
                "r_port_1": {"from": "r_port_1", "cores": [0, 1]},
                "r_again": {"from": "r_port_1", "cores": [1]},
                "w_port_1": {"from": "w_port_1", "cores": [0, 1]},
            },
            "core 1's port r_port_1 is in ports ['r_again', 'r_port_1']",
        ),
    ],
)
def test_every_port_a_view_declares_is_in_exactly_one_port_of_its_memory(tmp_path, ports, missing):
    data = yaml.safe_load(TWO_L1S)
    _declare_ports(data, "l1_a", ports)
    with pytest.raises(ValueError, match=re.escape(missing)):
        _accelerator(tmp_path, yaml.safe_dump(data))


def _declare_ports(data: dict, memory: str, ports: dict) -> None:
    """Give ``memory`` ``ports``, each connection reaching it through all of them."""
    data["memories"][memory]["ports"] = ports
    for connection in data["connections"]:
        if memory in connection["ends"]:
            connection["ends"][memory] = list(ports)


def _link(accelerator, sender: str, receiver: str, parallel: int = 0):
    """The link from memory ``sender`` to ``receiver``, the ``parallel``-th declared between them."""
    a, b = accelerator.memories[sender], accelerator.memories[receiver]
    (link,) = [
        d["cl"] for u, v, d in accelerator.network.edges(data=True) if (u, v) == (a, b) and d["cl"].parallel == parallel
    ]
    return link


def test_cores_of_one_memory_may_each_have_their_own_port(tmp_path):
    """One port per core: each reaches its own. A transfer over a connection through both cannot tell which it
    passes; one over a connection through one of them passes that one."""
    data = yaml.safe_load(TWO_L1S)
    _declare_ports(
        data,
        "l1_a",
        {
            f"{direction}_{core}": {"from": f"{direction}_port_1", "cores": [core]}
            for direction in ("r", "w")
            for core in (0, 1)
        },
    )
    data["connections"].append(
        {"type": "link", "ends": {"l1_a": ["r_1", "w_1"], "l1_b": ["r_port_1", "w_port_1"]}, "bandwidth": 32}
    )
    accelerator = _accelerator(tmp_path, yaml.safe_dump(data, sort_keys=False))
    l1_a = [port for port in accelerator.ports if port.key.memory == "l1_a"]
    assert sorted((port.key.port, port.core_ids) for port in l1_a) == [
        ("r_0", (0,)),
        ("r_1", (1,)),
        ("w_0", (0,)),
        ("w_1", (1,)),
    ]
    assert [port.key.port for port in accelerator.ports.ports_of(accelerator.get_core(1))] == ["r_1", "w_1"]
    l1_a, l1_b = accelerator.memories["l1_a"], accelerator.memories["l1_b"]
    through_both, through_one = _link(accelerator, "l1_a", "l1_b"), _link(accelerator, "l1_a", "l1_b", 1)
    with pytest.raises(ValueError, match=r"memory l1_a over .*: ports \['r_0', 'r_1'\] all serve output read"):
        accelerator.ports.memory_port(l1_a, READ, OUTPUT, [through_both])
    assert accelerator.ports.memory_port(l1_a, READ, OUTPUT, [through_one]).key.port == "r_1"
    assert accelerator.ports.memory_port(l1_b, WRITE, OUTPUT, [through_one]).key.port == "w_port_1"
    assert accelerator.ports.memory_port(l1_a, READ, OUTPUT, []) is None


def test_views_naming_their_ports_apart_leave_a_transfer_no_port(tmp_path):
    """Two ports of one memory serving an operand in one direction, as views naming their ports apart declare, say
    nothing about which a transfer passes."""
    core = yaml.safe_load(open("stream/inputs/testing/hardware/cores/shared_testing_core2.yaml"))
    for port in core["memories"]["sram_64KB"]["ports"]:
        port["name"] = port["name"].replace("_1", "_2")
    data = yaml.safe_load(TWO_L1S)
    data["cores"][1] = core
    _declare_ports(
        data,
        "l1_a",
        {
            f"{direction}_port_{core}": {"from": f"{direction}_port_{core}", "cores": [core - 1]}
            for direction in ("r", "w")
            for core in (1, 2)
        },
    )
    accelerator = _accelerator(tmp_path, yaml.safe_dump(data))
    link = _link(accelerator, "l1_a", "l1_b")
    with pytest.raises(ValueError, match=r"memory l1_a over .*: ports \['r_port_1', 'r_port_2'\] all serve output"):
        accelerator.ports.memory_port(accelerator.memories["l1_a"], READ, OUTPUT, [link])
    assert accelerator.ports.memory_port(accelerator.memories["l1_b"], WRITE, OUTPUT, [link]).key.port == "w_port_1"


class _Stub:
    """A node, tensor, core or memory with just the fields a test reads, hashable by identity."""

    def __init__(self, **fields):
        self.__dict__.update(fields)


def test_the_bits_a_position_moves_are_its_own_where_cores_share_a_memory():
    """A windowed transfer's bits out of one position are what that position hands, not what every position in the
    same memory hands: the port shares of two cores sharing a memory sum to the memory's, not twice it."""
    a, b = _Stub(name="l1_a"), _Stub(name="l1_b")
    space = object.__new__(DecisionSpace)
    space.hardware = _Stub(shares_memory=lambda one, other: one is other)
    space.overlaps = lambda tr: {(0, 0): 3, (1, 0): 5, (1, 1): 7}  # type: ignore[method-assign]
    tr = _Stub(inputs=[_Stub(operand_type=_Stub(bitwidth=8))], outputs=[None])
    choice = _Stub(sources=(a, a), targets=(b, b))
    assert space.moved_bits(tr, choice) == 15 * 8
    assert [space.moved_bits(tr, choice, source_at=i) for i in range(2)] == [3 * 8, 12 * 8]
    assert [space.moved_bits(tr, choice, target_at=j) for j in range(2)] == [8 * 8, 7 * 8]


def test_a_core_reads_a_copy_in_one_memory():
    """A copy lands in one memory per reading core, in the readers' core order, cores sharing a memory repeating it;
    a core reading it in two memories, as two operands of two readers, is rejected."""
    a, b = _Stub(name="l1_a"), _Stub(name="l1_b")
    cores = [_Stub(id=i) for i in range(3)]
    one, other, tensor = _Stub(name="conv1"), _Stub(name="conv2"), _Stub(name="x")
    assert reader_positions(tensor, [(one, cores[:2], (a, a)), (other, cores[1:], (a, b))]) == (a, a, b)
    with pytest.raises(ValueError, match="core 1 reads x in l1_b for conv2 but in l1_a for conv1"):
        reader_positions(tensor, [(one, cores[:2], (a, a)), (other, cores[1:], (b, b))])


def test_fifo_depth_and_descriptors_count_a_tensor_once_per_position():
    """A tensor two cores hold in one memory takes a fifo, and its descriptors, on each: twice the tiles."""
    model = create_solver(SolverBackend.ORTOOLS_GSCIP, "positions")
    model.set_param(SolverParams.VERBOSITY, 0)
    memory = Memory.__new__(Memory)
    core = _Stub(id=0, kind="compute", core_type="zigzag.compute")
    memory.name, memory.views, memory.offchip, memory.capacity_bits = "l1_a", ((core, "sram"),), False, 0
    memory.ports = {}
    tensor = _Stub(name="x")
    transfer = _Stub(name="Transfer(x)", inputs=[_Stub(name="x0")], outputs=[tensor])
    transfer.tensors = [*transfer.inputs, *transfer.outputs]
    space = object.__new__(DecisionSpace)
    space.transfer_nodes = (transfer,)
    space.tensor_choices = {tensor: ((memory, memory),), transfer.inputs[0]: ((_Stub(offchip=True),),)}
    space.candidate_memories = lambda t: set(space.tensor_choices[t][0])  # type: ignore[method-assign]
    space.is_fixed = lambda t: True  # type: ignore[method-assign]
    space.fixed_choice = lambda t: space.tensor_choices[t][0]  # type: ignore[method-assign]
    space.stops = lambda t: range(-1, 1)  # type: ignore[method-assign]
    space.tiles_needed_levels = {(tensor, -1): 1, (tensor, 0): 4}
    space.bds_needed_levels = {}
    z_stop = {(tensor, stop): model.add_var(vtype=SolverVarType.BINARY, name=f"z{stop}") for stop in (-1, 0)}
    model.add_constr(z_stop[(tensor, -1)] == 0)
    model.add_constr(z_stop[(tensor, 0)] == 1)
    q = QuantityRegistry()
    ctx = FormulationContext(space, DecisionVariables({}, {}, z_stop, {}, {}), model, q, ResourceLedger())
    ObjectFifoDepth().build(ctx)
    BufferDescriptors().build(ctx)
    depth, descriptors = q.get("object_fifo_depth", memory).expr, q.get("buffer_descriptor_depth", memory).expr
    model.set_objective(depth + descriptors)
    model.optimize()
    assert (model.value(depth), model.value(descriptors)) == (8, 8)
    assert ctx.ledger.terms[(OBJECT_FIFO_DEPTH, memory)] == {"x": 2}


SHIPPED_HARDWARE = sorted(
    str(path)
    for path in Path("stream/inputs").glob("**/hardware/*.yaml")
    if "cores" in (yaml.safe_load(path.read_text()) or {})
)


@pytest.mark.parametrize("path", SHIPPED_HARDWARE)
def test_every_core_a_hardware_file_declares_is_in_its_accelerator(path):
    declared = yaml.safe_load(Path(path).read_text())["cores"]
    assert sorted(core.id for core in parse_accelerator(path).core_list) == sorted(declared)


@pytest.mark.parametrize("name", ["whole_array", "whole_array_strix"])
def test_aie_dram_is_an_off_chip_memory_with_a_shim_dma_and_every_mem_tile_reaches_it(name):
    """No core views DRAM: its ports are one shim tile's DMA, two 64-bit channels each way."""
    accelerator = parse_accelerator(f"stream/inputs/aie/hardware/{name}.yaml")
    dram = accelerator.offchip_memory
    assert dram.views == () and {core.kind for core in accelerator.core_list} == {"compute", "memory"}
    assert {
        name: (port.bits_per_cycle, port.core_ids) for port in declared_ports(dram) for name in [port.key.port]
    } == {
        "mm2s": (128, ()),
        "s2mm": (128, ()),
    }
    mem_tiles = {memory for memory in accelerator.memories.values() if memory.kind == "memory"}
    assert {v for u, v in accelerator.network.edges() if u is dram} == mem_tiles


def _node(op_type: str, inputs: int) -> ComputationNode:
    tensors = [Tensor.create(f"t{i}", bf16, (8, 8)) for i in range(inputs + 1)]
    maps = tuple(AffineMap.from_callable(lambda m, n: (m, n)) for _ in tensors)
    return ComputationNode(
        type=op_type, name=op_type, inputs=tuple(tensors[:-1]), outputs=(tensors[-1],), operand_mapping=maps
    )


def test_a_multiply_accumulates_bias_lands_where_its_output_does():
    """A Gemm's operands are its first two inputs; its bias is added into the output, on a core with an activation and
    a weights SRAM in the activations', as ZigZag, which prices no third operand, leaves it."""
    gemm = _node("Gemm", 3)
    assert [role_of(gemm, t) for t in (*gemm.inputs, *gemm.outputs)] == ["input1", "input2", OUTPUT, OUTPUT]
    accelerator = parse_accelerator(META_PROTOTYPE)
    core = accelerator.get_core(0)
    assert accelerator.memory_of(core, role_of(gemm, gemm.inputs[2])).name == "meta_prototype_0"


def test_every_input_of_an_elementwise_node_is_an_operand():
    add = _node("Add", 3)
    assert [role_of(add, t) for t in add.inputs] == ["input1", "input2", "input3"]


def test_views_declaring_a_shared_port_differently_are_rejected_when_parsed(tmp_path):
    """The disagreement surfaces when the description is read, not when a solve first asks for the ports."""
    core = yaml.safe_load(open("stream/inputs/testing/hardware/cores/shared_testing_core2.yaml"))
    core["memories"]["sram_64KB"]["ports"][0]["bandwidth_max"] = 256
    data = yaml.safe_load(TWO_L1S)
    data["cores"][1] = core
    with pytest.raises(ValueError, match="declare r_port_1 differently"):
        _accelerator(tmp_path, yaml.safe_dump(data))


def test_carried_state_lands_where_the_output_does():
    """A linear-attention state update reads two operands and carries its state, updated in place each step."""
    from stream.workload.blocks import build_block  # noqa: PLC0415

    node = next(n for n in build_block("linear_attention").get_computation_nodes() if n.name == "state_update")
    assert [role_of(node, t) for t in node.inputs] == ["input1", "input2", OUTPUT]
