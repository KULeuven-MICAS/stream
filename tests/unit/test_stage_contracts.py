"""Stages declare the context fields they read and write; a pipeline is checked before it runs and as it runs."""

import ast
import importlib
import re
from pathlib import Path

import pytest

from stream.profiling import timed
from stream.stages.context import StageContext, StageContractError
from stream.stages.stage import LeafStage, MainStage, Stage, check_contracts

CONTRACT = ("reads", "optional_reads", "writes", "result_reads", "result_writes")
ROW = re.compile(r"^\| `(\w+)` \|")


class _Produce(Stage):
    reads = ("source",)
    writes = ("made",)

    def run(self):
        self.ctx.set(made=self.ctx.get("source") + 1)
        yield from self.list_of_callables[0](self.list_of_callables[1:], self.ctx).run()


class _Consume(Stage):
    reads = ("made",)
    writes = ("used",)

    def run(self):
        self.ctx.set(used=self.ctx.get("made") * 2)
        yield from self.list_of_callables[0](self.list_of_callables[1:], self.ctx).run()


class _Wrap(Stage):
    result_reads = ("used",)
    result_writes = ("reported",)

    def run(self):
        for ctx in self.list_of_callables[0](self.list_of_callables[1:], self.ctx).run():
            ctx.set(reported=ctx.get("used"))
            yield ctx


class _Peek(Stage):
    reads = ("source",)

    def run(self):
        self.ctx.get("made")
        yield from self.list_of_callables[0](self.list_of_callables[1:], self.ctx).run()


class _Scribble(LeafStage):
    def run(self):
        self.ctx.set(scribbled=True)
        yield self.ctx


def test_a_pipeline_runs_when_every_read_is_written_before_it():
    (ctx,) = MainStage([_Wrap, _Produce, _Consume, LeafStage], StageContext.from_kwargs(source=1)).run()
    assert (ctx.get("made"), ctx.get("used"), ctx.get("reported")) == (2, 4, 4)


def test_the_check_names_the_stage_and_the_field_nothing_writes():
    with pytest.raises(StageContractError, match=r"_Consume reads \['made'\]"):
        MainStage([_Consume, _Produce, LeafStage], StageContext.from_kwargs(source=1))


def test_a_result_read_is_checked_against_what_the_stages_after_it_write():
    assert "reported" in check_contracts([_Wrap, _Produce, _Consume, LeafStage], {"source"})
    with pytest.raises(StageContractError, match=r"_Wrap reads \['used'\]"):
        check_contracts([_Wrap, _Produce, LeafStage], {"source"})


def test_a_wrapped_stage_is_checked_by_the_contract_of_the_stage_it_wraps():
    with pytest.raises(StageContractError, match="_Consume"):
        check_contracts([timed(_Consume), LeafStage], {"source"})


def test_a_running_stage_cannot_read_a_field_it_does_not_declare():
    pipeline = MainStage([_Produce, _Peek, LeafStage], StageContext.from_kwargs(source=1))
    with pytest.raises(StageContractError, match="_Peek reads 'made'"):
        pipeline.run()


def test_a_running_stage_cannot_write_a_field_it_does_not_declare():
    with pytest.raises(StageContractError, match=r"_Scribble writes \['scribbled'\]"):
        MainStage([_Scribble], StageContext()).run()


def test_outside_a_stage_every_field_is_open():
    (ctx,) = MainStage([_Produce, LeafStage], StageContext.from_kwargs(source=1)).run()
    ctx.set(anything=ctx.get("source"))
    assert ctx.get("anything") == 1


def _stream_stages() -> tuple[dict[str, type[Stage]], set[str]]:
    """Every Stage in ``stream.stages``, and the names of those in modules an optional dependency keeps from
    importing."""
    unimportable: set[str] = set()
    for path in sorted(Path("stream/stages").rglob("*.py")):
        try:
            importlib.import_module(".".join(path.with_suffix("").parts).removesuffix(".__init__"))
        except ModuleNotFoundError:
            unimportable |= {node.name for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef)}
    found: dict[str, type[Stage]] = {}
    pending = Stage.__subclasses__()
    while pending:
        stage = pending.pop()
        if stage.__module__.startswith("stream."):
            found[stage.__name__] = stage
        pending.extend(stage.__subclasses__())
    return found, unimportable


def _row(stage: type[Stage]) -> str:
    cells = [", ".join(f"`{field}`" for field in getattr(stage, part)) for part in CONTRACT]
    return f"| `{stage.__name__}` | " + " | ".join(cells) + " |"


def test_the_stage_reference_lists_every_stage_with_its_contract():
    stages, unimportable = _stream_stages()
    lines = Path("docs/source/stages.md").read_text().splitlines()
    documented = {m.group(1): line for line in lines if (m := ROW.match(line)) and m.group(1).endswith("Stage")}
    assert set(stages) <= set(documented), f"undocumented stages: {sorted(set(stages) - set(documented))}"
    assert set(documented) - set(stages) <= unimportable, f"no such stage: {sorted(set(documented) - set(stages))}"
    for name, stage in stages.items():
        assert documented[name] == _row(stage), f"stages.md has {name}'s contract wrong; it is\n{_row(stage)}"
