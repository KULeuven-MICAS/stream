import functools
import inspect
from abc import ABCMeta, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any, ClassVar, Protocol, runtime_checkable

from stream.stages.context import StageContext, StageContractError, running


class Stage(metaclass=ABCMeta):
    """A step of a pipeline, which runs the stages after it and yields what they yield.

    Its contract names the context fields it touches: it needs its ``reads`` and may use its ``optional_reads``,
    sets its ``writes`` before the stages after it run, and reads its ``result_reads`` and sets its ``result_writes``
    on the context those stages yield. :class:`MainStage` checks a pipeline's contracts before it runs, and while a
    stage runs the context rejects any field its contract leaves out.
    """

    reads: ClassVar[tuple[str, ...]] = ()
    optional_reads: ClassVar[tuple[str, ...]] = ()
    writes: ClassVar[tuple[str, ...]] = ()
    result_reads: ClassVar[tuple[str, ...]] = ()
    result_writes: ClassVar[tuple[str, ...]] = ()
    readable: ClassVar[frozenset[str]] = frozenset()
    writable: ClassVar[frozenset[str]] = frozenset()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
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


def _in_stage(init: Callable[..., None]) -> Callable[..., None]:
    @functools.wraps(init)
    def checked(self: Stage, *args: Any, **kwargs: Any) -> None:
        with running(type(self)):
            init(self, *args, **kwargs)

    return checked


def _run_in_stage(run: Callable[[Stage], Iterator[StageContext]]) -> Callable[[Stage], Iterator[StageContext]]:
    @functools.wraps(run)
    def checked(self: Stage) -> Iterator[StageContext]:
        results = run(self)
        while True:
            with running(type(self)):
                try:
                    result = next(results)
                except StopIteration:
                    return
            yield result

    return checked


def check_contracts(stages: Sequence["StageCallable"], fields: Iterable[str]) -> set[str]:
    """The fields a context holding ``fields`` holds after ``stages`` run, each stage running the ones after it;
    raises a :class:`StageContractError` naming the stage that reads a field neither the context nor a stage
    before it writes."""
    available = set(fields)
    if not stages:
        return available
    stage = inspect.unwrap(stages[0])
    if not (isinstance(stage, type) and issubclass(stage, Stage)):
        raise StageContractError(f"{stage!r} is not a Stage, so it declares no contract")
    _require(stage, stage.reads, available)
    after = check_contracts(stages[1:], available | set(stage.writes))
    _require(stage, stage.result_reads, after)
    return after | set(stage.result_writes)


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
    """Leaf stage class that doesn't do anything besides return the ctx."""

    def __init__(self, list_of_callables: list[StageCallable], ctx: StageContext):
        assert not list_of_callables, "LeafStage must have an empty list_of_callables"
        super().__init__(list_of_callables, ctx)

    def is_leaf(self) -> bool:
        return True

    def run(self):
        yield self.ctx
