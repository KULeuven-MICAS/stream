from abc import ABCMeta, abstractmethod
from collections.abc import Generator
from typing import Any, Protocol, runtime_checkable

from stream.cost_model.cost_model import StreamCostModelEvaluation


class Stage(metaclass=ABCMeta):
    kwargs: dict[str, Any]

    def __init__(
        self,
        list_of_callables: list["StageCallable"],
        **kwargs: Any,
    ):
        """
        @param list_of_callables: a list of callables, that must have a signature compatible with this __init__ function
        and return a Stage instance. This is used to flexibly build iterators upon other iterators.
        @param kwargs: any keyword arguments, irrelevant to the specific class in question but passed on down
        """
        self.kwargs = kwargs
        self.list_of_callables = list_of_callables
        if self.is_leaf() and list_of_callables not in ([], tuple(), set(), None):
            raise ValueError("Leaf runnable received a non empty list_of_callables")

        if list_of_callables in ([], tuple(), set(), None) and not self.is_leaf():
            raise ValueError(
                "List of callables empty on a non leaf runnable, so nothing can be generated. "
                "Final callable in list_of_callables must return Stage instances that have is_leaf() == True"
            )

    def __iter__(self):
        return self.run()

    def is_leaf(self) -> bool:
        """Returns true if the runnable is a leaf runnable, meaning that it does not use (or thus need)
        any substages to be able to yield a result. Final element in list_of_callables must always have
        is_leaf() == True, except for that final element that has an empty list_of_callables
        """
        return False

    @abstractmethod
    def run(self) -> Generator[tuple[StreamCostModelEvaluation, Any], None, None]: ...


@runtime_checkable
class StageCallable(Protocol):
    def __call__(self, list_of_callables: list["StageCallable"], **kwagrs: Any) -> Stage: ...


class MainStage:
    """! Not actually a Stage, as running it does return (not yields!) a list of results instead of a generator
    Can be used as the main entry point
    """

    def __init__(self, list_of_callables: list[StageCallable], **kwargs: Any):
        self.kwargs = kwargs
        self.list_of_callables = list_of_callables

    def run(self):
        answers: list[tuple[StreamCostModelEvaluation, Any]] = []
        for cme, extra_info in self.list_of_callables[0](self.list_of_callables[1:], **self.kwargs).run():
            answers.append((cme, extra_info))
        return answers
