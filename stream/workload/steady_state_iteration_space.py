from zigzag.datatypes import LayerDim


class IterationVariable:
    def __init__(self, dimension: LayerDim, relevancy: bool) -> None:
        self.dimension = dimension
        self.relevancy = relevancy


class SteadyStateIterationSpace:
    def __init__(self, iteration_space: list[IterationVariable]) -> None:
        self.iteration_space = iteration_space
