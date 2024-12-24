from zigzag.datatypes import LayerDim


class TemporalLoop:
    def __init__(self, dimension: LayerDim, size: int) -> None:
        self.dimension = dimension
        self.size = size
        self.type = "temporal"

    def __str__(self):
        return f"TemporalLoop({self.dimension},{self.size})"

    def __repr__(self):
        return str(self)

    def unpack(self):
        """Unpack `dimension` and `size`"""
        return (self.dimension, self.size)
