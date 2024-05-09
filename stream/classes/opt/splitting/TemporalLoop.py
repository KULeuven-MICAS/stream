class TemporalLoop:

    def __init__(self, dimension: str, size: int) -> None:
        self.dimension = dimension
        self.size = size
        self.type = "temporal"

    def __str__(self):
        return f"TemporalLoop({self.dimension},{self.size})"

    def __repr__(self):
        return str(self)
