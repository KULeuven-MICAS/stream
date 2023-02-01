from typing import List
from zigzag.classes.hardware.architecture.operational_unit import OperationalUnit


class PoolingUnit(OperationalUnit):
    def __init__(self, input_precision: List[int], energy_cost: float, area: float):
        """
        Initialize the Multiplier object.

        :param input_precision: The bit precision of the multiplication inputs.
        :param output_precision: The bit precision of the multiplication outputs.
        :param energy_cost: The energy cost of performing a single multiplication.
        :param area: The area of a single multiplier.
        """
        output_precision = input_precision
        super().__init__(input_precision, output_precision, energy_cost, area)