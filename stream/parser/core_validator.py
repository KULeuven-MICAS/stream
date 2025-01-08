from typing import Any

from zigzag.parser.accelerator_validator import AcceleratorValidator as ZigZagAcceleratorValidator


class CoreValidator(ZigZagAcceleratorValidator):
    """Validates a single Stream accelerator core from a user-provided yaml file attribute."""

    SCHEMA = ZigZagAcceleratorValidator.SCHEMA.copy()

    # Add custom schema rules for Stream accelerator cores here
    SCHEMA.update({"utilization": {"type": "float", "required": False}})
    SCHEMA.update({"type": {"type": "string", "required": False, "allowed": ["memory", "compute"]}})

    def __init__(self, data: Any):
        """Initialize Validator object, assign schema and store normalize user-given data"""
        super().__init__(data)
