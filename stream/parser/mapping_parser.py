from typing import Any

from zigzag.utils import open_yaml

from stream.parser.mapping_factory import MappingFactory
from stream.parser.mapping_validator import MappingValidator


class MappingParser:
    def __init__(self, mapping_yaml_path: str):
        self.mapping_yaml_path = mapping_yaml_path

    def run(self):
        mapping_data = self.parse_mapping_data()
        return self.parse_mapping(mapping_data)

    def parse_mapping_data(self) -> list[dict[str, Any]]:
        """Parse, validate and normalize workload mapping from a given yaml file path"""
        mapping_data = open_yaml(self.mapping_yaml_path)
        mapping_validator = MappingValidator(mapping_data)
        mapping_data = mapping_validator.normalized_data
        mapping_validate_success = mapping_validator.validate()
        if not mapping_validate_success:
            raise ValueError("Failed to validate user provided mapping.")
        return mapping_data

    def parse_mapping(self, mapping_data: list[dict[str, Any]]):
        mapping_factory = MappingFactory(mapping_data)
        all_mappings = mapping_factory.create()
        return all_mappings
