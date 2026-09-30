import sys
from unittest.mock import MagicMock

import pytest

from stream.parser.mapping_factory import MappingFactory


def test_a_named_kernel_without_the_aie_kernels_installed_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "stream.compiler.kernels.registry", None)
    factory = MappingFactory({}, MagicMock(), MagicMock())
    with pytest.raises(ModuleNotFoundError, match="stream-setup-aie"):
        factory.create_kernel({"kernel": {"name": "mm"}})


def test_a_layer_without_a_kernel_needs_no_aie_kernels(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "stream.compiler.kernels.registry", None)
    assert MappingFactory({}, MagicMock(), MagicMock()).create_kernel({}) is None
