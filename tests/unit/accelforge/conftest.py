import pytest


@pytest.fixture(autouse=True)
def in_a_scratch_directory(tmp_path, monkeypatch):
    """AccelForge's own ``map_workload_to_arch`` writes ``mapping.svg`` into the working directory."""
    monkeypatch.chdir(tmp_path)
