def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")


def pytest_addoption(parser):
    parser.addoption(
        "--keep-output",
        action="store_true",
        help="Keep generated outputs after tests finish",
    )
