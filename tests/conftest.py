def pytest_addoption(parser):
    parser.addoption(
        "--keep-output",
        action="store_true",
        help="Keep generated outputs after tests finish",
    )
