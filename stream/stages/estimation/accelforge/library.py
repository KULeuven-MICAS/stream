"""AccelForge, imported and run without changing the process: :func:`contained` undoes what its code does to the root
logger (its dependency ``hwcomponents`` calls ``logging.basicConfig`` and its root-level ``logging.info`` calls add a
handler), the warning filters and the environment. It searches in one process, so ties between equally good mappings
break the same way every run. The modules that build or run AccelForge objects import this first."""

import logging
import os
import warnings
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def contained() -> Iterator[None]:
    """Run AccelForge code, its own dependencies' deprecation warnings silenced, then restore the root logger's
    handlers and level and remove the environment variables it set."""
    handlers, level, environment = list(logging.root.handlers), logging.root.level, set(os.environ)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", module=r"accelforge\.")
        try:
            yield
        finally:
            logging.root.handlers[:] = handlers
            logging.root.setLevel(level)
            for name in set(os.environ) - environment:
                del os.environ[name]


with contained():
    from accelforge.util.parallel import set_n_parallel_jobs

    set_n_parallel_jobs(1)
