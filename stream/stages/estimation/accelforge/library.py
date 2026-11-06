"""AccelForge, imported once without the root logging handler, warning filters and environment variables its import
adds, and searching in one process so that ties between equally good mappings break the same way every run. The modules
that build or run AccelForge objects import this first."""

import logging
import os
import warnings


def _import_accelforge() -> None:
    handlers, environment = list(logging.root.handlers), set(os.environ)
    with warnings.catch_warnings():
        from accelforge.util.parallel import set_n_parallel_jobs  # noqa: PLC0415

    logging.root.handlers[:] = handlers
    for name in set(os.environ) - environment:
        del os.environ[name]
    set_n_parallel_jobs(1)


_import_accelforge()
