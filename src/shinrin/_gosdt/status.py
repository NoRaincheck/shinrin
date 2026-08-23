"""Status codes mirroring the vendored GOSDT engine's gosdt::Status enum."""

from enum import IntEnum


class Status(IntEnum):
    CONVERGED = 0
    TIMEOUT = 1
    NON_CONVERGENCE = 2
    FALSE_CONVERGENCE = 3
    UNINITIALIZED = 4


__all__ = ["Status"]
