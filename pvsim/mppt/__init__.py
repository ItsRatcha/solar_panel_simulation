"""MPPT algorithms. Importing this package registers all built-ins."""

from .base import MPPTAlgorithm, available_mppt, create_mppt, mppt_class, register_mppt
from . import po, inccond, global_scan, manual_duty  # noqa: F401  (registration)

__all__ = ["MPPTAlgorithm", "available_mppt", "create_mppt", "mppt_class", "register_mppt"]
