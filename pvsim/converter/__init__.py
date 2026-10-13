"""DC-DC converter models: averaged buck/boost (the engine's default) and a
switched state-space model for ripple and cross-checking."""

from __future__ import annotations

from .base import Converter, ConverterResult
from .boost_averaged import BoostAveraged, BoostParams
from .buck_averaged import BuckAveraged, BuckParams

TOPOLOGIES = ("buck", "boost")
PARAMS = {"buck": BuckParams, "boost": BoostParams}
AVERAGED = {"buck": BuckAveraged, "boost": BoostAveraged}


def converter_params(topology: str, overrides: dict | None = None):
    """Parameter dataclass for `topology` with `overrides` applied (unknown keys ignored)."""
    if topology not in PARAMS:
        raise ValueError(f"converter topology must be one of {TOPOLOGIES}, got {topology!r}")
    cls = PARAMS[topology]
    return cls.from_dict({**cls().to_dict(), **(overrides or {})})


def make_converter(topology: str = "buck", overrides: dict | None = None) -> Converter:
    """Averaged converter of the given topology."""
    params = converter_params(topology, overrides)
    return AVERAGED[topology](params)


__all__ = ["Converter", "ConverterResult", "BuckAveraged", "BuckParams", "BoostAveraged",
           "BoostParams", "TOPOLOGIES", "converter_params", "make_converter"]
