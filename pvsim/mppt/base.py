"""MPPT plugin interface and registry.

Every algorithm is written the way it would be on the ESP32:

    duty = algo.step(v_meas, i_meas)

called once per MPPT period with the *measured* panel voltage and current
(the duty applied during that period is algo.s.d).  Parameters (`self.p`)
and state (`self.s`) are flat dataclasses of ints/floats so they map 1:1 to
C structs in firmware_ref/mppt.h.

New algorithms (fuzzy, ANN, RL, ...) drop in with the decorator:

    @register_mppt("my_algo")
    class MyAlgo(MPPTAlgorithm):
        Params = MyParams
        State = MyState
        def step(self, v, i): ...
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import ClassVar

_REGISTRY: dict[str, type["MPPTAlgorithm"]] = {}


def register_mppt(name: str):
    def deco(cls):
        if name in _REGISTRY:
            raise ValueError(f"MPPT algorithm {name!r} already registered")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return deco


def available_mppt(hardware: bool = True) -> list[str]:
    """Registered names; hardware=False leaves out plugins that need a device (serial_esp32)."""
    return [n for n, c in _REGISTRY.items() if hardware or not c.hardware]


def mppt_class(name: str) -> type["MPPTAlgorithm"]:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown MPPT algorithm {name!r}; available: {available_mppt()}") from None


def create_mppt(name: str, **params) -> "MPPTAlgorithm":
    cls = mppt_class(name)
    return cls(cls.Params(**params))


@dataclass
class BaseParams:
    d_init: float = 0.5
    d_min: float = 0.0
    d_max: float = 0.95
    step: float = 0.005  # duty perturbation per MPPT period


@dataclass
class BaseState:
    d: float = 0.0
    v_prev: float = 0.0
    i_prev: float = 0.0
    p_prev: float = 0.0
    n: int = 0  # steps taken


class MPPTAlgorithm:
    name: ClassVar[str] = "base"
    hardware: ClassVar[bool] = False  # True: decisions come from an external device
    Params: ClassVar[type] = BaseParams
    State: ClassVar[type] = BaseState

    def __init__(self, params=None):
        self.p = params if params is not None else self.Params()
        self.reset()

    def reset(self) -> None:
        self.s = self.State()
        self.s.d = self.clip(self.p.d_init)

    @property
    def duty(self) -> float:
        return self.s.d

    def clip(self, d: float) -> float:
        if d < self.p.d_min:
            return self.p.d_min
        if d > self.p.d_max:
            return self.p.d_max
        return d

    def step(self, v: float, i: float) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    def _remember(self, v: float, i: float) -> None:
        self.s.v_prev = v
        self.s.i_prev = i
        self.s.p_prev = v * i
        self.s.n += 1

    # ------------------------------------------------------------ serialise
    def params_dict(self) -> dict:
        return asdict(self.p)

    @classmethod
    def param_fields(cls) -> list[str]:
        return [f.name for f in fields(cls.Params)]
