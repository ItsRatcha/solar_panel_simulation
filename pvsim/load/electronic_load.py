"""Programmable DC electronic load (CR / CC / CV / OFF).

Every mode is a non-decreasing, piecewise-linear sink current I(V), which
makes the operating point unique and lets the load be intersected with a
Thevenin source (E, R) in closed form:

    V = E - I(V) * R

CR : I = V / R
CC : I = I_set * min(1, V / v_min)       (current folds back near 0 V, as a
                                            real e-load does when the source
                                            cannot hold its voltage)
CV : I = clip((V - V_set) / r_cv, 0, i_max) (stiff sink above V_set)
OFF: I = 0
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

MODES = ("CR", "CC", "CV", "OFF")


@dataclass(frozen=True)
class ElectronicLoad:
    mode: str = "CR"
    setpoint: float = 5.0  # ohm (CR), A (CC), V (CV)
    v_min: float = 0.5  # CC fold-back voltage [V]
    r_cv: float = 0.02  # CV slope resistance [ohm]
    i_max: float = 10.0  # CV current limit [A]

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"load mode must be one of {MODES}, got {self.mode!r}")
        if self.mode == "CR" and self.setpoint <= 0:
            raise ValueError("CR setpoint must be > 0 ohm")
        if self.mode in ("CC", "CV") and self.setpoint < 0:
            raise ValueError("setpoint must be >= 0")

    @property
    def is_on(self) -> bool:
        return self.mode != "OFF"

    def current(self, v: float) -> float:
        if v <= 0.0:
            return 0.0
        m = self.mode
        if m == "CR":
            return v / self.setpoint
        if m == "CC":
            return self.setpoint * min(1.0, v / self.v_min)
        if m == "CV":
            return min(max((v - self.setpoint) / self.r_cv, 0.0), self.i_max)
        return 0.0

    def intersect(self, e: float, r: float) -> float:
        """Load voltage when fed from a Thevenin source of EMF e and resistance r."""
        if e <= 0.0:
            return 0.0
        m = self.mode
        if m == "OFF" or r <= 0.0:
            return e
        if m == "CR":
            return e * self.setpoint / (self.setpoint + r)
        if m == "CC":
            v = e - self.setpoint * r
            if v >= self.v_min:
                return v
            return e / (1.0 + self.setpoint * r / self.v_min)
        # CV
        vs = self.setpoint
        if e <= vs:
            return e
        v = (e * self.r_cv + vs * r) / (self.r_cv + r)
        if (v - vs) / self.r_cv > self.i_max:
            v = e - self.i_max * r
        return v

    def describe(self) -> str:
        unit = {"CR": "ohm", "CC": "A", "CV": "V", "OFF": ""}[self.mode]
        return "OFF" if self.mode == "OFF" else f"{self.mode} {self.setpoint:g} {unit}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ElectronicLoad":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})
