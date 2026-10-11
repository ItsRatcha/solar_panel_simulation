"""Time-varying load profiles (setpoint schedules for the electronic load)."""

from __future__ import annotations

import bisect
import csv
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .electronic_load import ElectronicLoad


@dataclass(frozen=True)
class LoadPoint:
    """Load setting that applies from time t [s].

    interp="linear" ramps the setpoint from the previous point to this one
    (only when both points use the same mode); "step" switches at t.
    """

    t: float
    mode: str
    setpoint: float
    interp: str = "step"


@dataclass
class LoadProfile:
    points: list[LoadPoint]
    base: ElectronicLoad = field(default_factory=ElectronicLoad)  # carries v_min, r_cv, i_max

    def __post_init__(self):
        if not self.points:
            raise ValueError("load profile needs at least one point")
        self.points = sorted(self.points, key=lambda p: p.t)
        self._t = [p.t for p in self.points]

    def _make(self, mode: str, sp: float) -> ElectronicLoad:
        b = self.base
        return ElectronicLoad(mode, sp, b.v_min, b.r_cv, b.i_max)

    def at(self, t: float) -> ElectronicLoad:
        k = bisect.bisect_right(self._t, t) - 1
        if k < 0:
            p = self.points[0]
            return self._make(p.mode, p.setpoint)
        p = self.points[k]
        if k + 1 < len(self.points):
            q = self.points[k + 1]
            if q.interp == "linear" and q.mode == p.mode and q.t > p.t:
                f = (t - p.t) / (q.t - p.t)
                return self._make(p.mode, p.setpoint + f * (q.setpoint - p.setpoint))
        return self._make(p.mode, p.setpoint)

    # ------------------------------------------------------------ builders
    @classmethod
    def constant(cls, load: ElectronicLoad) -> "LoadProfile":
        return cls([LoadPoint(0.0, load.mode, load.setpoint)], base=load)

    @classmethod
    def step(cls, mode: str, before: float, after: float, t_step: float) -> "LoadProfile":
        return cls([LoadPoint(0.0, mode, before), LoadPoint(t_step, mode, after)])

    # ----------------------------------------------------------------- I/O
    def to_dict(self) -> dict:
        return {"points": [asdict(p) for p in self.points], "base": self.base.to_dict()}

    @classmethod
    def from_dict(cls, d: dict) -> "LoadProfile":
        base = ElectronicLoad.from_dict(d["base"]) if "base" in d else ElectronicLoad()
        return cls([LoadPoint(**p) for p in d["points"]], base=base)

    @classmethod
    def from_csv(cls, path: str | Path) -> "LoadProfile":
        """CSV with header t,mode,setpoint[,interp]."""
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        pts = [
            LoadPoint(float(r["t"]), r["mode"].strip().upper(), float(r["setpoint"]),
                      (r.get("interp") or "step").strip().lower())
            for r in rows
        ]
        return cls(pts)
