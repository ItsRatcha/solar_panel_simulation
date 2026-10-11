"""Fault injection with labels for ML datasets.

A `FaultEvent` is active on one panel (or all panels) over a window of local
hours; while it is active the row's `fault_label` is the fault kind.  Each
kind perturbs one part of the channel:

    partial_shading_lvl1..3   extra shade on substrings (deeper per level)
    soiling                   uniform loss of irradiance on the whole panel
    bypass_diode_short        one substring's bypass diode shorted
    tracker_stuck             tracker motor stops (ignored, unlabeled, on fixed mounts)
    sensor_offset             offset on the PV-side INA226 (what the MPPT sees)
    open_circuit              panel disconnected from the converter

`FaultSchedule.apply` is called by the DayRunner every step, after the
environment refresh; it derives conditions from the unfaulted base, so the
perturbations never compound and are undone as soon as a window closes.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

from ..engine.live import Conditions, LiveChannel

NORMAL = "normal"

# default parameters per kind; a FaultEvent's params override these
FAULT_KINDS: dict[str, dict] = {
    "partial_shading_lvl1": {"depth": [0.0, 0.0, 0.3]},
    "partial_shading_lvl2": {"depth": [0.0, 0.0, 0.6]},
    "partial_shading_lvl3": {"depth": [0.0, 0.6, 0.9]},
    "soiling": {"loss": 0.2},
    "bypass_diode_short": {"substring": 0},
    "tracker_stuck": {},
    "sensor_offset": {"offset_v": 0.5, "offset_i": 0.1},
    "open_circuit": {},
}


@dataclass(frozen=True)
class FaultEvent:
    kind: str
    start_h: float
    end_h: float
    panel_id: str | None = None  # None: every panel
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in FAULT_KINDS:
            raise ValueError(f"unknown fault kind {self.kind!r}; choose from {list(FAULT_KINDS)}")
        if self.end_h <= self.start_h:
            raise ValueError(f"fault {self.kind}: end_h must be > start_h")
        unknown = set(self.params) - set(FAULT_KINDS[self.kind])
        if unknown:
            raise ValueError(f"fault {self.kind}: unknown params {sorted(unknown)}")

    @property
    def label(self) -> str:
        return self.kind

    def p(self, key: str):
        return self.params.get(key, FAULT_KINDS[self.kind][key])

    def active(self, t_s: float) -> bool:
        return self.start_h * 3600.0 <= t_s < self.end_h * 3600.0

    def applies_to(self, panel_id: str) -> bool:
        return self.panel_id is None or self.panel_id == panel_id

    def to_dict(self) -> dict:
        return {"kind": self.kind, "start_h": self.start_h, "end_h": self.end_h,
                "panel_id": self.panel_id, "params": dict(self.params)}

    @classmethod
    def from_dict(cls, d: dict) -> "FaultEvent":
        unknown = set(d) - {"kind", "start_h", "end_h", "panel_id", "params"}
        if unknown:
            raise ValueError(f"unknown fault fields: {sorted(unknown)}")
        return cls(**{**d, "params": dict(d.get("params") or {})})


def _combine(a: float, b: float) -> float:
    """Two independent blocked fractions."""
    return 1.0 - (1.0 - a) * (1.0 - b)


class FaultSchedule:
    """The faults of one panel; owns the channel/mount state they touch."""

    def __init__(self, events: list[FaultEvent]):
        self.events = list(events)
        self._sensor_cfg = None  # unfaulted PV-side sensor config
        self.label: str | None = None

    def active(self, t_s: float) -> list[FaultEvent]:
        return [e for e in self.events if e.active(t_s)]

    def apply(self, ch: LiveChannel, mount, base: Conditions, t_s: float) -> str | None:
        """Set channel/mount state for time t_s; returns the fault label or None."""
        if self._sensor_cfg is None:
            self._sensor_cfg = ch.sensor.cfg
        act = [e for e in self.active(t_s)  # a fixed mount cannot get stuck
               if e.kind != "tracker_stuck" or hasattr(mount, "stuck")]
        shade = list(base.shade)
        shorted = [False] * len(shade)
        stuck = open_c = False
        cfg = self._sensor_cfg
        for e in act:
            k = e.kind
            if k.startswith("partial_shading"):
                for i, d in enumerate(e.p("depth")[: len(shade)]):
                    shade[i] = _combine(shade[i], float(d))
            elif k == "soiling":
                shade = [_combine(s, float(e.p("loss"))) for s in shade]
            elif k == "bypass_diode_short":
                shorted[int(e.p("substring")) % len(shorted)] = True
            elif k == "tracker_stuck":
                stuck = True
            elif k == "sensor_offset":
                cfg = dataclasses.replace(cfg, offset_v=cfg.offset_v + float(e.p("offset_v")),
                                          offset_i=cfg.offset_i + float(e.p("offset_i")))
            elif k == "open_circuit":
                open_c = True
        ch.cond = Conditions(g=base.g, shade=tuple(round(s, 3) for s in shade), t_amb=base.t_amb)
        ch.shorted = tuple(shorted) if any(shorted) else None
        ch.open_circuit = open_c
        ch.sensor.cfg = cfg
        if hasattr(mount, "stuck"):
            mount.stuck = stuck
        # latest-starting active fault names the row
        self.label = max(act, key=lambda e: e.start_h).label if act else None
        return self.label


def parse_faults(items: list[dict]) -> list[FaultEvent]:
    return [FaultEvent.from_dict(d) for d in items]
