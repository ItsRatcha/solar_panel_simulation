"""Panel mounts: fixed tilt and a 2-axis sun tracker.

Orientation is (tilt, azimuth): tilt from horizontal [deg], azimuth of the
surface normal clockwise from north [deg].  The tracker aims its normal at
the sun, but only moves when the pointing error exceeds a deadband, moves at
a finite angular speed per axis, and returns to a stow position at night.
Its energy use is motor power while moving plus controller standby power.

The pointing error is the angle of incidence of the beam on the panel
normal (Duffie & Beckman, Solar Engineering of Thermal Processes, eq. 1.6.3).
A 2-axis tracker with zero pointing error maximises beam on the plane
(Duffie & Beckman sec. 1.7).
ASSUMPTION: every TrackerParams value (deadband, speed, tilt limit, motor
and standby power, stow) is a design placeholder for the prototype tracker.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from ..environment.solar_position import SunPosition


def pointing_error(tilt: float, az: float, zenith: float, sun_az: float) -> float:
    """Angle between the panel normal and the sun direction [deg]."""
    t, z = math.radians(tilt), math.radians(zenith)
    c = math.cos(z) * math.cos(t) + math.sin(z) * math.sin(t) * math.cos(math.radians(sun_az - az))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _wrap180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


class FixedMount:
    kind = "fixed"

    def __init__(self, tilt: float = 14.0, azimuth: float = 180.0):
        self.tilt, self.azimuth = tilt, azimuth
        self.energy_wh = 0.0

    def reset(self) -> None:
        self.energy_wh = 0.0

    def update(self, sun: SunPosition, dt: float) -> tuple[float, float]:
        return self.tilt, self.azimuth

    @property
    def orientation(self) -> tuple[float, float]:
        return self.tilt, self.azimuth

    @property
    def moving(self) -> bool:
        return False


@dataclass
class TrackerParams:
    deadband: float = 2.0  # [deg] pointing error that triggers a move
    speed: float = 0.5  # [deg/s] per axis
    tilt_max: float = 80.0
    p_motor: float = 4.0  # [W] while any axis moves
    p_standby: float = 0.3  # [W] controller + sensors, always on
    min_elevation: float = 3.0  # below this the tracker stows
    stow_tilt: float = 0.0
    stow_azimuth: float = 90.0  # face east, ready for sunrise

    def to_dict(self) -> dict:
        return asdict(self)


class TwoAxisTracker:
    kind = "two_axis"

    def __init__(self, params: TrackerParams | None = None):
        self.p = params or TrackerParams()
        self.reset()

    def reset(self) -> None:
        self.tilt, self.azimuth = self.p.stow_tilt, self.p.stow_azimuth
        self.energy_wh = 0.0
        self.motor_s = 0.0
        self._target: tuple[float, float] | None = None
        self.stuck = False  # fault injection hook

    @property
    def orientation(self) -> tuple[float, float]:
        return self.tilt, self.azimuth

    @property
    def moving(self) -> bool:
        return self._target is not None

    def update(self, sun: SunPosition, dt: float) -> tuple[float, float]:
        p = self.p
        self.energy_wh += p.p_standby * dt / 3600.0
        if self.stuck:
            return self.orientation
        if sun.elevation < p.min_elevation:
            target = (p.stow_tilt, p.stow_azimuth)
            if (abs(self.tilt - target[0]) > 0.05 or abs(_wrap180(self.azimuth - target[1])) > 0.05):
                self._target = target
        else:
            err = pointing_error(self.tilt, self.azimuth, sun.zenith, sun.azimuth)
            if err > p.deadband:
                self._target = (min(sun.zenith, p.tilt_max), sun.azimuth)
            elif self._target is not None:  # keep following the sun while moving
                self._target = (min(sun.zenith, p.tilt_max), sun.azimuth)
        if self._target is None:
            return self.orientation
        step = p.speed * dt
        dt_ = self._target[0] - self.tilt
        da = _wrap180(self._target[1] - self.azimuth)
        # azimuth is irrelevant when the panel is flat: move it for free-ish
        self.tilt += max(-step, min(step, dt_))
        self.azimuth = (self.azimuth + max(-step, min(step, da))) % 360.0
        move_time = min(max(abs(dt_), abs(da)) / p.speed, dt) if p.speed > 0 else 0.0
        self.motor_s += move_time
        self.energy_wh += p.p_motor * move_time / 3600.0
        if abs(dt_) <= step and abs(da) <= step:
            self._target = None
        return self.orientation


def make_mount(kind: str, **kw):
    if kind == "fixed":
        return FixedMount(**kw)
    if kind in ("two_axis", "tracker"):
        return TwoAxisTracker(TrackerParams(**kw) if kw else None)
    raise ValueError(f"unknown mount {kind!r}")
