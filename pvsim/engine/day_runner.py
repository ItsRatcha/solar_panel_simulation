"""Run a LiveChannel through a simulated day (sunrise to sunset).

Every MPPT step advances the clock by `channel.dt`.  The environment (sun,
clouds, temperature, shadows) and the mount orientation are refreshed every
`env_dt` seconds of simulated time, which keeps the cost of rebuilding I-V
curves bounded.  Energies are integrated as the day goes.

The run covers sunrise - margin to sunset + margin unless a window
(t_start, t_end in seconds after midnight) is given.  An optional load
profile (times in seconds after midnight) re-programs the e-load every step,
and an optional fault schedule perturbs the channel (see data.faults).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..environment.day import DayEnvironment, EnvSample
from ..environment.irradiance import POA
from ..load.profiles import LoadProfile
from .live import LiveChannel, Snapshot


@dataclass
class DayEnergy:
    pv_wh: float = 0.0  # energy taken from the panel
    mpp_wh: float = 0.0  # best possible (true MPP all day)
    out_wh: float = 0.0  # delivered to the load
    poa_wh_m2: float = 0.0  # plane-of-array insolation
    mount_wh: float = 0.0  # tracker motor + standby

    @property
    def mppt_eff(self) -> float:
        """Energy-based MPPT efficiency, integral(P_pv) / integral(P_mpp)
        (EN 50530:2010+A1:2013; IEC 62891:2020)."""
        return self.pv_wh / self.mpp_wh if self.mpp_wh > 1e-9 else 0.0

    @property
    def net_wh(self) -> float:
        return self.out_wh - self.mount_wh


@dataclass
class DaySnapshot:
    snap: Snapshot
    clock_s: float  # seconds after local midnight
    env: EnvSample
    poa: POA
    tilt: float
    azimuth: float
    energy: DayEnergy

    @property
    def clock_h(self) -> float:
        return self.clock_s / 3600.0


class DayRunner:
    def __init__(self, channel: LiveChannel, env: DayEnvironment, mount, env_dt: float = 1.0,
                 margin_s: float = 300.0, t_start: float | None = None, t_end: float | None = None,
                 load_profile: LoadProfile | None = None, faults=None):
        self.ch, self.env, self.mount = channel, env, mount
        self.env_dt = env_dt
        self.t_start = env.sunrise_s - margin_s if t_start is None else t_start
        self.t_end = env.sunset_s + margin_s if t_end is None else t_end
        if self.t_end <= self.t_start:
            raise ValueError("day window is empty (t_end <= t_start)")
        self.load_profile = load_profile
        self.faults = faults  # data.faults.FaultSchedule or None
        self.fault_label: str | None = None
        self.reset()

    def reset(self) -> None:
        self.ch.reset()
        self.mount.reset()
        self.clock = self.t_start
        self.energy = DayEnergy()
        self._next_env = self.clock
        self._env: EnvSample | None = None
        self._poa: POA | None = None
        self._refresh_env(force=True)

    @property
    def done(self) -> bool:
        return self.clock >= self.t_end

    @property
    def progress(self) -> float:
        return (self.clock - self.t_start) / (self.t_end - self.t_start)

    def _refresh_env(self, force: bool = False) -> None:
        if not force and self.clock < self._next_env:
            return
        dt = max(self.env_dt, self.ch.dt) if not force else 0.0
        s = self.env.sample(self.clock)
        tilt, az = self.mount.update(s.sun, dt)
        cond, poa = self.env.conditions(s, tilt, az)
        self.ch.cond = self._base_cond = cond
        self._env, self._poa = s, poa
        self._next_env = self.clock + self.env_dt

    def step(self) -> DaySnapshot:
        self._refresh_env()
        if self.faults is not None:
            self.fault_label = self.faults.apply(self.ch, self.mount, self._base_cond, self.clock)
        if self.load_profile is not None:
            self.ch.load = self.load_profile.at(self.clock)
        snap = self.ch.step()
        dt_h = self.ch.dt / 3600.0
        e = self.energy
        e.pv_wh += snap.p_pv * dt_h
        e.mpp_wh += snap.p_mpp * dt_h
        e.out_wh += snap.p_out * dt_h
        e.poa_wh_m2 += self._poa.total * dt_h
        e.mount_wh = self.mount.energy_wh
        self.clock += self.ch.dt
        tilt, az = self.mount.orientation
        return DaySnapshot(snap, self.clock, self._env, self._poa, tilt, az, e)
