"""Run a Scenario headless: all panels in lockstep, rows into a Recorder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from ..data.recorder import Recorder
from ..engine.day_runner import DayEnergy, DayRunner
from .model import Scenario


@dataclass
class RunResult:
    scenario: Scenario
    frame: pd.DataFrame  # internal column keys (see data.schema)
    energy: dict[str, DayEnergy]  # per panel_id
    sky: str

    def summary(self) -> str:
        lines = [f"{self.scenario.name}  {self.scenario.date}  sky={self.sky}  rows={len(self.frame)}",
                 f"{'panel':<14}{'algo':<18}{'mpp Wh':>9}{'pv Wh':>9}{'out Wh':>9}{'mount Wh':>10}"
                 f"{'net Wh':>9}{'mppt %':>8}"]
        for p in self.scenario.panels:
            e = self.energy[p.panel_id]
            lines.append(f"{p.panel_id:<14}{p.algorithm:<18}{e.mpp_wh:9.2f}{e.pv_wh:9.2f}{e.out_wh:9.2f}"
                         f"{e.mount_wh:10.2f}{e.net_wh:9.2f}{100 * e.mppt_eff:8.2f}")
        return "\n".join(lines)


def run_scenario(sc: Scenario, extended: bool = False,
                 progress: Callable[[float], None] | None = None) -> RunResult:
    env = sc.environment()
    t_start = sc.start_h * 3600.0 if sc.start_h is not None else None
    t_end = sc.end_h * 3600.0 if sc.end_h is not None else None
    runners, out_sensors = [], []
    for k in range(len(sc.panels)):
        ch, mount, out_sensor, profile = sc.build_channel(k)
        runners.append(DayRunner(ch, env, mount, env_dt=sc.env_dt, margin_s=sc.margin_s,
                                 t_start=t_start, t_end=t_end, load_profile=profile,
                                 faults=sc.fault_schedule(k)))
        out_sensors.append(out_sensor)
    rec = Recorder(env.midnight, sc.tz_hours, extended=extended)
    ids = [p.panel_id for p in sc.panels]
    lead = runners[0]
    n = 0
    while not lead.done:
        for pid, r, so in zip(ids, runners, out_sensors):
            ds = r.step()
            rec.record(pid, r.ch, ds, so, label=r.fault_label)
        n += 1
        if progress is not None and n % 500 == 0:
            progress(lead.progress)
    if progress is not None:
        progress(1.0)
    return RunResult(sc, rec.frame(), {pid: r.energy for pid, r in zip(ids, runners)}, env.sky)
