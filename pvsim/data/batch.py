"""Labeled dataset generator: N randomized days, one CSV per day + manifest.

Every random choice for day `i` comes from `default_rng([seed, i])`, so a day
does not depend on how many days are generated or which worker runs it, and
the manifest carries no timestamps: re-running with the same arguments gives
byte-identical files.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from ..environment.weather import SKY_TYPES, WeatherConfig
from ..scenario.model import PanelSpec, Scenario
from .export import write_csv
from .faults import FAULT_KINDS

SKY_WEIGHTS = (0.35, 0.30, 0.20, 0.15)  # same order as SKY_TYPES
PANELS = (("fixed", "fixed"), ("tracker", "tracker"))
MAX_FAULTS = 2


def _r(x: float, nd: int = 3) -> float:
    return round(float(x), nd)


def make_day_scenario(index: int, seed: int, year: int = 2026, period: float = 1.0) -> Scenario:
    """Day `index` of a batch: random date, sky, temperatures, load and faults."""
    rng = np.random.default_rng([seed, index])
    day = date(year, 1, 1) + timedelta(days=int(rng.integers(0, 365)))
    sky = str(rng.choice(SKY_TYPES, p=SKY_WEIGHTS))
    t_min = _r(rng.uniform(22.0, 27.0), 2)
    t_max = _r(t_min + rng.uniform(6.0, 10.0), 2)
    w_seed = int(rng.integers(0, 2**31 - 1))
    r_load = _r(rng.uniform(2.0, 4.0), 2)

    faults = []
    for _ in range(int(rng.integers(0, MAX_FAULTS + 1))):
        kind = str(rng.choice(list(FAULT_KINDS)))
        start = _r(rng.uniform(8.0, 15.5), 2)
        end = _r(start + rng.uniform(0.5, 3.0), 2)
        pid = "tracker" if kind == "tracker_stuck" else PANELS[int(rng.integers(0, len(PANELS)))][0]
        faults.append({"kind": kind, "start_h": start, "end_h": end, "panel_id": pid, "params": {}})

    return Scenario(
        name=f"batch_s{seed}_d{index:03d}",
        date=day.isoformat(),
        weather=WeatherConfig(sky=sky, seed=w_seed, t_min=t_min, t_max=t_max),
        faults=faults,
        panels=[PanelSpec(pid, mount=m, algorithm="po", load={"mode": "CR", "setpoint": r_load})
                for pid, m in PANELS],
        mppt_period=period,
        seed=seed * 100_000 + index,
    )


def _run_day(job: tuple) -> dict:
    """Worker: simulate one day and write its CSV (top-level so it pickles under spawn)."""
    from ..scenario.runner import run_scenario  # keep worker import light until needed

    index, seed, out_dir, rule, period, extended, window = job
    sc = make_day_scenario(index, seed, period=period)
    if window is not None:
        sc.start_h, sc.end_h = window
    res = run_scenario(sc, extended=extended)
    fname = f"day_{index:03d}.csv"
    write_csv(res.frame, Path(out_dir) / fname, rule)
    labels = res.frame.fault_label.value_counts()
    return {
        "index": index, "file": fname, "sky": res.sky, "scenario": sc.to_dict(),
        "rows_raw": int(len(res.frame)),
        "label_counts_raw": {str(k): int(v) for k, v in sorted(labels.items())},
        "energy_wh": {pid: {"mpp": _r(e.mpp_wh, 4), "pv": _r(e.pv_wh, 4), "out": _r(e.out_wh, 4),
                            "mount": _r(e.mount_wh, 4)} for pid, e in res.energy.items()},
    }


def run_batch(days: int, seed: int, out_dir: str | Path, resample: str = "1min",
              period: float = 1.0, extended: bool = False, workers: int = 1,
              window: tuple[float, float] | None = None, progress=None) -> Path:
    """Generate `days` labeled CSVs plus manifest.json in out_dir; returns the manifest path."""
    if days < 1:
        raise ValueError("days must be >= 1")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(i, seed, str(out), resample, period, extended, window) for i in range(days)]
    results: list[dict] = []
    if workers <= 1:
        for j in jobs:
            results.append(_run_day(j))
            if progress is not None:
                progress(len(results) / days)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for r in ex.map(_run_day, jobs):  # map keeps job order
                results.append(r)
                if progress is not None:
                    progress(len(results) / days)

    manifest = {
        "generator": "pvsim batch",
        "config": {"days": days, "seed": seed, "resample": resample, "period": period,
                   "extended": extended, "window": list(window) if window else None,
                   "sky_weights": dict(zip(SKY_TYPES, SKY_WEIGHTS)), "fault_kinds": list(FAULT_KINDS)},
        "days": results,
    }
    path = out / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    return path
