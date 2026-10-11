"""Command line interface.

    pvsim run SCENARIO.json|PRESET --out run.csv [--extended] [--resample 1min] [--period 1] [--seed 3]
    pvsim presets [--write presets/]
    pvsim schema [--extended]
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

from .data.export import RESAMPLE_CHOICES, write_csv
from .data.schema import describe
from .scenario.io import load_scenario, save_scenario
from .scenario.model import Scenario
from .scenario.presets import PRESETS, get_preset
from .scenario.runner import run_scenario


def resolve_scenario(ref: str) -> Scenario:
    """A JSON path, or the name of a built-in preset."""
    p = Path(ref)
    if p.suffix.lower() == ".json" or p.exists():
        return load_scenario(p)
    return get_preset(ref)


def _progress(frac: float) -> None:
    sys.stderr.write(f"\r  {100 * frac:5.1f} %")
    if frac >= 1.0:
        sys.stderr.write("\n")
    sys.stderr.flush()


def cmd_run(a: argparse.Namespace) -> int:
    sc = resolve_scenario(a.scenario)
    if a.period is not None:
        sc.mppt_period = a.period
    if a.seed is not None:  # overrides both the scenario and the weather seed
        sc.seed = a.seed
        sc.weather = dataclasses.replace(sc.weather, seed=a.seed)
    res = run_scenario(sc, extended=a.extended, progress=None if a.quiet else _progress)
    print(res.summary())
    if a.out:
        path = write_csv(res.frame, a.out, a.resample)
        print(f"wrote {path}")
    if a.save_scenario:
        print(f"wrote {save_scenario(sc, a.save_scenario)}")
    return 0


def cmd_presets(a: argparse.Namespace) -> int:
    for name, make in PRESETS.items():
        doc = " ".join((make.__doc__ or "").strip().split("\n\n")[0].split())
        print(f"{name:<28}{doc}")
        if a.write:
            save_scenario(make(), Path(a.write) / f"{name}.json")
    if a.write:
        print(f"wrote {len(PRESETS)} presets to {a.write}")
    return 0


def cmd_schema(a: argparse.Namespace) -> int:
    print(describe(a.extended))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="pvsim", description="PV + buck + MPPT simulator")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run one scenario headless and export CSV")
    r.add_argument("scenario", help="scenario JSON file or preset name")
    r.add_argument("--out", "-o", help="CSV output path")
    r.add_argument("--extended", action="store_true", help="append ground-truth columns")
    r.add_argument("--resample", choices=RESAMPLE_CHOICES, default="raw")
    r.add_argument("--period", type=float, help="MPPT period [s] (overrides the scenario)")
    r.add_argument("--seed", type=int, help="seed for weather and sensors (overrides the scenario)")
    r.add_argument("--save-scenario", help="also write the effective scenario JSON here")
    r.add_argument("--quiet", "-q", action="store_true", help="no progress output")
    r.set_defaults(func=cmd_run)

    p = sub.add_parser("presets", help="list built-in scenarios")
    p.add_argument("--write", metavar="DIR", help="write each preset as JSON into DIR")
    p.set_defaults(func=cmd_presets)

    s = sub.add_parser("schema", help="print the CSV column schema")
    s.add_argument("--extended", action="store_true")
    s.set_defaults(func=cmd_schema)
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        return a.func(a)
    except (ValueError, FileNotFoundError) as e:
        print(f"pvsim: error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
