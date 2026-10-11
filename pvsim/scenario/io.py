"""Scenario JSON files."""

from __future__ import annotations

import json
from pathlib import Path

from .model import Scenario


def save_scenario(sc: Scenario, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sc.to_dict(), indent=2) + "\n", encoding="utf-8", newline="\n")
    return path


def load_scenario(path: str | Path) -> Scenario:
    return Scenario.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
