"""Scenario: everything needed to reproduce one simulated run.

A scenario is one day at one site with one or more panels.  All panels see
the same environment (sun, clouds, temperature, shadows) and step in
lockstep, so their rows interleave by timestamp in the export.  Every random
source (weather, sensors) is derived from `seed`, so a scenario plus its
seed fully determines the output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from ..converter import TOPOLOGIES, make_converter
from ..data.faults import FaultEvent, FaultSchedule, parse_faults
from ..engine.live import LiveChannel
from ..environment.day import SHADE_PRESETS, DayEnvironment, ShadeEvent
from ..environment.solar_position import Site
from ..environment.weather import WeatherConfig
from ..load.electronic_load import ElectronicLoad
from ..load.profiles import LoadProfile
from ..mppt import create_mppt
from ..pv.datasheet import PanelDatasheet
from ..sensors.ina226 import INA226, INA226Config
from ..tracking.mounts import make_mount

SCHEMA_VERSION = 1


@dataclass
class PanelSpec:
    panel_id: str = "fixed"
    mount: str = "fixed"  # "fixed" | "tracker"
    mount_params: dict = field(default_factory=dict)  # tilt/azimuth, or TrackerParams fields
    algorithm: str = "po"
    mppt_params: dict = field(default_factory=dict)
    datasheet: dict = field(default_factory=dict)  # PanelDatasheet overrides
    topology: str = "buck"  # "buck" | "boost"
    buck: dict = field(default_factory=dict)  # BuckParams overrides (used when topology == "buck")
    boost: dict = field(default_factory=dict)  # BoostParams overrides (used when topology == "boost")
    sensor: dict = field(default_factory=dict)  # INA226Config overrides (both sensors)
    load: dict = field(default_factory=lambda: {"mode": "CR", "setpoint": 3.0})
    load_profile: dict | None = None  # LoadProfile.to_dict(); times in s after midnight

    def to_dict(self) -> dict:
        return {
            "panel_id": self.panel_id, "mount": self.mount, "mount_params": dict(self.mount_params),
            "algorithm": self.algorithm, "mppt_params": dict(self.mppt_params),
            "datasheet": dict(self.datasheet), "topology": self.topology,
            "buck": dict(self.buck), "boost": dict(self.boost), "sensor": dict(self.sensor),
            "load": dict(self.load), "load_profile": self.load_profile,
        }

    def __post_init__(self):
        if self.topology not in TOPOLOGIES:
            raise ValueError(f"topology must be one of {TOPOLOGIES}, got {self.topology!r}")

    @property
    def converter_params(self) -> dict:
        """Parameter overrides of the active topology."""
        return self.boost if self.topology == "boost" else self.buck

    @classmethod
    def from_dict(cls, d: dict) -> "PanelSpec":
        known = cls.__dataclass_fields__
        unknown = set(d) - set(known)
        if unknown:
            raise ValueError(f"unknown panel fields: {sorted(unknown)}")
        return cls(**d)


@dataclass
class Scenario:
    name: str = "scenario"
    date: str = "2026-03-21"  # YYYY-MM-DD, local
    weather: WeatherConfig = field(default_factory=lambda: WeatherConfig(sky="clear"))
    shading: str | list = "none"  # SHADE_PRESETS name or list of ShadeEvent dicts
    faults: list = field(default_factory=list)  # FaultEvent dicts (data.faults)
    panels: list[PanelSpec] = field(default_factory=lambda: [PanelSpec()])
    mppt_period: float = 1.0  # [s]
    env_dt: float = 1.0  # [s] environment refresh
    margin_s: float = 300.0
    start_h: float | None = None  # optional window (local hours)
    end_h: float | None = None
    seed: int = 1
    latitude: float = Site.latitude
    longitude: float = Site.longitude
    tz_hours: float = Site.tz_hours
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self):
        if self.mppt_period <= 0:
            raise ValueError("mppt_period must be > 0")
        ids = [p.panel_id for p in self.panels]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError(f"panel_ids must be non-empty and unique, got {ids}")
        for e in self.fault_events():
            if e.panel_id is not None and e.panel_id not in ids:
                raise ValueError(f"fault {e.kind} targets unknown panel {e.panel_id!r}")

    # ------------------------------------------------------------- builders
    @property
    def day(self) -> date:
        return date.fromisoformat(self.date)

    def shade_events(self) -> list[ShadeEvent]:
        if isinstance(self.shading, str):
            if self.shading not in SHADE_PRESETS:
                raise ValueError(f"unknown shading preset {self.shading!r}; choose from {list(SHADE_PRESETS)}")
            return list(SHADE_PRESETS[self.shading])
        return [ShadeEvent(**{**e, "depth": tuple(e["depth"])}) for e in self.shading]

    def fault_events(self) -> list[FaultEvent]:
        return parse_faults(self.faults)

    def fault_schedule(self, k: int) -> FaultSchedule | None:
        pid = self.panels[k].panel_id
        events = [e for e in self.fault_events() if e.applies_to(pid)]
        return FaultSchedule(events) if events else None

    def site(self) -> Site:
        return Site(self.latitude, self.longitude, self.tz_hours)

    def environment(self) -> DayEnvironment:
        return DayEnvironment(self.day, self.weather, self.shade_events(), self.site())

    def sensor_seeds(self, k: int) -> tuple[int, int]:
        """Input/output INA226 seeds for panel k, derived from the scenario seed."""
        base = self.seed * 1000 + 10 * k
        return base + 1, base + 2

    def build_channel(self, k: int) -> tuple[LiveChannel, object, INA226, LoadProfile | None]:
        p = self.panels[k]
        s_in, s_out = self.sensor_seeds(k)
        cfg = INA226Config.from_dict(p.sensor)
        ch = LiveChannel(
            datasheet=PanelDatasheet.from_dict({**PanelDatasheet().to_dict(), **p.datasheet}),
            conv=make_converter(p.topology, p.converter_params),
            load=ElectronicLoad.from_dict(p.load),
            algo=create_mppt(p.algorithm, **p.mppt_params),
            sensor=INA226(cfg, seed=s_in),
            dt=self.mppt_period,
        )
        mount = make_mount(p.mount, **p.mount_params)
        profile = LoadProfile.from_dict(p.load_profile) if p.load_profile else None
        return ch, mount, INA226(cfg, seed=s_out), profile

    # ------------------------------------------------------------ serialise
    def to_dict(self) -> dict:
        shading = self.shading if isinstance(self.shading, str) else [
            {**e, "depth": list(e["depth"])} for e in self.shading]
        return {
            "schema_version": self.schema_version, "name": self.name, "date": self.date,
            "seed": self.seed, "mppt_period": self.mppt_period, "env_dt": self.env_dt,
            "margin_s": self.margin_s, "start_h": self.start_h, "end_h": self.end_h,
            "latitude": self.latitude, "longitude": self.longitude, "tz_hours": self.tz_hours,
            "weather": self.weather.to_dict(), "shading": shading,
            "faults": [e.to_dict() for e in self.fault_events()],
            "panels": [p.to_dict() for p in self.panels],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Scenario":
        d = dict(d)
        ver = d.pop("schema_version", SCHEMA_VERSION)
        if ver > SCHEMA_VERSION:
            raise ValueError(f"scenario schema_version {ver} is newer than supported ({SCHEMA_VERSION})")
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown scenario fields: {sorted(unknown)}")
        if "weather" in d:
            d["weather"] = WeatherConfig.from_dict(d["weather"])
        if "panels" in d:
            d["panels"] = [PanelSpec.from_dict(p) for p in d["panels"]]
        if "faults" in d:
            d["faults"] = [dict(e) for e in d["faults"]]
        if isinstance(d.get("shading"), list):
            d["shading"] = [dict(e) for e in d["shading"]]
        return cls(**d)
