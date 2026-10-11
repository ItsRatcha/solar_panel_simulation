"""One simulated day: sun, sky, temperature and shadows as functions of time.

`DayEnvironment.sample(t)` gives everything the panel needs at t seconds
after local midnight; `conditions(sample, tilt, az)` turns it into the
per-substring irradiance seen by a panel with that orientation.

Shading events model objects (tree, pole, bird) that block the *beam* on
some substrings; diffuse light still reaches the shaded cells, so the
effective blocked fraction is depth * beam / POA.
"""

from __future__ import annotations

import math

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta

from ..engine.live import Conditions
from .irradiance import POA, Irradiance, clearsky_ineichen, poa_hay_davies
from .solar_position import Site, SunPosition, sun_position, sunrise_sunset
from .weather import Weather, WeatherConfig


@dataclass(frozen=True)
class ShadeEvent:
    """Shadow covering substrings from start_h to end_h (local hours)."""

    start_h: float
    end_h: float
    depth: tuple[float, ...]  # blocked beam fraction per substring at full shadow
    ramp_s: float = 300.0  # shadow edge sweeps in/out over this time
    label: str = "partial_shading"

    def factor(self, t_s: float) -> float:
        a, b = self.start_h * 3600.0, self.end_h * 3600.0
        if t_s <= a or t_s >= b:
            return 0.0
        r = max(min(self.ramp_s, (b - a) / 2), 1e-9)
        return min(1.0, (t_s - a) / r, (b - t_s) / r)

    def to_dict(self) -> dict:
        return asdict(self)


SHADE_PRESETS: dict[str, list[ShadeEvent]] = {
    "none": [],
    "tree_morning": [ShadeEvent(7.0, 9.0, (0.8, 0.3, 0.0), 900.0, "partial_shading_tree")],
    "pole_noon": [ShadeEvent(11.5, 12.5, (0.0, 0.7, 0.0), 300.0, "partial_shading_pole")],
    "building_evening": [ShadeEvent(16.0, 18.5, (0.0, 0.6, 0.95), 1200.0, "partial_shading_building")],
}


@dataclass(frozen=True)
class EnvSample:
    t: float  # seconds after local midnight
    when: datetime
    sun: SunPosition
    irr: Irradiance  # actual GHI/DNI/DHI
    clear: Irradiance  # clear-sky GHI/DNI/DHI
    kc: float
    t_amb: float
    shade_depth: tuple[float, ...]
    shade_label: str


@dataclass
class DayEnvironment:
    day: date = field(default_factory=lambda: date(2026, 3, 21))
    weather: WeatherConfig = field(default_factory=WeatherConfig)
    shading: list[ShadeEvent] = field(default_factory=list)
    site: Site = field(default_factory=Site)
    albedo: float = 0.2
    n_sub: int = 3

    def __post_init__(self):
        rise, sset = sunrise_sunset(self.day, self.site)
        self.midnight = datetime(self.day.year, self.day.month, self.day.day)
        self.sunrise_s = (rise - self.midnight).total_seconds()
        self.sunset_s = (sset - self.midnight).total_seconds()
        self.wx = Weather(self.weather, self.sunrise_s / 3600, self.sunset_s / 3600)
        self.doy = self.day.timetuple().tm_yday

    @property
    def sky(self) -> str:
        return self.wx.sky

    def sample(self, t: float) -> EnvSample:
        when = self.midnight + timedelta(seconds=t)
        sun = sun_position(when, self.site)
        clear = clearsky_ineichen(sun.zenith, self.doy, self.weather.linke, self.site.altitude_m)
        kc = self.wx.kc(t)
        # Beam is blocked faster than diffuse: DNI scales with a steep function
        # of kc, and DHI makes up the rest of GHI (continuous in kc).
        ghi = clear.ghi * kc
        fb = min(max((kc - 0.25) / 0.75, 0.0), 1.2) ** 1.5
        dni = clear.dni * fb
        cos_z = max(math.cos(math.radians(sun.zenith)), 0.0)
        dni = min(dni, ghi / cos_z) if cos_z > 0.01 else 0.0
        irr = Irradiance(ghi, dni, max(ghi - dni * cos_z, 0.0))
        depth = [0.0] * self.n_sub
        label = "normal"
        for ev in self.shading:
            f = ev.factor(t)
            if f > 0:
                label = ev.label
                for k, d in enumerate(ev.depth[: self.n_sub]):
                    depth[k] = max(depth[k], d * f)
        return EnvSample(t, when, sun, irr, clear, kc, self.wx.t_ambient(t), tuple(depth), label)

    def poa(self, s: EnvSample, tilt: float, azimuth: float) -> POA:
        return poa_hay_davies(s.irr, tilt, azimuth, s.sun.zenith, s.sun.azimuth, self.doy, self.albedo)

    def conditions(self, s: EnvSample, tilt: float, azimuth: float,
                   g_step: float = 0.5, t_step: float = 0.05) -> tuple[Conditions, POA]:
        """Panel conditions; G and T are quantized so the I-V cache gets hits."""
        poa = self.poa(s, tilt, azimuth)
        g = round(poa.total / g_step) * g_step if poa.total > 0 else 0.0
        beam_frac = poa.beam / poa.total if poa.total > 1e-9 else 0.0
        shade = tuple(round(d * beam_frac, 3) for d in s.shade_depth)
        t_amb = round(s.t_amb / t_step) * t_step
        return Conditions(g=g, shade=shade, t_amb=t_amb), poa
