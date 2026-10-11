"""Synthetic weather for one day: cloud cover and ambient temperature.

Clouds are modelled as a clear-sky index kc(t) = GHI / GHI_clear sampled at
1 s, built from Poisson-arriving cloud passages (exponential duration,
random optical depth, smooth edges) on top of a slowly varying base level.
Ambient temperature follows a diurnal curve (minimum at sunrise, maximum
mid-afternoon) that is pulled down when the sky is cloudy.  Everything is
deterministic for a given seed.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

SKY_TYPES = ("clear", "partly_cloudy", "cloudy", "overcast")


@dataclass(frozen=True)
class SkyParams:
    base_kc: float  # clear-sky index between clouds
    clouds_per_hour: float
    mean_duration_s: float
    depth: tuple[float, float]  # range of kc inside a cloud
    edge_s: float  # edge ramp time
    t_swing: float  # fraction of the clear-day temperature swing


SKY_PRESETS: dict[str, SkyParams] = {
    "clear": SkyParams(1.0, 0.0, 60.0, (0.9, 1.0), 10.0, 1.0),
    "partly_cloudy": SkyParams(1.0, 8.0, 120.0, (0.25, 0.6), 15.0, 0.9),
    "cloudy": SkyParams(0.85, 20.0, 300.0, (0.2, 0.5), 30.0, 0.7),
    "overcast": SkyParams(0.3, 6.0, 600.0, (0.15, 0.3), 120.0, 0.4),
}


@dataclass(frozen=True)
class WeatherConfig:
    sky: str = "partly_cloudy"  # one of SKY_TYPES, or "random"
    seed: int = 1
    t_min: float = 25.5  # ambient minimum near sunrise [degC]
    t_max: float = 34.0  # ambient maximum on a clear day [degC]
    t_peak_hour: float = 14.5
    linke: float = 4.5

    def to_dict(self) -> dict:
        return asdict(self)


class Weather:
    """Precomputed clear-sky index and temperature for one day (1 s grid)."""

    def __init__(self, cfg: WeatherConfig, sunrise_h: float, sunset_h: float):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        sky = cfg.sky
        if sky == "random":
            sky = str(rng.choice(SKY_TYPES, p=[0.25, 0.4, 0.25, 0.1]))
        if sky not in SKY_PRESETS:
            raise ValueError(f"unknown sky type {cfg.sky!r}; choose from {SKY_TYPES} or 'random'")
        self.sky = sky
        self.params = sp = SKY_PRESETS[sky]
        self.sunrise_h, self.sunset_h = sunrise_h, sunset_h
        self._kc = self._make_kc(sp, rng)
        self._tnoise = self._make_noise(rng)

    # ----------------------------------------------------------- generation
    @staticmethod
    def _smooth(x: np.ndarray, width: int) -> np.ndarray:
        if width <= 1:
            return x
        k = np.ones(width) / width
        pad = np.pad(x, (width, width), mode="edge")
        return np.convolve(pad, k, mode="same")[width:-width]

    def _make_kc(self, sp: SkyParams, rng: np.random.Generator) -> np.ndarray:
        n = 86400
        # slowly wandering base level (+-5 % over ~20 min)
        walk = np.cumsum(rng.normal(0.0, 1.0, n // 600 + 2))
        walk = (walk - walk.mean()) / (np.abs(walk).max() + 1e-9) * 0.05
        base = np.interp(np.arange(n), np.arange(len(walk)) * 600, walk) + sp.base_kc
        kc = base.copy()
        if sp.clouds_per_hour > 0:
            t = 0.0
            rate = sp.clouds_per_hour / 3600.0
            while True:
                t += rng.exponential(1.0 / rate)
                if t >= n:
                    break
                dur = rng.exponential(sp.mean_duration_s) + 10.0
                depth = rng.uniform(*sp.depth)
                a, b = int(t), min(int(t + dur), n)
                kc[a:b] = np.minimum(kc[a:b], depth)
        kc = self._smooth(kc, int(sp.edge_s))
        # small fast flicker inside clouds
        kc = kc + rng.normal(0.0, 0.01, n) * (kc < sp.base_kc - 0.1)
        return np.clip(kc, 0.05, 1.1)

    @staticmethod
    def _make_noise(rng: np.random.Generator) -> np.ndarray:
        # mean-reverting AR(1) on a 1 min grid: ~0.5 degC std, ~20 min memory
        n, a, sd = 86400 // 60 + 2, math.exp(-1 / 20.0), 0.5
        e = rng.normal(0.0, sd * math.sqrt(1 - a * a), n)
        x = np.empty(n)
        x[0] = rng.normal(0.0, sd)
        for k in range(1, n):
            x[k] = a * x[k - 1] + e[k]
        return x

    # ------------------------------------------------------------- queries
    def kc(self, t_s: float) -> float:
        """Clear-sky index at t_s seconds after local midnight."""
        k = int(min(max(t_s, 0.0), 86399.0))
        return float(self._kc[k])

    def mean_kc(self, t0_s: float, t1_s: float) -> float:
        a, b = int(max(t0_s, 0)), int(min(t1_s, 86400))
        return float(self._kc[a:b].mean()) if b > a else 1.0

    def t_ambient(self, t_s: float) -> float:
        """Ambient temperature [degC] at t_s seconds after local midnight."""
        c = self.cfg
        h = t_s / 3600.0
        rise, peak = self.sunrise_h, c.t_peak_hour
        swing = (c.t_max - c.t_min) * self.params.t_swing
        if rise <= h <= peak:  # warming: half cosine from min to max
            x = (h - rise) / (peak - rise)
            frac = 0.5 - 0.5 * math.cos(math.pi * x)
        elif h > peak:  # cooling towards the next morning's minimum
            frac = math.exp(-(h - peak) / 5.0)
        else:  # before sunrise: tail of the previous evening
            frac = math.exp(-(h + 24 - peak) / 5.0)
        # a cloud passing now cools the air slightly
        frac *= 0.9 + 0.1 * self.kc(t_s) if rise <= h <= self.sunset_h else 1.0
        k = int(min(max(t_s / 60.0, 0), len(self._tnoise) - 1))
        return c.t_min + swing * frac + float(self._tnoise[k])
