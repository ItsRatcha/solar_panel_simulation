"""Synthetic weather for one day: cloud cover and ambient temperature.

Clouds
    Clear-sky index kc(t) = GHI / GHI_clear on a 1 s grid, generated as a
    two-state (clear / cloudy) process with Poisson cloud arrivals and
    exponentially distributed cloud durations -- the structure of Morf's
    stochastic two-state solar irradiance model (Morf 1998, Solar Energy
    62(2):101-112; Morf 2011, Solar Energy 85:985-999).  Bright et al. (2015,
    Solar Energy 115:229-242) use the same Markov clear/cloudy idea at 1 min.
    Cloud edges are a linear ramp of `edge_s` seconds (transit of the cloud
    edge across the module).

Temperature
    Parton & Logan (1981, Agric. Meteorol. 23:205-216): truncated sine
    between sunrise and sunset, exponential decay at night, coefficients
    for air at 150 cm: a = 1.86 h (lag of Tmax after solar noon),
    b = 2.20 (night decay), c = -0.17 h (lag of Tmin after sunrise).
    Cloud cover reduces the diurnal temperature range mainly by lowering
    Tmax while Tmin is nearly unchanged (Dai, Trenberth & Karl 1999,
    J. Climate 12:2451-2473); the sky type scales Tmax - Tmin by `t_swing`.

ASSUMPTIONS (not taken from a paper; calibrate against site data):
    the per-sky numbers in SKY_PRESETS, the random-sky probabilities, and
    the default Tmin/Tmax (Bangkok-like dry season).
Everything is deterministic for a given seed.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

SKY_TYPES = ("clear", "partly_cloudy", "cloudy", "overcast")
RANDOM_SKY_P = (0.25, 0.4, 0.25, 0.1)  # ASSUMPTION

# Parton & Logan (1981), Table 1, air temperature at 150 cm
PL_A = 1.86  # lag of Tmax after solar noon [h]
PL_B = 2.20  # nighttime decay coefficient [-]
PL_C = -0.17  # lag of Tmin after sunrise [h]


@dataclass(frozen=True)
class SkyParams:
    base_kc: float  # clear-sky index in the "clear" state
    clouds_per_hour: float  # Poisson arrival rate of the cloudy state
    mean_duration_s: float  # mean of the exponential cloudy-state duration
    depth: tuple[float, float]  # uniform range of kc in the cloudy state
    edge_s: float  # cloud-edge ramp time
    t_swing: float  # fraction of the clear-day Tmax - Tmin (Dai et al. 1999)


SKY_PRESETS: dict[str, SkyParams] = {  # ASSUMPTION: values to be calibrated
    "clear": SkyParams(1.0, 0.0, 60.0, (0.9, 1.0), 10.0, 1.0),
    "partly_cloudy": SkyParams(1.0, 8.0, 120.0, (0.25, 0.6), 15.0, 0.9),
    "cloudy": SkyParams(0.85, 20.0, 300.0, (0.2, 0.5), 30.0, 0.7),
    "overcast": SkyParams(0.3, 6.0, 600.0, (0.15, 0.3), 120.0, 0.4),
}


@dataclass(frozen=True)
class WeatherConfig:
    sky: str = "partly_cloudy"  # one of SKY_TYPES, or "random"
    seed: int = 1
    t_min: float = 25.5  # daily minimum [degC]           ASSUMPTION (site)
    t_max: float = 34.0  # daily maximum, clear sky [degC] ASSUMPTION (site)
    # Linke turbidity for Ineichen-Perez clear sky; None = monthly Remund
    # (2003) climatology for the date (see irradiance.linke_turbidity)
    linke: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "WeatherConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class Weather:
    """Precomputed clear-sky index for one day (1 s grid) and air temperature."""

    def __init__(self, cfg: WeatherConfig, sunrise_h: float, sunset_h: float):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        sky = cfg.sky
        if sky == "random":
            sky = str(rng.choice(SKY_TYPES, p=list(RANDOM_SKY_P)))
        if sky not in SKY_PRESETS:
            raise ValueError(f"unknown sky type {cfg.sky!r}; choose from {SKY_TYPES} or 'random'")
        self.sky = sky
        self.params = sp = SKY_PRESETS[sky]
        self.sunrise_h, self.sunset_h = sunrise_h, sunset_h
        self._kc = self._make_kc(sp, rng)

    # ----------------------------------------------------------- generation
    @staticmethod
    def _smooth(x: np.ndarray, width: int) -> np.ndarray:
        if width <= 1:
            return x
        k = np.ones(width) / width
        pad = np.pad(x, (width, width), mode="edge")
        return np.convolve(pad, k, mode="same")[width:-width]

    def _make_kc(self, sp: SkyParams, rng: np.random.Generator) -> np.ndarray:
        """Two-state process: Poisson cloud arrivals, exponential durations (Morf 1998)."""
        n = 86400
        kc = np.full(n, sp.base_kc)
        if sp.clouds_per_hour > 0:
            t = 0.0
            rate = sp.clouds_per_hour / 3600.0
            while True:
                t += rng.exponential(1.0 / rate)
                if t >= n:
                    break
                dur = rng.exponential(sp.mean_duration_s)
                depth = rng.uniform(*sp.depth)
                a, b = int(t), min(int(t + dur) + 1, n)
                kc[a:b] = np.minimum(kc[a:b], depth)
        return np.clip(self._smooth(kc, int(sp.edge_s)), 0.0, None)

    # ------------------------------------------------------------- queries
    def kc(self, t_s: float) -> float:
        """Clear-sky index at t_s seconds after local midnight."""
        k = int(min(max(t_s, 0.0), 86399.0))
        return float(self._kc[k])

    def mean_kc(self, t0_s: float, t1_s: float) -> float:
        a, b = int(max(t0_s, 0)), int(min(t1_s, 86400))
        return float(self._kc[a:b].mean()) if b > a else 1.0

    def t_ambient(self, t_s: float) -> float:
        """Air temperature [degC] at t_s s after midnight (Parton & Logan 1981)."""
        c = self.cfg
        h = t_s / 3600.0
        rise, sset = self.sunrise_h, self.sunset_h
        t_min = c.t_min
        t_max = c.t_min + (c.t_max - c.t_min) * self.params.t_swing
        y = sset - rise  # day length [h]

        def day(m: float) -> float:  # m = hours after (sunrise + c); eq. 1, p. 206
            return t_min + (t_max - t_min) * math.sin(math.pi * m / (y + 2 * PL_A))

        if rise + PL_C < h < sset:
            return day(h - rise - PL_C)
        t_sn = day(sset - rise - PL_C)  # temperature at sunset
        n = h - sset if h >= sset else h + 24.0 - sset  # hours after sunset
        z = 24.0 - y  # night length Z [h]; eq. 2, p. 206
        return t_min + (t_sn - t_min) * math.exp(-PL_B * n / z)
