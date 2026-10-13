"""Built-in scenarios (also written to presets/*.json by `pvsim presets --write`)."""

from __future__ import annotations

from typing import Callable

from ..environment.weather import WeatherConfig
from .model import PanelSpec, Scenario

NOON_S = 12 * 3600.0


def clear_day_fixed_vs_tracker() -> Scenario:
    """Full clear day: fixed 14 deg south panel vs 2-axis tracker, both P&O."""
    return Scenario(
        name="clear_day_fixed_vs_tracker",
        weather=WeatherConfig(sky="clear", seed=1),
        panels=[PanelSpec("fixed", mount="fixed", algorithm="po"),
                PanelSpec("tracker", mount="tracker", algorithm="po")],
    )


def partial_shading() -> Scenario:
    """Midday shadow over one substring: plain P&O stays on the high-voltage
    local peak while the global-scan variant moves to the global MPP."""
    return Scenario(
        name="partial_shading",
        weather=WeatherConfig(sky="clear", seed=1),
        shading=[{"start_h": 11.5, "end_h": 12.5, "depth": [0.0, 0.0, 0.7], "ramp_s": 600.0,
                  "label": "partial_shading"}],
        start_h=11.0, end_h=13.0,
        panels=[PanelSpec("po", mount="fixed", algorithm="po"),
                PanelSpec("global_scan", mount="fixed", algorithm="global_scan",
                          mppt_params={"scan_period": 600})],
    )


def cloudy_day() -> Scenario:
    """Broken-cloud day: fast irradiance ramps stress the tracking dynamics."""
    return Scenario(
        name="cloudy_day",
        weather=WeatherConfig(sky="cloudy", seed=7),
        panels=[PanelSpec("po", mount="fixed", algorithm="po"),
                PanelSpec("inccond_var", mount="fixed", algorithm="inccond_variable")],
    )


def load_step() -> Scenario:
    """CR load steps 3 -> 8 -> 2 ohm around noon.  At 8 ohm the buck cannot
    present R_mpp (R_in >= R_load), so mpp_unreachable is set for that hour."""
    profile = {"points": [
        {"t": 0.0, "mode": "CR", "setpoint": 3.0, "interp": "step"},
        {"t": NOON_S - 1800.0, "mode": "CR", "setpoint": 8.0, "interp": "step"},
        {"t": NOON_S + 1800.0, "mode": "CR", "setpoint": 2.0, "interp": "step"},
    ]}
    return Scenario(
        name="load_step",
        weather=WeatherConfig(sky="clear", seed=1),
        start_h=11.0, end_h=13.0,
        panels=[PanelSpec("po", mount="fixed", algorithm="po", load_profile=profile)],
    )


def buck_vs_boost() -> Scenario:
    """Same panel and 20 ohm load behind a buck and a boost.  A buck only raises
    the reflected resistance (R_in >= R_load), so it stays off the MPP and is
    flagged mpp_unreachable; a boost lowers it (R_in = R_load (1-D)^2) and tracks."""
    load = {"mode": "CR", "setpoint": 20.0}
    return Scenario(
        name="buck_vs_boost",
        weather=WeatherConfig(sky="clear", seed=1),
        start_h=10.0, end_h=14.0,
        panels=[PanelSpec("buck", mount="fixed", algorithm="po", topology="buck", load=dict(load)),
                PanelSpec("boost", mount="fixed", algorithm="po", topology="boost", load=dict(load))],
    )


PRESETS: dict[str, Callable[[], Scenario]] = {
    "clear_day_fixed_vs_tracker": clear_day_fixed_vs_tracker,
    "partial_shading": partial_shading,
    "cloudy_day": cloudy_day,
    "load_step": load_step,
    "buck_vs_boost": buck_vs_boost,
}


def get_preset(name: str) -> Scenario:
    try:
        return PRESETS[name]()
    except KeyError:
        raise ValueError(f"unknown preset {name!r}; choose from {list(PRESETS)}") from None
