from datetime import date, datetime

import pytest

from pvsim.environment.day import DayEnvironment, ShadeEvent
from pvsim.environment.irradiance import (Irradiance, clearsky_ineichen, erbs, linke_turbidity,
                                         poa_hay_davies)
from pvsim.environment.solar_position import Site, solar_noon, sun_position, sunrise_sunset
from pvsim.environment.weather import PL_A, PL_C, SKY_TYPES, Weather, WeatherConfig

SITE = Site()


def _hours(dt: datetime) -> float:
    return dt.hour + dt.minute / 60 + dt.second / 3600


def test_sun_position_equinox_noon_pathum_thani():
    noon = solar_noon(date(2026, 3, 21), SITE)
    s = sun_position(noon, SITE)
    # at the equinox the noon zenith is close to the latitude
    assert s.zenith == pytest.approx(14.07, abs=0.6)
    assert 11.8 < _hours(noon) < 12.5
    assert s.is_up


def test_sunrise_sunset_plausible():
    rise, sset = sunrise_sunset(date(2026, 3, 21), SITE)
    assert 6.1 < _hours(rise) < 6.6
    assert 18.2 < _hours(sset) < 18.7


@pytest.mark.parametrize("when", [datetime(2026, 3, 21, 8, 0), datetime(2026, 6, 21, 12, 30),
                                  datetime(2026, 12, 21, 16, 45), datetime(2026, 9, 1, 10, 15)])
def test_sun_position_matches_pvlib(when):
    pvlib = pytest.importorskip("pvlib")
    import pandas as pd

    t = pd.DatetimeIndex([pd.Timestamp(when).tz_localize("Etc/GMT-7")])
    ref = pvlib.solarposition.get_solarposition(t, SITE.latitude, SITE.longitude, altitude=SITE.altitude_m)
    s = sun_position(when, SITE)
    assert s.zenith == pytest.approx(float(ref["apparent_zenith"].iloc[0]), abs=0.1) or \
        s.zenith == pytest.approx(float(ref["zenith"].iloc[0]), abs=0.1)
    assert s.azimuth == pytest.approx(float(ref["azimuth"].iloc[0]), abs=0.1)


def test_clearsky_plausible():
    cs = clearsky_ineichen(15.0, 80, linke=4.5)
    assert 850 < cs.ghi < 1050
    assert 600 < cs.dni < 950
    assert cs.ghi == pytest.approx(cs.dhi + cs.dni * 0.9659258, rel=1e-6)
    assert clearsky_ineichen(95.0, 80).ghi == 0.0


def test_erbs_conserves_ghi():
    for ghi in (50, 300, 700, 950):
        r = erbs(ghi, 30.0, 80)
        assert r.dhi + r.dni * 0.8660254 == pytest.approx(ghi, rel=1e-6)


def test_poa_horizontal_equals_ghi():
    irr = Irradiance(800.0, 700.0, 800.0 - 700.0 * 0.8660254)
    p = poa_hay_davies(irr, 0.0, 180.0, 30.0, 120.0, 80)
    assert p.total == pytest.approx(800.0, rel=1e-6)


def test_weather_deterministic_and_seed_sensitive():
    a = Weather(WeatherConfig(sky="partly_cloudy", seed=5), 6.3, 18.5)
    b = Weather(WeatherConfig(sky="partly_cloudy", seed=5), 6.3, 18.5)
    c = Weather(WeatherConfig(sky="partly_cloudy", seed=6), 6.3, 18.5)
    ts = range(0, 86400, 97)
    assert [a.kc(t) for t in ts] == [b.kc(t) for t in ts]
    assert [a.kc(t) for t in ts] != [c.kc(t) for t in ts]


def test_sky_types_ordered_by_brightness():
    means = [Weather(WeatherConfig(sky=s, seed=2), 6.3, 18.5).mean_kc(6.5 * 3600, 18 * 3600)
             for s in SKY_TYPES]
    assert means == sorted(means, reverse=True)
    assert Weather(WeatherConfig(sky="random", seed=2), 6.3, 18.5).sky in SKY_TYPES
    with pytest.raises(ValueError):
        Weather(WeatherConfig(sky="foggy"), 6.3, 18.5)


def test_ambient_temperature_diurnal():
    w = Weather(WeatherConfig(sky="clear", seed=1), 6.3, 18.5)
    assert w.t_ambient(6.4 * 3600) < w.t_ambient(14.5 * 3600) - 5
    temps = [w.t_ambient(t) for t in range(0, 86400, 600)]
    assert 22 < min(temps) and max(temps) < 37


def test_ambient_temperature_parton_logan_peak():
    # Eq. 1 (Parton & Logan 1981, p. 206): sin(pi m / (Y + 2a)) peaks at
    # m = (Y + 2a)/2 hours after Tmin (sunrise + c), where T = Tmax.
    rise, sset = 6.3, 18.5
    w = Weather(WeatherConfig(sky="clear", seed=1), rise, sset)
    t_peak = rise + PL_C + (sset - rise + 2 * PL_A) / 2
    t_max = w.cfg.t_min + (w.cfg.t_max - w.cfg.t_min) * w.params.t_swing
    assert w.t_ambient(t_peak * 3600) == pytest.approx(t_max, abs=1e-9)
    assert w.t_ambient((t_peak - 0.5) * 3600) < t_max
    assert w.t_ambient((t_peak + 0.5) * 3600) < t_max


def test_linke_turbidity_matches_pvlib():
    pd = pytest.importorskip("pandas")
    pvlib = pytest.importorskip("pvlib")
    days = pd.date_range("2026-01-01", "2026-12-31", freq="7D").append(
        pd.date_range("2028-02-25", "2028-03-05", freq="D"))  # leap year
    ref = pvlib.clearsky.lookup_linke_turbidity(days, SITE.latitude, SITE.longitude)
    for d, r in zip(days, ref):
        assert linke_turbidity(d.date()) == pytest.approx(r, abs=1e-6)


def test_day_environment_uses_monthly_linke():
    assert DayEnvironment(day=date(2026, 1, 15)).linke < 3.7
    assert DayEnvironment(day=date(2026, 7, 15)).linke > 5.2
    assert DayEnvironment(day=date(2026, 7, 15), weather=WeatherConfig(linke=4.5)).linke == 4.5


def test_shade_event_ramps():
    ev = ShadeEvent(10.0, 11.0, (0.8, 0.0, 0.0), ramp_s=600)
    assert ev.factor(9.9 * 3600) == 0.0
    assert ev.factor(10 * 3600 + 300) == pytest.approx(0.5)
    assert ev.factor(10.5 * 3600) == 1.0
    assert ev.factor(11 * 3600 - 300) == pytest.approx(0.5)
    assert ev.factor(11.1 * 3600) == 0.0


def test_shade_blocks_beam_only():
    env = DayEnvironment(weather=WeatherConfig(sky="clear"),
                         shading=[ShadeEvent(11.0, 13.0, (1.0, 0.0, 0.0), 60)])
    s = env.sample(12 * 3600)
    assert s.shade_label == "partial_shading"
    cond, poa = env.conditions(s, 14.0, 180.0)
    assert 0.5 < cond.shade[0] < 1.0  # diffuse still reaches the shaded substring
    assert cond.shade[0] == pytest.approx(poa.beam / poa.total, abs=2e-3)
    assert cond.shade[1] == 0.0


def test_night_has_no_irradiance():
    env = DayEnvironment()
    cond, poa = env.conditions(env.sample(2 * 3600), 14.0, 180.0)
    assert cond.g == 0.0 and poa.total == 0.0
