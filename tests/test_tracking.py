import math
from datetime import date

import pytest

from pvsim.environment.day import DayEnvironment
from pvsim.environment.weather import WeatherConfig
from pvsim.tracking.mounts import FixedMount, TrackerParams, TwoAxisTracker, make_mount, pointing_error


def _clear_day():
    return DayEnvironment(day=date(2026, 3, 21), weather=WeatherConfig(sky="clear", seed=1))


def test_tracker_poa_not_below_fixed_under_clear_sky():
    env = _clear_day()
    fixed, trk = FixedMount(), TwoAxisTracker()
    t, worse = env.sunrise_s, 0
    while t < env.sunset_s:
        s = env.sample(t)
        of = fixed.update(s.sun, 10.0)
        ot = trk.update(s.sun, 10.0)
        # Only where beam dominates: in diffuse light a sun-pointing panel can
        # lose to a flatter one (Kelly & Gibson 2009, Solar Energy 83(11):2092-2102,
        # doi:10.1016/j.solener.2009.08.009).
        if s.irr.dni * math.sin(math.radians(s.sun.elevation)) > s.irr.dhi:
            worse += env.poa(s, *ot).total < env.poa(s, *of).total - 1.0
        t += 10.0
    assert worse == 0


def test_tracker_follows_sun_within_deadband():
    env = _clear_day()
    trk = TwoAxisTracker()
    for t in range(int(env.sunrise_s), 12 * 3600, 2):
        trk.update(env.sample(t).sun, 2.0)
    s = env.sample(12 * 3600)
    assert pointing_error(*trk.orientation, s.sun.zenith, s.sun.azimuth) <= TrackerParams().deadband + 0.5


def test_tracker_energy_accounting():
    trk = TwoAxisTracker(TrackerParams(p_motor=4.0, p_standby=0.5, speed=1.0))
    sun = _clear_day().sample(9 * 3600).sun  # tracker starts stowed -> must move
    for _ in range(3600):
        trk.update(sun, 1.0)
    assert trk.motor_s > 0
    assert trk.energy_wh == pytest.approx(0.5 + 4.0 * trk.motor_s / 3600, rel=1e-9)
    assert not trk.moving


def test_stuck_tracker_does_not_move():
    trk = TwoAxisTracker()
    trk.stuck = True
    before = trk.orientation
    trk.update(_clear_day().sample(10 * 3600).sun, 10.0)
    assert trk.orientation == before and trk.motor_s == 0


def test_tracker_stows_at_night():
    env = _clear_day()
    trk = TwoAxisTracker()
    for t in range(16 * 3600, int(env.sunset_s + 3600), 2):
        trk.update(env.sample(t).sun, 2.0)
    assert trk.orientation == pytest.approx((0.0, 90.0), abs=0.1)


def test_make_mount():
    assert make_mount("fixed", tilt=10).tilt == 10
    assert isinstance(make_mount("tracker"), TwoAxisTracker)
    with pytest.raises(ValueError):
        make_mount("nope")
