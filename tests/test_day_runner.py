import numpy as np
import pytest

from pvsim.engine.day_runner import DayRunner
from pvsim.engine.live import LiveChannel
from pvsim.environment.day import DayEnvironment
from pvsim.environment.weather import WeatherConfig
from pvsim.load.electronic_load import ElectronicLoad
from pvsim.mppt import create_mppt
from pvsim.pv.panel import Panel
from pvsim.tracking.mounts import FixedMount, TwoAxisTracker


def _run(mount, dt=10.0):
    ch = LiveChannel(algo=create_mppt("po", d_init=0.3), dt=dt, load=ElectronicLoad("CR", 3.0))
    r = DayRunner(ch, DayEnvironment(weather=WeatherConfig(sky="clear", seed=1)), mount, env_dt=dt)
    last = None
    while not r.done:
        last = r.step()
    return r, last


@pytest.fixture(scope="module")
def fixed_run():
    return _run(FixedMount())


def test_day_runs_sunrise_to_sunset_and_harvests(fixed_run):
    r, last = fixed_run
    e = r.energy
    assert last.clock_s >= r.env.sunset_s
    assert r.progress >= 1.0
    assert 150 < e.mpp_wh < 400  # 50 W panel on a clear tropical equinox day
    assert e.pv_wh <= e.mpp_wh * 1.001
    assert e.mppt_eff > 0.9
    assert 0 < e.out_wh < e.pv_wh


def test_tracker_beats_fixed_on_clear_day(fixed_run):
    rf, _ = fixed_run
    rt, _ = _run(TwoAxisTracker())
    assert rt.energy.mpp_wh > rf.energy.mpp_wh * 1.15
    assert rt.energy.mount_wh > 0 and rf.energy.mount_wh == 0
    assert rt.energy.net_wh > rf.energy.net_wh


def test_fast_curve_mpp_matches_exact_sweep():
    p = Panel()
    for g in [(1000.0,) * 3, (600.0, 420.0, 180.0), (300.0, 150.0, 30.0)]:
        c = p.curve(g, 40.0)
        v = np.linspace(0, c.v_oc, 1500)
        pv = v * np.array([p.current_exact(x, g, 40.0) for x in v])
        assert c.mpp.p >= float(pv.max()) * 0.9995
        assert c.mpp.p <= float(pv.max()) * 1.0005
