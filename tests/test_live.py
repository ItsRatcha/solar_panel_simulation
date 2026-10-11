import pytest

from pvsim.converter.buck_averaged import BuckAveraged, BuckParams
from pvsim.engine.live import Conditions, LiveChannel
from pvsim.load.electronic_load import ElectronicLoad
from pvsim.mppt import create_mppt
from pvsim.sensors.ina226 import INA226, INA226Config


def test_live_channel_tracks_mpp():
    ch = LiveChannel(conv=BuckAveraged(BuckParams.ideal()), sensor=INA226(INA226Config.ideal()),
                     algo=create_mppt("inccond", d_init=0.3, d_max=1.0))
    snaps = [ch.step() for _ in range(400)]
    s = snaps[-1]
    assert s.tracking_eff >= 0.99
    assert s.t == pytest.approx(400 * ch.dt)
    assert s.reach.reachable and not s.collapsed


def test_conditions_change_curve_and_power():
    ch = LiveChannel()
    p_full = ch.step().p_mpp
    ch.cond = Conditions(g=1000.0, shade=(0.0, 0.0, 0.75), t_amb=30.0)
    s = ch.step()
    assert s.p_mpp < p_full
    assert len(s.curve.local_maxima) >= 2


def test_energy_bookkeeping_in_snapshot():
    ch = LiveChannel()
    for _ in range(50):
        s = ch.step()
    assert s.p_pv == pytest.approx(s.p_out + s.p_loss, rel=1e-6)
    assert 0.8 < s.conv_eff < 1.0


def test_unreachable_flag_for_high_resistance_load():
    ch = LiveChannel(load=ElectronicLoad("CR", 50.0))
    s = ch.step()
    assert not s.reach.reachable


def test_dark_panel_is_safe():
    ch = LiveChannel(cond=Conditions(g=0.0))
    s = ch.step()
    assert s.p_pv == 0.0 and s.tracking_eff == 0.0
