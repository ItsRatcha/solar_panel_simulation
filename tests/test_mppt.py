import numpy as np
import pytest

from pvsim.converter.buck_averaged import BuckAveraged, BuckParams
from pvsim.load.electronic_load import ElectronicLoad
from pvsim.mppt import available_mppt, create_mppt
from pvsim.mppt.harness import run_fixed_conditions
from pvsim.pv.datasheet import PanelDatasheet
from pvsim.pv.panel import Panel
from pvsim.sensors.ina226 import INA226, INA226Config

LOCAL = ["po", "po_variable", "inccond", "inccond_variable"]


@pytest.fixture(scope="module")
def panel():
    return Panel(PanelDatasheet())


def steady_ratio(trace, curve, last=100):
    return float(np.mean(trace.p_pv_true[-last:])) / curve.mpp.p


def test_registry_lists_builtins():
    names = available_mppt()
    for n in LOCAL + ["global_scan", "manual"]:
        assert n in names
    with pytest.raises(KeyError):
        create_mppt("does_not_exist")


@pytest.mark.parametrize("name", LOCAL)
@pytest.mark.parametrize("g,t", [(1000.0, 25.0), (600.0, 45.0), (250.0, 30.0)])
def test_local_algorithms_reach_99pct_ideal(panel, name, g, t):
    curve = panel.curve(g, t)
    conv = BuckAveraged(BuckParams.ideal())
    algo = create_mppt(name, d_init=0.3, d_max=1.0)
    tr = run_fixed_conditions(curve, conv, ElectronicLoad("CR", 3.0), algo, n_steps=600)
    assert steady_ratio(tr, curve) >= 0.99


@pytest.mark.parametrize("name", LOCAL + ["global_scan"])
def test_realistic_converter_and_sensors(panel, name):
    curve = panel.curve(850.0, 40.0)
    conv = BuckAveraged(BuckParams())
    algo = create_mppt(name, d_init=0.3)
    sensor = INA226(INA226Config(), seed=3)
    tr = run_fixed_conditions(curve, conv, ElectronicLoad("CR", 3.0), algo, n_steps=800, sensor=sensor)
    assert steady_ratio(tr, curve, last=200) >= 0.97


def _shaded(panel):
    curve = panel.curve((1000.0, 1000.0, 250.0), 25.0)
    assert len(curve.local_maxima) >= 2
    gm = curve.mpp
    local = max((m for m in curve.local_maxima if abs(m.v - gm.v) > 1.0), key=lambda m: m.p)
    assert local.v > gm.v  # trap peak sits on the high-voltage side
    return curve, local


def test_po_trapped_at_local_peak_under_shading(panel):
    curve, local = _shaded(panel)
    conv = BuckAveraged(BuckParams.ideal())
    algo = create_mppt("po", d_init=0.2, d_max=1.0)
    tr = run_fixed_conditions(curve, conv, ElectronicLoad("CR", 3.0), algo, n_steps=600)
    p_ss = np.mean(tr.p_pv_true[-100:])
    assert p_ss == pytest.approx(local.p, rel=0.02)
    assert p_ss < 0.9 * curve.mpp.p


def test_global_scan_finds_gmpp_under_shading(panel):
    curve, _ = _shaded(panel)
    conv = BuckAveraged(BuckParams.ideal())
    algo = create_mppt("global_scan", d_init=0.2, d_max=1.0)
    tr = run_fixed_conditions(curve, conv, ElectronicLoad("CR", 3.0), algo, n_steps=600)
    assert steady_ratio(tr, curve) >= 0.98


def test_global_scan_rescans_after_power_change(panel):
    conv = BuckAveraged(BuckParams.ideal())
    algo = create_mppt("global_scan", d_init=0.2, d_max=1.0, scan_period=10_000, dp_trigger=0.25)
    run_fixed_conditions(panel.curve(1000.0, 25.0), conv, ElectronicLoad("CR", 3.0), algo, 300)
    scans = algo.s.n_scans
    shaded, _ = _shaded(panel)
    tr = run_fixed_conditions(shaded, conv, ElectronicLoad("CR", 3.0), algo, 400, reset=False)
    assert algo.s.n_scans > scans
    assert steady_ratio(tr, shaded) >= 0.98


def test_manual_duty_is_constant(panel):
    algo = create_mppt("manual", duty=0.42)
    tr = run_fixed_conditions(panel.curve(800.0, 30.0), BuckAveraged(), ElectronicLoad("CR", 3.0), algo, 20)
    assert np.all(tr.duty == 0.42)


def test_duty_stays_within_limits(panel):
    conv = BuckAveraged(BuckParams())
    for name in LOCAL + ["global_scan"]:
        algo = create_mppt(name, d_init=0.5, d_min=0.1, d_max=0.9)
        tr = run_fixed_conditions(panel.curve(1000.0, 25.0), conv, ElectronicLoad("CR", 50.0), algo, 300)
        assert tr.duty.min() >= 0.1 and tr.duty.max() <= 0.9


def test_algorithm_state_is_flat_scalars():
    """C-portability: state holds only ints/floats (no arrays, lists or objects)."""
    for name in available_mppt():
        algo = create_mppt(name)
        for k, v in vars(algo.s).items():
            assert isinstance(v, (int, float)), f"{name}.{k} is {type(v).__name__}"
