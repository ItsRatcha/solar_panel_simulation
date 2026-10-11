import numpy as np
import pytest

from pvsim.converter.buck_averaged import BuckAveraged, BuckParams
from pvsim.engine.operating_point import mpp_reachability, solve_operating_point
from pvsim.load.electronic_load import ElectronicLoad
from pvsim.load.profiles import LoadProfile, LoadPoint
from pvsim.pv.datasheet import PanelDatasheet
from pvsim.pv.panel import Panel


@pytest.fixture(scope="module")
def panel():
    return Panel(PanelDatasheet())


@pytest.fixture(scope="module")
def ideal():
    return BuckAveraged(BuckParams.ideal())


@pytest.fixture(scope="module")
def real():
    return BuckAveraged(BuckParams())


# ------------------------------------------------------------------ loads
@pytest.mark.parametrize(
    "load",
    [
        ElectronicLoad("CR", 4.0),
        ElectronicLoad("CC", 2.0),
        ElectronicLoad("CV", 6.0),
        ElectronicLoad("OFF", 0.0),
    ],
)
def test_load_is_monotone_and_thevenin_intersection_exact(load):
    v = np.linspace(0, 30, 3001)
    i = np.array([load.current(x) for x in v])
    assert np.all(np.diff(i) >= -1e-12)
    for e in [0.0, 0.3, 3.0, 7.0, 15.0]:
        for r in [0.0, 0.05, 1.0]:
            vo = load.intersect(e, r)
            assert 0.0 <= vo <= e + 1e-12
            assert vo + load.current(vo) * r == pytest.approx(e, abs=1e-9)


def test_cc_load_holds_setpoint_with_enough_voltage():
    ld = ElectronicLoad("CC", 1.5)
    assert ld.current(5.0) == pytest.approx(1.5)
    assert ld.current(0.0) == 0.0


# -------------------------------------------------------------- converter
def test_ideal_buck_vout_is_d_vin(ideal):
    for d in [0.1, 0.5, 0.9]:
        r = ideal.output(18.0, d, ElectronicLoad("CR", 5.0))
        assert r.v_out == pytest.approx(d * 18.0, rel=1e-12)
        assert r.p_in == pytest.approx(r.p_out, rel=1e-12)
        assert r.p_loss == pytest.approx(0.0, abs=1e-12)


def test_ideal_reflected_resistance(panel, ideal):
    c = panel.curve(1000.0, 25.0)
    for d, rl in [(0.5, 2.0), (0.8, 4.0), (0.3, 1.0)]:
        op = solve_operating_point(c, ideal, d, ElectronicLoad("CR", rl))
        assert op.v_pv / op.i_pv == pytest.approx(rl / d**2, rel=1e-4)


def test_real_buck_losses_and_efficiency(real):
    r = real.output(18.0, 0.55, ElectronicLoad("CR", 3.0))
    assert r.mode == "CCM"
    assert all(v >= 0 for v in r.losses.values())
    assert 0.85 < r.efficiency < 0.99
    assert r.v_out < 0.55 * 18.0


def test_dcm_at_light_load(real):
    d = 0.3
    r = real.output(18.0, d, ElectronicLoad("CR", 500.0))
    assert r.mode == "DCM"
    assert r.v_out > d * 18.0  # DCM pushes Vout above D*Vin


def test_converter_off_below_uvlo(real):
    r = real.output(real.p.v_uvlo * 0.5, 0.5, ElectronicLoad("CR", 3.0))
    assert r.mode == "off" and r.i_in == 0.0 and r.p_out == 0.0


# ------------------------------------------------------- operating point
def _random_load(rng):
    m = rng.choice(["CR", "CC", "CV", "OFF"])
    sp = {"CR": rng.uniform(0.5, 60), "CC": rng.uniform(0.05, 6), "CV": rng.uniform(1, 20), "OFF": 0.0}[m]
    return ElectronicLoad(m, sp)


@pytest.mark.parametrize("params", [BuckParams.ideal(), BuckParams(), BuckParams(synchronous=True)])
def test_energy_balance_every_step(panel, params):
    conv = BuckAveraged(params)
    rng = np.random.default_rng(0)
    for _ in range(300):
        g = tuple(rng.uniform(0, 1100, 3)) if rng.random() < 0.3 else rng.uniform(0, 1100)
        c = panel.curve(g, rng.uniform(20, 70))
        d = rng.uniform(0, 1)
        op = solve_operating_point(c, conv, d, _random_load(rng))
        r = op.conv
        assert abs(r.p_in - r.p_out - r.p_loss) <= 1e-9 * max(r.p_in, 1e-12)
        assert abs(op.i_pv - r.i_in) < 1e-6
        assert op.v_pv == pytest.approx(r.v_in)
        assert 0.0 <= op.v_pv <= c.v_oc + 1e-9
        assert r.p_out >= 0 and r.p_loss >= 0


def test_increasing_duty_pulls_panel_voltage_down(panel, real):
    c = panel.curve(900.0, 40.0)
    load = ElectronicLoad("CR", 3.0)
    v = [solve_operating_point(c, real, d, load).v_pv for d in np.linspace(0.05, 0.95, 40)]
    assert np.all(np.diff(v) <= 1e-9)


def test_duty_sweep_reaches_true_mpp(panel, real):
    c = panel.curve(1000.0, 25.0)
    load = ElectronicLoad("CR", 3.0)
    p = max(solve_operating_point(c, real, d, load).p_pv for d in np.linspace(0.3, 0.95, 400))
    assert p >= 0.995 * c.mpp.p


def test_warm_start_gives_same_answer(panel, real):
    c = panel.curve(700.0, 35.0)
    load = ElectronicLoad("CR", 3.0)
    cold = solve_operating_point(c, real, 0.6, load)
    warm = solve_operating_point(c, real, 0.6, load, v_guess=cold.v_pv * 0.97)
    assert warm.v_pv == pytest.approx(cold.v_pv, abs=1e-6)


def test_cc_load_output_current(panel, real):
    c = panel.curve(1000.0, 25.0)
    op = solve_operating_point(c, real, 0.5, ElectronicLoad("CC", 2.0))
    assert op.conv.i_out == pytest.approx(2.0, rel=1e-6)


def test_cv_load_holds_voltage(panel, real):
    c = panel.curve(1000.0, 25.0)
    op = solve_operating_point(c, real, 0.7, ElectronicLoad("CV", 12.0))
    assert op.conv.v_out == pytest.approx(12.0, abs=0.1)
    assert op.conv.i_out > 0.5


def test_mpp_unreachable_flag(panel, real):
    c = panel.curve(1000.0, 25.0)
    bad = mpp_reachability(c, real, ElectronicLoad("CR", 50.0))
    assert not bad.reachable and "R" in bad.reason
    good = mpp_reachability(c, real, ElectronicLoad("CR", 3.0))
    assert good.reachable


def test_dark_panel_operating_point(panel, real):
    c = panel.curve(0.0, 25.0)
    op = solve_operating_point(c, real, 0.5, ElectronicLoad("CR", 3.0))
    assert op.p_pv == 0.0 and op.conv.p_out == 0.0


# --------------------------------------------------------------- profiles
def test_load_profile_step_and_ramp():
    prof = LoadProfile(
        [
            LoadPoint(0.0, "CR", 5.0),
            LoadPoint(10.0, "CR", 2.0),
            LoadPoint(20.0, "CR", 4.0, interp="linear"),
            LoadPoint(30.0, "CC", 1.0),
        ]
    )
    assert prof.at(-1.0).setpoint == 5.0
    assert prof.at(9.99).setpoint == 5.0
    assert prof.at(10.0).setpoint == 2.0
    assert prof.at(15.0).setpoint == pytest.approx(3.0)  # ramp toward the 20 s point
    assert prof.at(25.0).setpoint == 4.0
    assert prof.at(31.0).mode == "CC"
    assert LoadProfile.from_dict(prof.to_dict()).to_dict() == prof.to_dict()
