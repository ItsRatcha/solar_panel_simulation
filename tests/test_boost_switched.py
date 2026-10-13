import numpy as np
import pytest

from pvsim.converter import (
    TOPOLOGIES, BoostAveraged, BoostParams, BuckAveraged, BuckParams, converter_params, make_converter,
)
from pvsim.converter.switched import simulate_switched
from pvsim.engine.operating_point import mpp_reachability, solve_operating_point
from pvsim.load.electronic_load import ElectronicLoad
from pvsim.pv.datasheet import PanelDatasheet
from pvsim.pv.panel import Panel


@pytest.fixture(scope="module")
def panel():
    return Panel(PanelDatasheet())


@pytest.fixture(scope="module")
def curve(panel):
    return panel.curve(1000.0, 25.0)


@pytest.fixture(scope="module")
def boost():
    return BoostAveraged(BoostParams())


# ----------------------------------------------------------- factory
def test_factory():
    assert TOPOLOGIES == ("buck", "boost")
    assert isinstance(make_converter(), BuckAveraged)
    b = make_converter("boost", {"l": 100e-6, "unknown": 1})
    assert isinstance(b, BoostAveraged) and b.p.l == 100e-6
    assert isinstance(converter_params("buck"), BuckParams)
    with pytest.raises(ValueError):
        make_converter("cuk")


# ------------------------------------------------------ averaged boost
@pytest.mark.parametrize("d", [0.1, 0.4, 0.7, 0.9])
def test_ideal_boost_reflected_resistance(curve, d):
    conv = BoostAveraged(BoostParams.ideal())
    op = solve_operating_point(curve, conv, d, ElectronicLoad("CR", 20.0))
    assert op.v_pv / op.i_pv == pytest.approx(20.0 * (1 - d) ** 2, rel=1e-6)
    assert op.conv.v_out == pytest.approx(op.v_pv / (1 - d), rel=1e-9)
    assert op.conv.efficiency == pytest.approx(1.0)


def _random_load(rng):
    m = rng.choice(["CR", "CC", "CV", "OFF"])
    sp = {"CR": rng.uniform(0.5, 300), "CC": rng.uniform(0.05, 6), "CV": rng.uniform(1, 60), "OFF": 0.0}[m]
    return ElectronicLoad(m, sp)


@pytest.mark.parametrize("params", [BoostParams.ideal(), BoostParams(), BoostParams(synchronous=True)])
def test_boost_energy_balance_every_step(panel, params):
    conv = BoostAveraged(params)
    rng = np.random.default_rng(1)
    for _ in range(300):
        g = tuple(rng.uniform(0, 1100, 3)) if rng.random() < 0.3 else rng.uniform(0, 1100)
        c = panel.curve(g, rng.uniform(20, 70))
        op = solve_operating_point(c, conv, rng.uniform(0, 1), _random_load(rng))
        r = op.conv
        assert abs(r.p_in - r.p_out - r.p_loss) <= 1e-9 * max(r.p_in, 1e-12)
        assert abs(op.i_pv - r.i_in) < 1e-6
        assert 0.0 <= op.v_pv <= c.v_oc + 1e-9
        assert r.p_out >= 0 and r.p_loss >= -1e-12
        assert r.v_out <= params.v_out_max + 1e-9


def test_boost_modes(curve, boost):
    assert solve_operating_point(curve, boost, 0.5, ElectronicLoad("CR", 20.0)).conv.mode == "CCM"
    assert solve_operating_point(curve, boost, 0.2, ElectronicLoad("CR", 200.0)).conv.mode == "DCM"
    ovp = solve_operating_point(curve, boost, 0.5, ElectronicLoad("OFF", 0.0)).conv
    assert ovp.mode == "OVP" and ovp.v_out == pytest.approx(60.0)
    off = solve_operating_point(curve, boost, 0.0, ElectronicLoad("CR", 5.0))
    assert off.conv.mode == "off" and off.i_pv > 1.0  # diode path still conducts
    assert off.conv.v_out == pytest.approx(off.v_pv - 0.45 - off.conv.i_out * 0.05)


def test_boost_duty_pulls_panel_voltage_down(panel, boost):
    c = panel.curve(900.0, 40.0)
    load = ElectronicLoad("CR", 20.0)
    v = [solve_operating_point(c, boost, d, load).v_pv for d in np.linspace(0.0, 0.9, 60)]
    assert np.all(np.diff(v) <= 1e-9)


def test_boost_duty_sweep_reaches_mpp(curve, boost):
    load = ElectronicLoad("CR", 20.0)
    p = max(solve_operating_point(curve, boost, d, load).p_pv for d in np.linspace(0.0, 0.9, 400))
    assert p >= 0.995 * curve.mpp.p


@pytest.mark.parametrize("load, ok, word", [
    (ElectronicLoad("CR", 20.0), True, ""),
    (ElectronicLoad("CR", 3.0), False, "too low"),
    (ElectronicLoad("CR", 2000.0), False, "too high"),
    (ElectronicLoad("CC", 1.0), True, ""),
    (ElectronicLoad("CC", 3.5), False, "CC"),
    (ElectronicLoad("CV", 30.0), True, ""),
    (ElectronicLoad("CV", 10.0), False, "CV"),
])
def test_boost_reachability(curve, boost, load, ok, word):
    r = mpp_reachability(curve, boost, load)
    assert r.reachable is ok
    assert word in r.reason


# ------------------------------------------------------- switched model
CASES = [
    (BuckParams(), 0.6, ElectronicLoad("CR", 3.0)),
    (BuckParams(synchronous=True), 0.6, ElectronicLoad("CR", 3.0)),
    (BuckParams(), 0.3, ElectronicLoad("CR", 30.0)),  # DCM
    (BoostParams(), 0.45, ElectronicLoad("CR", 20.0)),
    (BoostParams(synchronous=True), 0.45, ElectronicLoad("CR", 20.0)),
    (BoostParams(), 0.2, ElectronicLoad("CR", 200.0)),  # DCM
    (BoostParams(), 0.4, ElectronicLoad("CC", 1.5)),
]


@pytest.mark.parametrize("params, d, load", CASES)
def test_switched_agrees_with_averaged(curve, params, d, load):
    conv = make_converter("boost" if isinstance(params, BoostParams) else "buck", params.to_dict())
    avg = solve_operating_point(curve, conv, d, load).conv
    sw = simulate_switched(conv, d, load, curve)
    assert sw.mode == avg.mode
    assert sw.v_in_avg == pytest.approx(avg.v_in, rel=0.01)
    assert sw.v_out_avg == pytest.approx(avg.v_out, rel=0.01)
    assert sw.p_in == pytest.approx(avg.p_in, rel=0.02)
    assert sw.p_out == pytest.approx(avg.p_out, rel=0.02)
    assert sw.efficiency == pytest.approx(avg.efficiency, abs=0.01)
    assert sw.ripple_il == pytest.approx(avg.ripple_il, rel=0.03)
    for k, v in avg.losses.items():
        assert sw.losses[k] == pytest.approx(v, rel=0.1, abs=5e-3), k
    assert abs(sw.energy_residual) < 1e-3
    assert len(sw.t) == len(sw.i_l) == 20 * 100


@pytest.mark.parametrize("topology, d, load", [("buck", 0.6, ElectronicLoad("CR", 3.0)),
                                               ("boost", 0.45, ElectronicLoad("CR", 20.0))])
def test_switched_settles_from_rest(curve, topology, d, load):
    conv = make_converter(topology)
    ref = simulate_switched(conv, d, load, curve)
    cold = simulate_switched(conv, d, load, curve, init=(0.0, 0.0, curve.v_oc), n_settle=4000)
    assert cold.v_in_avg == pytest.approx(ref.v_in_avg, rel=2e-3)
    assert cold.p_out == pytest.approx(ref.p_out, rel=5e-3)


def test_switched_ideal_source_and_ideal_parts():
    conv = BoostAveraged(BoostParams.ideal())
    sw = simulate_switched(conv, 0.5, ElectronicLoad("CR", 20.0), 12.0, n_settle=4000)
    assert sw.v_out_avg == pytest.approx(24.0, rel=2e-3)
    assert sw.efficiency == pytest.approx(1.0, abs=1e-3)  # lossless; the rest is residual ringing
    assert sw.v_in_avg == pytest.approx(12.0) and sw.ripple_v_in == 0.0
    assert sw.t[0] == pytest.approx(sw.t[1] - sw.t[0]) and sw.t[-1] == pytest.approx(20 / 40e3)
