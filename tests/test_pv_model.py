import numpy as np
import pytest

from pvsim.pv.datasheet import (
    PanelDatasheet,
    cell_temperature_noct,
    fit_single_diode,
)
from pvsim.pv.panel import Panel
from pvsim.pv.single_diode import current_from_voltage, voltage_from_current


@pytest.fixture(scope="module")
def ds():
    return PanelDatasheet()


@pytest.fixture(scope="module")
def panel(ds):
    return Panel(ds)


def rel(a, b):
    return abs(a - b) / abs(b)


def test_fit_passes_through_datasheet_points(ds):
    p = fit_single_diode(ds)
    assert p.r_s > 0 and p.r_sh > 0 and p.i_0 > 0
    assert rel(float(current_from_voltage(0.0, p)), ds.i_sc) < 1e-6
    assert abs(float(current_from_voltage(ds.v_oc, p))) < 1e-6
    assert rel(float(current_from_voltage(ds.v_mp, p)), ds.i_mp) < 1e-6


def test_lambert_forms_are_inverses(ds):
    p = fit_single_diode(ds)
    v = np.linspace(0, ds.v_oc, 50)
    i = current_from_voltage(v, p)
    assert np.allclose(voltage_from_current(i, p), v, atol=1e-6)


@pytest.mark.parametrize("n_bypass", [1, 2, 3])
def test_stc_matches_datasheet_within_2pct(ds, n_bypass):
    d = PanelDatasheet(n_bypass=n_bypass)
    c = Panel(d).curve(1000.0, 25.0)
    assert rel(c.v_oc, d.v_oc) < 0.02
    assert rel(c.i_sc, d.i_sc) < 0.02
    assert rel(c.mpp.p, d.v_mp * d.i_mp) < 0.02
    assert rel(c.mpp.v, d.v_mp) < 0.02


def test_temperature_coefficients(panel, ds):
    c25 = panel.curve(1000.0, 25.0)
    c50 = panel.curve(1000.0, 50.0)
    dvoc = (c50.v_oc - c25.v_oc) / 25.0
    disc = (c50.i_sc - c25.i_sc) / 25.0
    assert dvoc == pytest.approx(ds.beta_voc, rel=0.05)
    assert disc == pytest.approx(ds.alpha_isc, rel=0.1)
    # Power must drop with temperature at roughly the datasheet coefficient
    dp_pct = (c50.mpp.p - c25.mpp.p) / c25.mpp.p / 25.0 * 100
    assert -0.6 < dp_pct < -0.3


def test_irradiance_scaling(panel):
    c1000 = panel.curve(1000.0, 25.0)
    c500 = panel.curve(500.0, 25.0)
    assert c500.i_sc == pytest.approx(c1000.i_sc / 2, rel=0.01)
    assert c500.v_oc < c1000.v_oc
    assert 0.45 < c500.mpp.p / c1000.mpp.p < 0.5


def test_dark_panel_produces_nothing(panel):
    c = panel.curve(0.0, 25.0)
    assert c.mpp.p == pytest.approx(0.0, abs=1e-9)
    assert float(c.current(5.0)) == pytest.approx(0.0, abs=1e-9)


def test_noct_cell_temperature():
    assert cell_temperature_noct(20.0, 800.0, 45.0) == pytest.approx(45.0)
    assert cell_temperature_noct(30.0, 0.0, 45.0) == pytest.approx(30.0)


def test_curve_is_monotone_and_interpolates(panel):
    c = panel.curve(800.0, 40.0)
    assert np.all(np.diff(c.v) > 0)
    assert np.all(np.diff(c.i) <= 1e-12)
    # interpolation agrees with the exact (root-solved) panel current
    for v in [0.0, 5.0, 12.0, 17.0, 19.5]:
        assert float(c.current(v)) == pytest.approx(panel.current_exact(v, 800.0, 40.0), abs=2e-3)


def test_partial_shading_creates_multiple_peaks(panel):
    c = panel.curve((1000.0, 1000.0, 300.0), 25.0)
    assert len(c.local_maxima) >= 2
    unshaded = panel.curve(1000.0, 25.0)
    assert len(unshaded.local_maxima) == 1
    assert c.mpp.p < unshaded.mpp.p


def test_global_mpp_matches_brute_force_sweep(panel):
    g = (1000.0, 600.0, 250.0)
    c = panel.curve(g, 30.0)
    v = np.linspace(0, c.v_oc, 20001)
    p = np.array([vv * panel.current_exact(vv, g, 30.0) for vv in v[::20]])
    p_brute = p.max()
    assert c.mpp.p == pytest.approx(p_brute, rel=2e-3)
    assert c.mpp.p >= p_brute * (1 - 1e-6) or c.mpp.p == pytest.approx(p_brute, rel=2e-3)
    # the global peak is the largest local maximum
    assert c.mpp.p == pytest.approx(max(m.p for m in c.local_maxima), rel=1e-3)


def test_bypass_diode_clamps_shaded_substring(panel, ds):
    # Fully shaded substring is bypassed: panel still delivers ~2/3 power
    c = panel.curve((1000.0, 1000.0, 0.0), 25.0)
    full = panel.curve(1000.0, 25.0)
    assert 0.55 < c.mpp.p / full.mpp.p < 0.7
    # at full current the bypassed substring contributes about -Vf
    assert c.v_oc == pytest.approx(full.v_oc * 2 / 3, rel=0.05)


def test_curve_cache_reuses_objects(panel):
    a = panel.curve(900.0, 33.0)
    b = panel.curve(900.0, 33.0)
    assert a is b
