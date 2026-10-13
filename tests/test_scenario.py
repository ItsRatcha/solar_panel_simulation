"""Phase 5: scenario JSON, headless runner, recorder, CSV export, CLI."""

import json

import numpy as np
import pandas as pd
import pytest

from pvsim.cli import main
from pvsim.data.export import resample, to_csv_frame, write_csv
from pvsim.data.schema import column_keys, column_names
from pvsim.scenario.io import load_scenario, save_scenario
from pvsim.scenario.model import PanelSpec, Scenario
from pvsim.scenario.presets import PRESETS, get_preset
from pvsim.scenario.runner import run_scenario


def _short(**kw) -> Scenario:
    """Two panels over a short noon window at a coarse MPPT period: fast."""
    base = dict(name="short", mppt_period=10.0, start_h=11.5, end_h=12.0,
                panels=[PanelSpec("fixed"), PanelSpec("tracker", mount="tracker", algorithm="inccond")])
    base.update(kw)
    return Scenario(**base)


@pytest.fixture(scope="module")
def short_run():
    return run_scenario(_short(), extended=True)


# ---------------------------------------------------------------- scenario
@pytest.mark.parametrize("name", list(PRESETS))
def test_json_round_trip(tmp_path, name):
    sc = get_preset(name)
    path = save_scenario(sc, tmp_path / f"{name}.json")
    back = load_scenario(path)
    assert back.to_dict() == sc.to_dict()
    assert json.loads(path.read_text())["schema_version"] == 1


def test_scenario_rejects_bad_input():
    d = _short().to_dict()
    with pytest.raises(ValueError, match="unknown scenario fields"):
        Scenario.from_dict({**d, "bogus": 1})
    with pytest.raises(ValueError, match="newer"):
        Scenario.from_dict({**d, "schema_version": 99})
    with pytest.raises(ValueError, match="unique"):
        Scenario(panels=[PanelSpec("a"), PanelSpec("a")])
    with pytest.raises(ValueError, match="unknown panel fields"):
        PanelSpec.from_dict({"panel_id": "a", "colour": "blue"})


def test_shipped_preset_files_match_code():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "presets"
    for name in PRESETS:
        assert load_scenario(root / f"{name}.json").to_dict() == get_preset(name).to_dict(), name


# ------------------------------------------------------------------ runner
def test_rows_and_window(short_run):
    f = short_run.frame
    assert list(f.columns) == column_keys(extended=True)
    n_steps = int(0.5 * 3600 / 10.0)
    assert len(f) == 2 * n_steps
    assert set(f.panel_id) == {"fixed", "tracker"}
    t = f.timestamp
    assert t.min() == pd.Timestamp("2026-03-21 11:30:00+07:00")
    assert t.max() < pd.Timestamp("2026-03-21 12:00:00+07:00")
    assert t.is_monotonic_increasing


def test_measured_close_to_truth_and_energy(short_run):
    f = short_run.frame
    assert np.allclose(f.v_pv, f.v_pv_true, atol=0.1)
    assert ((f.p_out_true <= f.p_pv_true + 1e-9)).all()
    for pid, e in short_run.energy.items():
        assert 0 < e.out_wh < e.pv_wh <= e.mpp_wh * 1.0001, pid
    assert short_run.energy["tracker"].mount_wh > 0
    assert short_run.energy["fixed"].mount_wh == 0
    assert "fixed" in short_run.summary()


def test_determinism():
    a = run_scenario(_short(end_h=11.6)).frame
    b = run_scenario(_short(end_h=11.6)).frame
    pd.testing.assert_frame_equal(a, b)
    c = run_scenario(_short(end_h=11.6, seed=2)).frame
    assert not np.array_equal(a.v_pv.to_numpy(), c.v_pv.to_numpy())  # sensor noise follows the seed


def test_load_profile_and_label():
    sc = get_preset("load_step")
    sc.mppt_period, sc.start_h, sc.end_h = 10.0, 11.25, 11.75
    f = run_scenario(sc, extended=True).frame
    before = f[f.timestamp.dt.minute < 30]
    after = f[f.timestamp.dt.minute >= 30]
    assert (before.load_setpoint == 3.0).all() and (after.load_setpoint == 8.0).all()
    assert after.mpp_unreachable.all() and not before.mpp_unreachable.any()
    assert (f.fault_label == "normal").all()


def test_partial_shading_trap():
    """The headline check: P&O is trapped on the local peak, global scan is not."""
    sc = get_preset("partial_shading")
    sc.start_h, sc.end_h = 11.4, 12.25
    f = run_scenario(sc, extended=True).frame
    shaded = f[f.timestamp >= pd.Timestamp("2026-03-21 12:00:00+07:00")]
    assert (shaded.fault_label == "partial_shading").all()
    sums = shaded.groupby("panel_id")[["p_pv_true", "p_mpp_true"]].sum()
    eff = sums.p_pv_true / sums.p_mpp_true
    assert eff["po"] < 0.85
    assert eff["global_scan"] > 0.93


# ------------------------------------------------------------------ export
def test_resample_1min(short_run):
    f = short_run.frame
    r = resample(f, "1min")
    assert len(r) == 2 * 30
    assert list(r.columns) == list(f.columns)
    g = f[(f.panel_id == "fixed") & (f.timestamp < pd.Timestamp("2026-03-21 11:31:00+07:00"))]
    row = r[(r.panel_id == "fixed")].iloc[0]
    assert row.p_pv == pytest.approx(g.p_pv.mean())  # mean of P, not mean V * mean I
    assert row.mppt_efficiency == pytest.approx(g.p_pv_true.sum() / g.p_mpp_true.sum())
    assert row.load_mode == "CR" and row.algorithm == "po"
    assert r.load_state.dtype.kind == "i" and r.mpp_unreachable.dtype == bool
    assert resample(f, "raw") is f
    with pytest.raises(ValueError):
        resample(f, "5min")


def test_csv_columns(tmp_path, short_run):
    path = write_csv(short_run.frame, tmp_path / "x.csv", "1s")
    df = pd.read_csv(path)
    assert list(df.columns) == column_names(extended=True)
    assert df.timestamp.iloc[0] == "2026-03-21T11:30:00.000+07:00"
    base = to_csv_frame(short_run.frame[column_keys(False)])
    assert list(base.columns) == column_names(extended=False)


# --------------------------------------------------------------------- CLI
def test_cli_run_and_presets(tmp_path, capsys):
    assert main(["presets", "--write", str(tmp_path / "p")]) == 0
    sc_path = tmp_path / "p" / "load_step.json"
    assert sc_path.exists()
    out = tmp_path / "run.csv"
    rc = main(["run", str(sc_path), "--period", "30", "--seed", "4", "--resample", "1min",
               "-q", "-o", str(out), "--save-scenario", str(tmp_path / "eff.json")])
    assert rc == 0
    df = pd.read_csv(out)
    assert list(df.columns) == column_names(extended=False)
    assert len(df) == 120
    eff = load_scenario(tmp_path / "eff.json")
    assert eff.mppt_period == 30 and eff.seed == 4 and eff.weather.seed == 4
    assert "load_step" in capsys.readouterr().out
    assert main(["run", "no_such_preset"]) == 2
    assert main(["schema"]) == 0


def test_topology_field_round_trip_and_old_files():
    from pvsim.converter import BoostAveraged, BuckAveraged

    sc = get_preset("buck_vs_boost")
    assert Scenario.from_dict(sc.to_dict()).to_dict() == sc.to_dict()
    assert isinstance(sc.build_channel(0)[0].conv, BuckAveraged)
    assert isinstance(sc.build_channel(1)[0].conv, BoostAveraged)
    old = PanelSpec().to_dict()
    del old["topology"], old["boost"]  # files written before the boost option
    assert PanelSpec.from_dict(old).topology == "buck"
    with pytest.raises(ValueError, match="topology"):
        PanelSpec(topology="cuk")
    sc = Scenario(panels=[PanelSpec("b", topology="boost", boost={"l": 100e-6}, buck={"l": 1.0})])
    assert sc.build_channel(0)[0].conv.p.l == 100e-6
