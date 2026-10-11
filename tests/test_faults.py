"""Phase 6: fault injection, labels and the `pvsim batch` dataset generator."""

import json

import numpy as np
import pandas as pd
import pytest

from pvsim.cli import main
from pvsim.data.batch import make_day_scenario, run_batch
from pvsim.data.faults import FAULT_KINDS, FaultEvent
from pvsim.scenario.model import PanelSpec, Scenario
from pvsim.scenario.runner import run_scenario

T_FAULT = pd.Timestamp("2026-03-21 12:00:00+07:00")


def _faulted(kind: str, panel_id=None, **params) -> pd.DataFrame:
    """Fixed + tracker around noon with one fault from 12:00 to the end."""
    sc = Scenario(name=kind, mppt_period=2.0, start_h=11.9, end_h=12.2,
                  panels=[PanelSpec("fixed"), PanelSpec("tracker", mount="tracker")],
                  faults=[{"kind": kind, "start_h": 12.0, "end_h": 13.0, "panel_id": panel_id,
                           "params": params}])
    return run_scenario(sc, extended=True).frame


def _split(f: pd.DataFrame, pid: str):
    g = f[f.panel_id == pid]
    return g[g.timestamp < T_FAULT], g[g.timestamp >= T_FAULT]


# ------------------------------------------------------------------ events
def test_fault_event_validation_and_round_trip():
    with pytest.raises(ValueError, match="unknown fault kind"):
        FaultEvent("lightning", 1, 2)
    with pytest.raises(ValueError, match="end_h"):
        FaultEvent("soiling", 2, 2)
    with pytest.raises(ValueError, match="unknown params"):
        FaultEvent("soiling", 1, 2, params={"depth": 1})
    with pytest.raises(ValueError, match="unknown fault fields"):
        FaultEvent.from_dict({"kind": "soiling", "start_h": 1, "end_h": 2, "when": 0})
    e = FaultEvent("soiling", 10, 11, "fixed", {"loss": 0.4})
    assert FaultEvent.from_dict(e.to_dict()) == e
    assert e.p("loss") == 0.4 and FaultEvent("soiling", 1, 2).p("loss") == 0.2
    assert e.active(10 * 3600) and not e.active(11 * 3600)
    with pytest.raises(ValueError, match="unknown panel"):
        Scenario(faults=[e.to_dict()], panels=[PanelSpec("other")])


def test_scenario_with_faults_round_trips():
    sc = make_day_scenario(3, seed=5)
    back = Scenario.from_dict(json.loads(json.dumps(sc.to_dict())))
    assert back.to_dict() == sc.to_dict()


# ----------------------------------------------------------------- effects
@pytest.mark.parametrize("kind", [k for k in FAULT_KINDS if k != "tracker_stuck"])
def test_each_fault_labels_rows_of_its_panel_only(kind):
    f = _faulted(kind, panel_id="fixed")
    before, during = _split(f, "fixed")
    assert (before.fault_label == "normal").all()
    assert (during.fault_label == kind).all()
    assert (f[f.panel_id == "tracker"].fault_label == "normal").all()


def test_shading_soiling_and_diode_reduce_available_power():
    def ratio(kind):
        before, during = _split(_faulted(kind), "fixed")
        return during.p_mpp_true.mean() / before.p_mpp_true.mean()

    r = {k: ratio(k) for k in ("partial_shading_lvl1", "partial_shading_lvl2",
                               "partial_shading_lvl3", "soiling", "bypass_diode_short")}
    assert 1.0 > r["partial_shading_lvl1"] > r["partial_shading_lvl2"] > r["partial_shading_lvl3"]
    assert r["soiling"] == pytest.approx(0.8, abs=0.04)
    assert 0.55 < r["bypass_diode_short"] < 0.8  # one of three substrings lost


def test_open_circuit_and_sensor_offset():
    _, during = _split(_faulted("open_circuit"), "fixed")
    assert (during.p_pv_true == 0).all() and (during.p_mpp_true == 0).all()

    f = _faulted("sensor_offset", offset_v=0.5)
    before, during = _split(f, "fixed")
    assert abs((before.v_pv - before.v_pv_true).mean()) < 0.05
    assert (during.v_pv - during.v_pv_true).mean() == pytest.approx(0.5, abs=0.05)


def test_tracker_stuck_freezes_tracker_and_ignores_fixed_mount():
    f = _faulted("tracker_stuck")  # applies to both panels
    before, during = _split(f, "tracker")
    assert np.ptp(before.tilt_deg) > 0.1
    assert np.ptp(during.tilt_deg) == 0 and (during.fault_label == "tracker_stuck").all()
    assert (f[f.panel_id == "fixed"].fault_label == "normal").all()


def test_fault_window_closes():
    sc = Scenario(mppt_period=2.0, start_h=11.9, end_h=12.2,
                  faults=[{"kind": "open_circuit", "start_h": 12.0, "end_h": 12.1}])
    f = run_scenario(sc, extended=True).frame
    after = f[f.timestamp >= pd.Timestamp("2026-03-21 12:06:00+07:00")]
    assert (after.fault_label == "normal").all() and (after.p_pv_true > 20).all()


# ------------------------------------------------------------------- batch
def test_day_scenarios_are_seeded_and_independent():
    a, b = make_day_scenario(2, 7), make_day_scenario(2, 7)
    assert a.to_dict() == b.to_dict()
    assert make_day_scenario(2, 8).to_dict() != a.to_dict()
    days = [make_day_scenario(i, 1) for i in range(40)]
    assert len({d.date for d in days}) > 30
    assert len({d.weather.sky for d in days}) == 4
    for d in days:
        for e in d.fault_events():
            assert e.kind != "tracker_stuck" or e.panel_id == "tracker"
            assert 8.0 <= e.start_h < e.end_h <= 18.5


def _files(d):
    return {p.name: p.read_bytes() for p in sorted(d.iterdir())}


def test_batch_is_deterministic_across_workers(tmp_path):
    kw = dict(days=3, seed=1, resample="1min", period=10.0, window=(11.0, 11.5))
    m = run_batch(out_dir=tmp_path / "a", workers=1, **kw)
    run_batch(out_dir=tmp_path / "b", workers=2, **kw)
    a, b = _files(tmp_path / "a"), _files(tmp_path / "b")
    assert sorted(a) == ["day_000.csv", "day_001.csv", "day_002.csv", "manifest.json"]
    assert a == b

    man = json.loads(m.read_text())
    assert [d["file"] for d in man["days"]] == ["day_000.csv", "day_001.csv", "day_002.csv"]
    for d in man["days"]:
        sc = Scenario.from_dict(d["scenario"])
        assert sc.to_dict() == make_day_scenario(d["index"], 1, period=10.0).to_dict() | {
            "start_h": 11.0, "end_h": 11.5}
        df = pd.read_csv(tmp_path / "a" / d["file"])
        assert len(df) == 2 * 30 and set(df.panel_id) == {"fixed", "tracker"}
        assert set(df.fault_label) <= {"normal", *FAULT_KINDS}


def test_cli_batch(tmp_path, capsys):
    out = tmp_path / "ds"
    assert main(["batch", "--days", "1", "--seed", "2", "--period", "60", "-q", "--out", str(out)]) == 0
    assert (out / "manifest.json").exists() and (out / "day_000.csv").exists()
    assert "manifest.json" in capsys.readouterr().out
    assert main(["batch", "--days", "0", "-q", "--out", str(out)]) == 2
