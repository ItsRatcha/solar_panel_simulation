import math
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")


@pytest.fixture(scope="module")
def window():
    from PySide6.QtWidgets import QApplication

    from pvsim.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    w = MainWindow()
    yield w
    w.timer.stop()
    w.close()
    app.processEvents()


def test_manual_mode_steps(window):
    for _ in range(10):
        window._tick()
    assert not window.day_mode
    assert len(window.hist["t"]) > 10


def test_full_day_runs_to_sunset_then_back_to_manual(window):
    w = window
    w.cmb_period.setCurrentIndex(2)  # 10 s
    w.cmb_sky.setCurrentIndex(w.cmb_sky.findData("clear"))
    w.cmb_mode.setCurrentIndex(1)
    w.cmb_speed.setCurrentText("Max")
    assert w.day_mode
    for _ in range(5000):
        if not w.timer.isActive():
            break
        w._tick()
    assert w.runner.done
    e = w.runner.energy
    assert e.pv_wh > 100 and e.mppt_eff > 0.85
    assert "Day finished" in w.statusBar().currentMessage()
    assert 6 < w.hist["t"][0] < 7 and 18 < w.hist["t"][-1] < 19

    w.preset_shading("po")  # a manual preset leaves day mode
    assert not w.day_mode and w.ch.dt == pytest.approx(0.1)


def _run_to_end(w, max_ticks=5000):
    w.cmb_speed.setCurrentText("Max")
    for _ in range(max_ticks):
        if not w.timer.isActive():
            break
        w._tick()


def test_csv_export_after_day(window, tmp_path):
    w = window  # previous test ran a day, then went back to manual: the recording is kept
    out = w.export_csv_to(tmp_path / "day.csv", "1min")
    text = out.read_text().splitlines()
    assert len(text) > 100 and "," in text[0]


def test_ab_compare_manual(window):
    w = window
    w.preset_shading_ab()
    assert w.ch_b is not None and w.ch_b.algo.name == "global_scan"
    for _ in range(40):
        w._tick()
    assert w.last_b is not None
    assert not math.isnan(w.hist["p_pv_b"][-1])
    assert w.ch_b.cond is w.ch.cond
    w.chk_b.setChecked(False)
    assert w.ch_b is None


def test_param_form_changes_panel(window):
    w = window
    w.f_ds.widgets["p_max"].setText("200")
    w.f_ds.widgets["p_max"].setModified(True)
    w.f_ds.widgets["p_max"].editingFinished.emit()
    assert w.ch.datasheet.p_max == pytest.approx(200)
    w.f_ds.widgets["p_max"].setText("abc")
    with pytest.raises(ValueError, match="p_max"):
        w.f_ds.values()
    from pvsim.pv.datasheet import PanelDatasheet

    w.f_ds.set_values(PanelDatasheet().to_dict())
    w._apply_datasheet()
    assert w.ch.datasheet == PanelDatasheet()


def test_scenario_preset_two_panels_and_round_trip(window, tmp_path):
    from pvsim.scenario.presets import get_preset

    w = window
    w.apply_scenario(get_preset("partial_shading"))
    assert w.day_mode and w.runner_b is not None
    assert [p.algorithm for p in w.scenario().panels] == ["po", "global_scan"]
    _run_to_end(w)
    assert w.runner.done and w.runner_b.done
    assert w.runner_b.energy.pv_wh > w.runner.energy.pv_wh * 0.9
    assert "B:" in w.statusBar().currentMessage()

    path = w.save_scenario_to(tmp_path / "sc.json")
    sc1 = w.scenario()
    w.preset_normal()
    assert not w.day_mode
    sc2 = w.load_scenario_from(path)
    assert w.day_mode
    assert w.scenario().to_dict() == sc1.to_dict() == sc2.to_dict()
    w.timer.stop()
