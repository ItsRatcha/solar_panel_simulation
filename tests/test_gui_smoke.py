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
