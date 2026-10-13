"""Main window: controls on the left, live curves and time plots on the right.

Two modes:
- Manual: you move the sun/temperature/shade sliders yourself.
- Full day: a scenario (sun path, clouds, temperature, scripted shadows,
  fixed mount or tracker) drives the panel from sunrise to sunset and the
  window shows the clock and the energy collected so far.

Panel B (optional) runs next to panel A under the same sun and load, with its
own MPPT algorithm (and, in day mode, its own mount), so the two can be
compared directly.  Day runs are recorded and can be exported to CSV; the
current settings can be saved and loaded as scenario JSON.

The engine runs on the GUI thread from a 33 ms timer, with a per-frame wall
time budget.  The engine is pure Python (GIL-bound), so a worker thread would
not run it any faster and would only add locking around the shared channel.
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QDate, Qt, QTimer
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDateEdit, QDoubleSpinBox, QFileDialog, QFormLayout, QGridLayout,
    QGroupBox, QHBoxLayout, QInputDialog, QLabel, QMainWindow, QMessageBox, QPushButton,
    QScrollArea, QSlider, QSpinBox, QSplitter, QToolBox, QVBoxLayout, QWidget,
)

from ..converter.buck_averaged import BuckAveraged, BuckParams
from ..data.export import RESAMPLE_CHOICES, write_csv
from ..data.recorder import Recorder
from ..engine.day_runner import DayRunner, DaySnapshot
from ..engine.live import Conditions, LiveChannel, Snapshot
from ..environment.day import SHADE_PRESETS
from ..environment.weather import WeatherConfig
from ..load.electronic_load import ElectronicLoad
from ..mppt import available_mppt, create_mppt
from ..pv.datasheet import PanelDatasheet
from ..scenario.io import load_scenario, save_scenario
from ..scenario.model import PanelSpec, Scenario
from ..scenario.presets import PRESETS, get_preset
from ..sensors.ina226 import AVG_CHOICES, CONV_TIMES, INA226, INA226Config
from ..tracking.mounts import TrackerParams
from .controls import ParamForm

HISTORY = 1500  # samples kept in the time plots (manual mode)
HISTORY_DAY = 8000  # whole day at one sample per DAY_RECORD_S
DAY_RECORD_S = 10.0  # day mode: plot one sample every 10 s of clock time
FRAME_BUDGET_S = 0.030  # stop stepping when a frame has used this much wall time
LIVE_DT = 0.1  # MPPT period in manual mode [s]
TRAIL = 40  # recent operating points drawn on the curves
SPEEDS = {"Slow (1 step/frame)": 1, "Normal (5)": 5, "Fast (25)": 25, "Very fast (100)": 100}
# day mode: simulated seconds per real second ("Max" = as fast as the CPU allows)
DAY_SPEEDS = {"1 min / s": 60, "10 min / s": 600, "1 h / s": 3600, "Max": 1e9}
DAY_PERIODS = {"1 s (accurate)": 1.0, "5 s": 5.0, "10 s (fast)": 10.0}
SKY_LABELS = {"clear": "Clear", "partly_cloudy": "Partly cloudy", "cloudy": "Cloudy",
              "overcast": "Overcast", "random": "Random (from seed)"}
SHADE_LABELS = {"none": "No shading", "tree_morning": "Tree, 07:00-09:00",
                "pole_noon": "Pole, 11:30-12:30", "building_evening": "Building, 16:00-18:30"}
MOUNT_LABELS = {"fixed": "Fixed (tilt/azimuth below)", "tracker": "2-axis sun tracker"}
VARIABLE_STEP = ("po_variable", "inccond_variable")
ALGO_LABELS = {
    "po": "P&O (fixed step)",
    "po_variable": "P&O (variable step)",
    "inccond": "IncCond (fixed step)",
    "inccond_variable": "IncCond (variable step)",
    "global_scan": "P&O + global scan",
    "manual": "Manual duty (no MPPT)",
}
LOAD_UNITS = {"CR": ("Ω", 3.0, 0.1, 1000.0), "CC": ("A", 2.0, 0.0, 10.0), "CV": ("V", 6.0, 0.0, 40.0)}
FIXED_DEFAULTS = {"tilt": 14.0, "azimuth": 180.0}
HIST_KEYS = ("t", "p_mpp", "p_pv", "p_out", "duty", "v_pv", "v_mpp", "v_out", "g", "t_amb", "t_cell",
             "p_pv_b", "p_mpp_b", "duty_b", "eff", "eff_b", "conv_eff",
             "sun_el", "sun_az", "tilt", "az", "tilt_b", "az_b")
B_KEYS = ("p_pv_b", "p_mpp_b", "duty_b", "eff_b", "tilt_b", "az_b")
NAN = float("nan")

C_CURVE = (90, 160, 255)
C_MPP = (255, 200, 0)
C_OP = (40, 220, 90)
C_MEAS = (255, 90, 90)
C_LOCAL = (255, 140, 0)
C_B = (230, 80, 230)
C_SUN = (255, 170, 40)


def _hhmm(t_s: float, sec: bool = False) -> str:
    t = int(t_s)
    h, m, x = t // 3600, t % 3600 // 60, t % 60
    return f"{h:02d}:{m:02d}:{x:02d}" if sec else f"{h:02d}:{m:02d}"


def _conv_label(t: float) -> str:
    return f"{t * 1e6:.0f} µs" if t < 1e-3 else f"{t * 1e3:.3g} ms"


def _eff(p: float, ref: float) -> float:
    """Ratio in percent, NaN when the reference is too small to mean anything."""
    return 100.0 * p / ref if ref > 0.5 else NAN


def _default_scenario() -> Scenario:
    return Scenario(name="gui", weather=WeatherConfig(sky="partly_cloudy", seed=1),
                    panels=[PanelSpec("A")])


class LabeledSlider(QWidget):
    """Horizontal slider with a live value label; works in float units."""

    def __init__(self, lo, hi, value, scale=1.0, fmt="{:.0f}", unit="", tooltip=""):
        super().__init__()
        self.scale, self.fmt, self.unit = scale, fmt, unit
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(int(round(lo / scale)), int(round(hi / scale)))
        self.slider.setValue(int(round(value / scale)))
        self.label = QLabel()
        self.label.setMinimumWidth(70)
        self.label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.label)
        self.slider.valueChanged.connect(self._update_label)
        self._update_label()
        if tooltip:
            self.setToolTip(tooltip)

    @property
    def valueChanged(self):
        return self.slider.valueChanged

    def value(self) -> float:
        return self.slider.value() * self.scale

    def setValue(self, v: float) -> None:
        self.slider.setValue(int(round(v / self.scale)))

    def show_value(self, v: float) -> None:
        """Display a value without emitting valueChanged (label shows the exact value)."""
        self.slider.blockSignals(True)
        self.setValue(v)
        self.slider.blockSignals(False)
        self.label.setText(self.fmt.format(v) + self.unit)

    def _update_label(self):
        self.label.setText(self.fmt.format(self.value()) + self.unit)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PV + Buck + MPPT simulator")
        self.resize(1500, 1050)

        self.ch = LiveChannel()
        self.ch_b: LiveChannel | None = None
        self.trail: deque = deque(maxlen=TRAIL)
        self.hist = self._make_hist(HISTORY)
        self.last: Snapshot | None = None
        self.last_b: Snapshot | None = None
        self.runner: DayRunner | None = None
        self.runner_b: DayRunner | None = None
        self.day: DaySnapshot | None = None
        self.day_b: DaySnapshot | None = None
        self.rec: Recorder | None = None  # rows of the last day run (CSV export)
        self.out_sensors: list[INA226] = []
        self.ids: list[str] = []
        self.sc_running: Scenario | None = None
        self.base_sc = _default_scenario()  # template for fields the widgets don't show
        self._custom_shading: list | None = None
        self._events: list[tuple[float, object]] = []  # manual mode: (sim time, action)
        self._next_rec = -1.0

        self.timer = QTimer(self)
        self.timer.setInterval(33)  # ~30 frames per second
        self.timer.timeout.connect(self._tick)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_controls())
        splitter.addWidget(self._build_plots())
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([430, 1070])
        self.setCentralWidget(splitter)
        self._build_menus()

        self._apply_all()
        self._tick_once_and_draw()
        self.toggle_run()

    @staticmethod
    def _make_hist(maxlen: int) -> dict[str, deque]:
        return {k: deque(maxlen=maxlen) for k in HIST_KEYS}

    @property
    def day_mode(self) -> bool:
        return self.runner is not None

    def _channels(self) -> list[LiveChannel]:
        return [c for c in (self.ch, self.ch_b) if c is not None]

    # ------------------------------------------------------------------ controls
    def _build_controls(self) -> QWidget:
        root = QWidget()
        v = QVBoxLayout(root)

        # --- run
        g = QGroupBox("Simulation")
        f = QGridLayout(g)
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(["Manual (you move the sliders)", "Full day (sunrise → sunset)"])
        self.cmb_mode.setToolTip("Full day: sun, clouds, temperature and shadows change by themselves.")
        self.cmb_mode.currentIndexChanged.connect(self._mode_changed)
        self.btn_run = QPushButton("Start")
        self.btn_run.clicked.connect(self.toggle_run)
        btn_reset = QPushButton("Reset")
        btn_reset.setToolTip("Restart the MPPT and clear the plots (day mode: back to sunrise).")
        btn_reset.clicked.connect(self.reset)
        self.cmb_speed = QComboBox()
        self.lbl_time = QLabel("t = 0.0 s")
        f.addWidget(QLabel("Mode"), 0, 0)
        f.addWidget(self.cmb_mode, 0, 1)
        f.addWidget(self.btn_run, 1, 0)
        f.addWidget(btn_reset, 1, 1)
        f.addWidget(QLabel("Speed"), 2, 0)
        f.addWidget(self.cmb_speed, 2, 1)
        f.addWidget(self.lbl_time, 3, 0, 1, 2)
        self._fill_speeds(day=False)
        v.addWidget(g)

        # --- full day settings (only visible in day mode)
        self.grp_day = g = QGroupBox("Full day")
        f = QFormLayout(g)
        self.de_date = QDateEdit(QDate(2026, 3, 21))
        self.de_date.setCalendarPopup(True)
        self.de_date.setDisplayFormat("yyyy-MM-dd")
        self.cmb_sky = QComboBox()
        for k, lbl in SKY_LABELS.items():
            self.cmb_sky.addItem(lbl, k)
        self.cmb_sky.setCurrentIndex(self.cmb_sky.findData("partly_cloudy"))
        self.sp_seed = QSpinBox()
        self.sp_seed.setRange(0, 999999)
        self.sp_seed.setValue(1)
        self.sp_seed.setToolTip("Same seed = same clouds and temperature, so runs can be compared.")
        self.cmb_dayshade = QComboBox()
        for k in SHADE_PRESETS:
            self.cmb_dayshade.addItem(SHADE_LABELS.get(k, k), k)
        self.cmb_mount = QComboBox()
        for k, lbl in MOUNT_LABELS.items():
            self.cmb_mount.addItem(lbl, k)
        self.cmb_period = QComboBox()
        for k, val in DAY_PERIODS.items():
            self.cmb_period.addItem(k, val)
        self.cmb_period.setToolTip("Time between MPPT steps. 1 s is close to the ESP32;\n"
                                   "10 s simulates a day about 10x faster.")
        btn_day = QPushButton("Start new day")
        btn_day.setToolTip("Apply these settings and run from sunrise to sunset.")
        btn_day.clicked.connect(self.start_day)
        f.addRow("Date", self.de_date)
        f.addRow("Sky", self.cmb_sky)
        f.addRow("Weather seed", self.sp_seed)
        f.addRow("Shading", self.cmb_dayshade)
        f.addRow("Mount (A)", self.cmb_mount)
        f.addRow("MPPT period", self.cmb_period)
        f.addRow(btn_day)
        g.hide()
        v.addWidget(g)

        # --- environment
        g = QGroupBox("Sun and shading")
        f = QFormLayout(g)
        self.sl_g = LabeledSlider(0, 1200, 1000, 10, unit=" W/m²",
                                  tooltip="Sunlight hitting the panel (1000 W/m² = full sun).")
        self.sl_t = LabeledSlider(10, 45, 30, 1, fmt="{:.1f}", unit=" °C",
                                  tooltip="Air temperature around the panel.")
        f.addRow("Irradiance", self.sl_g)
        f.addRow("Ambient temp", self.sl_t)
        self.sl_shade = []
        for k in range(3):
            s = LabeledSlider(0, 100, 0, 1, unit=" %",
                              tooltip=f"How much of cell group {k + 1} (12 cells, 1 bypass diode) is shaded.")
            self.sl_shade.append(s)
            f.addRow(f"Shade group {k + 1}", s)
        self.lbl_tcell = QLabel()
        f.addRow("Cell temp", self.lbl_tcell)
        for w in (self.sl_g, self.sl_t, *self.sl_shade):
            w.valueChanged.connect(self._apply_conditions)
        v.addWidget(g)

        # --- load
        g = QGroupBox("Electronic load")
        f = QFormLayout(g)
        self.cmb_load = QComboBox()
        self.cmb_load.addItems(["CR", "CC", "CV"])
        self.cmb_load.setToolTip("CR = constant resistance, CC = constant current, CV = constant voltage.")
        self.sp_load = QDoubleSpinBox()
        self.sp_load.setDecimals(2)
        self.sp_load.setSingleStep(0.5)
        self.cmb_load.currentTextChanged.connect(self._load_mode_changed)
        self.sp_load.valueChanged.connect(self._apply_load)
        f.addRow("Mode", self.cmb_load)
        f.addRow("Setpoint", self.sp_load)
        v.addWidget(g)

        # --- MPPT
        g = QGroupBox("MPPT algorithm (panel A)")
        f = QFormLayout(g)
        self.cmb_algo = QComboBox()
        for name in available_mppt():
            self.cmb_algo.addItem(ALGO_LABELS.get(name, name), name)
        self.sp_step = QDoubleSpinBox()
        self.sp_step.setDecimals(4)
        self.sp_step.setRange(0.0005, 0.05)
        self.sp_step.setSingleStep(0.001)
        self.sp_step.setValue(0.005)
        self.lbl_step = QLabel("Step size")
        self.sl_duty = LabeledSlider(0, 0.95, 0.5, 0.005, fmt="{:.3f}",
                                     tooltip="Duty cycle used by 'Manual duty'.")
        self.cmb_algo.currentIndexChanged.connect(self._apply_algo)
        self.sp_step.valueChanged.connect(self._apply_algo)
        self.sl_duty.valueChanged.connect(self._manual_duty)
        f.addRow("Algorithm", self.cmb_algo)
        f.addRow(self.lbl_step, self.sp_step)
        f.addRow("Manual duty", self.sl_duty)
        v.addWidget(g)

        # --- A/B compare
        g = QGroupBox("Compare: panel B")
        f = QFormLayout(g)
        self.chk_b = QCheckBox("Run panel B next to A")
        self.chk_b.setToolTip("B sees the same sun, shade and load as A.\n"
                              "Pick a different algorithm (or, in day mode, a different mount).")
        self.cmb_algo_b = QComboBox()
        for name in available_mppt():
            self.cmb_algo_b.addItem(ALGO_LABELS.get(name, name), name)
        self.cmb_algo_b.setCurrentIndex(self.cmb_algo_b.findData("global_scan"))
        self.cmb_mount_b = QComboBox()
        for k, lbl in MOUNT_LABELS.items():
            self.cmb_mount_b.addItem(lbl, k)
        self.cmb_mount_b.setCurrentIndex(self.cmb_mount_b.findData("tracker"))
        self.cmb_mount_b.setToolTip("Only used in full-day mode.")
        self.chk_b.toggled.connect(self._b_toggled)
        self.cmb_algo_b.currentIndexChanged.connect(self._apply_algo_b)
        self.cmb_mount_b.currentIndexChanged.connect(self._mount_b_changed)
        f.addRow(self.chk_b)
        f.addRow("Algorithm (B)", self.cmb_algo_b)
        f.addRow("Mount (B)", self.cmb_mount_b)
        v.addWidget(g)

        # --- realism
        g = QGroupBox("Realism")
        f = QVBoxLayout(g)
        self.chk_losses = QCheckBox("Converter losses (MOSFET, diode, inductor…)")
        self.chk_losses.setChecked(True)
        self.chk_sensor = QCheckBox("INA226 sensor noise + resolution")
        self.chk_sensor.setChecked(True)
        self.chk_losses.toggled.connect(self._apply_converter)
        self.chk_sensor.toggled.connect(self._apply_sensor)
        f.addWidget(self.chk_losses)
        f.addWidget(self.chk_sensor)
        v.addWidget(g)

        v.addWidget(self._build_advanced())

        # --- presets
        g = QGroupBox("Try these")
        f = QVBoxLayout(g)
        for text, fn, tip in (
            ("Full sun, normal", self.preset_normal, "1000 W/m², no shade, 3 Ω load, P&O."),
            ("Shading trap (P&O)", lambda: self.preset_shading("po"),
             "One cell group 75 % shaded. P&O gets stuck on the small peak."),
            ("Shading + global scan", lambda: self.preset_shading("global_scan"),
             "Same shading; global scan sweeps first and finds the real peak."),
            ("Shading: P&O (A) vs global scan (B)", self.preset_shading_ab,
             "Both algorithms side by side on the same shaded panel."),
            ("Cloud step: 1000 → 300 W/m²", self.preset_cloud,
             "Sudden drop in sunlight at t = 4 s; watch it re-track."),
            ("Fast cloud ramp", self.preset_cloud_ramp,
             "1000 → 200 W/m² over 2 s, then back: can the MPPT keep up?"),
            ("Load step: 3 → 8 Ω", self.preset_load_step,
             "Load resistance jumps at t = 4 s; at 8 Ω the MPP is out of the buck's reach."),
            ("Load too big (unreachable)", self.preset_unreachable,
             "50 Ω load: a buck can't pull the panel down to its MPP."),
            ("Full day: clouds + tree shadow", self.preset_day,
             "Partly cloudy equinox day, a tree shades the panel 07:00-09:00, fixed mount."),
            ("Full day: fixed vs tracker", self.preset_ab,
             "Clear day; A on a fixed mount, B on a 2-axis tracker. Compare the energy."),
        ):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            f.addWidget(b)
        v.addWidget(g)
        v.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(root)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(410)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        return scroll

    def _build_advanced(self) -> QWidget:
        self.grp_adv = g = QGroupBox("Advanced parameters")
        g.setCheckable(True)
        g.setChecked(False)
        lay = QVBoxLayout(g)
        self.tb_adv = QToolBox()
        self.f_ds = ParamForm(PanelDatasheet().to_dict(), skip=("n_bypass",), tips={
            "ideality": "Empty = fitted from the datasheet.",
            "noct": "Nominal operating cell temperature [°C].",
        })
        self.f_buck = ParamForm(BuckParams().to_dict())
        conv = {("t_conv_bus", t): _conv_label(t) for t in CONV_TIMES}
        conv.update({("t_conv_shunt", t): _conv_label(t) for t in CONV_TIMES})
        self.f_sensor = ParamForm(INA226Config().to_dict(),
                                  choices={"avg": AVG_CHOICES, "t_conv_bus": CONV_TIMES,
                                           "t_conv_shunt": CONV_TIMES},
                                  labels=conv)
        self.f_fixed = ParamForm(FIXED_DEFAULTS, tips={"azimuth": "180 = facing south."})
        self.f_tracker = ParamForm(TrackerParams().to_dict())
        for form, title in ((self.f_ds, "Panel datasheet"), (self.f_buck, "Buck converter"),
                            (self.f_sensor, "INA226 sensor"), (self.f_fixed, "Fixed mount (next day)"),
                            (self.f_tracker, "Tracker (next day)")):
            self.tb_adv.addItem(form, title)
        self.f_ds.changed.connect(self._apply_datasheet)
        self.f_buck.changed.connect(self._apply_converter)
        self.f_sensor.changed.connect(self._apply_sensor)
        note = QLabel("Panel, converter and sensor changes apply at once. "
                      "Mount changes apply on the next 'Start new day'.")
        note.setWordWrap(True)
        lay.addWidget(note)
        lay.addWidget(self.tb_adv)
        self.tb_adv.setVisible(False)
        note.setVisible(False)
        g.toggled.connect(self.tb_adv.setVisible)
        g.toggled.connect(note.setVisible)
        return g

    def _build_menus(self):
        m = self.menuBar().addMenu("&File")
        for text, fn, key in (("Save scenario…", self._save_dialog, QKeySequence.Save),
                              ("Load scenario…", self._load_dialog, QKeySequence.Open),
                              ("Export day to CSV…", self._export_dialog, None)):
            a = m.addAction(text)
            a.triggered.connect(fn)
            if key is not None:
                a.setShortcut(key)
        m.addSeparator()
        a = m.addAction("Quit")
        a.setShortcut(QKeySequence.Quit)
        a.triggered.connect(self.close)
        m = self.menuBar().addMenu("&Scenarios")
        for name, build in PRESETS.items():
            a = m.addAction(name.replace("_", " "))
            a.setToolTip((build.__doc__ or "").strip())
            a.triggered.connect(lambda _=False, n=name: self.apply_scenario(get_preset(n)))

    def _fill_speeds(self, day: bool):
        self.cmb_speed.blockSignals(True)
        self.cmb_speed.clear()
        self.cmb_speed.addItems(DAY_SPEEDS if day else SPEEDS)
        self.cmb_speed.setCurrentIndex(1)
        self.cmb_speed.setToolTip(
            "Simulated time per real second. Max = as fast as your computer allows." if day else
            "MPPT steps simulated per screen frame (one step = 0.1 s).")
        self.cmb_speed.blockSignals(False)

    # --------------------------------------------------------------------- plots
    def _build_plots(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)

        # readouts
        self.lbl_big = QLabel()
        self.lbl_big.setTextFormat(Qt.RichText)
        self.lbl_big.setStyleSheet("font-size: 15px;")
        self.lbl_warn = QLabel()
        self.lbl_warn.setWordWrap(True)
        self.lbl_warn.setStyleSheet("color: white; background: #b03030; padding: 4px; font-weight: bold;")
        self.lbl_warn.hide()
        v.addWidget(self.lbl_big)
        v.addWidget(self.lbl_warn)

        glw = pg.GraphicsLayoutWidget()
        v.addWidget(glw, 1)

        def curve_plot(title, ylabel, yunit):
            p = glw.addPlot(title=title)
            p.setLabel("bottom", "Panel voltage", "V")
            p.setLabel("left", ylabel, yunit)
            p.showGrid(x=True, y=True, alpha=0.3)
            p.addLegend(offset=(-10, 10))
            return p

        self.p_iv = curve_plot("I-V curve (current vs voltage)", "Current", "A")
        self.p_pv = curve_plot("P-V curve (power vs voltage) — the MPPT climbs this hill", "Power", "W")
        self.items = {}
        for name, plot, is_p in (("iv", self.p_iv, False), ("pv", self.p_pv, True)):
            self.items[name] = dict(
                curve=plot.plot(pen=pg.mkPen(C_CURVE, width=2), name="Panel curve"),
                curve_b=plot.plot(pen=pg.mkPen(C_B, width=1.5, style=Qt.DashLine),
                                  name="Panel B curve" if is_p else None),
                local=plot.plot(pen=None, symbol="o", symbolSize=11, symbolBrush=None,
                                symbolPen=pg.mkPen(C_LOCAL, width=2), name="Other peaks" if is_p else None),
                trail=plot.plot(pen=pg.mkPen((40, 220, 90, 110), width=1), symbol="o", symbolSize=4,
                                symbolBrush=(40, 220, 90, 110), symbolPen=None,
                                name="Recent operating points"),
                mpp=plot.plot(pen=None, symbol="star", symbolSize=20, symbolBrush=C_MPP,
                              symbolPen=pg.mkPen("k"), name="True max power point"),
                mpp_b=plot.plot(pen=None, symbol="star", symbolSize=14, symbolBrush=C_B,
                                symbolPen=pg.mkPen("k")),
                op=plot.plot(pen=None, symbol="o", symbolSize=14, symbolBrush=C_OP,
                             symbolPen=pg.mkPen("k"), name="Operating point (true)"),
                op_b=plot.plot(pen=None, symbol="t", symbolSize=14, symbolBrush=C_B,
                               symbolPen=pg.mkPen("k"), name="Panel B operating point"),
                meas=plot.plot(pen=None, symbol="x", symbolSize=12, symbolPen=pg.mkPen(C_MEAS, width=2),
                               name="What the INA226 measures"),
            )

        def time_plot(title, ylabel, yunit=None, legend=True, colspan=1):
            p = glw.addPlot(title=title, colspan=colspan)
            p.setLabel("left", ylabel, yunit)
            p.showGrid(x=True, y=True, alpha=0.3)
            if legend:
                p.addLegend(offset=(10, 10))
            return p

        def line(plot, color, name=None, dash=False, width=2):
            pen = pg.mkPen(color, width=width, style=Qt.DashLine if dash else Qt.SolidLine)
            return plot.plot(pen=pen, name=name, connect="finite")

        glw.nextRow()
        self.p_pow = time_plot("Power over time", "Power", "W", colspan=2)
        self.l_pmpp = line(self.p_pow, C_MPP, "Best possible (true MPP)", dash=True)
        self.l_ppv = line(self.p_pow, C_CURVE, "Panel power (P_in)")
        self.l_pout = line(self.p_pow, C_OP, "Power to load (P_out)")
        self.l_pmpp_b = line(self.p_pow, C_B, "B best possible", dash=True, width=1)
        self.l_ppv_b = line(self.p_pow, C_B, "B panel power")

        glw.nextRow()
        self.p_duty = time_plot("Duty cycle chosen by the MPPT", "Duty D")
        self.l_duty = line(self.p_duty, (200, 120, 255), "A")
        self.l_duty_b = line(self.p_duty, C_B, "B", dash=True)

        self.p_volt = time_plot("Voltages", "Voltage", "V")
        self.l_vmpp = line(self.p_volt, C_MPP, "V at true MPP", dash=True)
        self.l_vpv = line(self.p_volt, C_CURVE, "Panel V (in)")
        self.l_vout = line(self.p_volt, C_OP, "Load V (out)")

        glw.nextRow()
        self.p_eff = time_plot("Efficiency over time", "Efficiency", "%")
        self.l_eff = line(self.p_eff, C_CURVE, "A MPPT (P_pv ÷ P_mpp)")
        self.l_eff_b = line(self.p_eff, C_B, "B MPPT")
        self.l_conv = line(self.p_eff, C_OP, "A converter (P_out ÷ P_pv)", width=1)

        self.p_ang = time_plot("Sun & panel angles (full day)", "Angle", "°")
        self.l_sun_el = line(self.p_ang, C_SUN, "Sun elevation")
        self.l_sun_az = line(self.p_ang, C_SUN, "Sun azimuth", dash=True, width=1)
        self.l_tilt = line(self.p_ang, C_CURVE, "A tilt")
        self.l_az = line(self.p_ang, C_CURVE, "A azimuth", dash=True, width=1)
        self.l_tilt_b = line(self.p_ang, C_B, "B tilt")
        self.l_az_b = line(self.p_ang, C_B, "B azimuth", dash=True, width=1)

        glw.nextRow()
        self.p_irr = time_plot("Sunlight on the panel", "Irradiance", "W/m²", legend=False)
        self.l_irr = line(self.p_irr, C_SUN)

        self.p_temp = time_plot("Temperature", "Temperature", "°C")
        self.l_tamb = line(self.p_temp, (120, 200, 255), "Air")
        self.l_tcell = line(self.p_temp, (255, 90, 90), "Cell")

        self.time_plots = (self.p_pow, self.p_duty, self.p_volt, self.p_eff, self.p_ang,
                           self.p_irr, self.p_temp)
        for p in self.time_plots:
            p.setDownsampling(auto=True, mode="peak")
            p.setClipToView(True)
            if p not in (self.p_duty, self.p_pow):
                p.setXLink(self.p_duty)
        # pyqtgraph's x-link maps by pixel position, so the double-width power plot
        # would show twice the time span; keep its range equal by hand instead
        self._syncing = False
        self.p_duty.sigXRangeChanged.connect(lambda _, r: self._sync_x(self.p_pow, r))
        self.p_pow.sigXRangeChanged.connect(lambda _, r: self._sync_x(self.p_duty, r))
        self._set_time_axis(day=False)

        # duty first: the other plots clip to its x range, so it must update before them
        self.lines = (
            (self.l_duty, "duty"), (self.l_duty_b, "duty_b"),
            (self.l_pmpp, "p_mpp"), (self.l_ppv, "p_pv"), (self.l_pout, "p_out"),
            (self.l_pmpp_b, "p_mpp_b"), (self.l_ppv_b, "p_pv_b"),
            (self.l_vmpp, "v_mpp"), (self.l_vpv, "v_pv"), (self.l_vout, "v_out"),
            (self.l_eff, "eff"), (self.l_eff_b, "eff_b"), (self.l_conv, "conv_eff"),
            (self.l_sun_el, "sun_el"), (self.l_sun_az, "sun_az"), (self.l_tilt, "tilt"),
            (self.l_az, "az"), (self.l_tilt_b, "tilt_b"), (self.l_az_b, "az_b"),
            (self.l_irr, "g"), (self.l_tamb, "t_amb"), (self.l_tcell, "t_cell"),
        )

        hint = QLabel(
            "<b>How to read this:</b> the <span style='color:#ffc800'>★ star</span> is the best possible point. "
            "The <span style='color:#28dc5a'>● green dot</span> is where the panel actually runs "
            "(<span style='color:#e650e6'>▼ magenta</span> = panel B). "
            "The MPPT moves the duty cycle to push the green dot onto the star. "
            "Higher duty → lower panel voltage (dot moves left)."
        )
        hint.setWordWrap(True)
        hint.setTextFormat(Qt.RichText)
        v.addWidget(hint)
        return w

    def _sync_x(self, plot, rng):
        if self._syncing:
            return
        self._syncing = True
        try:
            plot.setXRange(*rng, padding=0)
        finally:
            self._syncing = False

    def _set_time_axis(self, day: bool):
        for p in self.time_plots:
            if day:
                p.setLabel("bottom", "Time of day", "h")
            else:
                p.setLabel("bottom", "Time", "s")
            p.getAxis("bottom").enableAutoSIPrefix(not day)
        self.p_duty.enableAutoRange(axis="x")

    # -------------------------------------------------------- parameter objects
    def _try(self, what: str, fn):
        """Run fn(); on a bad parameter show why in the status bar and return None."""
        try:
            return fn()
        except (ValueError, TypeError, RuntimeError) as e:
            self.statusBar().showMessage(f"{what}: {e}", 8000)
            return None

    def _datasheet(self) -> PanelDatasheet | None:
        return self._try("Panel datasheet", lambda: PanelDatasheet.from_dict(
            {**PanelDatasheet().to_dict(), **self.f_ds.values()}))

    def _buck_params(self) -> BuckParams | None:
        if not self.chk_losses.isChecked():
            return BuckParams.ideal()
        return self._try("Buck converter", lambda: BuckParams.from_dict(self.f_buck.values()))

    def _sensor_cfg(self) -> INA226Config | None:
        if not self.chk_sensor.isChecked():
            return INA226Config.ideal()
        return self._try("INA226", lambda: INA226Config.from_dict(self.f_sensor.values()))

    def _algo_kw(self, name: str) -> dict:
        if name == "manual":
            return {"duty": self.sl_duty.value()}
        if name in VARIABLE_STEP:
            # the step adapts by itself; the box sets its upper limit
            return {"step_max": self.sp_step.value()}
        return {"step": self.sp_step.value()}

    def _make_algo(self, name: str, d0: float):
        return create_mppt(name, d_init=d0, **self._algo_kw(name))

    def _make_manual_b(self) -> LiveChannel:
        a = self.ch
        return LiveChannel(
            datasheet=a.datasheet,
            conv=BuckAveraged(self._buck_params() or BuckParams()),
            load=a.load,
            algo=self._make_algo(self.cmb_algo_b.currentData(), 0.3),
            sensor=INA226(self._sensor_cfg() or INA226Config()),
            cond=a.cond,
            dt=LIVE_DT,
        )

    # ------------------------------------------------------------- apply changes
    def _apply_all(self):
        self._load_mode_changed(self.cmb_load.currentText())
        self._apply_conditions()
        self._apply_converter()
        self._apply_sensor()
        self._apply_algo()

    def _apply_conditions(self, *_):
        if self.day_mode:
            return  # the day environment drives the panel
        self.ch.cond = Conditions(
            g=self.sl_g.value(),
            shade=tuple(s.value() / 100.0 for s in self.sl_shade),
            t_amb=self.sl_t.value(),
        )
        if self.ch_b is not None:
            self.ch_b.cond = self.ch.cond
        self.lbl_tcell.setText(f"{self.ch.t_cell():.1f} °C  (panel heats up in the sun)")
        if not self.timer.isActive():
            self._redraw_curves_only()

    def _set_load_widgets(self, mode: str, setpoint: float | None = None):
        unit, default, lo, hi = LOAD_UNITS[mode]
        self.sp_load.blockSignals(True)
        self.sp_load.setRange(lo, hi)
        self.sp_load.setSuffix(" " + unit)
        self.sp_load.setValue(default if setpoint is None else setpoint)
        self.sp_load.blockSignals(False)

    def _load_mode_changed(self, mode: str):
        self._set_load_widgets(mode)
        self._apply_load()

    def _apply_load(self, *_):
        load = ElectronicLoad(self.cmb_load.currentText(), self.sp_load.value())
        for c in self._channels():
            c.load = load

    def _apply_converter(self, *_):
        self.f_buck.setEnabled(self.chk_losses.isChecked())
        params = self._buck_params()
        if params is None:
            return
        for c in self._channels():
            c.conv = BuckAveraged(params)

    def _apply_sensor(self, *_):
        self.f_sensor.setEnabled(self.chk_sensor.isChecked())
        cfg = self._sensor_cfg()
        if cfg is None:
            return
        for c in self._channels():
            c.sensor = INA226(cfg)

    def _apply_datasheet(self, *_):
        ds = self._datasheet()
        if ds is None:
            return
        for c in self._channels():
            if self._try("Panel datasheet", lambda c=c: c.set_datasheet(ds)) is None and c.datasheet is not ds:
                return
        self.statusBar().showMessage(f"Panel: {ds.name}, {ds.p_max:g} W", 4000)
        if not self.timer.isActive():
            self._redraw_curves_only()

    def _apply_algo(self, *_):
        name = self.cmb_algo.currentData()
        d0 = self.ch.algo.duty if self.last is not None else 0.3
        self.ch.algo = self._make_algo(name, d0)
        variable = name in VARIABLE_STEP
        self.sl_duty.setEnabled(name == "manual" or self.cmb_algo_b.currentData() == "manual")
        self.sp_step.setEnabled(name != "manual")
        self.lbl_step.setText("Max step" if variable else "Step size")
        self.sp_step.setToolTip(
            "Variable step: the step shrinks near the peak and grows far from it,\n"
            "up to this maximum." if variable else
            "How much the duty cycle changes per MPPT step.\n"
            "Bigger = faster but more wobble around the peak.")
        self._apply_algo_b()

    def _apply_algo_b(self, *_):
        self.sl_duty.setEnabled("manual" in (self.cmb_algo.currentData(), self.cmb_algo_b.currentData()))
        if self.ch_b is not None:
            self.ch_b.algo = self._make_algo(self.cmb_algo_b.currentData(), self.ch_b.algo.duty)

    def _manual_duty(self, *_):
        for c in self._channels():
            if c.algo.name == "manual":
                c.algo.set_duty(self.sl_duty.value())

    def _b_toggled(self, on: bool):
        if self.day_mode:
            self.start_day()
            return
        self.ch_b = self._make_manual_b() if on else None
        self.reset()

    def _mount_b_changed(self, *_):
        if self.day_mode and self.chk_b.isChecked():
            self.start_day()

    # ------------------------------------------------------------------ presets
    def _set(self, g=1000, t=30, shade=(0, 0, 0), mode="CR", setpoint=3.0, algo="po"):
        if self.cmb_mode.currentIndex() != 0:
            self.cmb_mode.setCurrentIndex(0)
        for w, val in ((self.sl_g, g), (self.sl_t, t), *zip(self.sl_shade, shade)):
            w.show_value(val)
        self.cmb_load.blockSignals(True)
        self.cmb_load.setCurrentText(mode)
        self.cmb_load.blockSignals(False)
        self._set_load_widgets(mode, setpoint)
        self.cmb_algo.blockSignals(True)
        self.cmb_algo.setCurrentIndex(self.cmb_algo.findData(algo))
        self.cmb_algo.blockSignals(False)
        self._apply_conditions()
        self._apply_load()
        self._apply_algo()

    def _at(self, t: float, fn):
        """Manual mode: run fn when the simulated time reaches t."""
        self._events.append((t, fn))
        self._events.sort(key=lambda e: e[0])

    def preset_normal(self):
        self._set()
        self.reset(d_init=0.3)

    def preset_shading(self, algo):
        self._set(shade=(0, 0, 75), algo=algo)
        self.reset(d_init=0.2)

    def preset_shading_ab(self):
        self.chk_b.blockSignals(True)
        self.chk_b.setChecked(True)
        self.chk_b.blockSignals(False)
        self.cmb_algo_b.blockSignals(True)
        self.cmb_algo_b.setCurrentIndex(self.cmb_algo_b.findData("global_scan"))
        self.cmb_algo_b.blockSignals(False)
        self._set(shade=(0, 0, 75), algo="po")
        self.ch_b = self._make_manual_b()
        self.reset(d_init=0.2)

    def preset_cloud(self):
        self._set(g=1000)
        self.reset(d_init=0.3)
        self._at(4.0, lambda: self.sl_g.setValue(300))

    def preset_cloud_ramp(self):
        self._set(g=1000)
        self.reset(d_init=0.3)
        n = 20
        for k in range(1, n + 1):
            down = 1000.0 - 800.0 * k / n
            self._at(3.0 + 0.1 * k, lambda g=down: self.sl_g.setValue(g))
            self._at(9.0 + 0.1 * k, lambda g=1200.0 - down: self.sl_g.setValue(g))

    def preset_load_step(self):
        self._set()
        self.reset(d_init=0.3)
        self._at(4.0, lambda: self.sp_load.setValue(8.0))

    def preset_unreachable(self):
        self._set(setpoint=50.0)
        self.reset(d_init=0.3)

    def preset_day(self):
        self._set()
        self.base_sc = _default_scenario()
        self.de_date.setDate(QDate(2026, 3, 21))
        self.cmb_sky.setCurrentIndex(self.cmb_sky.findData("partly_cloudy"))
        self.sp_seed.setValue(3)
        self.cmb_dayshade.setCurrentIndex(self.cmb_dayshade.findData("tree_morning"))
        self.cmb_mount.setCurrentIndex(self.cmb_mount.findData("fixed"))
        self.cmb_period.setCurrentIndex(0)
        self.cmb_mode.setCurrentIndex(1)  # builds the day and starts it

    def preset_ab(self):
        self.apply_scenario(get_preset("clear_day_fixed_vs_tracker"))

    # ---------------------------------------------------------------- scenarios
    def _panel_spec(self, pid: str, algo: str, mount: str, base: PanelSpec | None) -> PanelSpec:
        mount_params = self.f_fixed.values() if mount == "fixed" else self.f_tracker.values()
        mppt = dict(base.mppt_params) if base is not None and base.algorithm == algo else {}
        mppt.update(self._algo_kw(algo))
        buck = self.f_buck.values() if self.chk_losses.isChecked() else BuckParams.ideal().to_dict()
        sensor = self.f_sensor.values() if self.chk_sensor.isChecked() else INA226Config.ideal().to_dict()
        return PanelSpec(
            panel_id=pid, mount=mount, mount_params=mount_params, algorithm=algo, mppt_params=mppt,
            datasheet=self.f_ds.values(), buck=buck, sensor=sensor,
            load={"mode": self.cmb_load.currentText(), "setpoint": self.sp_load.value()},
            load_profile=base.load_profile if base is not None else None,
        )

    def scenario(self) -> Scenario:
        """The scenario the widgets describe (fields they don't show come from `base_sc`).

        Raises ValueError if a parameter box does not parse.
        """
        base = self.base_sc
        qd = self.de_date.date()
        weather = WeatherConfig.from_dict({**base.weather.to_dict(), "sky": self.cmb_sky.currentData(),
                                           "seed": self.sp_seed.value()})
        shade = self.cmb_dayshade.currentData()
        shading = list(self._custom_shading or []) if shade == "custom" else shade
        bp = list(base.panels)
        id_a = bp[0].panel_id if bp else "A"
        panels = [self._panel_spec(id_a, self.cmb_algo.currentData(), self.cmb_mount.currentData(),
                                   bp[0] if bp else None)]
        if self.chk_b.isChecked():
            id_b = bp[1].panel_id if len(bp) > 1 else ("B2" if id_a == "B" else "B")
            panels.append(self._panel_spec(id_b, self.cmb_algo_b.currentData(),
                                           self.cmb_mount_b.currentData(), bp[1] if len(bp) > 1 else None))
        ids = {p.panel_id for p in panels}
        faults = [dict(e) for e in base.faults if e.get("panel_id") in (None, *ids)]
        return Scenario(
            name=base.name, date=f"{qd.year():04d}-{qd.month():02d}-{qd.day():02d}", weather=weather,
            shading=shading, faults=faults, panels=panels, mppt_period=float(self.cmb_period.currentData()),
            env_dt=base.env_dt, margin_s=base.margin_s, start_h=base.start_h, end_h=base.end_h,
            seed=base.seed, latitude=base.latitude, longitude=base.longitude, tz_hours=base.tz_hours,
        )

    def _select_period(self, period: float):
        idx = next((i for i in range(self.cmb_period.count())
                    if abs(self.cmb_period.itemData(i) - period) < 1e-9), -1)
        if idx < 0:
            self.cmb_period.addItem(f"{period:g} s", float(period))
            idx = self.cmb_period.count() - 1
        self.cmb_period.setCurrentIndex(idx)

    def _set_mount_form(self, p: PanelSpec):
        if p.mount == "fixed":
            self.f_fixed.set_values({**FIXED_DEFAULTS, **p.mount_params})
        else:
            self.f_tracker.set_values({**TrackerParams().to_dict(), **p.mount_params})

    def apply_scenario(self, sc: Scenario):
        """Show a scenario in the widgets and start it in full-day mode."""
        self.timer.stop()
        self.base_sc = sc
        blocked = (self.de_date, self.cmb_sky, self.sp_seed, self.cmb_dayshade, self.cmb_mount,
                   self.cmb_period, self.cmb_algo, self.sp_step, self.cmb_load, self.sp_load,
                   self.chk_losses, self.chk_sensor, self.chk_b, self.cmb_algo_b, self.cmb_mount_b)
        for w in blocked:
            w.blockSignals(True)
        try:
            d = sc.day
            self.de_date.setDate(QDate(d.year, d.month, d.day))
            idx = self.cmb_sky.findData(sc.weather.sky)
            if idx >= 0:
                self.cmb_sky.setCurrentIndex(idx)
            self.sp_seed.setValue(sc.weather.seed if sc.weather.seed is not None else sc.seed)
            if isinstance(sc.shading, str):
                self._custom_shading = None
                self.cmb_dayshade.setCurrentIndex(max(0, self.cmb_dayshade.findData(sc.shading)))
            else:
                self._custom_shading = [dict(e) for e in sc.shading]
                if self.cmb_dayshade.findData("custom") < 0:
                    self.cmb_dayshade.addItem("Custom (from scenario)", "custom")
                self.cmb_dayshade.setCurrentIndex(self.cmb_dayshade.findData("custom"))
            self._select_period(sc.mppt_period)

            pa = sc.panels[0]
            for p, cmb_algo, cmb_mount in ((pa, self.cmb_algo, self.cmb_mount),
                                           *(((sc.panels[1], self.cmb_algo_b, self.cmb_mount_b),)
                                             if len(sc.panels) > 1 else ())):
                cmb_algo.setCurrentIndex(max(0, cmb_algo.findData(p.algorithm)))
                cmb_mount.setCurrentIndex(cmb_mount.findData("fixed" if p.mount == "fixed" else "tracker"))
            mp = pa.mppt_params
            for key in ("step", "step_max"):
                if key in mp:
                    self.sp_step.setValue(mp[key])
            if "duty" in mp:
                self.sl_duty.show_value(mp["duty"])
            self._set_mount_form(pa)
            if len(sc.panels) > 1 and (sc.panels[1].mount == "fixed") != (pa.mount == "fixed"):
                self._set_mount_form(sc.panels[1])
            self.chk_b.setChecked(len(sc.panels) > 1)

            mode = pa.load.get("mode", "CR")
            self.cmb_load.setCurrentText(mode)
            self._set_load_widgets(mode, pa.load.get("setpoint"))
            buck = BuckParams.from_dict(pa.buck)
            self.chk_losses.setChecked(buck != BuckParams.ideal())
            if buck != BuckParams.ideal():
                self.f_buck.set_values(buck.to_dict())
            cfg = INA226Config.from_dict(pa.sensor)
            self.chk_sensor.setChecked(not cfg.is_ideal)
            if not cfg.is_ideal:
                self.f_sensor.set_values(cfg.to_dict())
            self.f_ds.set_values({**PanelDatasheet().to_dict(), **pa.datasheet})
        finally:
            for w in blocked:
                w.blockSignals(False)
        self.f_buck.setEnabled(self.chk_losses.isChecked())
        self.f_sensor.setEnabled(self.chk_sensor.isChecked())
        if self.cmb_mode.currentIndex() != 1:
            self.cmb_mode.setCurrentIndex(1)  # starts the day through _mode_changed
        else:
            self.start_day()
        if len(sc.panels) > 2:
            self.statusBar().showMessage(
                f"{sc.name}: only the first two of {len(sc.panels)} panels are shown "
                f"(run `pvsim run` for all of them).", 10000)

    def save_scenario_to(self, path) -> Path:
        out = save_scenario(self.scenario(), path)
        self.statusBar().showMessage(f"Scenario saved to {out}", 6000)
        return out

    def load_scenario_from(self, path) -> Scenario:
        sc = load_scenario(path)
        self.apply_scenario(sc)
        return sc

    def export_csv_to(self, path, rule: str = "raw") -> Path:
        if self.rec is None or self.rec.n_rows == 0:
            raise ValueError("nothing recorded yet: run a full day first")
        out = write_csv(self.rec.frame(), path, rule)
        self.statusBar().showMessage(f"{self.rec.n_rows} rows exported to {out}", 6000)
        return out

    def _save_dialog(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save scenario", "scenario.json", "Scenario (*.json)")
        if path:
            try:
                self.save_scenario_to(path)
            except (OSError, ValueError, TypeError) as e:
                QMessageBox.warning(self, "Save scenario", str(e))

    def _load_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load scenario", "", "Scenario (*.json)")
        if path:
            try:
                self.load_scenario_from(path)
            except (OSError, ValueError, TypeError, KeyError) as e:
                QMessageBox.warning(self, "Load scenario", f"Could not load {path}:\n{e}")

    def _export_dialog(self):
        if self.rec is None or self.rec.n_rows == 0:
            QMessageBox.information(self, "Export CSV", "Nothing recorded yet: run a full day first.")
            return
        rule, ok = QInputDialog.getItem(self, "Export CSV", "Resample to", list(RESAMPLE_CHOICES), 0, False)
        if not ok:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export CSV", "pvsim_day.csv", "CSV (*.csv)")
        if path:
            try:
                self.export_csv_to(path, rule)
            except (OSError, ValueError) as e:
                QMessageBox.warning(self, "Export CSV", str(e))

    # ----------------------------------------------------------------- day mode
    def _mode_changed(self, idx: int):
        day = idx == 1
        self.timer.stop()
        self._events.clear()
        self.grp_day.setVisible(day)
        for w in (self.sl_g, self.sl_t, *self.sl_shade):
            w.setEnabled(not day)
        self._fill_speeds(day)
        self._set_time_axis(day)
        if day:
            self.start_day()
            return
        self.runner = self.runner_b = self.day = self.day_b = None
        self.ch.dt = LIVE_DT
        self.ch.shorted, self.ch.open_circuit = None, False  # clear injected faults
        self.ch_b = self._make_manual_b() if self.chk_b.isChecked() else None
        self.hist = self._make_hist(HISTORY)
        self.statusBar().clearMessage()
        self._apply_conditions()
        self._apply_load()
        self.reset()
        self.toggle_run()

    def start_day(self):
        if self.cmb_mode.currentIndex() != 1:
            self.cmb_mode.setCurrentIndex(1)  # comes back here through _mode_changed
            return
        self.timer.stop()
        try:
            sc = self.scenario()
            env = sc.environment()
            built = [sc.build_channel(k) for k in range(len(sc.panels))]
        except (ValueError, TypeError, RuntimeError, KeyError) as e:
            msg = f"Cannot start the day: {e}"
            if self.runner is None:
                self.cmb_mode.setCurrentIndex(0)  # back to manual
            self.statusBar().showMessage(msg)
            return
        t_start = sc.start_h * 3600.0 if sc.start_h is not None else None
        t_end = sc.end_h * 3600.0 if sc.end_h is not None else None
        runners = [DayRunner(ch, env, mount, env_dt=sc.env_dt, margin_s=sc.margin_s,
                             t_start=t_start, t_end=t_end, load_profile=profile,
                             faults=sc.fault_schedule(k))
                   for k, (ch, mount, _, profile) in enumerate(built)]
        self.ch, self.runner = runners[0].ch, runners[0]
        if len(runners) > 1:
            self.ch_b, self.runner_b = runners[1].ch, runners[1]
        else:
            self.ch_b = self.runner_b = None
        self.out_sensors = [b[2] for b in built]
        self.ids = [p.panel_id for p in sc.panels]
        self.sc_running = sc
        self.rec = Recorder(env.midnight, sc.tz_hours, extended=True)
        self.day = self.day_b = self.last = self.last_b = None
        self.hist = self._make_hist(HISTORY_DAY)
        self.trail.clear()
        self._next_rec = -1.0
        self.statusBar().showMessage(
            f"{sc.name} {env.day.isoformat()}: sunrise {_hhmm(env.sunrise_s)}, sunset {_hhmm(env.sunset_s)}, "
            f"sky: {SKY_LABELS.get(env.sky, env.sky)}, panels: {', '.join(self.ids)}")
        self._tick_once_and_draw()
        self.p_duty.enableAutoRange(axis="x")
        self.timer.start()
        self.btn_run.setText("Pause")

    def _gain(self) -> float:
        """Net energy of B relative to A [%] (NaN until A has produced something)."""
        a, b = self.runner.energy.net_wh, self.runner_b.energy.net_wh
        return 100.0 * (b - a) / a if a > 0.01 else NAN

    def _day_finished(self):
        self.timer.stop()
        self.btn_run.setText("New day")
        e = self.runner.energy
        msg = (f"Day finished: panel gave {e.pv_wh:.1f} Wh of {e.mpp_wh:.1f} Wh possible "
               f"(MPPT {e.mppt_eff * 100:.1f} %), {e.out_wh:.1f} Wh reached the load, "
               f"mount used {e.mount_wh:.2f} Wh, net {e.net_wh:.1f} Wh.")
        if self.runner_b is not None:
            eb = self.runner_b.energy
            msg += (f"  |  B: {eb.pv_wh:.1f} Wh (MPPT {eb.mppt_eff * 100:.1f} %), "
                    f"net {eb.net_wh:.1f} Wh = {self._gain():+.1f} % vs A.")
        self.statusBar().showMessage(msg)

    # ----------------------------------------------------------------- run loop
    def toggle_run(self):
        if self.day_mode and self.runner.done:
            self.start_day()
            return
        if self.timer.isActive():
            self.timer.stop()
            self.btn_run.setText("Start")
        else:
            self.timer.start()
            self.btn_run.setText("Pause")

    def reset(self, *_, d_init: float | None = None):
        if self.day_mode:
            running = self.timer.isActive()
            self.start_day()  # back to sunrise with fresh channels and recorder
            if not running and self.day_mode:
                self.timer.stop()
                self.btn_run.setText("Start")
            return
        for c in self._channels():
            if d_init is not None:
                c.algo.p.d_init = d_init
            c.reset()
        self._manual_duty()
        self._events.clear()
        for d in self.hist.values():
            d.clear()
        self.trail.clear()
        self.last_b = None
        self._tick_once_and_draw()

    def _step(self) -> bool:
        """Advance one MPPT period. Returns False when the day is over."""
        h = self.hist
        sb = None
        if self.day_mode:
            if self.runner.done:
                return False
            ds = self.day = self.runner.step()
            self.rec.record(self.ids[0], self.ch, ds, self.out_sensors[0], label=self.runner.fault_label)
            if self.runner_b is not None:
                dsb = self.day_b = self.runner_b.step()
                self.rec.record(self.ids[1], self.ch_b, dsb, self.out_sensors[1],
                                label=self.runner_b.fault_label)
                sb = dsb.snap
            s, x, g, t_amb = ds.snap, ds.clock_h, ds.poa.total, ds.env.t_amb
            record = ds.clock_s >= self._next_rec
            if record:
                self._next_rec = ds.clock_s + DAY_RECORD_S
                sun = ds.env.sun
                for k, val in (("sun_el", sun.elevation), ("sun_az", sun.azimuth),
                               ("tilt", ds.tilt), ("az", ds.azimuth),
                               ("tilt_b", self.day_b.tilt if sb is not None else NAN),
                               ("az_b", self.day_b.azimuth if sb is not None else NAN)):
                    h[k].append(val)
        else:
            while self._events and self._events[0][0] <= self.ch.t + 1e-9:
                self._events.pop(0)[1]()
            s = self.ch.step()
            if self.ch_b is not None:
                self.ch_b.cond = self.ch.cond
                sb = self.ch_b.step()
            x, g, t_amb, record = s.t, self.ch.cond.g, self.ch.cond.t_amb, True
            for k in ("sun_el", "sun_az", "tilt", "az", "tilt_b", "az_b"):
                h[k].append(NAN)
        self.last, self.last_b = s, sb
        self.trail.append((s.v_pv, s.i_pv))
        if record:
            for k, val in (("t", x), ("p_mpp", s.p_mpp), ("p_pv", s.p_pv), ("p_out", s.p_out),
                           ("duty", s.duty), ("v_pv", s.v_pv), ("v_mpp", s.curve.mpp.v),
                           ("v_out", s.v_out), ("g", g), ("t_amb", t_amb), ("t_cell", s.t_cell),
                           ("eff", _eff(s.p_pv, s.p_mpp)), ("conv_eff", _eff(s.p_out, s.p_pv))):
                h[k].append(val)
            if sb is not None:
                for k, val in (("p_pv_b", sb.p_pv), ("p_mpp_b", sb.p_mpp), ("duty_b", sb.duty),
                               ("eff_b", _eff(sb.p_pv, sb.p_mpp))):
                    h[k].append(val)
            else:
                for k in ("p_pv_b", "p_mpp_b", "duty_b", "eff_b"):
                    h[k].append(NAN)
        return True

    def _steps_per_frame(self) -> int:
        if not self.day_mode:
            return SPEEDS.get(self.cmb_speed.currentText(), 5)
        sim_s = DAY_SPEEDS.get(self.cmb_speed.currentText(), 600) * self.timer.interval() / 1000.0
        return int(max(1, min(round(sim_s / self.ch.dt), 1e6)))

    def _tick(self):
        t0 = time.perf_counter()
        for _ in range(self._steps_per_frame()):
            if not self._step():
                self._draw()
                self._day_finished()
                return
            if self.day_mode and time.perf_counter() - t0 > FRAME_BUDGET_S:
                break
        self._draw()

    def _tick_once_and_draw(self):
        if self._step():
            self._draw()

    def _redraw_curves_only(self):
        # conditions changed while paused: show the new curve, keep the old point
        if self.last is None:
            return
        self._draw_curve(self.ch.curve())

    # --------------------------------------------------------------------- draw
    def _draw_curve(self, curve, curve_b=None):
        local = [m for m in curve.local_maxima if abs(m.v - curve.mpp.v) > 0.5]
        for name, y_of in (("iv", lambda v, i: i), ("pv", lambda v, i: v * i)):
            it = self.items[name]
            it["curve"].setData(curve.v, y_of(curve.v, curve.i))
            it["mpp"].setData([curve.mpp.v], [y_of(curve.mpp.v, curve.mpp.i)])
            if local:
                it["local"].setData([m.v for m in local], [y_of(m.v, m.i) for m in local])
            else:
                it["local"].setData([], [])
            if curve_b is not None:
                it["curve_b"].setData(curve_b.v, y_of(curve_b.v, curve_b.i))
                it["mpp_b"].setData([curve_b.mpp.v], [y_of(curve_b.mpp.v, curve_b.mpp.i)])
            else:
                it["curve_b"].setData([], [])
                it["mpp_b"].setData([], [])

    def _draw(self):
        s, sb = self.last, self.last_b
        if s is None:
            return
        # in manual mode B sees the same conditions, so its curve is A's curve
        self._draw_curve(s.curve, sb.curve if sb is not None and self.day_mode else None)
        tv = np.array([p[0] for p in self.trail])
        ti = np.array([p[1] for p in self.trail])
        for name, y_of in (("iv", lambda v, i: i), ("pv", lambda v, i: v * i)):
            it = self.items[name]
            it["trail"].setData(tv, y_of(tv, ti))
            it["op"].setData([s.v_pv], [y_of(s.v_pv, s.i_pv)])
            it["meas"].setData([s.v_meas], [y_of(s.v_meas, s.i_meas)])
            if sb is not None:
                it["op_b"].setData([sb.v_pv], [y_of(sb.v_pv, sb.i_pv)])
            else:
                it["op_b"].setData([], [])

        h = {k: np.fromiter(d, float) for k, d in self.hist.items()}
        hide = set()
        if self.ch_b is None:
            hide.update(B_KEYS)
        if not self.day_mode:
            hide.update(("sun_el", "sun_az", "tilt", "az", "tilt_b", "az_b", "p_mpp_b"))
        for line, key in self.lines:
            if key in hide:
                line.setData([], [])
            else:
                line.setData(h["t"], h[key])

        ds = self.day if self.day_mode else None
        if ds is not None:
            self.sl_g.show_value(ds.poa.total)
            self.sl_t.show_value(ds.env.t_amb)
            for w, sh in zip(self.sl_shade, self.ch.cond.shade):
                w.show_value(100.0 * sh)
            self.lbl_time.setText(f"Clock {_hhmm(ds.clock_s, sec=True)}   "
                                  f"({min(self.runner.progress, 1.0) * 100:.0f} % of the day)")
        else:
            self.lbl_time.setText(f"Simulated time: {s.t:.1f} s   ({len(h['t'])} samples shown)")
        self.lbl_tcell.setText(f"{s.t_cell:.1f} °C  (panel heats up in the sun)")

        te = s.tracking_eff * 100
        te_col = "#28c050" if te >= 98 else ("#e0a000" if te >= 90 else "#e04040")
        rows = ""
        if ds is not None:
            rows += self._day_row(ds)
        if sb is not None:
            rows += self._ab_row(s, sb)
        self.lbl_big.setText(
            f"<table cellspacing='10'><tr>"
            f"<td><b>MPPT efficiency</b><br><span style='font-size:22px;color:{te_col}'>{te:.1f} %</span>"
            f"<br><small>panel power ÷ best possible</small></td>"
            f"<td><b>Panel (input)</b><br>{s.v_pv:.2f} V × {s.i_pv:.3f} A<br>= <b>{s.p_pv:.2f} W</b>"
            f" &nbsp;(best {s.p_mpp:.2f} W)</td>"
            f"<td><b>Duty</b><br><span style='font-size:22px'>{s.duty:.3f}</span></td>"
            f"<td><b>Load (output)</b><br>{s.v_out:.2f} V × {s.i_out:.3f} A<br>= <b>{s.p_out:.2f} W</b></td>"
            f"<td><b>Converter</b><br>efficiency {s.conv_eff * 100:.1f} %<br>losses {s.p_loss:.2f} W"
            f" &nbsp;[{s.conv_mode}]</td>"
            f"</tr>{rows}</table>"
        )
        warn = []
        if not s.reach.reachable and s.p_mpp > 0.5:
            warn.append("MPP UNREACHABLE — " + s.reach.reason
                        + ". Lower the load resistance (or raise its current) so the buck can reach the peak.")
        if s.collapsed and (ds is None or s.p_mpp > 0.5):
            warn.append("Converter is OFF: panel voltage collapsed below the converter's minimum input.")
        self.lbl_warn.setText("\n".join(warn))
        self.lbl_warn.setVisible(bool(warn))

    def _label(self, k: int) -> str:
        """'id: algorithm, mount' for panel k of the running scenario (manual: algorithm only)."""
        if self.day_mode and self.sc_running is not None:
            p = self.sc_running.panels[k]
            return f"{p.panel_id}: {ALGO_LABELS.get(p.algorithm, p.algorithm)}, {p.mount}"
        cmb = self.cmb_algo if k == 0 else self.cmb_algo_b
        return f"{'AB'[k]}: {cmb.currentText()}"

    def _ab_row(self, s: Snapshot, sb: Snapshot) -> str:
        c = "<span style='color:#e650e6'>"
        if not self.day_mode:
            return (
                f"<tr><td><b>A vs B</b><br><small>same sun and load</small></td>"
                f"<td><b>{self._label(0)}</b><br>{s.p_pv:.2f} W, MPPT {s.tracking_eff * 100:.1f} %</td>"
                f"<td colspan='2'>{c}<b>{self._label(1)}</b></span><br>{sb.p_pv:.2f} W, "
                f"MPPT {sb.tracking_eff * 100:.1f} %, duty {sb.duty:.3f}</td>"
                f"<td><b>B − A</b><br><span style='font-size:18px'>{sb.p_pv - s.p_pv:+.2f} W</span></td></tr>"
            )
        e, eb = self.runner.energy, self.runner_b.energy
        gain = self._gain()
        col = "#28c050" if gain >= 0 else "#e04040"
        return (
            f"<tr><td><b>A vs B</b><br><small>energy so far</small></td>"
            f"<td><b>{self._label(0)}</b><br>{e.pv_wh:.1f} Wh, MPPT {e.mppt_eff * 100:.1f} %"
            f"<br><small>mount {e.mount_wh:.2f} Wh, net <b>{e.net_wh:.1f} Wh</b></small></td>"
            f"<td colspan='2'>{c}<b>{self._label(1)}</b></span><br>{eb.pv_wh:.1f} Wh, "
            f"MPPT {eb.mppt_eff * 100:.1f} %<br><small>mount {eb.mount_wh:.2f} Wh, "
            f"net <b>{eb.net_wh:.1f} Wh</b></small></td>"
            f"<td><b>B vs A (net)</b><br><span style='font-size:22px;color:{col}'>{gain:+.1f} %</span></td></tr>"
        )

    def _day_row(self, ds: DaySnapshot) -> str:
        e = ds.energy
        eff = e.mppt_eff * 100
        col = "#28c050" if eff >= 98 else ("#e0a000" if eff >= 90 else "#e04040")
        env = self.runner.env
        shade = ("<br><b style='color:#ff8c00'>shadow on panel</b>"
                 if ds.env.shade_label != "normal" and any(self.ch.cond.shade) else "")
        return (
            f"<tr>"
            f"<td><b>Day so far</b><br><span style='font-size:22px;color:{col}'>{eff:.1f} %</span>"
            f"<br><small>MPPT efficiency (energy)</small></td>"
            f"<td><b>Energy from panel</b><br><b>{e.pv_wh:.1f} Wh</b> of {e.mpp_wh:.1f} Wh possible"
            f"<br><small>lost to tracking: {e.mpp_wh - e.pv_wh:.2f} Wh</small></td>"
            f"<td><b>Clock</b><br><span style='font-size:22px'>{_hhmm(ds.clock_s)}</span>"
            f"<br><small>{SKY_LABELS.get(env.sky, env.sky)}, sun {ds.env.sun.elevation:.0f}° up</small>{shade}</td>"
            f"<td><b>Energy to load</b><br><b>{e.out_wh:.1f} Wh</b>"
            f"<br><small>sunlight {e.poa_wh_m2 / 1000:.2f} kWh/m²</small></td>"
            f"<td><b>Mount</b> tilt {ds.tilt:.0f}°, az {ds.azimuth:.0f}°<br>"
            f"motor + standby {e.mount_wh:.2f} Wh<br><b>net {e.net_wh:.1f} Wh</b></td>"
            f"</tr>"
        )
