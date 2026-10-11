"""Main window: controls on the left, live curves and time plots on the right.

Two modes:
- Manual: you move the sun/temperature/shade sliders yourself.
- Full day: a DayRunner drives the panel from sunrise to sunset (sun path,
  clouds, temperature, scripted shadows, fixed mount or tracker) and the
  window shows the clock and the energy collected so far.
"""

from __future__ import annotations

import time
from collections import deque
from datetime import date

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QDate, Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDateEdit, QDoubleSpinBox, QFormLayout, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QMainWindow, QPushButton, QScrollArea, QSlider, QSpinBox, QSplitter,
    QVBoxLayout, QWidget,
)

from ..converter.buck_averaged import BuckAveraged, BuckParams
from ..engine.day_runner import DayRunner, DaySnapshot
from ..engine.live import Conditions, LiveChannel, Snapshot
from ..environment.day import SHADE_PRESETS, DayEnvironment
from ..environment.weather import WeatherConfig
from ..load.electronic_load import ElectronicLoad
from ..mppt import available_mppt, create_mppt
from ..sensors.ina226 import INA226, INA226Config
from ..tracking.mounts import make_mount

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
HIST_KEYS = ("t", "p_mpp", "p_pv", "p_out", "duty", "v_pv", "v_mpp", "v_out", "g", "t_amb", "t_cell")

C_CURVE = (90, 160, 255)
C_MPP = (255, 200, 0)
C_OP = (40, 220, 90)
C_MEAS = (255, 90, 90)
C_LOCAL = (255, 140, 0)


def _hhmm(t_s: float, sec: bool = False) -> str:
    t = int(t_s)
    h, m, x = t // 3600, t % 3600 // 60, t % 60
    return f"{h:02d}:{m:02d}:{x:02d}" if sec else f"{h:02d}:{m:02d}"


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
        self.resize(1500, 1000)

        self.ch = LiveChannel()
        self.trail: deque = deque(maxlen=TRAIL)
        self.hist = self._make_hist(HISTORY)
        self.last: Snapshot | None = None
        self.runner: DayRunner | None = None
        self.day: DaySnapshot | None = None
        self._next_rec = -1.0

        self.timer = QTimer(self)
        self.timer.setInterval(33)  # ~30 frames per second
        self.timer.timeout.connect(self._tick)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_controls())
        splitter.addWidget(self._build_plots())
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([420, 1080])
        self.setCentralWidget(splitter)

        self._apply_all()
        self._tick_once_and_draw()
        self.toggle_run()

    @staticmethod
    def _make_hist(maxlen: int) -> dict[str, deque]:
        return {k: deque(maxlen=maxlen) for k in HIST_KEYS}

    @property
    def day_mode(self) -> bool:
        return self.runner is not None

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
        self.cmb_mount.addItem("Fixed, 14° tilt facing south", "fixed")
        self.cmb_mount.addItem("2-axis sun tracker", "tracker")
        self.cmb_period = QComboBox()
        self.cmb_period.addItems(DAY_PERIODS)
        self.cmb_period.setToolTip("Time between MPPT steps. 1 s is close to the ESP32;\n"
                                   "10 s simulates a day about 10x faster.")
        btn_day = QPushButton("Start new day")
        btn_day.setToolTip("Apply these settings and run from sunrise to sunset.")
        btn_day.clicked.connect(self.start_day)
        f.addRow("Date", self.de_date)
        f.addRow("Sky", self.cmb_sky)
        f.addRow("Weather seed", self.sp_seed)
        f.addRow("Shading", self.cmb_dayshade)
        f.addRow("Mount", self.cmb_mount)
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
        g = QGroupBox("MPPT algorithm")
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
                                     tooltip="Duty cycle used in Manual mode.")
        self.cmb_algo.currentIndexChanged.connect(self._apply_algo)
        self.sp_step.valueChanged.connect(self._apply_algo)
        self.sl_duty.valueChanged.connect(self._manual_duty)
        f.addRow("Algorithm", self.cmb_algo)
        f.addRow(self.lbl_step, self.sp_step)
        f.addRow("Manual duty", self.sl_duty)
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

        # --- presets
        g = QGroupBox("Try these")
        f = QVBoxLayout(g)
        for text, fn, tip in (
            ("Full sun, normal", self.preset_normal, "1000 W/m², no shade, 3 Ω load, P&O."),
            ("Shading trap (P&O)", lambda: self.preset_shading("po"),
             "One cell group 75 % shaded. P&O gets stuck on the small peak."),
            ("Shading + global scan", lambda: self.preset_shading("global_scan"),
             "Same shading; global scan sweeps first and finds the real peak."),
            ("Cloud: 1000 → 300 W/m²", self.preset_cloud, "Sudden drop in sunlight; watch it re-track."),
            ("Load too big (unreachable)", self.preset_unreachable,
             "50 Ω load: a buck can't pull the panel down to its MPP."),
            ("Full day: clouds + tree shadow", self.preset_day,
             "Partly cloudy equinox day, a tree shades the panel 07:00-09:00, fixed mount."),
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
        scroll.setMinimumWidth(400)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        return scroll

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
                local=plot.plot(pen=None, symbol="o", symbolSize=11, symbolBrush=None,
                                symbolPen=pg.mkPen(C_LOCAL, width=2), name="Other peaks" if is_p else None),
                trail=plot.plot(pen=pg.mkPen((40, 220, 90, 110), width=1), symbol="o", symbolSize=4,
                                symbolBrush=(40, 220, 90, 110), symbolPen=None,
                                name="Recent operating points"),
                mpp=plot.plot(pen=None, symbol="star", symbolSize=20, symbolBrush=C_MPP,
                              symbolPen=pg.mkPen("k"), name="True max power point"),
                op=plot.plot(pen=None, symbol="o", symbolSize=14, symbolBrush=C_OP,
                             symbolPen=pg.mkPen("k"), name="Operating point (true)"),
                meas=plot.plot(pen=None, symbol="x", symbolSize=12, symbolPen=pg.mkPen(C_MEAS, width=2),
                               name="What the INA226 measures"),
            )

        glw.nextRow()
        self.p_pow = glw.addPlot(title="Power over time", colspan=2)
        self.p_pow.setLabel("left", "Power", "W")
        self.p_pow.showGrid(x=True, y=True, alpha=0.3)
        self.p_pow.addLegend(offset=(10, 10))
        self.l_pmpp = self.p_pow.plot(pen=pg.mkPen(C_MPP, width=2, style=Qt.DashLine),
                                      name="Best possible (true MPP)")
        self.l_ppv = self.p_pow.plot(pen=pg.mkPen(C_CURVE, width=2), name="Panel power (P_in)")
        self.l_pout = self.p_pow.plot(pen=pg.mkPen(C_OP, width=2), name="Power to load (P_out)")

        glw.nextRow()
        self.p_duty = glw.addPlot(title="Duty cycle chosen by the MPPT")
        self.p_duty.setLabel("left", "Duty D")
        self.p_duty.showGrid(x=True, y=True, alpha=0.3)
        self.l_duty = self.p_duty.plot(pen=pg.mkPen((200, 120, 255), width=2))

        self.p_volt = glw.addPlot(title="Voltages")
        self.p_volt.setLabel("left", "Voltage", "V")
        self.p_volt.showGrid(x=True, y=True, alpha=0.3)
        self.p_volt.addLegend(offset=(10, 10))
        self.l_vmpp = self.p_volt.plot(pen=pg.mkPen(C_MPP, width=2, style=Qt.DashLine), name="V at true MPP")
        self.l_vpv = self.p_volt.plot(pen=pg.mkPen(C_CURVE, width=2), name="Panel V (in)")
        self.l_vout = self.p_volt.plot(pen=pg.mkPen(C_OP, width=2), name="Load V (out)")

        glw.nextRow()
        self.p_irr = glw.addPlot(title="Sunlight on the panel")
        self.p_irr.setLabel("left", "Irradiance", "W/m²")
        self.p_irr.showGrid(x=True, y=True, alpha=0.3)
        self.l_irr = self.p_irr.plot(pen=pg.mkPen((255, 170, 40), width=2))

        self.p_temp = glw.addPlot(title="Temperature")
        self.p_temp.setLabel("left", "Temperature", "°C")
        self.p_temp.showGrid(x=True, y=True, alpha=0.3)
        self.p_temp.addLegend(offset=(10, 10))
        self.l_tamb = self.p_temp.plot(pen=pg.mkPen((120, 200, 255), width=2), name="Air")
        self.l_tcell = self.p_temp.plot(pen=pg.mkPen((255, 90, 90), width=2), name="Cell")

        self.time_plots = (self.p_pow, self.p_duty, self.p_volt, self.p_irr, self.p_temp)
        for p in self.time_plots:
            p.setDownsampling(auto=True, mode="peak")
            p.setClipToView(True)
            if p is not self.p_duty:
                p.setXLink(self.p_duty)
        self._set_time_axis(day=False)

        hint = QLabel(
            "<b>How to read this:</b> the <span style='color:#ffc800'>★ star</span> is the best possible point. "
            "The <span style='color:#28dc5a'>● green dot</span> is where the panel actually runs. "
            "The MPPT moves the duty cycle to push the green dot onto the star. "
            "Higher duty → lower panel voltage (dot moves left)."
        )
        hint.setWordWrap(True)
        hint.setTextFormat(Qt.RichText)
        v.addWidget(hint)
        return w

    def _set_time_axis(self, day: bool):
        for p in self.time_plots:
            if day:
                p.setLabel("bottom", "Time of day", "h")
            else:
                p.setLabel("bottom", "Time", "s")
            p.getAxis("bottom").enableAutoSIPrefix(not day)
        self.p_duty.enableAutoRange(axis="x")

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
        self.lbl_tcell.setText(f"{self.ch.t_cell():.1f} °C  (panel heats up in the sun)")
        if not self.timer.isActive():
            self._redraw_curves_only()

    def _load_mode_changed(self, mode: str):
        unit, default, lo, hi = LOAD_UNITS[mode]
        self.sp_load.blockSignals(True)
        self.sp_load.setRange(lo, hi)
        self.sp_load.setSuffix(" " + unit)
        self.sp_load.setValue(default)
        self.sp_load.blockSignals(False)
        self._apply_load()

    def _apply_load(self, *_):
        self.ch.load = ElectronicLoad(self.cmb_load.currentText(), self.sp_load.value())

    def _apply_converter(self, *_):
        self.ch.conv = BuckAveraged(BuckParams() if self.chk_losses.isChecked() else BuckParams.ideal())

    def _apply_sensor(self, *_):
        cfg = INA226Config() if self.chk_sensor.isChecked() else INA226Config.ideal()
        self.ch.sensor = INA226(cfg)

    def _apply_algo(self, *_):
        name = self.cmb_algo.currentData()
        d0 = self.ch.algo.duty if self.last is not None else 0.3
        kw = dict(d_init=d0)
        variable = name in VARIABLE_STEP
        if name == "manual":
            kw["duty"] = self.sl_duty.value()
        elif variable:
            # the step adapts by itself; the box sets its upper limit
            kw["step_max"] = self.sp_step.value()
        else:
            kw["step"] = self.sp_step.value()
        self.ch.algo = create_mppt(name, **kw)
        manual = name == "manual"
        self.sl_duty.setEnabled(manual)
        self.sp_step.setEnabled(not manual)
        self.lbl_step.setText("Max step" if variable else "Step size")
        self.sp_step.setToolTip(
            "Variable step: the step shrinks near the peak and grows far from it,\n"
            "up to this maximum." if variable else
            "How much the duty cycle changes per MPPT step.\n"
            "Bigger = faster but more wobble around the peak.")

    def _manual_duty(self, *_):
        if self.cmb_algo.currentData() == "manual":
            self.ch.algo.set_duty(self.sl_duty.value())

    # ------------------------------------------------------------------ presets
    def _set(self, g=1000, t=30, shade=(0, 0, 0), mode="CR", setpoint=3.0, algo="po"):
        if self.cmb_mode.currentIndex() != 0:
            self.cmb_mode.setCurrentIndex(0)
        for w, val in ((self.sl_g, g), (self.sl_t, t), *zip(self.sl_shade, shade)):
            w.show_value(val)
        self.cmb_load.blockSignals(True)
        self.cmb_load.setCurrentText(mode)
        self.cmb_load.blockSignals(False)
        self._load_mode_changed(mode)
        self.sp_load.setValue(setpoint)
        self.cmb_algo.blockSignals(True)
        self.cmb_algo.setCurrentIndex(self.cmb_algo.findData(algo))
        self.cmb_algo.blockSignals(False)
        self._apply_conditions()
        self._apply_load()
        self._apply_algo()

    def preset_normal(self):
        self._set()
        self.reset(d_init=0.3)

    def preset_shading(self, algo):
        self._set(shade=(0, 0, 75), algo=algo)
        self.reset(d_init=0.2)

    def preset_cloud(self):
        self._set(g=1000)
        self.reset(d_init=0.3)
        QTimer.singleShot(4000, lambda: None if self.day_mode else self.sl_g.setValue(300))

    def preset_unreachable(self):
        self._set(setpoint=50.0)
        self.reset(d_init=0.3)

    def preset_day(self):
        self._set()
        self.de_date.setDate(QDate(2026, 3, 21))
        self.cmb_sky.setCurrentIndex(self.cmb_sky.findData("partly_cloudy"))
        self.sp_seed.setValue(3)
        self.cmb_dayshade.setCurrentIndex(self.cmb_dayshade.findData("tree_morning"))
        self.cmb_mount.setCurrentIndex(self.cmb_mount.findData("fixed"))
        self.cmb_period.setCurrentIndex(0)
        self.cmb_mode.setCurrentIndex(1)  # builds the day and starts it

    # ----------------------------------------------------------------- day mode
    def _mode_changed(self, idx: int):
        day = idx == 1
        self.timer.stop()
        self.grp_day.setVisible(day)
        for w in (self.sl_g, self.sl_t, *self.sl_shade):
            w.setEnabled(not day)
        self._fill_speeds(day)
        self._set_time_axis(day)
        if day:
            self.start_day()
            return
        self.runner = self.day = None
        self.ch.dt = LIVE_DT
        self.hist = self._make_hist(HISTORY)
        self.statusBar().clearMessage()
        self._apply_conditions()
        self.reset()
        self.toggle_run()

    def start_day(self):
        if self.cmb_mode.currentIndex() != 1:
            self.cmb_mode.setCurrentIndex(1)  # comes back here through _mode_changed
            return
        self.timer.stop()
        qd = self.de_date.date()
        env = DayEnvironment(
            day=date(qd.year(), qd.month(), qd.day()),
            weather=WeatherConfig(sky=self.cmb_sky.currentData(), seed=self.sp_seed.value()),
            shading=list(SHADE_PRESETS[self.cmb_dayshade.currentData()]),
        )
        period = DAY_PERIODS[self.cmb_period.currentText()]
        self.ch.dt = period
        self.runner = DayRunner(self.ch, env, make_mount(self.cmb_mount.currentData()), env_dt=period)
        if self.cmb_algo.currentData() == "manual":
            self.ch.algo.set_duty(self.sl_duty.value())
        self.hist = self._make_hist(HISTORY_DAY)
        self.trail.clear()
        self._next_rec = -1.0
        self.statusBar().showMessage(
            f"{env.day.isoformat()}: sunrise {_hhmm(env.sunrise_s)}, sunset {_hhmm(env.sunset_s)}, "
            f"sky: {SKY_LABELS.get(env.sky, env.sky)}")
        self._tick_once_and_draw()
        self.p_duty.enableAutoRange(axis="x")
        self.timer.start()
        self.btn_run.setText("Pause")

    def _day_finished(self):
        self.timer.stop()
        self.btn_run.setText("New day")
        e = self.runner.energy
        self.statusBar().showMessage(
            f"Day finished: panel gave {e.pv_wh:.1f} Wh of {e.mpp_wh:.1f} Wh possible "
            f"(MPPT {e.mppt_eff * 100:.1f} %), {e.out_wh:.1f} Wh reached the load, "
            f"mount used {e.mount_wh:.2f} Wh, net {e.net_wh:.1f} Wh.")

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
        if d_init is not None:
            self.ch.algo.p.d_init = d_init
        if self.day_mode:
            self.runner.reset()  # back to sunrise
            self._next_rec = -1.0
            if not self.timer.isActive():
                self.btn_run.setText("Start")
        else:
            self.ch.reset()
        if self.cmb_algo.currentData() == "manual":
            self.ch.algo.set_duty(self.sl_duty.value())
        for d in self.hist.values():
            d.clear()
        self.trail.clear()
        self._tick_once_and_draw()

    def _step(self) -> bool:
        """Advance one MPPT period. Returns False when the day is over."""
        if self.day_mode:
            if self.runner.done:
                return False
            ds = self.day = self.runner.step()
            s, x, g, t_amb = ds.snap, ds.clock_h, ds.poa.total, ds.env.t_amb
            record = ds.clock_s >= self._next_rec
            if record:
                self._next_rec = ds.clock_s + DAY_RECORD_S
        else:
            s = self.ch.step()
            x, g, t_amb, record = s.t, self.ch.cond.g, self.ch.cond.t_amb, True
        self.last = s
        self.trail.append((s.v_pv, s.i_pv))
        if record:
            h = self.hist
            for k, val in (("t", x), ("p_mpp", s.p_mpp), ("p_pv", s.p_pv), ("p_out", s.p_out),
                           ("duty", s.duty), ("v_pv", s.v_pv), ("v_mpp", s.curve.mpp.v),
                           ("v_out", s.v_out), ("g", g), ("t_amb", t_amb), ("t_cell", s.t_cell)):
                h[k].append(val)
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
    def _draw_curve(self, curve):
        local = [m for m in curve.local_maxima if abs(m.v - curve.mpp.v) > 0.5]
        for name, y_of in (("iv", lambda v, i: i), ("pv", lambda v, i: v * i)):
            it = self.items[name]
            it["curve"].setData(curve.v, y_of(curve.v, curve.i))
            it["mpp"].setData([curve.mpp.v], [y_of(curve.mpp.v, curve.mpp.i)])
            if local:
                it["local"].setData([m.v for m in local], [y_of(m.v, m.i) for m in local])
            else:
                it["local"].setData([], [])

    def _draw(self):
        s = self.last
        if s is None:
            return
        self._draw_curve(s.curve)
        tv = np.array([p[0] for p in self.trail])
        ti = np.array([p[1] for p in self.trail])
        for name, y_of in (("iv", lambda v, i: i), ("pv", lambda v, i: v * i)):
            it = self.items[name]
            it["trail"].setData(tv, y_of(tv, ti))
            it["op"].setData([s.v_pv], [y_of(s.v_pv, s.i_pv)])
            it["meas"].setData([s.v_meas], [y_of(s.v_meas, s.i_meas)])

        h = {k: np.fromiter(d, float) for k, d in self.hist.items()}
        for line, key in ((self.l_pmpp, "p_mpp"), (self.l_ppv, "p_pv"), (self.l_pout, "p_out"),
                          (self.l_duty, "duty"), (self.l_vmpp, "v_mpp"), (self.l_vpv, "v_pv"),
                          (self.l_vout, "v_out"), (self.l_irr, "g"), (self.l_tamb, "t_amb"),
                          (self.l_tcell, "t_cell")):
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
            f"</tr>{self._day_row(ds) if ds is not None else ''}</table>"
        )
        warn = []
        if not s.reach.reachable and s.p_mpp > 0.5:
            warn.append("MPP UNREACHABLE — " + s.reach.reason
                        + ". Lower the load resistance (or raise its current) so the buck can reach the peak.")
        if s.collapsed and (ds is None or s.p_mpp > 0.5):
            warn.append("Converter is OFF: panel voltage collapsed below the converter's minimum input.")
        self.lbl_warn.setText("\n".join(warn))
        self.lbl_warn.setVisible(bool(warn))

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
