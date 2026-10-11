"""Main window: controls on the left, live curves and time plots on the right."""

from __future__ import annotations

from collections import deque

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QMainWindow, QPushButton, QScrollArea, QSlider, QSplitter, QVBoxLayout, QWidget,
)

from ..converter.buck_averaged import BuckAveraged, BuckParams
from ..engine.live import Conditions, LiveChannel, Snapshot
from ..load.electronic_load import ElectronicLoad
from ..mppt import available_mppt, create_mppt
from ..sensors.ina226 import INA226, INA226Config

HISTORY = 1500  # samples kept in the time plots
TRAIL = 40  # recent operating points drawn on the curves
SPEEDS = {"Slow (1 step/frame)": 1, "Normal (5)": 5, "Fast (25)": 25, "Very fast (100)": 100}
ALGO_LABELS = {
    "po": "P&O (fixed step)",
    "po_variable": "P&O (variable step)",
    "inccond": "IncCond (fixed step)",
    "inccond_variable": "IncCond (variable step)",
    "global_scan": "P&O + global scan",
    "manual": "Manual duty (no MPPT)",
}
LOAD_UNITS = {"CR": ("Ω", 3.0, 0.1, 1000.0), "CC": ("A", 2.0, 0.0, 10.0), "CV": ("V", 6.0, 0.0, 40.0)}

C_CURVE = (90, 160, 255)
C_MPP = (255, 200, 0)
C_OP = (40, 220, 90)
C_MEAS = (255, 90, 90)
C_LOCAL = (255, 140, 0)


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

    def _update_label(self):
        self.label.setText(self.fmt.format(self.value()) + self.unit)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PV + Buck + MPPT simulator")
        self.resize(1500, 950)

        self.ch = LiveChannel()
        self.trail: deque = deque(maxlen=TRAIL)
        self.hist = {k: deque(maxlen=HISTORY) for k in
                     ("t", "p_mpp", "p_pv", "p_out", "duty", "v_pv", "v_mpp", "v_out")}
        self.last: Snapshot | None = None

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_controls())
        splitter.addWidget(self._build_plots())
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([420, 1080])
        self.setCentralWidget(splitter)

        self.timer = QTimer(self)
        self.timer.setInterval(33)  # ~30 frames per second
        self.timer.timeout.connect(self._tick)

        self._apply_all()
        self._tick_once_and_draw()
        self.toggle_run()

    # ------------------------------------------------------------------ controls
    def _build_controls(self) -> QWidget:
        root = QWidget()
        v = QVBoxLayout(root)

        # --- run
        g = QGroupBox("Simulation")
        f = QGridLayout(g)
        self.btn_run = QPushButton("Start")
        self.btn_run.clicked.connect(self.toggle_run)
        btn_reset = QPushButton("Reset MPPT + plots")
        btn_reset.setToolTip("Restart the MPPT from its initial duty and clear the time plots.")
        btn_reset.clicked.connect(self.reset)
        self.cmb_speed = QComboBox()
        self.cmb_speed.addItems(SPEEDS)
        self.cmb_speed.setCurrentIndex(1)
        self.cmb_speed.setToolTip("MPPT steps simulated per screen frame (one step = 0.1 s).")
        self.lbl_time = QLabel("t = 0.0 s")
        f.addWidget(self.btn_run, 0, 0)
        f.addWidget(btn_reset, 0, 1)
        f.addWidget(QLabel("Speed"), 1, 0)
        f.addWidget(self.cmb_speed, 1, 1)
        f.addWidget(self.lbl_time, 2, 0, 1, 2)
        v.addWidget(g)

        # --- environment
        g = QGroupBox("Sun and shading")
        f = QFormLayout(g)
        self.sl_g = LabeledSlider(0, 1200, 1000, 10, unit=" W/m²",
                                  tooltip="Sunlight hitting the panel (1000 W/m² = full sun).")
        self.sl_t = LabeledSlider(10, 45, 30, 1, unit=" °C", tooltip="Air temperature around the panel.")
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
        self.sp_step.setToolTip("How much the duty cycle changes per MPPT step.\n"
                                "Bigger = faster but more wobble around the peak.")
        self.sl_duty = LabeledSlider(0, 0.95, 0.5, 0.005, fmt="{:.3f}",
                                     tooltip="Duty cycle used in Manual mode.")
        self.cmb_algo.currentIndexChanged.connect(self._apply_algo)
        self.sp_step.valueChanged.connect(self._apply_algo)
        self.sl_duty.valueChanged.connect(self._manual_duty)
        f.addRow("Algorithm", self.cmb_algo)
        f.addRow("Step size", self.sp_step)
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
        self.p_pow.setLabel("bottom", "Time", "s")
        self.p_pow.setLabel("left", "Power", "W")
        self.p_pow.showGrid(x=True, y=True, alpha=0.3)
        self.p_pow.addLegend(offset=(10, 10))
        self.l_pmpp = self.p_pow.plot(pen=pg.mkPen(C_MPP, width=2, style=Qt.DashLine),
                                      name="Best possible (true MPP)")
        self.l_ppv = self.p_pow.plot(pen=pg.mkPen(C_CURVE, width=2), name="Panel power (P_in)")
        self.l_pout = self.p_pow.plot(pen=pg.mkPen(C_OP, width=2), name="Power to load (P_out)")

        glw.nextRow()
        self.p_duty = glw.addPlot(title="Duty cycle chosen by the MPPT")
        self.p_duty.setLabel("bottom", "Time", "s")
        self.p_duty.setLabel("left", "Duty D")
        self.p_duty.showGrid(x=True, y=True, alpha=0.3)
        self.l_duty = self.p_duty.plot(pen=pg.mkPen((200, 120, 255), width=2))

        self.p_volt = glw.addPlot(title="Voltages")
        self.p_volt.setLabel("bottom", "Time", "s")
        self.p_volt.setLabel("left", "Voltage", "V")
        self.p_volt.showGrid(x=True, y=True, alpha=0.3)
        self.p_volt.addLegend(offset=(10, 10))
        self.l_vmpp = self.p_volt.plot(pen=pg.mkPen(C_MPP, width=2, style=Qt.DashLine), name="V at true MPP")
        self.l_vpv = self.p_volt.plot(pen=pg.mkPen(C_CURVE, width=2), name="Panel V (in)")
        self.l_vout = self.p_volt.plot(pen=pg.mkPen(C_OP, width=2), name="Load V (out)")
        self.p_volt.setXLink(self.p_duty)
        self.p_pow.setXLink(self.p_duty)

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

    # ------------------------------------------------------------- apply changes
    def _apply_all(self):
        self._load_mode_changed(self.cmb_load.currentText())
        self._apply_conditions()
        self._apply_converter()
        self._apply_sensor()
        self._apply_algo()

    def _apply_conditions(self, *_):
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
        if name == "manual":
            kw["duty"] = self.sl_duty.value()
        else:
            kw["step"] = self.sp_step.value()
        self.ch.algo = create_mppt(name, **kw)
        manual = name == "manual"
        self.sl_duty.setEnabled(manual)
        self.sp_step.setEnabled(not manual)

    def _manual_duty(self, *_):
        if self.cmb_algo.currentData() == "manual":
            self.ch.algo.set_duty(self.sl_duty.value())

    # ------------------------------------------------------------------ presets
    def _set(self, g=1000, t=30, shade=(0, 0, 0), mode="CR", setpoint=3.0, algo="po"):
        for w, val in ((self.sl_g, g), (self.sl_t, t), *zip(self.sl_shade, shade)):
            w.blockSignals(True)
            w.setValue(val)
            w.blockSignals(False)
            w._update_label()
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
        QTimer.singleShot(4000, lambda: self.sl_g.setValue(300))

    def preset_unreachable(self):
        self._set(setpoint=50.0)
        self.reset(d_init=0.3)

    # ----------------------------------------------------------------- run loop
    def toggle_run(self):
        if self.timer.isActive():
            self.timer.stop()
            self.btn_run.setText("Start")
        else:
            self.timer.start()
            self.btn_run.setText("Pause")

    def reset(self, *_, d_init: float | None = None):
        if d_init is not None:
            self.ch.algo.p.d_init = d_init
        self.ch.reset()
        if self.cmb_algo.currentData() == "manual":
            self.ch.algo.set_duty(self.sl_duty.value())
        for d in self.hist.values():
            d.clear()
        self.trail.clear()
        self._tick_once_and_draw()

    def _step(self):
        s = self.ch.step()
        self.last = s
        h = self.hist
        h["t"].append(s.t)
        h["p_mpp"].append(s.p_mpp)
        h["p_pv"].append(s.p_pv)
        h["p_out"].append(s.p_out)
        h["duty"].append(s.duty)
        h["v_pv"].append(s.v_pv)
        h["v_mpp"].append(s.curve.mpp.v)
        h["v_out"].append(s.v_out)
        self.trail.append((s.v_pv, s.i_pv))

    def _tick(self):
        for _ in range(SPEEDS[self.cmb_speed.currentText()]):
            self._step()
        self._draw()

    def _tick_once_and_draw(self):
        self._step()
        self._draw()

    def _redraw_curves_only(self):
        # conditions changed while paused: show the new curve, keep the old point
        if self.last is None:
            return
        curve = self.ch.curve()
        self._draw_curve(curve)

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
        self.l_pmpp.setData(h["t"], h["p_mpp"])
        self.l_ppv.setData(h["t"], h["p_pv"])
        self.l_pout.setData(h["t"], h["p_out"])
        self.l_duty.setData(h["t"], h["duty"])
        self.l_vmpp.setData(h["t"], h["v_mpp"])
        self.l_vpv.setData(h["t"], h["v_pv"])
        self.l_vout.setData(h["t"], h["v_out"])

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
            f"</tr></table>"
        )
        warn = []
        if not s.reach.reachable:
            warn.append("MPP UNREACHABLE — " + s.reach.reason
                        + ". Lower the load resistance (or raise its current) so the buck can reach the peak.")
        if s.collapsed:
            warn.append("Converter is OFF: panel voltage collapsed below the converter's minimum input.")
        self.lbl_warn.setText("\n".join(warn))
        self.lbl_warn.setVisible(bool(warn))
