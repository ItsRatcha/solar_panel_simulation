"""Step-by-step closed loop for interactive use (GUI, notebooks).

One `step()` = one MPPT period: build/reuse the I-V curve for the current
conditions, solve the operating point for the applied duty, measure it with
the INA226 model, and let the MPPT algorithm choose the next duty.
Conditions, load, converter, sensor and algorithm can be swapped between
steps.  No Qt imports here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..converter.base import Converter
from ..converter.buck_averaged import BuckAveraged
from ..load.electronic_load import ElectronicLoad
from ..mppt import MPPTAlgorithm, create_mppt
from ..pv.datasheet import PanelDatasheet, cell_temperature_noct
from ..pv.panel import IVCurve, Panel, _zero_curve
from ..sensors.ina226 import INA226
from .operating_point import Reachability, mpp_reachability, solve_operating_point


@dataclass
class Snapshot:
    t: float
    duty: float  # duty applied during this step
    duty_next: float  # duty chosen by the MPPT for the next step
    v_pv: float
    i_pv: float
    v_meas: float
    i_meas: float
    v_out: float
    i_out: float
    p_out: float
    p_loss: float
    losses: dict
    conv_mode: str
    collapsed: bool
    curve: IVCurve
    t_cell: float
    reach: Reachability

    @property
    def p_pv(self) -> float:
        return self.v_pv * self.i_pv

    @property
    def p_mpp(self) -> float:
        return self.curve.mpp.p

    @property
    def tracking_eff(self) -> float:
        return self.p_pv / self.p_mpp if self.p_mpp > 1e-9 else 0.0

    @property
    def conv_eff(self) -> float:
        return self.p_out / self.p_pv if self.p_pv > 1e-9 else 0.0


@dataclass
class Conditions:
    g: float = 1000.0  # unshaded plane-of-array irradiance [W/m^2]
    shade: tuple[float, ...] = (0.0, 0.0, 0.0)  # blocked fraction per substring, 0..1
    t_amb: float = 30.0  # [degC]

    def g_substrings(self) -> tuple[float, ...]:
        return tuple(self.g * (1.0 - min(max(s, 0.0), 1.0)) for s in self.shade)


@dataclass
class LiveChannel:
    datasheet: PanelDatasheet = field(default_factory=PanelDatasheet)
    conv: Converter = field(default_factory=BuckAveraged)
    load: ElectronicLoad = field(default_factory=lambda: ElectronicLoad("CR", 3.0))
    algo: MPPTAlgorithm = field(default_factory=lambda: create_mppt("po", d_init=0.3))
    sensor: INA226 = field(default_factory=INA226)
    cond: Conditions = field(default_factory=Conditions)
    dt: float = 0.1  # MPPT period [s]

    def __post_init__(self):
        self.panel = Panel(self.datasheet)
        self.t = 0.0
        self._v_prev: float | None = None
        self._reach_key = None
        self._reach: Reachability | None = None
        self.shorted: tuple[bool, ...] | None = None  # fault hook: shorted bypass diodes
        self.open_circuit = False  # fault hook: panel disconnected

    def reset(self) -> None:
        self.t = 0.0
        self._v_prev = None
        self.algo.reset()

    def t_cell(self) -> float:
        return cell_temperature_noct(self.cond.t_amb, self.cond.g, self.datasheet.noct)

    def curve(self) -> IVCurve:
        if self.open_circuit:
            return _zero_curve()
        return self.panel.curve(self.cond.g_substrings(), self.t_cell(), self.shorted)

    def _reachability(self, curve: IVCurve) -> Reachability:
        key = (id(curve), self.load, id(self.conv))
        if key != self._reach_key:
            self._reach = mpp_reachability(curve, self.conv, self.load)
            self._reach_key = key
        return self._reach

    def step(self) -> Snapshot:
        curve = self.curve()
        t_cell = self.t_cell()
        d = self.algo.duty
        op = solve_operating_point(curve, self.conv, d, self.load, v_guess=self._v_prev)
        self._v_prev = None if op.collapsed else op.v_pv
        m = self.sensor.measure(op.v_pv, op.i_pv)
        d_next = self.algo.step(m.v, m.i)
        self.t += self.dt
        c = op.conv
        return Snapshot(
            t=self.t, duty=d, duty_next=d_next,
            v_pv=op.v_pv, i_pv=op.i_pv, v_meas=m.v, i_meas=m.i,
            v_out=c.v_out, i_out=c.i_out, p_out=c.p_out, p_loss=c.p_loss,
            losses=dict(c.losses), conv_mode=c.mode, collapsed=op.collapsed,
            curve=curve, t_cell=t_cell, reach=self._reachability(curve),
        )
