"""Perturb & Observe (hill climbing) on the buck duty cycle.

For a buck feeding the load, raising D lowers the panel voltage.  P&O keeps
perturbing D in the same direction while power rises and reverses when it
falls (Femia, Petrone, Spagnuolo & Vitelli, IEEE TPEL 20(4):963-973, 2005).
Variable step uses step = clip(N * |dP/dV|, step_min, step_max) (Liu, Duan,
Liu, Liu & Kang, IEEE Trans. Ind. Electron. 55(7):2622-2628, 2008).
ASSUMPTION: the step sizes and N are untuned defaults; Femia et al. show
step and sampling period must be matched to the converter dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass

from .base import BaseParams, BaseState, MPPTAlgorithm, register_mppt


@dataclass
class POState(BaseState):
    dir: int = 1  # +1: increase D (lower V), -1: decrease D


@register_mppt("po")
class PerturbObserve(MPPTAlgorithm):
    State = POState

    def _step_size(self, dv: float, dp: float) -> float:
        return self.p.step

    def step(self, v: float, i: float) -> float:
        s = self.s
        p = v * i
        if s.n > 0:
            dp = p - s.p_prev
            if dp < 0.0:
                s.dir = -s.dir
            s.d = self.clip(s.d + s.dir * self._step_size(v - s.v_prev, dp))
        else:
            s.d = self.clip(s.d + s.dir * self.p.step)
        self._remember(v, i)
        return s.d


@dataclass
class POVarParams(BaseParams):
    n_gain: float = 0.002  # duty per W/V
    step_min: float = 0.0005
    step_max: float = 0.02


@register_mppt("po_variable")
class PerturbObserveVariable(PerturbObserve):
    Params = POVarParams

    def _step_size(self, dv: float, dp: float) -> float:
        p = self.p
        if abs(dv) < 1e-6:
            return p.step_min
        st = p.n_gain * abs(dp / dv)
        return min(max(st, p.step_min), p.step_max)
