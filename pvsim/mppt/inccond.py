"""Incremental Conductance (Hussein, Muta, Hoshino & Osakada, IEE Proc.
Gener. Transm. Distrib. 142(1):59-64, 1995).

At the MPP dP/dV = 0, i.e. dI/dV = -I/V.  With g = dI/dV + I/V:
    g > 0 : left of MPP (V too low)  -> raise V -> decrease D
    g < 0 : right of MPP (V too high) -> lower V -> increase D
    |g| <= eps_g : hold
When V did not change (dV = 0) the sign of dI tells whether irradiance rose
(dI > 0 -> MPP moved to higher V) or fell.

Variable step (inccond_variable): step = clip(N * |dP/dV|, step_min,
step_max), Liu, Duan, Liu, Liu & Kang, IEEE Trans. Ind. Electron.
55(7):2622-2628, 2008.
ASSUMPTION: N, step_min, step_max and the 4 x step back-off when the input
collapses (V <= eps_v) are untuned defaults; tune step and period to the
converter dynamics (Femia et al., IEEE TPEL 20(4):963-973, 2005).
"""

from __future__ import annotations

from dataclasses import dataclass

from .base import BaseParams, BaseState, MPPTAlgorithm, register_mppt


@dataclass
class IncCondParams(BaseParams):
    eps_g: float = 0.0  # [S] conductance dead band
    eps_v: float = 1e-6  # [V] treat |dV| below this as zero
    eps_i: float = 1e-6  # [A]


@register_mppt("inccond")
class IncrementalConductance(MPPTAlgorithm):
    Params = IncCondParams
    State = BaseState

    def _step_size(self, dv: float, di: float, v: float, i: float) -> float:
        return self.p.step

    def step(self, v: float, i: float) -> float:
        s, p = self.s, self.p
        if v <= p.eps_v:
            # input collapsed: back off the load
            s.d = self.clip(s.d - 4.0 * p.step)
            self._remember(v, i)
            return s.d
        if s.n == 0:
            s.d = self.clip(s.d + p.step)
            self._remember(v, i)
            return s.d

        dv = v - s.v_prev
        di = i - s.i_prev
        direction = 0  # +1 raise V, -1 lower V
        if abs(dv) < p.eps_v:
            if di > p.eps_i:
                direction = 1
            elif di < -p.eps_i:
                direction = -1
        else:
            g = di / dv + i / v
            if g > p.eps_g:
                direction = 1
            elif g < -p.eps_g:
                direction = -1
        if direction != 0:
            s.d = self.clip(s.d - direction * self._step_size(dv, di, v, i))
        self._remember(v, i)
        return s.d


@dataclass
class IncCondVarParams(IncCondParams):
    n_gain: float = 0.002
    step_min: float = 0.0005
    step_max: float = 0.02


@register_mppt("inccond_variable")
class IncrementalConductanceVariable(IncrementalConductance):
    Params = IncCondVarParams

    def _step_size(self, dv: float, di: float, v: float, i: float) -> float:
        p = self.p
        if abs(dv) < p.eps_v:
            return p.step_min
        dp_dv = i + v * di / dv
        return min(max(p.n_gain * abs(dp_dv), p.step_min), p.step_max)
