"""Operating point of panel + converter + load for a given duty cycle.

The panel voltage V settles where the panel current equals the current the
converter draws:

    f(V) = I_pv(V) - I_in(V, D, load) = 0

A stable point (input capacitor voltage returns after a disturbance) has f
falling through zero.  With a warm start (the previous step's voltage) the
root next to it is taken, which mimics the continuity of the input-capacitor
voltage; otherwise the highest-voltage stable root is taken (start-up from
open circuit).  If no root exists the input collapses into undervoltage
lockout and the converter is reported off (`collapsed`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from ..converter.base import Converter, ConverterResult
from ..load.electronic_load import ElectronicLoad
from ..pv.panel import IVCurve


@dataclass
class OperatingPoint:
    v_pv: float
    i_pv: float
    conv: ConverterResult
    collapsed: bool = False

    @property
    def p_pv(self) -> float:
        return self.v_pv * self.i_pv


def _off_point(curve: IVCurve, conv: Converter, d: float, collapsed: bool) -> OperatingPoint:
    r = ConverterResult(curve.v_oc, 0.0, 0.0, 0.0, d, "off", {})
    return OperatingPoint(curve.v_oc, 0.0, r, collapsed)


def solve_operating_point(
    curve: IVCurve,
    conv: Converter,
    d: float,
    load: ElectronicLoad,
    v_guess: float | None = None,
) -> OperatingPoint:
    d = conv.clip_duty(d)
    v_oc = curve.v_oc
    v_lo = max(conv.v_uvlo, 1e-9 * v_oc)
    if v_oc <= 0.0 or v_oc <= v_lo or curve.i_sc <= 0.0:
        return _off_point(curve, conv, d, collapsed=False)

    def f(v: float) -> float:
        return float(curve.current(v)) - conv.output(v, d, load).i_in

    f_oc = f(v_oc)
    if f_oc >= 0.0:  # converter draws nothing even at Voc
        return _finish(curve, conv, d, load, v_oc)

    root = None
    if v_guess is not None and v_lo < v_guess < v_oc:
        root = _root_near(f, v_guess, v_lo, v_oc, f_oc)
    if root is None:
        root = _highest_root(f, v_lo, v_oc, f_oc)
    if root is None:
        return _off_point(curve, conv, d, collapsed=True)
    return _finish(curve, conv, d, load, root)


def _finish(curve, conv, d, load, v) -> OperatingPoint:
    r = conv.output(v, d, load)
    return OperatingPoint(v, r.i_in if r.mode != "off" else 0.0, r)


def _root_near(f, v0, v_lo, v_oc, f_oc):
    span = v_oc - v_lo
    step = 0.01 * span
    f0 = f(v0)
    if f0 == 0.0:
        return v0
    if f0 > 0.0:  # root is above v0
        lo, hi = v0, v0
        while True:
            hi = min(hi + step, v_oc)
            fh = f_oc if hi >= v_oc else f(hi)
            if fh < 0.0:
                return brentq(f, lo, hi, xtol=1e-10)
            if hi >= v_oc:
                return None
            lo, step = hi, step * 2
    lo, hi = v0, v0  # f0 < 0: root is below v0
    while True:
        lo = max(lo - step, v_lo)
        fl = f(lo)
        if fl > 0.0:
            return brentq(f, lo, hi, xtol=1e-10)
        if lo <= v_lo:
            return None
        hi, step = lo, step * 2


def _highest_root(f, v_lo, v_oc, f_oc, n: int = 64):
    grid = np.linspace(v_lo, v_oc, n)
    f_hi = f_oc
    for k in range(n - 2, -1, -1):
        fk = f(grid[k])
        if fk > 0.0:
            return brentq(f, grid[k], grid[k + 1], xtol=1e-10)
        if fk == 0.0:
            return float(grid[k])
        f_hi = fk
    return None


@dataclass
class Reachability:
    reachable: bool
    reason: str
    v_pv_at_dmax: float


def mpp_reachability(curve: IVCurve, conv: Converter, load: ElectronicLoad,
                     tol: float = 0.01) -> Reachability:
    """Can some duty in [d_min, d_max] pull the panel down to its (global) MPP voltage?

    A buck reflects its load to the input as roughly R_in = R_load / D^2 >= R_load,
    so the panel can be loaded at most as hard as D = d_max allows.  Raising D
    lowers the panel voltage, so the MPP is reachable iff V_pv(d_max) <= V_mpp.
    """
    if curve.mpp.p <= 0.0:
        return Reachability(True, "no power available", curve.v_oc)
    op = solve_operating_point(curve, conv, conv.d_max, load)
    v_mpp = curve.mpp.v
    if op.collapsed:
        return Reachability(True, "input collapses at D_max (load demands more than P_mpp)", op.v_pv)
    if op.v_pv <= v_mpp * (1.0 + tol):
        return Reachability(True, "", op.v_pv)
    if load.mode == "CR":
        r_min = load.setpoint / conv.d_max**2
        reason = (f"R_load too high: R_in >= R_load/D_max^2 = {r_min:.1f} ohm "
                  f"> R_mpp = {curve.r_mpp:.1f} ohm")
    elif load.mode == "CC":
        reason = (f"CC setpoint too low: I_set = {load.setpoint:g} A cannot absorb "
                  f"P_mpp = {curve.mpp.p:.1f} W at the output voltage the buck can reach")
    elif load.mode == "CV":
        reason = (f"CV setpoint too high: V_set/D_max = {load.setpoint / conv.d_max:.1f} V "
                  f"> V_mpp = {v_mpp:.1f} V")
    else:
        reason = "load is off"
    reason += f" (at D_max = {conv.d_max:.2f} the panel sits at {op.v_pv:.2f} V)"
    return Reachability(False, reason, op.v_pv)
