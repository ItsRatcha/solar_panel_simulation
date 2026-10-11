"""Closed loop panel -> converter -> load under fixed conditions.

Used by tests, parameter sweeps and the firmware-equivalence checks.  One
iteration = one MPPT period: solve the steady operating point for the duty
currently applied, measure it, let the algorithm choose the next duty.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..converter.base import Converter
from ..engine.operating_point import solve_operating_point
from ..load.electronic_load import ElectronicLoad
from ..pv.panel import IVCurve
from ..sensors.ina226 import INA226
from .base import MPPTAlgorithm


@dataclass
class Trace:
    duty: np.ndarray
    v_pv_true: np.ndarray
    i_pv_true: np.ndarray
    v_meas: np.ndarray
    i_meas: np.ndarray
    p_out: np.ndarray

    @property
    def p_pv_true(self) -> np.ndarray:
        return self.v_pv_true * self.i_pv_true


def run_fixed_conditions(
    curve: IVCurve,
    conv: Converter,
    load: ElectronicLoad,
    algo: MPPTAlgorithm,
    n_steps: int,
    sensor: INA226 | None = None,
    reset: bool = True,
) -> Trace:
    if reset:
        algo.reset()
    out = np.zeros((6, n_steps))
    v_prev = None
    for k in range(n_steps):
        d = algo.duty
        op = solve_operating_point(curve, conv, d, load, v_guess=v_prev)
        v_prev = op.v_pv
        if sensor is not None:
            m = sensor.measure(op.v_pv, op.i_pv)
            vm, im = m.v, m.i
        else:
            vm, im = op.v_pv, op.i_pv
        out[:, k] = (d, op.v_pv, op.i_pv, vm, im, op.conv.p_out)
        algo.step(vm, im)
    return Trace(*out)
