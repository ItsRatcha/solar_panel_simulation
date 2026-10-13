"""Switched (cycle-by-cycle) buck / boost model for ripple and cross-checking.

The averaged models (`buck_averaged`, `boost_averaged`) are what the engine
steps; this model resolves each switching period instead, for a short window,
to show ripple and to check that the averaged models' cycle averages hold.

State x = (iL, vC_out, vC_in), integrated with fixed-step RK4 at
`steps_per_cycle` substeps per period; the on/off instants fall exactly on
substep edges.  Within one switch state the circuit is piecewise linear except
for the panel (I-V curve) and the load (piecewise linear sink):

    buck  on : L diL/dt = v_pv - iL (Rds + R_L) - v_out        i_sw = iL
          off: L diL/dt = -Vf - iL R_L - v_out  (diode, iL >= 0)
               L diL/dt = -iL (Rds_low + R_L) - v_out  (synchronous)
               i_sw = 0
    boost on : L diL/dt = v_pv - iL (Rds + R_L)                 i_node = 0
          off: L diL/dt = v_pv - iL R_L - Vf - v_out (diode, iL >= 0)
               L diL/dt = v_pv - iL (Rds_low + R_L) - v_out (synchronous)
               i_node = iL;  i_sw = iL always (inductor on the input side)

    C_out dvC_out/dt = i_node - I_load(v_out),  v_out = vC_out + ESR_out (i_node - I_load)
    C_in  dvC_in/dt  = I_pv(v_pv) - i_sw,        v_pv  = vC_in  + ESR_in  (I_pv - i_sw)

(buck: i_node = iL).  Both ESR node equations are solved exactly: the load
through `ElectronicLoad.intersect`, the panel on the linear segment of the
sampled I-V curve.  An asynchronous inductor current is clamped at zero, which
gives DCM by itself.

Switching overlap and gate-drive energy cannot appear in an ideal-switch
circuit, so they are taken from the input capacitor as discrete energies at
each edge (E = 1/2 V_block |iL| t_r or t_f, plus Qg Vg per turn-on).  This
matches the averaged models' 1/2 V I (tr+tf) fsw and Qg Vg fsw terms.

Controller behaviour: no pulse in a period that starts below UVLO (buck: off;
boost: diode path), and a boost skips the pulse while vC_out >= v_out_max.

Every loss is integrated from the waveforms, so over the window
    E_in = E_out + sum(E_loss) + delta(E_stored)
holds up to the integration error (reported as `energy_residual`).
"""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass, field

import numpy as np

from ..load.electronic_load import ElectronicLoad
from .base import Converter, ConverterResult


@dataclass
class SwitchedResult:
    t: np.ndarray  # [s] from the start of the recorded window, every substep
    i_l: np.ndarray
    v_in: np.ndarray  # panel (converter input) node voltage
    v_out: np.ndarray
    duty: float
    mode: str  # "CCM", "DCM", "off" (no pulses) or "OVP" (pulses skipped)
    v_in_avg: float
    i_in_avg: float  # source (panel) current
    v_out_avg: float
    i_out_avg: float
    p_in: float  # source power, window average
    p_out: float  # load power, window average
    losses: dict[str, float] = field(default_factory=dict)  # window averages [W]
    ripple_il: float = 0.0  # peak-to-peak over the last period
    ripple_v_out: float = 0.0
    ripple_v_in: float = 0.0
    energy_residual: float = 0.0  # (E_in - E_out - E_loss - dE_stored) / E_in

    @property
    def p_loss(self) -> float:
        return sum(self.losses.values())

    @property
    def efficiency(self) -> float:
        return self.p_out / self.p_in if self.p_in > 0 else 0.0

    def to_converter_result(self) -> ConverterResult:
        """Cycle-averaged view, comparable with the averaged models' output."""
        return ConverterResult(self.v_in_avg, self.i_in_avg, self.v_out_avg, self.i_out_avg,
                               self.duty, self.mode, dict(self.losses), self.ripple_il)


class _PanelNode:
    """Panel node with input-capacitor ESR: v = vc + esr (I_pv(v) - i_sw), solved exactly."""

    def __init__(self, curve, esr: float):
        self.v = [float(x) for x in curve.v]
        self.i = [float(x) for x in curve.i]
        self.v_oc, self.i_sc = float(curve.v_oc), float(curve.i_sc)
        self.esr = esr
        n = len(self.v)
        self.slope = [(self.i[k + 1] - self.i[k]) / (self.v[k + 1] - self.v[k]) for k in range(n - 1)]

    def segment(self, v: float) -> tuple[float, float]:
        """(a, b) with I_pv = a + b v on the segment containing v."""
        if v >= self.v[-1]:
            return 0.0, 0.0
        if v <= self.v[0]:
            return self.i_sc, 0.0
        k = bisect_right(self.v, v) - 1
        b = self.slope[k]
        return self.i[k] - b * self.v[k], b

    def solve(self, vc: float, i_sw: float) -> tuple[float, float]:
        if self.esr <= 0.0:
            a, b = self.segment(vc)
            return vc, a + b * vc
        v = vc
        for _ in range(8):
            a, b = self.segment(v)
            v_new = (vc + self.esr * (a - i_sw)) / (1.0 - self.esr * b)
            if abs(v_new - v) < 1e-12:
                break
            v = v_new
        a, b = self.segment(v)
        return v, a + b * v


def simulate_switched(
    conv: Converter,
    d: float,
    load: ElectronicLoad,
    source,
    *,
    n_cycles: int = 20,
    n_settle: int = 300,
    steps_per_cycle: int = 100,
    init: tuple[float, float, float] | None = None,
) -> SwitchedResult:
    """Simulate `n_settle` + `n_cycles` switching periods at fixed duty `d`.

    conv   : BuckAveraged or BoostAveraged (its params and topology are used)
    source : an IVCurve (panel) or a float (ideal voltage source; C_in unused)
    init   : (iL, vC_out, vC_in); default = the averaged model's operating point
    The returned waveforms and averages cover the last `n_cycles` periods.
    """
    p = conv.p
    boost = conv.topology == "boost"
    d = conv.clip_duty(d)
    sync = p.synchronous
    vf = 0.0 if sync else p.v_f
    l_, r_l, rds = p.l, p.r_l, p.rds_on
    r_off = p.rds_on_low + r_l if sync else r_l  # resistance in the freewheel path
    c_out, esr_out = p.c_out, p.esr_out
    c_in = p.c_in
    t_sw = 1.0 / p.f_sw
    ideal_src = not hasattr(source, "current")
    if ideal_src:
        v_src = float(source)
        node = None
    else:
        node = _PanelNode(source, p.esr_in)
    v_out_max = getattr(p, "v_out_max", math.inf)

    if init is None:
        init = _averaged_init(conv, d, load, source, boost)
    i_l, vc_out, vc_in = (float(x) for x in init)
    if ideal_src:
        vc_in = v_src

    def input_node(vc, i_sw):
        if ideal_src:
            return v_src, i_sw
        return node.solve(vc, i_sw)

    def out_node(vc, i_node):
        v = load.intersect(vc + esr_out * i_node, esr_out) if esr_out > 0 else max(vc, 0.0)
        return v, load.current(v)

    def deriv(on, il, vco, vci):
        """Derivatives plus the node quantities used for loss bookkeeping."""
        if boost:
            i_sw = il
            i_node = 0.0 if on else il
        else:
            i_sw = il if on else 0.0
            i_node = il
        v_pv, i_pv = input_node(vci, i_sw)
        v_o, i_o = out_node(vco, i_node)
        if boost:
            dil = v_pv - il * (r_l + rds) if on else v_pv - il * r_off - vf - v_o
        else:
            dil = v_pv - il * (rds + r_l) - v_o if on else -vf - il * r_off - v_o
        dil /= l_
        if not sync and not on and il <= 0.0 and dil < 0.0:
            dil = 0.0  # diode blocks: DCM idle interval
        dvo = (i_node - i_o) / c_out
        dvi = 0.0 if ideal_src else (i_pv - i_sw) / c_in
        return dil, dvo, dvi, v_pv, i_pv, i_sw, v_o, i_o, i_node

    def power_terms(on, il, q):
        v_pv, i_pv, i_sw, v_o, i_o, i_node = q[3], q[4], q[5], q[6], q[7], q[8]
        il2 = il * il
        cond_on = il2 * rds if on else 0.0
        diode = 0.0
        low = 0.0
        if not on:
            if sync:
                low = il2 * p.rds_on_low
            else:
                diode = vf * max(il, 0.0)
        p_in = v_pv * (i_sw if ideal_src else i_pv)
        esr_in = 0.0 if ideal_src else p.esr_in * (i_pv - i_sw) ** 2
        esr_o = esr_out * (i_node - i_o) ** 2
        return (p_in, v_o * i_o, cond_on, il2 * r_l, diode, low, esr_in, esr_o)

    names = ("switch_cond", "inductor_dcr", "diode", "low_side_cond", "esr_in", "esr_out")
    n_total = n_settle + n_cycles
    steps = max(int(steps_per_cycle), 4)
    acc = [0.0] * 8  # p_in, p_out, then the six conduction-type losses (integrated energy)
    e_switching = e_gate = 0.0
    acc_v_in = acc_i_in = acc_v_out = acc_i_out = 0.0
    rec_t, rec_il, rec_vin, rec_vout = [], [], [], []
    pulses = skipped = 0
    hit_zero = False
    e_stored0 = 0.0
    t = 0.0

    def stored(il, vco, vci):
        e = 0.5 * l_ * il * il + 0.5 * c_out * vco * vco
        return e if ideal_src else e + 0.5 * c_in * vci * vci

    def edge_energy(il, vco, vci, t_edge):
        """Overlap energy of one switch edge; returns (energy, new vC_in)."""
        if t_edge <= 0.0:
            return 0.0, vci
        if boost:
            v_o, _ = out_node(vco, max(il, 0.0))
            v_block = v_o + vf
        else:
            v_block = input_node(vci, 0.0)[0] + vf
        e = 0.5 * v_block * abs(il) * t_edge
        return e, vci

    for cyc in range(n_total):
        record = cyc >= n_settle
        if record and cyc == n_settle:
            e_stored0 = stored(i_l, vc_out, vc_in)
        v_now = input_node(vc_in, i_l if boost else 0.0)[0]
        pulse = 0.0 < d and v_now >= p.v_uvlo and not (boost and vc_out >= v_out_max)
        if pulse and d >= 1.0:
            n_on = steps
        elif pulse:
            n_on = min(max(round(steps * d), 1), steps - 1)
        else:
            n_on = 0
        if pulse:
            pulses += record
        elif d > 0.0:
            skipped += record
        seg = ((True, n_on, d * t_sw if pulse else 0.0), (False, steps - n_on, (1.0 - d) * t_sw if pulse else t_sw))
        for on, n, span in seg:
            if n == 0:
                continue
            if on:  # turn-on edge and gate charge
                e_on, _ = edge_energy(i_l, vc_out, vc_in, p.t_r)
                e_edge = e_on + p.q_g * p.v_g
                vc_in = _take_energy(vc_in, e_edge, c_in, ideal_src)
                if record:
                    e_switching += e_on
                    e_gate += p.q_g * p.v_g
            h = span / n
            for _ in range(n):
                k1 = deriv(on, i_l, vc_out, vc_in)
                k2 = deriv(on, i_l + 0.5 * h * k1[0], vc_out + 0.5 * h * k1[1], vc_in + 0.5 * h * k1[2])
                k3 = deriv(on, i_l + 0.5 * h * k2[0], vc_out + 0.5 * h * k2[1], vc_in + 0.5 * h * k2[2])
                k4 = deriv(on, i_l + h * k3[0], vc_out + h * k3[1], vc_in + h * k3[2])
                il_new = i_l + h / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
                vco_new = vc_out + h / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
                vci_new = vc_in + h / 6.0 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
                if not sync and il_new < 0.0:
                    il_new = 0.0
                    hit_zero = hit_zero or record
                if record:
                    # Simpson's rule over the substep, with k2/k3 as the midpoint estimate
                    q_mid = deriv(on, 0.5 * (i_l + il_new), 0.5 * (vc_out + vco_new), 0.5 * (vc_in + vci_new))
                    q_end = deriv(on, il_new, vco_new, vci_new)
                    a = power_terms(on, i_l, k1)
                    b = power_terms(on, 0.5 * (i_l + il_new), q_mid)
                    c = power_terms(on, il_new, q_end)
                    for j in range(8):
                        acc[j] += h / 6.0 * (a[j] + 4.0 * b[j] + c[j])
                    acc_v_in += h / 6.0 * (k1[3] + 4.0 * q_mid[3] + q_end[3])
                    acc_i_in += h / 6.0 * ((k1[5] if ideal_src else k1[4]) + 4.0 * (q_mid[5] if ideal_src else q_mid[4])
                                           + (q_end[5] if ideal_src else q_end[4]))
                    acc_v_out += h / 6.0 * (k1[6] + 4.0 * q_mid[6] + q_end[6])
                    acc_i_out += h / 6.0 * (k1[7] + 4.0 * q_mid[7] + q_end[7])
                i_l, vc_out, vc_in = il_new, vco_new, vci_new
                t += h
                if record:
                    q = deriv(on, i_l, vc_out, vc_in)
                    rec_t.append(t)
                    rec_il.append(i_l)
                    rec_vin.append(q[3])
                    rec_vout.append(q[6])
            if on:  # turn-off edge
                e_off, _ = edge_energy(i_l, vc_out, vc_in, p.t_f)
                vc_in = _take_energy(vc_in, e_off, c_in, ideal_src)
                if record:
                    e_switching += e_off

    t_win = n_cycles * t_sw
    e_loss = dict(zip(names, acc[2:]))
    e_loss["switching"] = e_switching
    e_loss["gate"] = e_gate
    e_loss.pop("diode" if sync else "low_side_cond")
    e_in = acc[0] + (e_switching + e_gate if ideal_src else 0.0)
    e_out = acc[1]
    d_stored = stored(i_l, vc_out, vc_in) - e_stored0
    residual = (e_in - e_out - sum(e_loss.values()) - d_stored) / e_in if e_in > 0 else 0.0

    if pulses == 0:
        mode = "off"
    elif skipped > 0:
        mode = "OVP"
    elif hit_zero:
        mode = "DCM"
    else:
        mode = "CCM"
    tail = slice(-steps, None)
    arr = np.asarray
    return SwitchedResult(
        t=arr(rec_t) - n_settle * t_sw, i_l=arr(rec_il), v_in=arr(rec_vin), v_out=arr(rec_vout), duty=d, mode=mode,
        v_in_avg=acc_v_in / t_win, i_in_avg=acc_i_in / t_win,
        v_out_avg=acc_v_out / t_win, i_out_avg=acc_i_out / t_win,
        p_in=e_in / t_win, p_out=e_out / t_win,
        losses={k: v / t_win for k, v in e_loss.items()},
        ripple_il=float(np.ptp(rec_il[tail])), ripple_v_out=float(np.ptp(rec_vout[tail])),
        ripple_v_in=float(np.ptp(rec_vin[tail])), energy_residual=residual,
    )


def _take_energy(vc: float, e: float, c: float, ideal_src: bool) -> float:
    """Remove energy e from a capacitor at voltage vc (no-op for an ideal source)."""
    if ideal_src or e <= 0.0:
        return vc
    return math.sqrt(max(vc * vc - 2.0 * e / c, 0.0))


def _averaged_init(conv, d, load, source, boost):
    from ..engine.operating_point import solve_operating_point

    if hasattr(source, "current"):
        op = solve_operating_point(source, conv, d, load)
        r = op.conv
        v_in = op.v_pv
    else:
        r = conv.output(float(source), d, load)
        v_in = float(source)
    i_l = r.i_in if boost else r.i_out
    return i_l, r.v_out, v_in
