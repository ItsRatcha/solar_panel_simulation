"""Averaged (quasi-static) buck converter model with a loss breakdown.

The MPPT period (ms to s) is much longer than the converter's L-C time
constants (tens of microseconds), so each MPPT step sees the converter in
periodic steady state (Erickson & Maksimovic, Fundamentals of Power
Electronics, ch. 2-3 and 5).

CCM, with non-ideal elements folded into a Thevenin output source:

    Vout = E - Iout * Req
    E    = D*Vin - (1-D)*Vf              (asynchronous: freewheeling diode)
         = D*Vin                         (synchronous)
    Req  = D*Rds_on + R_L  [+ (1-D)*Rds_low if synchronous]

DCM (asynchronous only, light load), from volt-second balance:

    Ipk  = (Vin - Vout) * D / (L fsw)
    D2   = (Vin - Vout) * D / (Vout + Vf)
    Iout = Ipk (D + D2) / 2

Losses use the RMS inductor current IL_rms^2 = Iout^2 + dI^2/12:
switch and diode conduction, inductor DCR, input/output capacitor ESR,
switching overlap 1/2 Vin Iout (tr + tf) fsw, and gate drive Qg Vg fsw.
The input current follows from the power balance

    Iin = (Pout + sum P_loss) / Vin

so Pin = Pout + losses holds exactly at every step.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from scipy.optimize import brentq

from ..load.electronic_load import ElectronicLoad
from .base import Converter, ConverterResult


@dataclass
class BuckParams:
    l: float = 220e-6  # inductance [H]
    r_l: float = 0.05  # inductor DCR [ohm]
    c_in: float = 470e-6  # input capacitance [F]
    esr_in: float = 0.05  # [ohm]
    c_out: float = 470e-6  # output capacitance [F]
    esr_out: float = 0.05  # [ohm]
    rds_on: float = 0.03  # high-side MOSFET on-resistance [ohm]
    synchronous: bool = False
    rds_on_low: float = 0.03  # low-side MOSFET (synchronous only) [ohm]
    v_f: float = 0.45  # freewheeling (Schottky) diode drop [V]
    f_sw: float = 40e3  # switching frequency [Hz]
    t_r: float = 40e-9  # switch rise time [s]
    t_f: float = 40e-9  # switch fall time [s]
    q_g: float = 20e-9  # total gate charge [C]
    v_g: float = 10.0  # gate drive voltage [V]
    d_min: float = 0.0
    d_max: float = 0.95  # bootstrap / driver limit
    v_uvlo: float = 5.0  # converter stops switching below this input voltage [V]
    allow_dcm: bool = True

    @classmethod
    def ideal(cls) -> "BuckParams":
        return cls(r_l=0.0, esr_in=0.0, esr_out=0.0, rds_on=0.0, rds_on_low=0.0, v_f=0.0,
                   t_r=0.0, t_f=0.0, q_g=0.0, d_max=1.0, v_uvlo=0.0, allow_dcm=False)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "BuckParams":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


class BuckAveraged(Converter):
    def __init__(self, params: BuckParams | None = None):
        self.p = params or BuckParams()

    @property
    def d_min(self) -> float:
        return self.p.d_min

    @property
    def d_max(self) -> float:
        return self.p.d_max

    @property
    def v_uvlo(self) -> float:
        return self.p.v_uvlo

    def _off(self, v_in: float, d: float) -> ConverterResult:
        return ConverterResult(v_in, 0.0, 0.0, 0.0, d, "off", {})

    def output(self, v_in: float, d: float, load: ElectronicLoad) -> ConverterResult:
        p = self.p
        d = self.clip_duty(d)
        if v_in <= 0.0 or v_in < p.v_uvlo or d <= 0.0:
            return self._off(v_in, d)

        sync = p.synchronous
        vf = 0.0 if sync else p.v_f
        e = d * v_in - (1.0 - d) * vf
        req = d * p.rds_on + p.r_l + ((1.0 - d) * p.rds_on_low if sync else 0.0)
        v_out = load.intersect(e, req) if e > 0 else 0.0
        i_out = load.current(v_out)
        k_ripple = d / (p.l * p.f_sw) if p.l > 0 and p.f_sw > 0 else 0.0
        d_il = max(v_in - v_out - i_out * req, 0.0) * k_ripple

        if not sync and p.allow_dcm and k_ripple > 0 and i_out < d_il / 2.0:
            return self._dcm(v_in, d, load)

        il_rms2 = i_out**2 + d_il**2 / 12.0
        losses = {
            "switch_cond": d * il_rms2 * p.rds_on,
            "inductor_dcr": il_rms2 * p.r_l,
            "esr_out": d_il**2 / 12.0 * p.esr_out,
            "esr_in": max(d * il_rms2 - (d * i_out) ** 2, 0.0) * p.esr_in,
            "switching": 0.5 * v_in * i_out * (p.t_r + p.t_f) * p.f_sw,
            "gate": p.q_g * p.v_g * p.f_sw,
        }
        if sync:
            losses["low_side_cond"] = (1.0 - d) * il_rms2 * p.rds_on_low
        else:
            losses["diode"] = (1.0 - d) * i_out * vf
        p_out = v_out * i_out
        i_in = (p_out + sum(losses.values())) / v_in
        return ConverterResult(v_in, i_in, v_out, i_out, d, "CCM", losses, d_il)

    def _dcm(self, v_in: float, d: float, load: ElectronicLoad) -> ConverterResult:
        p = self.p
        vf = p.v_f
        lf = p.l * p.f_sw

        def i_conv(vo: float) -> float:
            ipk = (v_in - vo) * d / lf
            d2 = min((v_in - vo) * d / (vo + vf), 1.0 - d) if vo + vf > 0 else 1.0 - d
            return ipk * (d + d2) / 2.0

        if load.current(v_in) <= 0.0:
            v_out = v_in  # unloaded: output charges up to the input
        else:
            v_out = brentq(lambda vo: i_conv(vo) - load.current(vo), 0.0, v_in, xtol=1e-12)
        i_out = load.current(v_out)
        ipk = (v_in - v_out) * d / lf
        d2 = min((v_in - v_out) * d / (v_out + vf), 1.0 - d) if v_out + vf > 0 else 1.0 - d
        il_rms2 = ipk**2 * (d + d2) / 3.0
        sw_rms2 = ipk**2 * d / 3.0
        sw_avg = ipk * d / 2.0
        losses = {
            "switch_cond": sw_rms2 * p.rds_on,
            "inductor_dcr": il_rms2 * p.r_l,
            "esr_out": max(il_rms2 - i_out**2, 0.0) * p.esr_out,
            "esr_in": max(sw_rms2 - sw_avg**2, 0.0) * p.esr_in,
            "switching": 0.5 * v_in * ipk * p.t_f * p.f_sw,  # zero-current turn-on
            "gate": p.q_g * p.v_g * p.f_sw,
            "diode": vf * ipk * d2 / 2.0,
        }
        p_out = v_out * i_out
        i_in = (p_out + sum(losses.values())) / v_in
        return ConverterResult(v_in, i_in, v_out, i_out, d, "DCM", losses, ipk)
