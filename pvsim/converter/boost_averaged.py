"""Averaged (quasi-static) boost converter model with a loss breakdown.

Same assumptions as the buck (`buck_averaged`): the MPPT period is much
longer than the L-C time constants, so every MPPT step sees periodic steady
state (Erickson & Maksimovic, Fundamentals of Power Electronics, ch. 2-3, 5).

CCM, inductor volt-second balance with IL = Iout / (1-D), folded into a
Thevenin output source:

    Vout = E - Iout * Req
    E    = Vin/(1-D) - Vf                         (asynchronous: boost diode)
         = Vin/(1-D)                              (synchronous)
    Req  = (R_L + D*Rds_on [+ (1-D)*Rds_low]) / (1-D)^2

DCM (asynchronous only, light load), from volt-second balance:

    Ipk  = Vin D / (L fsw)
    D2   = Vin D / (Vout + Vf - Vin)
    Iout = Ipk D2 / 2

A boost cannot step down: when the load holds Vout below Vin the diode
conducts straight through (D = 0 path, E = Vin - Vf, Req = R_L).  The same
path conducts when the controller is off (below UVLO or D = 0), so unlike the
buck the panel is never disconnected from the load ("off" mode still carries
current).  An unloaded boost would pump Vout up without limit, so it is held
at `v_out_max` by over-voltage protection (pulse skipping, mode "OVP").

Losses use IL_rms^2 = IL^2 + dI^2/12: switch and diode (or low-side FET)
conduction, inductor DCR, input/output capacitor ESR, switching overlap
1/2 Vout IL (tr + tf) fsw (the switch blocks Vout), gate drive Qg Vg fsw.
The input current follows from the power balance, Iin = (Pout + losses)/Vin.

ASSUMPTION: the BoostParams component values are typical placeholders, not
taken from a specific board.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from scipy.optimize import brentq

from ..load.electronic_load import ElectronicLoad
from .base import Converter, ConverterResult


@dataclass
class BoostParams:
    l: float = 220e-6  # inductance [H]
    r_l: float = 0.05  # inductor DCR [ohm]
    c_in: float = 470e-6  # input capacitance [F]
    esr_in: float = 0.05  # [ohm]
    c_out: float = 470e-6  # output capacitance [F]
    esr_out: float = 0.05  # [ohm]
    rds_on: float = 0.03  # low-side (boost) MOSFET on-resistance [ohm]
    synchronous: bool = False
    rds_on_low: float = 0.03  # high-side rectifier MOSFET (synchronous only) [ohm]
    v_f: float = 0.45  # boost (Schottky) diode drop [V]
    f_sw: float = 40e3  # switching frequency [Hz]
    t_r: float = 40e-9  # switch rise time [s]
    t_f: float = 40e-9  # switch fall time [s]
    q_g: float = 20e-9  # total gate charge [C]
    v_g: float = 10.0  # gate drive voltage [V]
    d_min: float = 0.0
    d_max: float = 0.9  # practical boost limit (gain and losses blow up near 1)
    v_uvlo: float = 5.0  # controller stops switching below this input voltage [V]
    v_out_max: float = 60.0  # output over-voltage protection [V]
    allow_dcm: bool = True

    @classmethod
    def ideal(cls) -> "BoostParams":
        return cls(r_l=0.0, esr_in=0.0, esr_out=0.0, rds_on=0.0, rds_on_low=0.0, v_f=0.0,
                   t_r=0.0, t_f=0.0, q_g=0.0, d_max=0.95, v_uvlo=0.0, allow_dcm=False)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "BoostParams":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


class BoostAveraged(Converter):
    topology = "boost"
    conducts_when_off = True

    def __init__(self, params: BoostParams | None = None):
        self.p = params or BoostParams()

    @property
    def d_min(self) -> float:
        return self.p.d_min

    @property
    def d_max(self) -> float:
        return self.p.d_max

    @property
    def v_uvlo(self) -> float:
        return self.p.v_uvlo

    def output(self, v_in: float, d: float, load: ElectronicLoad) -> ConverterResult:
        p = self.p
        d = self.clip_duty(d)
        if v_in <= 0.0:
            return ConverterResult(v_in, 0.0, 0.0, 0.0, d, "off", {})
        if v_in < p.v_uvlo or d <= 0.0:
            return self._passthrough(v_in, d, load)

        sync = p.synchronous
        vf = 0.0 if sync else p.v_f
        m = 1.0 - d
        e = v_in / m - vf
        r_series = p.r_l + d * p.rds_on + (m * p.rds_on_low if sync else 0.0)
        req = r_series / m**2
        v_out = load.intersect(e, req)
        if v_out > p.v_out_max:
            return self._ovp(v_in, d, load)
        i_out = load.current(v_out)
        i_l = i_out / m
        d_il = max(v_in - i_l * (p.r_l + d * p.rds_on), 0.0) * d / (p.l * p.f_sw) \
            if p.l > 0 and p.f_sw > 0 else 0.0

        if not sync and p.allow_dcm and d_il > 0 and i_l < d_il / 2.0:
            r = self._dcm(v_in, d, load)
            if r is not None:
                return r

        il_rms2 = i_l**2 + d_il**2 / 12.0
        losses = {
            "switch_cond": d * il_rms2 * p.rds_on,
            "inductor_dcr": il_rms2 * p.r_l,
            "esr_in": d_il**2 / 12.0 * p.esr_in,
            "esr_out": max(m * il_rms2 - i_out**2, 0.0) * p.esr_out,
            "switching": 0.5 * (v_out + vf) * i_l * (p.t_r + p.t_f) * p.f_sw,
            "gate": p.q_g * p.v_g * p.f_sw,
        }
        if sync:
            losses["low_side_cond"] = m * il_rms2 * p.rds_on_low
        else:
            losses["diode"] = i_out * vf
        return self._result(v_in, d, v_out, i_out, "CCM", losses, d_il)

    def _result(self, v_in, d, v_out, i_out, mode, losses, ripple) -> ConverterResult:
        p_out = v_out * i_out
        i_in = (p_out + sum(losses.values())) / v_in
        return ConverterResult(v_in, i_in, v_out, i_out, d, mode, losses, ripple)

    def _passthrough(self, v_in: float, d: float, load: ElectronicLoad) -> ConverterResult:
        """Not switching: panel -> inductor -> diode (or body diode) -> load."""
        p = self.p
        vf = p.v_f
        v_out = min(load.intersect(v_in - vf, p.r_l), p.v_out_max) if v_in > vf else 0.0
        i_out = load.current(v_out)
        losses = {"inductor_dcr": i_out**2 * p.r_l, "diode": i_out * vf}
        return self._result(v_in, d, v_out, i_out, "off", losses, 0.0)

    def _ovp(self, v_in: float, d: float, load: ElectronicLoad) -> ConverterResult:
        """Over-voltage protection: pulses are skipped to hold Vout at v_out_max."""
        p = self.p
        v_out = p.v_out_max
        i_out = load.current(v_out)
        losses = {
            "inductor_dcr": (i_out * v_out / max(v_in, 1e-12)) ** 2 * p.r_l,
            "diode": i_out * (0.0 if p.synchronous else p.v_f),
            "gate": p.q_g * p.v_g * p.f_sw,
        }
        return self._result(v_in, d, v_out, i_out, "OVP", losses, 0.0)

    def _dcm(self, v_in: float, d: float, load: ElectronicLoad) -> ConverterResult | None:
        """DCM operating point, or None when the load pulls past the CCM boundary
        (a stiff CV load right at the ideal-ratio voltage): then the CCM result stands."""
        p = self.p
        vf = p.v_f
        ipk = v_in * d / (p.l * p.f_sw)

        def d2_of(vo: float) -> float:
            return min(v_in * d / (vo + vf - v_in), 1.0 - d)

        def i_conv(vo: float) -> float:
            return ipk * d2_of(vo) / 2.0

        v_lo = v_in / (1.0 - d) - vf  # CCM/DCM boundary: D2 = 1 - D
        if i_conv(p.v_out_max) > load.current(p.v_out_max):
            return self._ovp(v_in, d, load)
        if i_conv(v_lo) < load.current(v_lo):
            return None
        v_out = brentq(lambda vo: i_conv(vo) - load.current(vo), v_lo, p.v_out_max, xtol=1e-12)
        i_out = load.current(v_out)
        d2 = d2_of(v_out)
        il_rms2 = ipk**2 * (d + d2) / 3.0
        dio_rms2 = ipk**2 * d2 / 3.0
        il_avg = ipk * (d + d2) / 2.0
        losses = {
            "switch_cond": ipk**2 * d / 3.0 * p.rds_on,
            "inductor_dcr": il_rms2 * p.r_l,
            "esr_in": max(il_rms2 - il_avg**2, 0.0) * p.esr_in,
            "esr_out": max(dio_rms2 - i_out**2, 0.0) * p.esr_out,
            "switching": 0.5 * (v_out + vf) * ipk * p.t_f * p.f_sw,  # zero-current turn-on
            "gate": p.q_g * p.v_g * p.f_sw,
            "diode": vf * i_out,
        }
        return self._result(v_in, d, v_out, i_out, "DCM", losses, ipk)
