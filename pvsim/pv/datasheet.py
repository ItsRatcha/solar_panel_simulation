"""Datasheet parameters and single-diode parameter extraction.

The five single-diode parameters (Iph, I0, Rs, Rsh, n) are extracted from the
values printed on a module datasheet at STC (1000 W/m^2, 25 degC, AM1.5):

    Voc, Isc, Vmp, Imp, Ns (cells in series)

With the ideality factor n fixed, the four remaining unknowns are found by
forcing the model through three points of the datasheet I-V curve
(short circuit, open circuit, maximum power) and imposing dP/dV = 0 at the
MPP (the approach of Villalva et al., 2009, solved here as a 2-D root problem
in (Rs, Rsh) because Iph and I0 follow linearly from the Isc/Voc equations).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.optimize import fsolve

# Physical constants
Q = 1.602176634e-19  # electron charge [C]
K = 1.380649e-23  # Boltzmann constant [J/K]
T_REF = 298.15  # STC cell temperature [K]
G_REF = 1000.0  # STC irradiance [W/m^2]


def thermal_voltage(t_kelvin: float) -> float:
    """kT/q [V]."""
    return K * t_kelvin / Q


@dataclass
class PanelDatasheet:
    """Values as printed on a PV module datasheet (STC unless noted).

    ASSUMPTION: the defaults are a generic 50 W / 36-cell module standing in
    for the panels in the EE06 proposal (~50 W mono, not yet purchased);
    replace them with the purchased module's datasheet.  The ratios are in
    the published ranges: Vmp/Voc = 0.833 vs Kpv = 0.816 measured by Baimel
    et al. (Electronics 8(3):321, 2019; Kpv 0.7-0.9 across panels), and
    Imp/Isc = 0.927 within the FSCC k = 0.9-0.98 (Suryakumari 2013, weak
    venue, background only).  NOCT 45 degC is the typical value used with the
    NOCT model (Skoplaki & Palyvos, Renew. Energy 34:23-29, 2009).
    """

    name: str = "Generic 50 W mono (36 cells)"
    p_max: float = 50.0  # [W]
    v_oc: float = 21.6  # [V]
    i_sc: float = 3.0  # [A]
    v_mp: float = 18.0  # [V]
    i_mp: float = 2.78  # [A]
    n_cells: int = 36  # cells in series
    n_bypass: int = 3  # bypass diodes = substrings
    noct: float = 45.0  # nominal operating cell temperature [degC]
    alpha_isc_pct: float = 0.05  # Isc temperature coefficient [%/K]
    beta_voc_pct: float = -0.32  # Voc temperature coefficient [%/K]
    gamma_pmax_pct: float = -0.42  # Pmax temperature coefficient [%/K] (informative)
    ideality: float | None = None  # force a diode ideality factor (None = auto)
    bypass_vf: float = 0.45  # bypass diode forward voltage [V]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PanelDatasheet":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    @property
    def alpha_isc(self) -> float:
        """Isc temperature coefficient [A/K]."""
        return self.alpha_isc_pct / 100.0 * self.i_sc

    @property
    def beta_voc(self) -> float:
        """Voc temperature coefficient [V/K]."""
        return self.beta_voc_pct / 100.0 * self.v_oc


@dataclass(frozen=True)
class SingleDiodeParams:
    """Single-diode parameters of a string of `n_cells` cells at given conditions.

    a = n * Ns * k * T / q is the modified ideality factor [V].
    """

    i_ph: float
    i_0: float
    r_s: float
    r_sh: float
    a: float
    n: float = field(default=1.3)
    n_cells: int = field(default=36)

    def scaled_to_cells(self, m: int) -> "SingleDiodeParams":
        """Parameters of a sub-string of m cells out of n_cells (same current)."""
        f = m / self.n_cells
        return SingleDiodeParams(
            self.i_ph, self.i_0, self.r_s * f, self.r_sh * f, self.a * f, self.n, m
        )


class FitError(RuntimeError):
    pass


def _iph_i0(isc: float, voc: float, rs: float, rsh: float, a: float) -> tuple[float, float]:
    """Iph and I0 from the short-circuit and open-circuit equations (linear in both)."""
    e_voc = math.exp(voc / a)
    e_isc = math.exp(isc * rs / a)
    i0 = (isc * (1.0 + rs / rsh) - voc / rsh) / (e_voc - e_isc)
    iph = voc / rsh + i0 * (e_voc - 1.0)
    return iph, i0


def _residuals(x, ds: PanelDatasheet, a: float):
    rs, rsh = math.exp(x[0]), math.exp(x[1])
    iph, i0 = _iph_i0(ds.i_sc, ds.v_oc, rs, rsh, a)
    vd = ds.v_mp + ds.i_mp * rs
    e = math.exp(vd / a)
    r1 = iph - i0 * (e - 1.0) - vd / rsh - ds.i_mp
    g = i0 / a * e + 1.0 / rsh  # dId/dVd
    didv = -g / (1.0 + g * rs)
    r2 = didv + ds.i_mp / ds.v_mp
    return [r1 / ds.i_mp, r2 * ds.v_mp / ds.i_mp]


def fit_single_diode(ds: PanelDatasheet) -> SingleDiodeParams:
    """Extract STC single-diode parameters from datasheet values."""
    vt = thermal_voltage(T_REF)
    n_candidates = [ds.ideality] if ds.ideality else [1.3, 1.2, 1.4, 1.1, 1.5, 1.0, 1.6, 1.8, 2.0]
    guesses = [(0.3, 300.0), (0.1, 100.0), (0.5, 1000.0), (0.05, 50.0), (1.0, 3000.0)]
    for n in n_candidates:
        a = n * ds.n_cells * vt
        for rs0, rsh0 in guesses:
            try:
                x, _info, ier, _msg = fsolve(
                    _residuals, [math.log(rs0), math.log(rsh0)], args=(ds, a), full_output=True,
                    xtol=1e-12,
                )
            except (OverflowError, ZeroDivisionError, ValueError):
                continue
            if ier != 1:
                continue
            res = np.abs(_residuals(x, ds, a))
            rs, rsh = math.exp(x[0]), math.exp(x[1])
            if res.max() > 1e-8 or not (1e-5 < rs < 10 * ds.v_oc / ds.i_sc) or rsh < 1.0:
                continue
            iph, i0 = _iph_i0(ds.i_sc, ds.v_oc, rs, rsh, a)
            if iph <= 0 or i0 <= 0:
                continue
            return SingleDiodeParams(iph, i0, rs, rsh, a, n, ds.n_cells)
    raise FitError(
        "Could not fit single-diode parameters to the datasheet values; "
        "check Voc > Vmp, Isc > Imp and a realistic fill factor."
    )


def params_at(
    ref: SingleDiodeParams, ds: PanelDatasheet, g: float, t_cell_c: float
) -> SingleDiodeParams:
    """Translate STC parameters to irradiance g [W/m^2] and cell temperature [degC].

    Iph  = (G/Gref) * (Iph_ref + alpha_Isc * dT)                (De Soto et al., 2006)
    a    = a_ref * T/Tref
    I0   = (Iph(Gref,T) - Voc(T)/Rsh) / (exp(Voc(T)/a) - 1),  Voc(T) = Voc + beta_Voc*dT
           (Villalva et al., 2009: makes the model honour the datasheet Voc coefficient)
    Rsh  = Rsh_ref * Gref/G                                     (De Soto et al., 2006)
    Rs   = Rs_ref
    """
    t = t_cell_c + 273.15
    dt = t - T_REF
    a = ref.a * t / T_REF
    iph_full = ref.i_ph + ds.alpha_isc * dt
    voc_t = ds.v_oc + ds.beta_voc * dt
    i0 = (iph_full - voc_t / ref.r_sh) / math.expm1(voc_t / a)
    g_eff = max(g, 0.0)
    iph = iph_full * g_eff / G_REF
    rsh = ref.r_sh * G_REF / max(g_eff, 1.0)
    return SingleDiodeParams(iph, i0, ref.r_s, rsh, a, ref.n, ref.n_cells)


def cell_temperature_noct(t_amb_c: float, g: float, noct: float) -> float:
    """NOCT cell-temperature model: Tc = Ta + (NOCT - 20)/800 * G."""
    return t_amb_c + (noct - 20.0) / 800.0 * g
