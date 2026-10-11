"""Explicit single-diode equations using the Lambert W function.

Implicit model (one string of cells):

    I = Iph - I0 * (exp((V + I*Rs)/a) - 1) - (V + I*Rs)/Rsh

Explicit solutions (Jain & Kapoor, 2004):

    I(V) = (Rsh*(Iph+I0) - V)/(Rs+Rsh)
           - (a/Rs) * W( Rs*Rsh*I0/(a*(Rs+Rsh)) * exp(Rsh*(Rs*(Iph+I0) + V)/(a*(Rs+Rsh))) )

    V(I) = Rsh*(Iph+I0) - I*(Rs+Rsh) - a * W( Rsh*I0/a * exp(Rsh*(Iph+I0-I)/a) )

The W arguments overflow float64 easily, so W is evaluated from the logarithm
of its argument (`lambertw_exp`).
"""

from __future__ import annotations

import numpy as np
from scipy.special import lambertw

from .datasheet import SingleDiodeParams


def lambertw_exp(x):
    """W(exp(x)) for real x, without overflow (principal branch)."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = x < 300.0
    if np.any(small):
        out[small] = np.real(lambertw(np.exp(x[small])))
    big = ~small
    if np.any(big):
        xb = x[big]
        w = xb - np.log(xb)  # asymptotic start
        for _ in range(6):  # Newton on w + ln(w) = x
            w = w - (w + np.log(w) - xb) / (1.0 + 1.0 / w)
        out[big] = w
    return out


def current_from_voltage(v, p: SingleDiodeParams):
    """Terminal current [A] for terminal voltage v [V] (array or scalar)."""
    v = np.asarray(v, dtype=float)
    if p.i_ph <= 0.0 and p.i_0 <= 0.0:
        return np.zeros_like(v)
    rs, rsh, a = p.r_s, p.r_sh, p.a
    rr = rs + rsh
    log_arg = np.log(rs * rsh * p.i_0 / (a * rr)) + rsh * (rs * (p.i_ph + p.i_0) + v) / (a * rr)
    return (rsh * (p.i_ph + p.i_0) - v) / rr - (a / rs) * lambertw_exp(log_arg)


def voltage_from_current(i, p: SingleDiodeParams):
    """Terminal voltage [V] for terminal current i [A] (array or scalar).

    For i > Iph the string is driven into reverse bias (negative voltage).
    """
    i = np.asarray(i, dtype=float)
    rs, rsh, a = p.r_s, p.r_sh, p.a
    log_arg = np.log(rsh * p.i_0 / a) + rsh * (p.i_ph + p.i_0 - i) / a
    return rsh * (p.i_ph + p.i_0) - i * (rs + rsh) - a * lambertw_exp(log_arg)
