"""PV module made of series substrings, each protected by a bypass diode.

Each substring of m = Ns / n_bypass cells is a single-diode string with
parameters scaled from the module (a, Rs, Rsh scale with m/Ns; Iph, I0 do
not).  Substrings are in series, so they carry the same current; the module
voltage is

    V(I) = sum_k max( V_k(I), -Vf )

where V_k is the explicit Lambert-W voltage of substring k at its own
irradiance and -Vf is the bypass diode clamp.  Evaluating V(I) on a dense
current grid gives the module I-V curve, which can have several local power
maxima when substrings see different irradiance.  Curves are cached, so a
static environment costs nothing per simulation step.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import NamedTuple, Sequence

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.signal import find_peaks

from .datasheet import PanelDatasheet, SingleDiodeParams, fit_single_diode, params_at
from .single_diode import voltage_from_current


class PowerPoint(NamedTuple):
    v: float
    i: float
    p: float


@dataclass
class IVCurve:
    """Sampled module I-V curve (v strictly increasing, 0 <= v <= Voc)."""

    v: np.ndarray
    i: np.ndarray
    v_oc: float
    i_sc: float
    mpp: PowerPoint
    local_maxima: list[PowerPoint] = field(default_factory=list)

    @property
    def p(self) -> np.ndarray:
        return self.v * self.i

    def current(self, v):
        """Module current at voltage v (linear interpolation, 0 above Voc)."""
        return np.interp(v, self.v, self.i, left=self.i_sc, right=0.0)

    def voltage(self, i):
        """Module voltage at current i (0 at or above Isc)."""
        return np.interp(i, self.i[::-1], self.v[::-1], left=self.v_oc, right=0.0)

    @property
    def r_mpp(self) -> float:
        return self.mpp.v / self.mpp.i if self.mpp.i > 0 else float("inf")


_ZERO_POINT = PowerPoint(0.0, 0.0, 0.0)


def _zero_curve() -> IVCurve:
    return IVCurve(np.array([0.0, 1e-9]), np.array([0.0, 0.0]), 0.0, 0.0, _ZERO_POINT, [])


class Panel:
    """A PV module with `n_bypass` substrings."""

    def __init__(self, datasheet: PanelDatasheet | None = None, cache_size: int = 256):
        self.ds = datasheet or PanelDatasheet()
        if self.ds.n_cells % self.ds.n_bypass:
            raise ValueError("n_cells must be divisible by n_bypass")
        self.ref = fit_single_diode(self.ds)
        self.cells_per_sub = self.ds.n_cells // self.ds.n_bypass
        self._cache: OrderedDict = OrderedDict()
        self._cache_size = cache_size

    # ------------------------------------------------------------------ params
    @property
    def n_sub(self) -> int:
        return self.ds.n_bypass

    def _g_tuple(self, g) -> tuple[float, ...]:
        if np.isscalar(g):
            return (float(g),) * self.n_sub
        g = tuple(float(x) for x in g)
        if len(g) != self.n_sub:
            raise ValueError(f"expected {self.n_sub} substring irradiances, got {len(g)}")
        return g

    def substring_params(self, g, t_cell_c: float) -> list[SingleDiodeParams]:
        return [
            params_at(self.ref, self.ds, gk, t_cell_c).scaled_to_cells(self.cells_per_sub)
            for gk in self._g_tuple(g)
        ]

    # ----------------------------------------------------------- exact physics
    def _v_of_i(self, i, subs: list[SingleDiodeParams], shorted: Sequence[bool]):
        i = np.asarray(i, dtype=float)
        v = np.zeros_like(i)
        for p, sh in zip(subs, shorted):
            if sh:  # shorted bypass diode: substring is short-circuited
                continue
            if p.i_ph <= 0:
                vk = np.where(i > 0, -self.ds.bypass_vf, 0.0)
            else:
                vk = np.maximum(voltage_from_current(i, p), -self.ds.bypass_vf)
            v = v + vk
        return v

    def current_exact(self, v: float, g, t_cell_c: float, shorted: Sequence[bool] | None = None) -> float:
        """Module current at voltage v by root-solving V(I) = v (slow, for checks)."""
        subs = self.substring_params(g, t_cell_c)
        shorted = shorted or (False,) * self.n_sub
        i_max = max(p.i_ph for p in subs)
        if i_max <= 0:
            return 0.0
        f = lambda i: float(self._v_of_i(i, subs, shorted)) - v  # noqa: E731
        if f(0.0) <= 0.0:
            return 0.0
        hi = i_max
        while f(hi) > 0:
            hi *= 1.05
        return brentq(f, 0.0, hi, xtol=1e-12)

    # ------------------------------------------------------------------- curve
    def curve(self, g, t_cell_c: float, shorted: Sequence[bool] | None = None) -> IVCurve:
        """I-V curve at substring irradiance(s) g [W/m^2] and cell temperature [degC]."""
        gt = self._g_tuple(g)
        shorted = tuple(bool(s) for s in (shorted or (False,) * self.n_sub))
        key = (tuple(round(x, 3) for x in gt), round(float(t_cell_c), 3), shorted)
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            return hit
        c = self._build_curve(gt, float(t_cell_c), shorted)
        self._cache[key] = c
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return c

    def _build_curve(self, g, t_cell_c, shorted) -> IVCurve:
        subs = self.substring_params(g, t_cell_c)
        active = [p for p, s in zip(subs, shorted) if not s]
        i_max = max((p.i_ph for p in active), default=0.0)
        if i_max <= 1e-9:
            return _zero_curve()
        grids = [np.linspace(0.0, i_max, 400)]
        for p in active:
            if p.i_ph > 0:
                grids.append(p.i_ph * (1.0 - np.geomspace(1e-5, 0.4, 120)))
        i = np.unique(np.concatenate(grids))
        v = self._v_of_i(i, subs, shorted)
        # sort by voltage ascending (V(I) is non-increasing)
        order = np.argsort(v, kind="stable")
        v, i = v[order], i[order]
        if v[-1] <= 0.0:
            return _zero_curve()
        i_sc = float(np.interp(0.0, v, i)) if v[0] < 0 else float(i[0])
        keep = v > 0.0
        v = np.concatenate(([0.0], v[keep]))
        i = np.concatenate(([i_sc], i[keep]))
        v, idx = np.unique(v, return_index=True)
        i = i[idx]
        i = np.minimum.accumulate(i)  # enforce monotone (guards round-off)
        v_oc = float(v[-1])

        p = v * i
        exact = lambda vv: vv * self._current_on(vv, subs, shorted, i_max)  # noqa: E731

        def refine(k: int) -> PowerPoint:
            lo, hi = v[max(k - 1, 0)], v[min(k + 1, len(v) - 1)]
            if hi <= lo:
                return PowerPoint(float(v[k]), float(i[k]), float(p[k]))
            r = minimize_scalar(lambda x: -exact(x), bounds=(lo, hi), method="bounded",
                                options={"xatol": 1e-6})
            vb = float(r.x)
            ib = self._current_on(vb, subs, shorted, i_max)
            if vb * ib < p[k]:
                return PowerPoint(float(v[k]), float(i[k]), float(p[k]))
            return PowerPoint(vb, ib, vb * ib)

        pmax = float(p.max())
        peaks, _ = find_peaks(np.concatenate(([0.0], p, [0.0])), prominence=0.005 * pmax)
        peaks = [k - 1 for k in peaks]
        if not peaks:
            peaks = [int(np.argmax(p))]
        maxima = [refine(k) for k in peaks]
        mpp = max(maxima, key=lambda m: m.p)
        return IVCurve(v, i, v_oc, i_sc, mpp, maxima)

    def _current_on(self, v: float, subs, shorted, i_max: float) -> float:
        f = lambda ii: float(self._v_of_i(ii, subs, shorted)) - v  # noqa: E731
        if f(0.0) <= 0.0:
            return 0.0
        hi = i_max
        while f(hi) > 0:
            hi *= 1.05
        return brentq(f, 0.0, hi, xtol=1e-12)
