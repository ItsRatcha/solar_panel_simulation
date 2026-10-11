"""P&O with periodic global duty sweep (for partial shading).

A coarse sweep of D over [d_min, d_max] finds the region of the global
maximum; P&O then refines locally.  A rescan is triggered every
`scan_period` steps or when power jumps by more than `dp_trigger` relative to
a slow reference (sudden shading / irradiance change).  Similar in spirit to
the periodic scanning approaches reviewed by Liu et al. (2015, RSER) and
Patel & Agarwal (IEEE TIE 2008).
"""

from __future__ import annotations

from dataclasses import dataclass

from .base import BaseParams, BaseState, MPPTAlgorithm, register_mppt

TRACK = 0
SCAN = 1


@dataclass
class GlobalScanParams(BaseParams):
    scan_points: int = 40
    scan_period: int = 1000  # MPPT steps between scans (0 = only on trigger)
    dp_trigger: float = 0.25  # relative power change that forces a rescan (0 = off)
    ref_alpha: float = 0.02  # EMA weight of the power reference
    scan_on_start: int = 1


@dataclass
class GlobalScanState(BaseState):
    dir: int = 1
    mode: int = TRACK
    scan_idx: int = 0
    best_p: float = 0.0
    best_d: float = 0.0
    since_scan: int = 0
    p_ref: float = 0.0
    n_scans: int = 0


@register_mppt("global_scan")
class GlobalScanPO(MPPTAlgorithm):
    Params = GlobalScanParams
    State = GlobalScanState

    def reset(self) -> None:
        super().reset()
        if self.p.scan_on_start:
            self._start_scan()

    def _scan_duty(self, k: int) -> float:
        p = self.p
        n = max(p.scan_points, 2)
        return p.d_min + k * (p.d_max - p.d_min) / (n - 1)

    def _start_scan(self) -> None:
        s = self.s
        s.mode = SCAN
        s.scan_idx = 0
        s.best_p = -1.0
        s.best_d = s.d
        s.n_scans += 1
        s.d = self.clip(self._scan_duty(0))

    def step(self, v: float, i: float) -> float:
        s, p = self.s, self.p
        pw = v * i
        if s.mode == SCAN:
            if pw > s.best_p:
                s.best_p = pw
                s.best_d = s.d
            s.scan_idx += 1
            if s.scan_idx < max(p.scan_points, 2):
                s.d = self.clip(self._scan_duty(s.scan_idx))
            else:
                s.mode = TRACK
                s.d = s.best_d
                s.since_scan = 0
                s.p_ref = s.best_p
                s.dir = 1
                pw = s.best_p  # next comparison is against the scan optimum
            self._remember(v, i)
            s.p_prev = pw
            return s.d

        # TRACK: local P&O plus rescan triggers
        s.since_scan += 1
        rescan = p.scan_period > 0 and s.since_scan >= p.scan_period
        if p.dp_trigger > 0 and s.p_ref > 0 and abs(pw - s.p_ref) > p.dp_trigger * s.p_ref:
            rescan = True
        if rescan:
            self._start_scan()
            self._remember(v, i)
            return s.d
        s.p_ref += p.ref_alpha * (pw - s.p_ref)
        if pw - s.p_prev < 0.0:
            s.dir = -s.dir
        s.d = self.clip(s.d + s.dir * p.step)
        self._remember(v, i)
        return s.d
