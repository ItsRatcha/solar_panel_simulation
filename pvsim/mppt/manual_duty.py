"""Fixed / manually set duty (open loop) - baseline for comparisons and sweeps."""

from __future__ import annotations

from dataclasses import dataclass

from .base import BaseParams, MPPTAlgorithm, register_mppt


@dataclass
class ManualParams(BaseParams):
    duty: float = 0.5


@register_mppt("manual")
class ManualDuty(MPPTAlgorithm):
    Params = ManualParams

    def reset(self) -> None:
        super().reset()
        self.s.d = self.clip(self.p.duty)

    def set_duty(self, d: float) -> None:
        self.p.duty = d
        self.s.d = self.clip(d)

    def step(self, v: float, i: float) -> float:
        self._remember(v, i)
        return self.s.d
