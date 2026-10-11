"""Converter interface shared by the averaged and switched models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..load.electronic_load import ElectronicLoad


@dataclass
class ConverterResult:
    """Steady (cycle-averaged) state of the converter at one operating point.

    Energy balance holds by construction: v_in * i_in == p_out + sum(losses).
    """

    v_in: float
    i_in: float
    v_out: float
    i_out: float
    duty: float
    mode: str  # "CCM", "DCM" or "off"
    losses: dict[str, float] = field(default_factory=dict)
    ripple_il: float = 0.0  # inductor peak-to-peak ripple [A]

    @property
    def p_in(self) -> float:
        return self.v_in * self.i_in

    @property
    def p_out(self) -> float:
        return self.v_out * self.i_out

    @property
    def p_loss(self) -> float:
        return sum(self.losses.values())

    @property
    def efficiency(self) -> float:
        return self.p_out / self.p_in if self.p_in > 0 else 0.0


class Converter(ABC):
    d_min: float
    d_max: float
    v_uvlo: float

    @abstractmethod
    def output(self, v_in: float, d: float, load: ElectronicLoad) -> ConverterResult:
        """Averaged converter state for input voltage v_in and duty d."""

    def clip_duty(self, d: float) -> float:
        return min(max(d, self.d_min), self.d_max)
