"""INA226 current/voltage/power monitor model (TI INA226 datasheet, SBOS547).

Bus voltage : 16-bit, LSB 1.25 mV, full scale 40.96 V (0..36 V rated)
Shunt       : 16-bit signed, LSB 2.5 uV, full scale +/-81.92 mV
Current     : V_shunt / R_shunt (calibration chosen so current LSB = 2.5 uV / R)
Power       : V_bus * I, LSB 25 x current LSB

Each reading is the average of `avg` conversions of the bus and the shunt
channel, so one sample takes avg * (t_conv_bus + t_conv_shunt) and the
conversion noise is reduced by sqrt(avg).  Noise figures are given per
single conversion at 1.1 ms and scaled by sqrt(1.1 ms / t_conv).
Gain/offset errors model calibration error or an injected sensor fault.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

BUS_LSB = 1.25e-3
BUS_FS = 40.96
SHUNT_LSB = 2.5e-6
SHUNT_FS = 81.92e-3
AVG_CHOICES = (1, 4, 16, 64, 128, 256, 512, 1024)
CONV_TIMES = (140e-6, 204e-6, 332e-6, 588e-6, 1.1e-3, 2.116e-3, 4.156e-3, 8.244e-3)


@dataclass
class INA226Config:
    r_shunt: float = 0.01  # [ohm]
    avg: int = 16
    t_conv_bus: float = 1.1e-3  # [s]
    t_conv_shunt: float = 1.1e-3  # [s]
    gain_err_v: float = 0.0  # fractional, e.g. 0.001 = +0.1 %
    gain_err_i: float = 0.0
    offset_v: float = 0.0  # [V]
    offset_i: float = 0.0  # [A]
    noise_v: float = 2e-3  # bus noise per conversion [V rms]
    noise_shunt: float = 5e-6  # shunt noise per conversion [V rms]
    quantize: bool = True

    def __post_init__(self):
        if self.avg not in AVG_CHOICES:
            raise ValueError(f"avg must be one of {AVG_CHOICES}")
        for t in (self.t_conv_bus, self.t_conv_shunt):
            if not any(abs(t - c) < 1e-9 for c in CONV_TIMES):
                raise ValueError(f"conversion time must be one of {CONV_TIMES}")
        if self.r_shunt <= 0:
            raise ValueError("r_shunt must be > 0")

    @property
    def sample_time(self) -> float:
        return self.avg * (self.t_conv_bus + self.t_conv_shunt)

    @property
    def current_lsb(self) -> float:
        return SHUNT_LSB / self.r_shunt

    @classmethod
    def ideal(cls) -> "INA226Config":
        return cls(noise_v=0.0, noise_shunt=0.0, quantize=False)

    @property
    def is_ideal(self) -> bool:
        return (not self.quantize and self.noise_v == 0 and self.noise_shunt == 0
                and self.gain_err_v == 0 and self.gain_err_i == 0
                and self.offset_v == 0 and self.offset_i == 0)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "INA226Config":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class Measurement:
    v: float
    i: float
    p: float
    saturated: bool = False


def _q(x: float, lsb: float) -> float:
    return round(x / lsb) * lsb


class INA226:
    def __init__(self, config: INA226Config | None = None, seed: int | None = None):
        self.cfg = config or INA226Config()
        self.rng = np.random.default_rng(seed)

    def measure(self, v_true: float, i_true: float) -> Measurement:
        c = self.cfg
        if c.is_ideal:
            return Measurement(v_true, i_true, v_true * i_true)

        k = 1.0 / np.sqrt(c.avg)
        v = v_true * (1.0 + c.gain_err_v) + c.offset_v
        if c.noise_v > 0:
            v += self.rng.normal(0.0, c.noise_v * k * np.sqrt(1.1e-3 / c.t_conv_bus))
        vs = (i_true * (1.0 + c.gain_err_i) + c.offset_i) * c.r_shunt
        if c.noise_shunt > 0:
            vs += self.rng.normal(0.0, c.noise_shunt * k * np.sqrt(1.1e-3 / c.t_conv_shunt))

        sat = False
        if v > BUS_FS:
            v, sat = BUS_FS, True
        elif v < 0.0:
            v = 0.0
        if abs(vs) > SHUNT_FS:
            vs, sat = float(np.copysign(SHUNT_FS, vs)), True

        if c.quantize:
            v = _q(v, BUS_LSB)
            vs = _q(vs, SHUNT_LSB)
        i = vs / c.r_shunt
        p = v * i
        if c.quantize:
            p = _q(p, 25.0 * c.current_lsb)
        return Measurement(float(v), float(i), float(p), sat)
