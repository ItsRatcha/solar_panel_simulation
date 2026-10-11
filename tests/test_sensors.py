import numpy as np
import pytest

from pvsim.sensors.ina226 import INA226, INA226Config


def test_bus_voltage_quantized_to_lsb():
    s = INA226(INA226Config(noise_v=0.0, noise_shunt=0.0))
    m = s.measure(12.3456789, 1.0)
    assert m.v / 1.25e-3 == pytest.approx(round(m.v / 1.25e-3), abs=1e-9)
    assert abs(m.v - 12.3456789) <= 1.25e-3 / 2 + 1e-12


def test_current_quantized_to_shunt_lsb():
    cfg = INA226Config(r_shunt=0.01, noise_v=0.0, noise_shunt=0.0)
    s = INA226(cfg)
    m = s.measure(10.0, 1.23456)
    lsb = 2.5e-6 / 0.01
    assert m.i / lsb == pytest.approx(round(m.i / lsb), abs=1e-6)
    assert abs(m.i - 1.23456) <= lsb / 2 + 1e-12


def test_shunt_range_clips():
    cfg = INA226Config(r_shunt=0.02, noise_v=0.0, noise_shunt=0.0)
    m = INA226(cfg).measure(10.0, 10.0)  # 200 mV across shunt > 81.92 mV
    assert m.i == pytest.approx(81.92e-3 / 0.02, rel=1e-6)
    assert m.saturated


def test_bus_range_clips():
    m = INA226(INA226Config(noise_v=0.0, noise_shunt=0.0)).measure(50.0, 0.0)
    assert m.v == pytest.approx(40.96, abs=1.25e-3)
    assert m.saturated


def test_gain_and_offset_errors():
    cfg = INA226Config(gain_err_v=0.01, offset_v=0.1, gain_err_i=-0.02, offset_i=0.05,
                       noise_v=0.0, noise_shunt=0.0, quantize=False)
    m = INA226(cfg).measure(10.0, 2.0)
    assert m.v == pytest.approx(10.0 * 1.01 + 0.1)
    assert m.i == pytest.approx(2.0 * 0.98 + 0.05)
    assert m.p == pytest.approx(m.v * m.i)


def test_averaging_reduces_noise():
    def std(avg):
        s = INA226(INA226Config(avg=avg, quantize=False, noise_v=5e-3), seed=1)
        return np.std([s.measure(12.0, 1.0).v for _ in range(4000)])

    ratio = std(1) / std(16)
    assert ratio == pytest.approx(4.0, rel=0.1)


def test_sample_time_from_conversion_settings():
    cfg = INA226Config(avg=16, t_conv_bus=1.1e-3, t_conv_shunt=1.1e-3)
    assert cfg.sample_time == pytest.approx(16 * 2.2e-3)


def test_invalid_settings_rejected():
    with pytest.raises(ValueError):
        INA226Config(avg=10)
    with pytest.raises(ValueError):
        INA226Config(t_conv_bus=1e-3)


def test_ideal_is_transparent():
    m = INA226(INA226Config.ideal()).measure(17.123456, 2.654321)
    assert m.v == 17.123456 and m.i == 2.654321


def test_seeded_noise_is_reproducible():
    a = INA226(INA226Config(), seed=7)
    b = INA226(INA226Config(), seed=7)
    assert [a.measure(12, 1).v for _ in range(5)] == [b.measure(12, 1).v for _ in range(5)]
