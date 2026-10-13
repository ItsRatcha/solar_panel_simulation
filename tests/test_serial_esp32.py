"""serial_esp32 HIL plugin, driven by a fake device instead of a board."""

from __future__ import annotations

import numpy as np
import pytest

from pvsim.converter import make_converter
from pvsim.load.electronic_load import ElectronicLoad
from pvsim.mppt import available_mppt, create_mppt
from pvsim.mppt.harness import run_fixed_conditions
from pvsim.mppt.serial_esp32 import FakeESP32
from pvsim.pv.datasheet import PanelDatasheet
from pvsim.pv.panel import Panel


@pytest.fixture(scope="module")
def curve():
    return Panel(PanelDatasheet()).curve(800.0, 30.0)


def test_registered_but_hidden_from_simulation_lists():
    assert "serial_esp32" in available_mppt()
    assert "serial_esp32" not in available_mppt(hardware=False)


def test_constructs_without_hardware():
    algo = create_mppt("serial_esp32", d_init=0.3)
    assert algo.duty == 0.3
    algo.reset()
    with pytest.raises(RuntimeError, match="no port"):
        algo.step(30.0, 5.0)


@pytest.mark.parametrize("name", ["po", "inccond", "global_scan"])
def test_hil_loop_matches_local_algorithm(curve, name):
    conv, load = make_converter("buck"), ElectronicLoad("CR", 3.0)
    ref = run_fixed_conditions(curve, conv, load, create_mppt(name, d_init=0.3), 400)

    hil = create_mppt("serial_esp32", d_init=0.3)
    dev = FakeESP32(name, d_init=0.3)
    hil.attach(dev)
    tr = run_fixed_conditions(curve, conv, load, hil, 400)
    assert dev.lines_in.count("R") == 2  # attach + harness reset
    np.testing.assert_array_equal(tr.duty, ref.duty)
    assert tr.p_pv_true[-100:].mean() > 0.98 * curve.mpp.p


def test_device_duty_is_clipped_and_comments_skipped():
    class Dev:
        def __init__(self):
            self.out = [b"# hello\n", b"0.4\n", b"\n", b"# dbg\n", b"1.7\n"]

        def write(self, data):
            return len(data)

        def readline(self):
            return self.out.pop(0) if self.out else b""

    algo = create_mppt("serial_esp32", d_max=0.9)
    algo.attach(Dev())
    assert algo.duty == 0.4  # initial duty taken from the device on reset
    assert algo.step(20.0, 1.0) == 0.9
    with pytest.raises(TimeoutError):
        algo.step(20.0, 1.0)


def test_pyserial_loopback_url():
    serial = pytest.importorskip("serial")
    algo = create_mppt("serial_esp32", port="loop://", timeout=0.05, reset_on_connect=0)
    algo.connect()
    # loop:// echoes the request; "V,I" is not a duty, which must be reported
    with pytest.raises(ValueError, match="unexpected reply"):
        algo.step(12.5, 3.0)
    algo.close()
    assert serial is not None
