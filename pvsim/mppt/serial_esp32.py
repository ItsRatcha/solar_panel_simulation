"""Hardware-in-the-loop MPPT: the duty decisions come from an ESP32 over serial.

The simulator stays the plant (panel, converter, load, INA226) and sends each
measurement to the board, which runs the firmware MPPT
(firmware_ref/mppt.c, sketch firmware_ref/esp32_mppt_hil) and answers with
the next duty.  Line protocol (ASCII, newline terminated):

    host -> "V,I"     measured panel voltage [V] and current [A]
    dev  -> "D"       next duty cycle
    host -> "R"       reset the firmware algorithm; dev -> its initial duty
    dev  -> "# ..."   comment line, ignored

Floats are sent with repr() (shortest round-trip form), so a device that
parses with strtod sees exactly the simulator's numbers.

The port opens lazily on the first step (or on connect()), so the algorithm
can be created, listed and reset without hardware.  For tests, attach any
object with write(bytes) / readline() -> bytes instead of a port, e.g.
FakeESP32 below, which emulates the firmware with a Python MPPT.

Requires pyserial (pip install pvsim[hil]) only when a real port is opened.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .base import BaseParams, MPPTAlgorithm, create_mppt, register_mppt


@dataclass
class SerialESP32Params(BaseParams):
    port: str = ""  # e.g. "COM5" or "/dev/ttyUSB0"; any pyserial URL works ("loop://", "socket://...")
    baud: int = 115200
    timeout: float = 1.0  # [s] per reply
    reset_on_connect: int = 1  # send "R" when the port opens and take the device's initial duty


@register_mppt("serial_esp32")
class SerialESP32(MPPTAlgorithm):
    Params = SerialESP32Params
    hardware: ClassVar[bool] = True  # needs a device; GUIs leave it out of their lists

    def __init__(self, params=None, link=None):
        self._link = link
        self._owns_link = False
        super().__init__(params)

    # ------------------------------------------------------------ link
    def attach(self, link) -> None:
        """Use a serial-like object (write(bytes), readline() -> bytes) as the device."""
        self.close()
        self._link = link
        self._owns_link = False
        self._sync_reset()

    def connect(self) -> None:
        """Open the configured port now (otherwise it opens on the first step)."""
        if self._link is not None:
            return
        if not self.p.port:
            raise RuntimeError("serial_esp32: no port set (Params.port) and no link attached")
        try:
            import serial  # pyserial
        except ImportError as e:  # pragma: no cover - depends on the environment
            raise ImportError("serial_esp32 needs pyserial: pip install pvsim[hil]") from e
        self._link = serial.serial_for_url(self.p.port, baudrate=self.p.baud, timeout=self.p.timeout)
        self._owns_link = True
        self._sync_reset()

    def close(self) -> None:
        if self._link is not None and self._owns_link:
            self._link.close()
        self._link = None
        self._owns_link = False

    def _sync_reset(self) -> None:
        if self.p.reset_on_connect:
            self._send("R")
            self.s.d = self.clip(self._read_duty())

    def _send(self, text: str) -> None:
        self._link.write((text + "\n").encode("ascii"))

    def _read_duty(self) -> float:
        while True:
            raw = self._link.readline()
            if not raw:
                raise TimeoutError(f"serial_esp32: no reply within {self.p.timeout} s")
            text = raw.decode("ascii", errors="replace").strip()
            if not text or text.startswith("#"):
                continue
            try:
                return float(text)
            except ValueError:
                raise ValueError(f"serial_esp32: unexpected reply {text!r}") from None

    # ------------------------------------------------------------ MPPT interface
    def reset(self) -> None:
        super().reset()
        if self._link is not None:
            self._sync_reset()

    def step(self, v: float, i: float) -> float:
        if self._link is None:
            self.connect()
        self._send(f"{float(v)!r},{float(i)!r}")
        self.s.d = self.clip(self._read_duty())
        self._remember(v, i)
        return self.s.d


class FakeESP32:
    """Serial-like stand-in for the board: runs `algo` (any pvsim MPPT, by name
    or instance) behind the line protocol, like esp32_mppt_hil.ino does."""

    def __init__(self, algo: str | MPPTAlgorithm = "po", **params):
        self.algo = create_mppt(algo, **params) if isinstance(algo, str) else algo
        self._out: list[bytes] = [b"# fake ESP32 ready\n"]
        self.lines_in: list[str] = []

    def write(self, data: bytes) -> int:
        for text in data.decode("ascii").splitlines():
            self.lines_in.append(text)
            if text == "R":
                self.algo.reset()
                d = self.algo.duty
            else:
                v, i = (float(x) for x in text.split(","))
                d = self.algo.step(v, i)
            self._out.append(f"{d!r}\n".encode("ascii"))
        return len(data)

    def readline(self) -> bytes:
        return self._out.pop(0) if self._out else b""

    def close(self) -> None:
        pass
