"""Turn DayRunner snapshots into schema rows.

The input side uses the channel's own INA226 reading (what the MPPT saw);
the output side gets a second INA226 per panel, as on the real rig.
Rows are collected column-wise and packed into DataFrame chunks so long
runs at short MPPT periods stay memory friendly.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from ..engine.day_runner import DaySnapshot
from ..engine.live import LiveChannel
from ..sensors.ina226 import INA226
from .schema import column_keys

CHUNK_ROWS = 50_000
SHADE_EPS = 0.01


class Recorder:
    def __init__(self, midnight: datetime, tz_hours: float, extended: bool = False):
        self.tz = timezone(timedelta(hours=tz_hours))
        self.midnight = midnight.replace(tzinfo=self.tz)
        self.extended = extended
        self.keys = column_keys(extended)
        self._cols: dict[str, list] = {k: [] for k in self.keys}
        self._chunks: list[pd.DataFrame] = []
        self.n_rows = 0

    @staticmethod
    def label_for(ch: LiveChannel, ds: DaySnapshot) -> str:
        """Default label: the shading event's label while it actually shades this panel."""
        if ds.env.shade_label != "normal" and max(ch.cond.shade, default=0.0) > SHADE_EPS:
            return ds.env.shade_label
        return "normal"

    def record(self, panel_id: str, ch: LiveChannel, ds: DaySnapshot, out_sensor: INA226,
               label: str | None = None) -> None:
        s = ds.snap
        load = ch.load
        mo = out_sensor.measure(s.v_out, s.i_out)
        t0 = ds.clock_s - ch.dt  # the sample describes the period that started here
        row = {
            "timestamp": t0,
            "panel_id": panel_id,
            "v_pv": s.v_meas,
            "i_pv": s.i_meas,
            "p_pv": s.v_meas * s.i_meas,
            "v_out": mo.v,
            "i_out": mo.i,
            "p_out": mo.v * mo.i,
            "duty": s.duty,
            "irradiance_poa": ds.poa.total,
            "t_panel": s.t_cell,
            "t_ambient": ch.cond.t_amb,
            "tilt_deg": ds.tilt,
            "azimuth_deg": ds.azimuth,
            "load_mode": load.mode,
            "load_setpoint": load.setpoint if load.is_on else 0.0,
            "load_state": int(load.is_on),
            "fault_label": label if label is not None else self.label_for(ch, ds),
        }
        if self.extended:
            row.update(
                v_pv_true=s.v_pv,
                p_pv_true=s.p_pv,
                p_out_true=s.p_out,
                p_mpp_true=s.p_mpp,
                mppt_efficiency=s.tracking_eff,
                conv_efficiency=s.conv_eff,
                mpp_unreachable=not s.reach.reachable,
                algorithm=ch.algo.name,
                ghi=ds.env.irr.ghi,
            )
        cols = self._cols
        for k in self.keys:
            cols[k].append(row[k])
        self.n_rows += 1
        if len(cols["timestamp"]) >= CHUNK_ROWS:
            self._flush()

    def _flush(self) -> None:
        if not self._cols["timestamp"]:
            return
        df = pd.DataFrame(self._cols, columns=self.keys)
        df["timestamp"] = pd.Timestamp(self.midnight) + pd.to_timedelta(df["timestamp"], unit="s")
        self._chunks.append(df)
        self._cols = {k: [] for k in self.keys}

    def frame(self) -> pd.DataFrame:
        """All rows so far, keyed by internal column keys, sorted by time then panel."""
        self._flush()
        if not self._chunks:
            return pd.DataFrame({k: [] for k in self.keys})
        df = pd.concat(self._chunks, ignore_index=True) if len(self._chunks) > 1 else self._chunks[0]
        self._chunks = [df]
        return df.sort_values(["timestamp", "panel_id"], kind="stable", ignore_index=True)
