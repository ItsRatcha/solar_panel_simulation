"""CSV columns for exported runs, defined in exactly one place.

Each column has an internal `key` (what the recorder fills in) and a CSV
`name` (the header).  To rename a column in the output, change its `name`
here; nothing else needs to change.

Rows are in long format: one row per panel per sample.  The base columns are
what the real IoT logger records, so v/i/p are *measured* values (INA226
model on both sides of the buck).  `extended=True` appends ground-truth
columns that only a simulator can provide.

`agg` says how a column is combined when resampling to coarser time steps:
    "mean"  numeric mean (power is the mean of power, never mean V * mean I)
    "mode"  most frequent value (categoricals, flags)
    ("ratio", num, den)  recomputed as sum(num) / sum(den) of the bucket
"""

from __future__ import annotations

from dataclasses import dataclass

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Column:
    key: str
    name: str
    dtype: str  # "datetime" | "float" | "int" | "str" | "bool"
    unit: str
    desc: str
    agg: object = "mean"


BASE_COLUMNS: tuple[Column, ...] = (
    Column("timestamp", "timestamp", "datetime", "", "local time (ISO 8601, with UTC offset)", "first"),
    Column("panel_id", "panel_id", "str", "", "panel / channel identifier", "first"),
    Column("v_pv", "v_pv", "float", "V", "panel voltage (input INA226)"),
    Column("i_pv", "i_pv", "float", "A", "panel current (input INA226)"),
    Column("p_pv", "p_pv", "float", "W", "panel power, v_pv * i_pv per sample"),
    Column("v_out", "v_out", "float", "V", "buck output voltage (output INA226)"),
    Column("i_out", "i_out", "float", "A", "buck output current (output INA226)"),
    Column("p_out", "p_out", "float", "W", "output power, v_out * i_out per sample"),
    Column("duty", "duty", "float", "", "PWM duty cycle applied during the sample"),
    Column("irradiance_poa", "irradiance_poa", "float", "W/m2", "plane-of-array irradiance (unshaded)"),
    Column("t_panel", "t_panel", "float", "degC", "cell temperature"),
    Column("t_ambient", "t_ambient", "float", "degC", "ambient temperature"),
    Column("tilt_deg", "tilt_deg", "float", "deg", "panel tilt from horizontal"),
    Column("azimuth_deg", "azimuth_deg", "float", "deg", "panel azimuth, clockwise from north"),
    Column("load_mode", "load_mode", "str", "", "electronic load mode CR/CC/CV/OFF", "mode"),
    Column("load_setpoint", "load_setpoint", "float", "ohm|A|V", "load setpoint (unit depends on mode)"),
    Column("load_state", "load_state", "int", "", "1 = load input enabled, 0 = off", "mode"),
    Column("fault_label", "fault_label", "str", "", "condition label for ML (normal, partial_shading_*, ...)",
           "mode"),
)

EXTENDED_COLUMNS: tuple[Column, ...] = (
    Column("v_pv_true", "v_pv_true", "float", "V", "true panel voltage"),
    Column("p_pv_true", "p_pv_true", "float", "W", "true panel power"),
    Column("p_out_true", "p_out_true", "float", "W", "true output power"),
    Column("p_mpp_true", "p_mpp_true", "float", "W", "true global maximum power of the panel"),
    Column("mppt_efficiency", "mppt_efficiency", "float", "", "p_pv_true / p_mpp_true",
           ("ratio", "p_pv_true", "p_mpp_true")),
    Column("conv_efficiency", "conv_efficiency", "float", "", "p_out_true / p_pv_true",
           ("ratio", "p_out_true", "p_pv_true")),
    Column("mpp_unreachable", "mpp_unreachable", "bool", "", "load cannot pull the panel to its MPP", "mode"),
    Column("algorithm", "algorithm", "str", "", "MPPT algorithm name", "mode"),
    Column("ghi", "ghi", "float", "W/m2", "global horizontal irradiance"),
)


def columns(extended: bool = False) -> tuple[Column, ...]:
    return BASE_COLUMNS + EXTENDED_COLUMNS if extended else BASE_COLUMNS


def column_names(extended: bool = False) -> list[str]:
    return [c.name for c in columns(extended)]


def column_keys(extended: bool = False) -> list[str]:
    return [c.key for c in columns(extended)]


def describe(extended: bool = True) -> str:
    """Markdown table of the schema (used by `pvsim schema` and the README)."""
    lines = ["| column | unit | description |", "|---|---|---|"]
    lines += [f"| {c.name} | {c.unit} | {c.desc} |" for c in columns(extended)]
    return "\n".join(lines)
