"""Data cleaning: range checks, spike detection, interpolation of inputs, dropping unusable target rows."""
import numpy as np
import pandas as pd

from config import INPUTS, SENSOR_COLS, TARGETS

PHYSICAL_LIMITS = {"pump_speed_rpm": (0, 4000), "valve_opening_pct": (0, 100), "heater_power_kw": (0, 600),
                   "coolant_flow_lpm": (0, 300), "feed_temp_c": (0, 120), "ambient_temp_c": (-40, 60),
                   "fluid_viscosity_cp": (0.1, 50), "temperature_c": (0, 200), "pressure_bar": (0, 30),
                   "flow_rate_m3h": (0, 100), "energy_kw": (0, 1500)}


def detect_spikes(s, rel=0.2, abs_frac=0.04):
    """Flag isolated spikes: a point far from BOTH neighbours while the neighbours agree with each other.
    This does not flag genuine setpoint steps (where the neighbours disagree)."""
    prev, nxt = s.shift(1), s.shift(-1)
    prev, nxt = prev.fillna(nxt), nxt.fillna(prev)      # if one neighbour is missing, compare with the other
    nb = (prev + nxt) / 2
    thr = np.maximum(rel * nb.abs(), abs_frac * (s.max() - s.min()))
    return ((s - nb).abs() > thr) & ((prev - nxt).abs() < thr / 2)


def clean(df):
    report = {"rows_in": len(df)}
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    report["missing_before"] = int(df[SENSOR_COLS].isna().sum().sum())
    report["out_of_range"] = 0
    for c, (lo, hi) in PHYSICAL_LIMITS.items():
        bad = (df[c] < lo) | (df[c] > hi)
        report["out_of_range"] += int(bad.sum()); df.loc[bad, c] = np.nan
    report["spikes_removed"] = 0
    for c in SENSOR_COLS:
        sp = detect_spikes(df[c]); report["spikes_removed"] += int(sp.sum()); df.loc[sp, c] = np.nan
    # Inputs: short gaps (<=3 steps = 15 min) are interpolated in time. Targets are LABELS: never invent labels -> drop rows.
    df = df.set_index("timestamp")
    df[INPUTS] = df[INPUTS].interpolate(method="time", limit=3, limit_area="inside")
    df = df.reset_index()
    df = df.dropna(subset=INPUTS + TARGETS).reset_index(drop=True)
    df["hour_of_day"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
    report["rows_out"] = len(df)
    return df, report
