"""Synthetic dataset for a heated process loop (pump -> heater -> cooler).

IMPORTANT: this data is SYNTHETIC. It is generated from simplified physical relationships
(affinity laws for the pump, an energy balance for the heater, a first-order cooling model).
It is NOT real plant data and must never be described as such.
"""
import numpy as np
import pandas as pd

from config import SENSOR_COLS


def process_model(pump_speed_rpm, valve_opening_pct, heater_power_kw, coolant_flow_lpm,
                  feed_temp_c, ambient_temp_c, fluid_viscosity_cp, fouling_index):
    """Noise-free 'ground truth' process (the stand-in for a plant / first-principles digital twin)."""
    pump, valve, heater, coolant, feed_t, amb_t, visc, foul = [
        np.asarray(x, dtype=float) for x in (pump_speed_rpm, valve_opening_pct, heater_power_kw, coolant_flow_lpm,
                                             feed_temp_c, ambient_temp_c, fluid_viscosity_cp, fouling_index)]
    # Flow: proportional to pump speed, ~sqrt(valve opening), reduced by fouling and viscosity
    flow = 0.0105 * pump * np.sqrt(valve / 100.0) * (1 - 0.30 * foul) / visc ** 0.25          # m3/h
    # Discharge pressure: ~ speed^2 (pump affinity law), higher when valve is closed or fouled
    pressure = 1.0 + 1.1e-6 * pump ** 2 * (1.6 - 0.9 * valve / 100.0) * (1 + 0.5 * foul) * visc ** 0.1  # bar
    # Heater energy balance: Q = m*cp*dT  (1 m3/h of water ~ 1.161 kW per K)
    heat_kw = heater * 0.92 * (1 - 0.15 * foul)
    t_hot = feed_t + heat_kw / (1.161 * flow)
    cooling = 0.5 * (1 - np.exp(-coolant / 150.0)) * (t_hot - 25.0)      # cooler removes part of the excess heat
    loss = 0.01 * (t_hot - amb_t)                                       # small heat loss to ambient
    temp = t_hot - cooling - loss                                       # outlet temperature, C
    # Electrical energy: heater + pump (power ~ speed^3) + chiller + base load
    pump_kw = 2.2e-9 * pump ** 3 * (1 + 0.4 * foul) * visc ** 0.15
    chiller_kw = 0.06 * coolant * (1 + 0.01 * (amb_t - 20.0))
    energy = heater + pump_kw + chiller_kw + 2.0
    return {"temperature_c": temp, "pressure_bar": pressure, "flow_rate_m3h": flow, "energy_kw": energy}


def _ar1(rng, n, phi=0.98, sigma=1.0):
    e = rng.normal(0, sigma, n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def generate_dataset(n=30000, seed=42, missing_rate=0.01, outlier_rate=0.003):
    """Return (dataframe, injection_log). 5-minute sampling, ~104 days for n=30000."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2025-01-01", periods=n, freq="5min")
    t_h = np.arange(n) * 5 / 60.0

    def setpoint(lo, hi):  # operators change setpoints every 2-8 hours
        out, i = np.empty(n), 0
        while i < n:
            L = int(rng.integers(24, 96)); out[i:i + L] = rng.uniform(lo, hi); i += L
        return out

    pump, valve = setpoint(1100, 2800), setpoint(30, 98)
    heater, coolant = setpoint(100, 330), setpoint(20, 120)
    feed_t = 45 + 3 * np.sin(2 * np.pi * (t_h - 3) / 24) + _ar1(rng, n, 0.97, 0.15)
    amb_t = 14 + 7 * np.sin(2 * np.pi * (t_h - 9) / 24) + _ar1(rng, n, 0.97, 0.1)
    visc = (0.9 + 2.2 * np.exp(-(feed_t - 30) / 18.0)) * (1 + rng.normal(0, 0.02, n))
    foul = ((np.arange(n) % 6000) / 6000.0) * 0.6          # builds up then a cleaning resets it

    truth = process_model(pump, valve, heater, coolant, feed_t, amb_t, visc, foul)
    df = pd.DataFrame({"timestamp": ts, "pump_speed_rpm": pump + rng.normal(0, 5, n),
                       "valve_opening_pct": np.clip(valve + rng.normal(0, 0.3, n), 0, 100),
                       "heater_power_kw": heater + rng.normal(0, 1.5, n),
                       "coolant_flow_lpm": coolant + rng.normal(0, 0.8, n),
                       "feed_temp_c": feed_t + rng.normal(0, 0.15, n),
                       "ambient_temp_c": amb_t + rng.normal(0, 0.2, n),
                       "fluid_viscosity_cp": visc * (1 + rng.normal(0, 0.01, n)),
                       "fouling_index": np.clip(foul + rng.normal(0, 0.005, n), 0, 1)})
    df["temperature_c"] = truth["temperature_c"] + rng.normal(0, 0.4, n)
    df["pressure_bar"] = truth["pressure_bar"] * (1 + rng.normal(0, 0.01, n))
    df["flow_rate_m3h"] = truth["flow_rate_m3h"] * (1 + rng.normal(0, 0.012, n))
    df["energy_kw"] = truth["energy_kw"] * (1 + rng.normal(0, 0.006, n))

    # Realistic data problems: missing values and isolated spikes (logged so the cleaning step can be audited)
    log = {"missing_injected": 0, "outliers_injected": 0}
    for c in SENSOR_COLS:
        m = rng.random(n) < missing_rate
        o = (rng.random(n) < outlier_rate) & ~m
        df.loc[o, c] = df.loc[o, c] * (1 + rng.choice([-1, 1], o.sum()) * rng.uniform(0.4, 0.8, o.sum()))
        df.loc[m, c] = np.nan
        log["missing_injected"] += int(m.sum()); log["outliers_injected"] += int(o.sum())
    return df, log
