"""Feature engineering using simple process knowledge (physically motivated interactions)."""
import numpy as np

from config import INPUTS

ENGINEERED = ["pump_x_sqrt_valve", "pump_speed_sq", "pump_speed_cu", "heater_per_throughput",
              "feed_ambient_delta", "hour_sin", "hour_cos"]
FEATURES = INPUTS + ENGINEERED


def add_features(df):
    out = df.copy()
    hyd = out["pump_speed_rpm"] * np.sqrt(out["valve_opening_pct"] / 100.0)          # throughput proxy
    out["pump_x_sqrt_valve"] = hyd
    out["pump_speed_sq"] = out["pump_speed_rpm"] ** 2 / 1e6                           # pressure ~ speed^2
    out["pump_speed_cu"] = out["pump_speed_rpm"] ** 3 / 1e9                           # pump power ~ speed^3
    out["heater_per_throughput"] = out["heater_power_kw"] / (hyd + 1.0)               # heat added per unit flow -> temperature rise
    out["feed_ambient_delta"] = out["feed_temp_c"] - out["ambient_temp_c"]
    hour = out["hour_of_day"] if "hour_of_day" in out else 12.0
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    return out
