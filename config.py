"""Central configuration: paths, column names, random seed."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
RANDOM_STATE = 42

# Sensor inputs (what the plant historian records and an operator can read)
CONTROLLABLE = ["pump_speed_rpm", "valve_opening_pct", "heater_power_kw", "coolant_flow_lpm"]
DISTURBANCE = ["feed_temp_c", "ambient_temp_c", "fluid_viscosity_cp", "fouling_index"]
INPUTS = CONTROLLABLE + DISTURBANCE
TARGETS = ["temperature_c", "pressure_bar", "flow_rate_m3h", "energy_kw"]
SENSOR_COLS = [c for c in INPUTS if c != "fouling_index"] + TARGETS  # columns that get missing values/outliers

# Default process constraints used by the optimiser (editable in the dashboard)
DEFAULT_CONSTRAINTS = {"temp_min": 58.0, "temp_max": 62.0, "flow_min": 10.0, "pressure_max": 9.0}
