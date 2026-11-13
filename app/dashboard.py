"""Streamlit dashboard. Run from the project root:  streamlit run app/dashboard.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from config import CONTROLLABLE, DEFAULT_CONSTRAINTS, DISTURBANCE, FIG_DIR, INPUTS, OUT_DIR
from src.data_generation import process_model
from src.optimize import check_constraints, optimize_operating_point
from src.predict import load_bundle, predict_one
from src.train import MODEL_NAMES

st.set_page_config(page_title="Industrial Process Prediction", layout="wide")
st.title("Industrial Process Parameter Prediction and Optimization")
st.warning("All data in this project is SYNTHETIC (simulated heated process loop), not real plant data.")


@st.cache_resource
def get_bundle():
    return load_bundle()


try:
    bundle = get_bundle()
except FileNotFoundError:
    st.error("No trained models found. Run `python run_pipeline.py` first."); st.stop()

UNITS = {"pump_speed_rpm": "rpm", "valve_opening_pct": "%", "heater_power_kw": "kW", "coolant_flow_lpm": "L/min",
         "feed_temp_c": "C", "ambient_temp_c": "C", "fluid_viscosity_cp": "cP", "fouling_index": "0-1"}
tab_pred, tab_opt, tab_perf = st.tabs(["Predict", "Optimize", "Model performance"])


def slider(col, key, default=None):
    lo, hi = bundle["train_ranges"][key]
    return col.slider(f"{key} ({UNITS[key]})", float(lo), float(hi), float(default if default is not None else (lo + hi) / 2))


with tab_pred:
    st.subheader("Predict outputs from sensor inputs")
    model = st.selectbox("Model", MODEL_NAMES, index=MODEL_NAMES.index(bundle["primary"]), key="pm")
    c1, c2 = st.columns(2)
    inputs = {k: slider(c1 if k in CONTROLLABLE else c2, k) for k in INPUTS}
    pred = predict_one(bundle, inputs, model)
    truth = {k: float(v) for k, v in process_model(**inputs).items()}
    m = st.columns(4)
    for col, (k, v) in zip(m, pred.items()):
        col.metric(k, f"{v:.2f}", f"simulator: {truth[k]:.2f}", delta_color="off")
    st.caption("'simulator' is the noise-free synthetic ground truth, available only because the data is simulated.")

with tab_opt:
    st.subheader("Find a candidate low-energy operating point")
    st.info("Prediction asks 'what will happen?'. Optimization searches for settings that minimise energy subject to "
            "constraints, using the model as a surrogate. The result is a CANDIDATE to validate, not a proven global optimum.")
    a, b = st.columns(2)
    ctx = {k: slider(a, k, {"feed_temp_c": 45, "ambient_temp_c": 14, "fluid_viscosity_cp": 1.9, "fouling_index": 0.3}[k])
           for k in DISTURBANCE}
    cons = {"temp_min": b.number_input("Min outlet temperature (C)", value=DEFAULT_CONSTRAINTS["temp_min"]),
            "temp_max": b.number_input("Max outlet temperature (C)", value=DEFAULT_CONSTRAINTS["temp_max"]),
            "flow_min": b.number_input("Min flow rate (m3/h)", value=DEFAULT_CONSTRAINTS["flow_min"]),
            "pressure_max": b.number_input("Max pressure (bar)", value=DEFAULT_CONSTRAINTS["pressure_max"])}
    om = b.selectbox("Surrogate model", MODEL_NAMES, index=MODEL_NAMES.index(bundle["primary"]), key="om")
    if st.button("Run optimization"):
        with st.spinner("Searching..."):
            res = optimize_operating_point(bundle, ctx, cons, model_name=om)
        st.dataframe(pd.DataFrame({"Recommended setpoint": res["setpoints"]}).round(2))
        st.dataframe(pd.DataFrame({"Model prediction": res["predicted"], "Simulator check": res["simulator"]}).round(2))
        ok = check_constraints(res["simulator"], cons)
        (st.success if all(ok.values()) else st.error)(f"Constraints satisfied in simulator: {ok}")

with tab_perf:
    st.subheader("Held-out test performance")
    f = OUT_DIR / "metrics.csv"
    if f.exists():
        st.dataframe(pd.read_csv(f).round(4))
    for img in sorted(FIG_DIR.glob("*.png")):
        st.image(str(img), caption=img.stem)
