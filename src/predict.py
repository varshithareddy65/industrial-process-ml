"""Inference: predict the four process parameters from new sensor inputs."""
import joblib
import numpy as np
import pandas as pd

from config import INPUTS, MODEL_DIR, TARGETS
from src.features import add_features


def load_bundle(path=None):
    return joblib.load(path or MODEL_DIR / "models.joblib")


def predict_frame(bundle, inputs_df, model_name=None):
    """inputs_df must contain the 8 INPUT columns (and optionally hour_of_day). Returns DataFrame of TARGETS."""
    model_name = model_name or bundle["primary"]
    df = inputs_df.copy()
    if "hour_of_day" not in df:
        df["hour_of_day"] = 12.0
    df = add_features(df)
    out = {t: bundle["models"][t][model_name].predict(df[bundle["feature_sets"][model_name]]) for t in TARGETS}
    return pd.DataFrame(out, index=inputs_df.index)


def predict_one(bundle, inputs: dict, model_name=None):
    missing = [c for c in INPUTS if c not in inputs]
    if missing:
        raise ValueError(f"Missing inputs: {missing}")
    row = pd.DataFrame([{**inputs}])
    return {k: float(v) for k, v in predict_frame(bundle, row, model_name).iloc[0].items()}
