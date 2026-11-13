import numpy as np
import pandas as pd
import pytest

from config import CONTROLLABLE, DISTURBANCE, INPUTS, TARGETS
from src.data_generation import generate_dataset, process_model
from src.features import FEATURES, add_features
from src.optimize import optimize_operating_point
from src.predict import predict_frame, predict_one
from src.preprocessing import clean, detect_spikes
from src.train import chrono_split, train_all

CTX = {"feed_temp_c": 45.0, "ambient_temp_c": 14.0, "fluid_viscosity_cp": 1.9, "fouling_index": 0.3}


@pytest.fixture(scope="module")
def small():
    raw, _ = generate_dataset(n=4000, seed=1)
    df, _ = clean(raw)
    df = add_features(df)
    tr, te = chrono_split(df)
    return df, tr, te, train_all(tr, rf_iter=1, xgb_iter=1, verbose=False)


def test_physics_monotonic():
    base = dict(pump_speed_rpm=2000, valve_opening_pct=70, heater_power_kw=200, coolant_flow_lpm=60, **CTX)
    hi = process_model(**{**base, "heater_power_kw": 300})
    lo = process_model(**base)
    assert hi["temperature_c"] > lo["temperature_c"] and hi["energy_kw"] > lo["energy_kw"]
    faster = process_model(**{**base, "pump_speed_rpm": 2400})
    assert faster["flow_rate_m3h"] > lo["flow_rate_m3h"] and faster["pressure_bar"] > lo["pressure_bar"]


def test_clean_removes_nans_and_spikes():
    raw, log = generate_dataset(n=3000, seed=2)
    df, rep = clean(raw)
    assert not df[INPUTS + TARGETS].isna().any().any()
    assert rep["spikes_removed"] > 0 and len(df) < len(raw)


def test_spike_detector_ignores_steps():
    s = pd.Series([10.0] * 10 + [50.0] * 10)
    assert not detect_spikes(s).any()
    s2 = s.copy(); s2.iloc[4] = 40.0
    assert detect_spikes(s2).iloc[4]


def test_split_is_chronological(small):
    _, tr, te, _ = small
    assert tr.timestamp.max() < te.timestamp.min()


def test_prediction_api(small):
    _, _, te, bundle = small
    out = predict_one(bundle, te.iloc[0][INPUTS].to_dict())
    assert set(out) == set(TARGETS) and all(np.isfinite(list(out.values())))
    with pytest.raises(ValueError):
        predict_one(bundle, {"pump_speed_rpm": 1000})


def test_optimizer_respects_bounds(small):
    _, _, _, bundle = small
    res = optimize_operating_point(bundle, CTX, popsize=8, maxiter=10)
    for k in CONTROLLABLE:
        lo, hi = bundle["train_ranges"][k]
        assert lo - 1e-6 <= res["setpoints"][k] <= hi + 1e-6
