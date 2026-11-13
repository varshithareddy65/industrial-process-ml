"""Model-based operating-point search.

PREDICTION answers "what will the process do if I set X?".
OPTIMISATION answers "which X gives the best objective while satisfying constraints?" and uses the prediction model
as a surrogate inside a search loop. The result is a CANDIDATE operating point that is only as trustworthy as the surrogate:
it is not a proven global optimum and must be validated (here against the simulator, in reality by plant trials / engineers).
"""
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from config import CONTROLLABLE, DEFAULT_CONSTRAINTS, DISTURBANCE
from src.data_generation import process_model
from src.predict import predict_frame

MARGINS = {"flow": 0.3, "temp": 1.0, "pressure": 0.2}   # temp margin ~ 1x the temperature RMSE (see scripts/margin_study.py)


def bounds_from_training(bundle):
    return {c: bundle["train_ranges"][c] for c in CONTROLLABLE}   # stay inside the data range: no extrapolation


def check_constraints(out, c, tol=1e-6):
    return {"flow": out["flow_rate_m3h"] >= c["flow_min"] - tol,
            "temp": c["temp_min"] - tol <= out["temperature_c"] <= c["temp_max"] + tol,
            "pressure": out["pressure_bar"] <= c["pressure_max"] + tol}


def optimize_operating_point(bundle, context, constraints=None, bounds=None, model_name=None, use_margins=True, margins=None,
                             seed=0, popsize=20, maxiter=50):
    """Minimise predicted energy_kw over CONTROLLABLE variables for fixed disturbance `context`."""
    c = {**DEFAULT_CONSTRAINTS, **(constraints or {})}
    bounds = bounds or bounds_from_training(bundle)
    m = (margins or MARGINS) if use_margins else {"flow": 0, "temp": 0, "pressure": 0}
    W = 500.0   # penalty per unit of constraint violation

    def fun(X):
        df = pd.DataFrame(X.T, columns=CONTROLLABLE)
        for k in DISTURBANCE:
            df[k] = context[k]
        p = predict_frame(bundle, df, model_name)
        viol = (np.maximum(0, c["flow_min"] + m["flow"] - p["flow_rate_m3h"])
                + np.maximum(0, c["temp_min"] + m["temp"] - p["temperature_c"])
                + np.maximum(0, p["temperature_c"] - (c["temp_max"] - m["temp"]))
                + np.maximum(0, p["pressure_bar"] - (c["pressure_max"] - m["pressure"])))
        return (p["energy_kw"] + W * viol).values

    res = differential_evolution(fun, [bounds[k] for k in CONTROLLABLE], vectorized=True, updating="deferred",
                                 popsize=popsize, maxiter=maxiter, seed=seed, tol=1e-6, polish=False)
    x = dict(zip(CONTROLLABLE, map(float, res.x)))
    pred = predict_frame(bundle, pd.DataFrame([{**x, **{k: context[k] for k in DISTURBANCE}}]), model_name).iloc[0].to_dict()
    truth = {k: float(v) for k, v in process_model(**x, **{k: context[k] for k in DISTURBANCE}).items()}
    return {"setpoints": x, "predicted": pred, "simulator": truth,
            "feasible_by_model": all(check_constraints(pred, c).values()),
            "feasible_by_simulator": all(check_constraints(truth, c).values())}


def historical_baseline(hist_df, context, constraints=None, k=500):
    """Best measured energy among the k historical rows with the most similar disturbances that met the constraints."""
    c = {**DEFAULT_CONSTRAINTS, **(constraints or {})}
    z = (hist_df[DISTURBANCE] - hist_df[DISTURBANCE].mean()) / hist_df[DISTURBANCE].std()
    zc = (pd.Series(context)[DISTURBANCE] - hist_df[DISTURBANCE].mean()) / hist_df[DISTURBANCE].std()
    near = hist_df.loc[((z - zc) ** 2).sum(axis=1).nsmallest(k).index]
    ok = near[(near.temperature_c.between(c["temp_min"], c["temp_max"])) & (near.flow_rate_m3h >= c["flow_min"])
              & (near.pressure_bar <= c["pressure_max"])]
    return None if ok.empty else float(ok.energy_kw.min())


def what_if(bundle, base_setpoints, context, variable, values, model_name=None):
    """Sweep one controllable variable, holding the rest fixed. Returns model predictions and simulator values."""
    df = pd.DataFrame([{**base_setpoints, **{k: context[k] for k in DISTURBANCE}} for _ in values])
    df[variable] = list(values)
    pred = predict_frame(bundle, df, model_name)
    sim = pd.DataFrame(process_model(**{c: df[c].values for c in CONTROLLABLE + DISTURBANCE}))
    return df[variable].values, pred, sim
