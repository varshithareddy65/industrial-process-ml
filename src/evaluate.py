"""Evaluation (MAE, MSE, RMSE, R2) and permutation feature importance."""
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import RANDOM_STATE, TARGETS
from src.train import MODEL_NAMES


def predict_model(bundle, df, target, name):
    return bundle["models"][target][name].predict(df[bundle["feature_sets"][name]])


def evaluate(bundle, test_df):
    rows = []
    for t in TARGETS:
        for n in MODEL_NAMES:
            p = predict_model(bundle, test_df, t, n)
            mse = mean_squared_error(test_df[t], p)
            rows.append({"target": t, "model": n, "MAE": mean_absolute_error(test_df[t], p), "MSE": mse,
                         "RMSE": float(np.sqrt(mse)), "R2": r2_score(test_df[t], p)})
    return pd.DataFrame(rows)


def feature_importance(bundle, test_df, name=None, n=4000, repeats=3):
    """Permutation importance on held-out data: drop in R2 when one feature is shuffled."""
    name = name or bundle["primary"]
    s = test_df.sample(min(n, len(test_df)), random_state=RANDOM_STATE)
    feats = bundle["feature_sets"][name]
    res = {}
    for t in TARGETS:
        r = permutation_importance(bundle["models"][t][name], s[feats], s[t], n_repeats=repeats,
                                   random_state=RANDOM_STATE, scoring="r2")
        res[t] = pd.Series(r.importances_mean, index=feats)
    return pd.DataFrame(res)
