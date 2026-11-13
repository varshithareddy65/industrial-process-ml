"""Model training: Linear Regression baselines, Random Forest and XGBoost (tuned with time-ordered CV)."""
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from config import INPUTS, MODEL_DIR, RANDOM_STATE, TARGETS
from src.features import FEATURES

LIN_RAW, LIN_ENG, RF, XGB = "Linear (raw features)", "Linear (engineered)", "Random Forest", "XGBoost"
MODEL_NAMES = [LIN_RAW, LIN_ENG, RF, XGB]
FEATURE_SETS = {LIN_RAW: INPUTS, LIN_ENG: FEATURES, RF: FEATURES, XGB: FEATURES}


def chrono_split(df, train_frac=0.8):
    """Time-ordered split. Random splitting of time series leaks the future into training."""
    cut = int(len(df) * train_frac)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def _search(est, params, n_iter, cv):
    return RandomizedSearchCV(est, params, n_iter=n_iter, cv=cv, scoring="neg_root_mean_squared_error",
                              random_state=RANDOM_STATE, n_jobs=1, refit=True)


def train_all(train_df, rf_iter=4, xgb_iter=6, verbose=True):
    cv = TimeSeriesSplit(n_splits=3)
    models, cv_rmse, best_params = {}, {}, {}
    for t in TARGETS:
        y = train_df[t].values
        models[t], cv_rmse[t], best_params[t] = {}, {}, {}
        for name in (LIN_RAW, LIN_ENG):
            est = make_pipeline(StandardScaler(), LinearRegression())
            Xf = train_df[FEATURE_SETS[name]]
            cv_rmse[t][name] = float(-cross_val_score(est, Xf, y, cv=cv, scoring="neg_root_mean_squared_error").mean())
            models[t][name] = est.fit(Xf, y)
        X = train_df[FEATURES]
        rf = _search(RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1, max_samples=0.6),
                     {"n_estimators": [100, 150], "max_depth": [10, 16], "min_samples_leaf": [3, 8],
                      "max_features": [0.6, 1.0]}, rf_iter, cv).fit(X, y)
        xg = _search(XGBRegressor(random_state=RANDOM_STATE, tree_method="hist", n_jobs=1),
                     {"n_estimators": [200, 400], "max_depth": [4, 6, 8], "learning_rate": [0.05, 0.1],
                      "subsample": [0.8, 1.0], "colsample_bytree": [0.7, 1.0]}, xgb_iter, cv).fit(X, y)
        for name, s in ((RF, rf), (XGB, xg)):
            models[t][name] = s.best_estimator_
            cv_rmse[t][name] = float(-s.best_score_)
            best_params[t][name] = s.best_params_
        if verbose:
            print(f"[train] {t}: " + ", ".join(f"{n}={cv_rmse[t][n]:.3f}" for n in MODEL_NAMES), flush=True)
    # Primary model chosen on TRAINING cross-validation only (never on the test set)
    score = {n: np.mean([cv_rmse[t][n] / train_df[t].std() for t in TARGETS]) for n in (RF, XGB)}
    primary = min(score, key=score.get)
    bundle = {"models": models, "feature_sets": FEATURE_SETS, "primary": primary, "cv_rmse": cv_rmse,
              "best_params": best_params, "train_ranges": {c: (float(train_df[c].quantile(.01)), float(train_df[c].quantile(.99)))
                                                            for c in INPUTS}}
    return bundle


def save_bundle(bundle, path=None):
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(bundle, path or MODEL_DIR / "models.joblib")
