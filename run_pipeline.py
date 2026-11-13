"""End-to-end pipeline: generate -> clean -> EDA -> features -> split -> train -> evaluate -> importance -> optimise."""
import json, time
import numpy as np
import pandas as pd

from config import CONTROLLABLE, DATA_DIR, DEFAULT_CONSTRAINTS, DISTURBANCE, OUT_DIR, RANDOM_STATE, TARGETS
from src import visualization as viz
from src.data_generation import generate_dataset
from src.evaluate import evaluate, feature_importance
from src.features import add_features
from src.optimize import historical_baseline, optimize_operating_point, what_if
from src.preprocessing import clean
from src.train import LIN_RAW, chrono_split, save_bundle, train_all

t0 = time.time()
def step(msg): print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

DATA_DIR.mkdir(exist_ok=True); OUT_DIR.mkdir(exist_ok=True)
raw, inj = generate_dataset(n=30000, seed=RANDOM_STATE)
raw.to_csv(DATA_DIR / "raw_sensor_data.csv", index=False); step(f"generated {raw.shape}, injected {inj}")
df, rep = clean(raw); df.to_csv(DATA_DIR / "clean_sensor_data.csv", index=False); step(f"cleaned: {rep}")
viz.correlation_heatmap(df); viz.eda_overview(df)
df = add_features(df)
train_df, test_df = chrono_split(df, 0.8)
step(f"split: train={len(train_df)} ({train_df.timestamp.min()} .. {train_df.timestamp.max()}), "
     f"test={len(test_df)} ({test_df.timestamp.min()} .. {test_df.timestamp.max()})")
bundle = train_all(train_df); save_bundle(bundle); step(f"trained; primary model (by training CV) = {bundle['primary']}")
metrics = evaluate(bundle, test_df); metrics.to_csv(OUT_DIR / "metrics.csv", index=False)
print(metrics.round(4).to_string(index=False), flush=True)
viz.actual_vs_predicted(bundle, test_df); viz.residual_distribution(bundle, test_df); viz.model_comparison(metrics)
imp = feature_importance(bundle, test_df); imp.to_csv(OUT_DIR / "feature_importance.csv"); viz.importance_plot(imp, bundle["primary"])
print(imp.round(4).to_string(), flush=True); step("evaluation + importance done")

# ---- optimisation experiment: 8 operating contexts taken from the test period ----
rows, first = [], None
ctx_rows = test_df.iloc[np.linspace(0, len(test_df) - 1, 8).astype(int)]
for model_name in (bundle["primary"], LIN_RAW):
    for i, (_, r) in enumerate(ctx_rows.iterrows()):
        ctx = {k: float(r[k]) for k in DISTURBANCE}
        res = optimize_operating_point(bundle, ctx, model_name=model_name)
        base = historical_baseline(train_df, ctx)
        e_true = res["simulator"]["energy_kw"]
        rows.append({"model": model_name, "context": i, **{k: round(v, 1) for k, v in res["setpoints"].items()},
                     "pred_energy": res["predicted"]["energy_kw"], "sim_energy": e_true,
                     "sim_temp": res["simulator"]["temperature_c"], "sim_flow": res["simulator"]["flow_rate_m3h"],
                     "sim_pressure": res["simulator"]["pressure_bar"], "feasible_by_model": res["feasible_by_model"],
                     "feasible_by_simulator": res["feasible_by_simulator"], "historical_best_energy": base,
                     "saving_pct_vs_hist": None if base is None else 100 * (base - e_true) / base})
        if first is None and model_name == bundle["primary"]:
            first = (ctx, res)
opt = pd.DataFrame(rows); opt.to_csv(OUT_DIR / "optimization_results.csv", index=False)
print(opt.round(2).to_string(index=False), flush=True); step("optimisation done")

ctx, res = first; sp = res["setpoints"]
sweeps = []
x, p, s = what_if(bundle, sp, ctx, "heater_power_kw", np.linspace(*bundle["train_ranges"]["heater_power_kw"], 40))
sweeps.append(("heater_power_kw", x, p, s, ["temperature_c"]))
x, p, s = what_if(bundle, sp, ctx, "pump_speed_rpm", np.linspace(*bundle["train_ranges"]["pump_speed_rpm"], 40))
sweeps.append(("pump_speed_rpm", x, p, s, ["flow_rate_m3h", "pressure_bar"]))
viz.whatif_plot(sweeps)

summary = {"injection": inj, "cleaning": rep, "n_train": len(train_df), "n_test": len(test_df), "primary_model": bundle["primary"],
           "cv_rmse": bundle["cv_rmse"], "best_params": {t: {k: {a: (float(b) if hasattr(b, "item") else b) for a, b in v.items()}
                                                           for k, v in d.items()} for t, d in bundle["best_params"].items()},
           "optimisation": {m: {"n": int(len(g)), "feasible_by_model": int(g.feasible_by_model.sum()),
                                "feasible_by_simulator": int(g.feasible_by_simulator.sum()),
                                "mean_saving_pct_vs_hist": float(g.saving_pct_vs_hist.mean())} for m, g in opt.groupby("model")}}
json.dump(summary, open(OUT_DIR / "results.json", "w"), indent=2, default=str)
step("DONE")
