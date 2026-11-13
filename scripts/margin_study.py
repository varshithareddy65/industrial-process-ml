"""How big must the safety margin be? Re-uses the saved models; 30 test-period contexts; validated against the simulator."""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from config import DATA_DIR, DISTURBANCE, OUT_DIR
from src.features import add_features
from src.optimize import historical_baseline, optimize_operating_point
from src.predict import load_bundle
from src.train import chrono_split

bundle = load_bundle()
df = add_features(pd.read_csv(DATA_DIR / "clean_sensor_data.csv", parse_dates=["timestamp"]))
train_df, test_df = chrono_split(df)
ctxs = test_df.iloc[np.linspace(0, len(test_df) - 1, 30).astype(int)]
rows = []
for tm in (0.0, 0.5, 1.0, 1.5):
    margins = {"flow": 0.3, "temp": tm, "pressure": 0.2}
    for _, r in ctxs.iterrows():
        ctx = {k: float(r[k]) for k in DISTURBANCE}
        res = optimize_operating_point(bundle, ctx, margins=margins)
        base = historical_baseline(train_df, ctx)
        rows.append({"temp_margin_C": tm, "feasible_sim": res["feasible_by_simulator"], "sim_energy": res["simulator"]["energy_kw"],
                     "worst_temp_violation_C": max(0, 58 - res["simulator"]["temperature_c"], res["simulator"]["temperature_c"] - 62),
                     "saving_pct": None if base is None else 100 * (base - res["simulator"]["energy_kw"]) / base})
out = pd.DataFrame(rows).groupby("temp_margin_C").agg(n=("feasible_sim", "size"), feasible=("feasible_sim", "sum"),
      mean_energy_kw=("sim_energy", "mean"), max_violation_C=("worst_temp_violation_C", "max"), mean_saving_pct=("saving_pct", "mean")).round(3)
out.to_csv(OUT_DIR / "margin_study.csv"); print(out.to_string())
