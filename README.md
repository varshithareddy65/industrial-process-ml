# Industrial Process Parameter Prediction and Optimization Using Machine Learning

Predicts **temperature, pressure, flow rate and energy consumption** of a heated process loop (pump -> heater -> cooler)
from sensor data, compares Linear Regression / Random Forest / XGBoost, explains the drivers, and searches for
low-energy operating points under constraints. Includes a Streamlit dashboard.

> **The data is SYNTHETIC.** No public industrial dataset could be downloaded in the build environment, so the data is
> generated from documented physical relationships (`src/data_generation.py`) with noise, missing values and outliers.
> It must never be presented as real plant data.

## Quick start
```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python run_pipeline.py            # generate data -> clean -> train -> evaluate -> optimise (about 5 min on 1 CPU core)
python -m pytest -q tests         # 6 unit tests
streamlit run app/dashboard.py    # dashboard at http://localhost:8501
python scripts/margin_study.py    # optional: safety-margin study for the optimiser
```

## Structure
| Path | Purpose |
|---|---|
| `config.py` | paths, column names, default constraints |
| `src/data_generation.py` | synthetic data + noise-free `process_model` (stand-in for a first-principles twin) |
| `src/preprocessing.py` | range checks, spike detection, interpolation, dropping unusable label rows |
| `src/features.py` | physically motivated engineered features |
| `src/train.py` | chronological split, Linear / RF / XGBoost, time-ordered CV tuning |
| `src/evaluate.py` | MAE, MSE, RMSE, R2, permutation importance |
| `src/predict.py` | inference from new sensor inputs |
| `src/optimize.py` | surrogate-based operating-point search, what-if sweeps |
| `src/visualization.py` | all plots (`outputs/figures/`) |
| `app/dashboard.py` | Streamlit dashboard (Predict / Optimize / Model performance) |
| `docs/DOCUMENTATION.md` | problem, data, maths, results, limitations |
| `docs/INTERVIEW_PREP.md` | 20 Q&A and pipeline walkthrough |

Results: `outputs/metrics.csv`, `outputs/optimization_results.csv`, `outputs/margin_study.csv`, `outputs/results.json`.
