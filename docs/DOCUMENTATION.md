# Documentation

## 1. Problem statement
Given historical sensor readings from a process loop, (a) predict four process parameters (outlet temperature, discharge
pressure, flow rate, electrical energy) from the measured inputs, (b) identify which inputs drive each output, and
(c) use the model to propose operating conditions with lower energy use that still satisfy process constraints.

## 2. Motivation and industrial relevance
Plants log large amounts of sensor data but operators rarely know how a setpoint change will propagate. A data-driven
model gives fast "what-if" answers (soft sensing, scenario analysis). This is the core of a **data-driven digital twin**:
a model kept consistent with the plant that can be queried before acting on the real asset.

## 3. Dataset (synthetic - state this openly)
30,000 rows at 5-minute sampling (2025-01-01 to 2025-04-15). No suitable public dataset could be downloaded from the
build environment, so data were simulated from simplified physics (`process_model`):
- flow ~ pump speed x sqrt(valve opening), reduced by fouling and viscosity
- pressure ~ pump speed^2 (affinity law), higher with closed valve and fouling
- temperature from an energy balance Q = m*cp*dT, minus cooler and ambient loss
- energy = heater + pump power (~speed^3) + chiller + base load

| Group | Columns |
|---|---|
| Controllable inputs | `pump_speed_rpm`, `valve_opening_pct`, `heater_power_kw`, `coolant_flow_lpm` |
| Disturbances | `feed_temp_c`, `ambient_temp_c`, `fluid_viscosity_cp`, `fouling_index` |
| Targets | `temperature_c`, `pressure_bar`, `flow_rate_m3h`, `energy_kw` |

Realism added: operator setpoint changes every 2-8 h, daily ambient/feed cycles, fouling that builds up and resets,
sensor noise on inputs and outputs, 3,317 injected missing values and 974 injected spikes.
**Targets are predicted only from the 8 inputs, never from each other** (no leakage).

## 4. Pipeline
1. **Cleaning**: physical range checks (24 out-of-range values), isolated-spike detector (1,006 points flagged; flags
   points far from both neighbours when the neighbours agree, so genuine setpoint steps are not removed), interpolation of
   input gaps up to 15 min, rows with missing/invalid *targets* dropped (labels are not invented). 30,000 -> 28,443 rows.
2. **EDA**: distributions, time series, correlation heatmap (`outputs/figures/01-03`).
3. **Features**: `pump*sqrt(valve)` (throughput proxy), `pump^2`, `pump^3`, `heater/(throughput)`, `feed-ambient`, hour sin/cos.
4. **Split**: chronological 80/20 - train 22,754 rows (to 2025-03-25), test 5,689 rows (after). 
5. **Models** (one per target): Linear (raw 8 inputs), Linear (engineered), Random Forest, XGBoost. RF and XGBoost tuned with
   `RandomizedSearchCV` + `TimeSeriesSplit(3)`. Primary model = best mean normalised CV RMSE on *training* folds (XGBoost).
6. **Evaluation** on the untouched test period: MAE, MSE, RMSE, R2.

## 5. Mathematical intuition
- **Linear regression**: y = Xb + e, b minimises sum of squared errors. Cannot represent curvature or interactions unless the features contain them (hence the gap between "raw" and "engineered").
- **Random Forest**: average of many decision trees, each trained on a bootstrap sample with random feature subsets. Averaging reduces variance. Predictions are piecewise-constant.
- **XGBoost**: gradient boosting. Trees are added sequentially, each fitting the gradient of the loss (residuals for squared error): F_m = F_(m-1) + lr * h_m. Regularisation (depth, learning rate, subsampling, L1/L2) controls overfitting.
- **Why these two**: tabular sensor data with non-linearities and interactions; no scaling needed; robust to outliers; give feature importance.
- **Metrics**: MAE = mean|e| (same units, robust); MSE = mean e^2; RMSE = sqrt(MSE) (penalises large errors); R2 = 1 - SSE/SST.

## 6. Results (actual, test period, 5,689 rows)
| Target | Model | MAE | RMSE | R2 |
|---|---|---|---|---|
| Temperature (C) | Linear (raw) | 1.389 | 1.879 | 0.8945 |
| | Linear (engineered) | 0.500 | 0.873 | 0.9772 |
| | Random Forest | 0.673 | 1.096 | 0.9641 |
| | **XGBoost** | 0.492 | 0.882 | 0.9768 |
| Pressure (bar) | Linear (raw) | 0.467 | 0.588 | 0.9553 |
| | Linear (engineered) | 0.221 | 0.282 | 0.9898 |
| | Random Forest | 0.226 | 0.303 | 0.9882 |
| | **XGBoost** | 0.158 | 0.227 | 0.9934 |
| Flow (m3/h) | Linear (raw) | 0.503 | 0.679 | 0.9721 |
| | Linear (engineered) | 0.222 | 0.328 | 0.9935 |
| | Random Forest | 0.268 | 0.383 | 0.9911 |
| | **XGBoost** | 0.186 | 0.293 | 0.9948 |
| Energy (kW) | Linear (raw) | 3.419 | 4.431 | 0.9960 |
| | **Linear (engineered)** | 1.896 | 2.786 | 0.9984 |
| | Random Forest | 3.432 | 4.568 | 0.9958 |
| | XGBoost | 2.795 | 3.882 | 0.9970 |

(MSE values are in `outputs/metrics.csv`.)

**Honest reading of these numbers**
- XGBoost is best for pressure and flow; for temperature it ties with engineered linear regression, and for energy the engineered linear model is better. Energy is almost linear in heater power, and my engineered features encode the physics, so a linear model does very well.
- Feature engineering mattered more than the algorithm (raw linear -> engineered linear cut temperature RMSE by 54%).
- R2 values are high because the synthetic process is smooth and quasi-steady-state. **Do not expect this on real plant data.**
- Single run, single seed, single chronological split: differences of a few percent are not statistically established.

### Feature importance (permutation, XGBoost, test data; drop in R2)
Temperature: heater_per_throughput 1.62, coolant_flow 0.24, feed_temp 0.14. Pressure: pump_speed 1.79, valve_opening 0.30,
fouling 0.046. Flow: pump_x_sqrt_valve 1.63, fouling 0.069. Energy: heater_power 1.86, pump_speed 0.10.
Values can exceed 1 and do not sum to 1: when one of several correlated/derived columns is shuffled the model sees an
impossible combination. `hour_cos` shows 0.035 for temperature: a mild spurious reliance (hour correlates with the daily feed-temperature cycle).

## 7. Optimization
**Prediction vs optimization**: prediction answers "what happens if I set X?". Optimization searches over X for the best
objective under constraints, using the model as a surrogate. The output is a *candidate* operating point, not a proven optimum.

Setup: minimise predicted `energy_kw` over the 4 controllable variables (bounded to the 1st-99th percentile of training data,
so no extrapolation) for fixed disturbances; constraints: temperature 58-62 C, flow >= 10 m3/h, pressure <= 9 bar; solver:
differential evolution with a penalty for violations; safety margins (flow +0.3, temperature +/-1.0 C, pressure -0.2).

Validation against the noise-free simulator (possible *only because the data is synthetic*; in reality use plant trials / engineering review):

| Surrogate | Contexts | Feasible per simulator | Mean saving vs best comparable historical point |
|---|---|---|---|
| XGBoost | 8 | 8 / 8 | 23.6% |
| Linear (raw) | 8 | 0 / 8 (flow 8.9-9.4 < 10) | 29.3% (invalid: constraints violated) |

Safety-margin study (30 contexts, XGBoost):
| Temp margin (C) | Feasible | Worst temp violation (C) | Mean energy (kW) |
|---|---|---|---|
| 0.0 | 11/30 | 1.19 | 227.8 |
| 0.5 | 19/30 | 0.37 | 233.8 |
| 1.0 | 30/30 | 0 | 242.8 |
| 1.5 | 30/30 | 0 | 249.3 |

Lessons: (1) an optimiser exploits surrogate errors and pushes to constraint edges, so margins comparable to the model RMSE (about 0.9 C) are needed; (2) robustness costs about 4% more energy; (3) a poor surrogate (raw linear) gives attractive but infeasible answers.
Caveats: the 1.0 C default was chosen after seeing this study on the same contexts, so the 30/30 figure is mildly optimistic. The "historical" baseline is weak because the synthetic operators change setpoints randomly, so the 23.6% saving should not be read as a realistic industrial saving.

## 8. Limitations and future work
- Synthetic, quasi-steady-state data: no process dynamics, so lag features/time-series models were not needed (real data would need them).
- Errors-in-variables: inputs are noisy, so there is irreducible error.
- Single seed/split; no confidence intervals. Future: repeated blocked CV, bootstrap intervals.
- Optimisation is single-objective and local to the data range; add multi-objective trade-offs, quantile/uncertainty-aware constraints, and process-engineer-defined limits.
- Real data: sensor drift, unit changes, control loops (feedback makes inputs depend on outputs -> causal confusion), covariate shift.
- Future: hybrid physics-informed models, SHAP, MLflow tracking, online retraining, Docker.

## 9. Verification performed
`pytest`: 6 tests pass. Pipeline ran end to end (about 250 s). Dashboard loaded and ran an optimisation with no exceptions using Streamlit's headless `AppTest` (no browser screenshots were taken; all figures are in `outputs/figures/`).
