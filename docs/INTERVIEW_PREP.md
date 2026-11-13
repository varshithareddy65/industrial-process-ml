# Interview preparation (Siemens industrial digital twin / process optimization)

**Say this first:** "The data is synthetic, generated from documented physical relationships, because I could not obtain a public
industrial dataset. I used it to build and validate the full pipeline, and I know real data would be harder."

## A. Pipeline in 60 seconds
Generate/clean data -> EDA -> engineer physics-based features -> chronological split -> train Linear, Random Forest, XGBoost
(time-ordered CV tuning) -> evaluate on the later test period (MAE/MSE/RMSE/R2) -> permutation importance -> use the model as a
surrogate to search for low-energy setpoints under constraints -> validate candidates against the simulator -> dashboard.

## B. The algorithms (what to be able to draw on a whiteboard)
- **Linear regression**: y = Xb; least squares. Needs features that already contain the non-linearity. 
- **Random Forest**: bagging of deep trees + random feature subsets; averaging cuts variance; piecewise-constant output; cannot extrapolate.
- **XGBoost**: boosting; each new tree fits the gradient of the loss of the current ensemble; learning rate shrinks each step; regularisation limits complexity.

## C. 20 questions
1. **Why regression?** The targets are continuous quantities, so we predict a number (not a class). Regression gives error in physical units (C, bar, m3/h, kW).
2. **Why chronological split?** Sensor data is a time series; random splitting puts neighbouring (near-identical) rows in train and test, leaking the future and inflating scores. Test = later period, like deployment.
3. **Why Random Forest and XGBoost?** Tabular, non-linear, interacting variables; no scaling needed; robust to outliers; give importances. They are strong, standard baselines. I also kept linear regression as a baseline.
4. **Which model won?** XGBoost for pressure (RMSE 0.227) and flow (0.293); engineered linear regression tied for temperature (0.873 vs 0.882) and beat all for energy (2.79 vs 3.88). So the simplest model with the right features was competitive. I would not claim XGBoost is always best.
5. **Why did engineered linear regression do so well?** My features encode known physics (pump^2, throughput, heat per unit flow). Energy is nearly linear in heater power. Feature engineering cut temperature RMSE from 1.88 to 0.87.
6. **Explain overfitting/underfitting.** Overfitting: low training error, high test error (model memorises noise); underfitting: high error on both (raw linear model misses curvature). Controls: chronological hold-out, time-ordered CV, tree depth, min leaf size, learning rate, subsampling.
7. **How did you detect overfitting?** Compared CV and test errors and used a hold-out period never used for tuning. Test RMSEs are close to CV RMSEs (e.g. XGBoost temperature CV 0.93 vs test 0.88). I did not plot learning curves, which would be a further check.
8. **How was the primary model chosen?** By mean normalised CV RMSE on training folds only, never on the test set, to avoid selecting on test data.
9. **What does R2 mean and what are its limits?** Fraction of variance explained, 1 - SSE/SST. High R2 can hide errors that matter operationally (temperature R2 0.977 still means about 0.9 C RMSE against a 4 C band). Always report RMSE/MAE in units.
10. **MAE vs RMSE?** MAE = average absolute error, robust. RMSE penalises large errors more; better when big misses are costly. Both reported.
11. **How did you handle missing values and outliers?** Input gaps up to 15 min interpolated; isolated spikes detected by comparing to both neighbours (so real setpoint steps are kept); rows with bad *targets* dropped because inventing labels would contaminate evaluation. 1,006 points flagged vs 974 injected; I did not measure per-point precision/recall.
12. **What is feature importance and what are the pitfalls?** I used permutation importance on test data: shuffle one feature, measure the R2 drop. Pitfalls: correlated/derived features share or distort importance (values above 1 appear), and importance is association, not causation. It showed heater/throughput drives temperature, pump speed drives pressure, heater power drives energy, matching the physics (a sanity check).
13. **Prediction vs optimization?** Prediction: "what will happen at setting X". Optimization: "which X minimises an objective subject to constraints". I use the predictor as a surrogate inside a search. The answer is a candidate that must be validated, not a proven global optimum.
14. **How did you optimise and why differential evolution?** Gradient-free, handles tree models (non-differentiable, piecewise constant), bounded variables, cheap here (4 variables). Constraint violations are penalised in the objective.
15. **What went wrong in optimisation and what did you learn?** With a 0.5 C temperature margin 2 of 8 candidates slightly violated the limit in the simulator, because model error (RMSE about 0.9 C) exceeded the margin. A margin study on 30 contexts: 0 C -> 11/30 feasible, 0.5 -> 19/30, 1.0 -> 30/30 (about 4% more energy than 0.5). The optimiser exploits surrogate errors and sits on constraint edges. Caveat: I chose 1.0 after seeing that study on the same contexts.
16. **Why did the raw linear surrogate fail the optimisation?** It looked feasible by its own predictions (8/8) but 0/8 in the simulator (flow about 9 vs 10 required). A biased surrogate gives attractive but infeasible answers, which is why validation against an independent reference matters.
17. **How do you know the saving is real?** I don't, beyond the synthetic setting. The 23.6% saving is against the best comparable historical point, but historical setpoints here are random, so the baseline is weak. In industry I would compare with operator practice in A/B plant trials.
18. **How does this connect to digital twins?** A digital twin is a model synchronised with a physical asset, used for monitoring, what-if analysis and optimisation. Here `process_model` plays the "physics twin" and the ML model is a fast data-driven surrogate; the what-if plot compares them. A real twin adds live data streaming, state estimation, model recalibration and drift monitoring.
19. **What would break on real plant data?** Dynamics (need lag features / time-series or state-space models), control loops (feedback makes outputs influence inputs), sensor drift and calibration changes, operating regimes not covered by history (trees cannot extrapolate), and non-stationarity. I would add blocked CV, drift monitoring and retraining.
20. **What would you do next?** Repeated blocked CV with confidence intervals, uncertainty-aware constraints (quantile regression/conformal), SHAP, hybrid physics-informed models, multi-objective optimisation, MLflow, and testing on a real dataset with a domain engineer.

## D. Things to say carefully (avoid overclaiming)
- Never call the dataset real. Never say the model "finds the optimal operating condition"; say "candidate low-energy operating points that satisfied the constraints in simulator validation".
- The high R2 comes from smooth synthetic physics.
- Differences between XGBoost and engineered linear regression are from one split/seed.
