# Cross-check against the ten uploaded course chapters

This checklist uses the lecture slide topics and distinguishes actual implemented methods from conceptual connections or methods unsuitable for this small tabular regression problem.

| Chapter | Lecture method | Status in this project | Evidence |
|---|---|---|---|
| 1 Introduction | Supervised learning and train/validation/test roles | **Used** | 3 geometric input features, continuous FE SCF target, 80/20 holdout |
| 2 Regression | Parametric linear prediction | **Used** | `Linear OLS`; SGD linear predictor |
| 3 Loss functions | MSE, MAE, residuals; outlier sensitivity | **Used** | MSE-fitting regressors; MAE for CV; RMSE, MAE and R² for independent test; residual plot |
| 4 Optimization | Iterative gradient-based optimization, learning rate, feature scaling, stopping criteria | **Used** via `SGDRegressor` with adaptive learning rate, chosen `eta0`, `max_iter` and `tol`; scaling inside pipeline | `Linear SGD`; hyperparameter search |
| 5 Least squares | Analytical solutions to linear regression | **Used** | sklearn `LinearRegression` and polynomial least squares |
| 6 Under/overfitting | Polynomial feature expansion and model complexity | **Used** | Degree 2 and 3 candidates; train vs CV-validation gaps |
| 7 Maximum likelihood | Gaussian noise -> least-squares; Laplace noise -> absolute error | **Theoretical framing only** | MSE fitting vs MAE validation; no probabilistic noise-model estimation |
| 8 Regularization | L2 Ridge, L1 Lasso; feature standardization; λ strength | **Used** | Ridge/Lasso pipelines, alpha grids, fold-specific scaling |
| 9 Generalization and model selection | Independent test, K-fold CV, validation-based grid search, final refit | **Used** | 112 training-development, 5-fold CV, 28 untouched test; test evaluated after selection |
| 10 Bias–variance | Interpreting training/validation gap and model flexibility | **Discussed, not empirically decomposed** | CV train/validation MAEs indicate model stability; no unsupported true-bias claims |

## Key lecture links to concrete design choices

- Chapter 4 p.62: standardizing **from training statistics** improves optimization conditioning. We put `StandardScaler` in each fold's Pipeline.
- Chapter 6 pp.12–15: polynomial feature expansion enhances a linear-in-parameters model. We compare degrees 2 and 3.
- Chapter 8 p.9: standardize before coefficient penalties. Both Ridge and Lasso have scaling inside the Pipeline.
- Chapter 9 p.17: **data leakage** is not prevented merely by disjoint filenames; preprocessing must be fitted inside training. We exclude the original authors' ANN output column and use fold-specific transformations.
- Chapter 9 pp.24–27: choose alpha/degree based on **validation loss**, not training loss or test outcomes.
- Chapter 9 pp.33–39: average K-fold validation loss, refit a fresh model on all development data, then test once on unseen cases.
- Chapter 10 pp.38–39: simple models may exhibit high bias, while flexible ones may exhibit high variance; training/validation gap is diagnostic but is **not** itself a numerical bias-variance decomposition.

## Lecture techniques intentionally not included

- **Dropout and deep-network early stopping** (Chapter 8): no deep neural network is trained here. Polynomial OLS/Ridge have analytical/numerical convex solvers; early stopping is not a natural comparison control.
- **K-fold checkpoint early stopping** (Chapter 9): needed only for iterative models monitored by epochs; `SGDRegressor` uses convergence stopping via `tol` instead.
- **New MLE model** (Chapter 7): Gaussian likelihood is mathematically connected to squared-loss OLS; adding a duplicate likelihood fit would not contribute new evaluation evidence.
- **Full statistical bias-variance decomposition** (Chapter 10): requires repeated independent datasets or a known generating function; cannot be estimated rigorously with this one fixed 140-case FE grid.

This is an honest coverage checklist: **not every method in the slides is compulsory or appropriate**, and the Mini-Project Brief values a defensible baseline and reproducible independent evaluation over needless complexity.