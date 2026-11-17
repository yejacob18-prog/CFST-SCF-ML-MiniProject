"""Reproducible ML mini-project: FE-based brace-saddle SCF of CFST T-joints.

DO NOT use authors' precomputed ANN predictions as an input/target.
The final 20% test set is used only after training-only cross-validation.
Source data DOI: https://doi.org/10.17632/rp7vm5627z.1 (CC BY 4.0).
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge, SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "scf_bs_compression.csv"
OUT = ROOT / "results"
FIG = ROOT / "figures"
SEED = 123
COLUMNS = ("beta", "gamma", "tau")
BOUNDS = ((0.3, 0.6), (20.0, 40.0), (0.4, 1.0))


def read_dataset():
    with DATA.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != [*COLUMNS, "SCF_FEA"]:
            raise ValueError(f"Unexpected columns: {reader.fieldnames}; ANN must not be an input.")
        rows = list(reader)
    if len(rows) != 140:
        raise ValueError(f"Expected 140 rows, found {len(rows)}")
    X = np.array([[float(row[c]) for c in COLUMNS] for row in rows], dtype=np.float64)
    y = np.array([float(row["SCF_FEA"]) for row in rows], dtype=np.float64)
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Dataset contains nonfinite input or target.")
    if np.unique(X, axis=0).shape[0] != 140:
        raise ValueError("Duplicate geometrical combinations detected.")
    expected_levels = [4, 5, 7]
    if [len(np.unique(X[:, i])) for i in range(3)] != expected_levels:
        raise ValueError("Parameter grid does not match the published T-joint sheet.")
    for i, (low, high) in enumerate(BOUNDS):
        if X[:, i].min() < low-1e-9 or X[:, i].max() > high+1e-9:
            raise ValueError(f"Input {COLUMNS[i]} out of published range.")
    return X, y


def pipe(model, degree=1):
    # Both expansion and scaling are fitted inside each CV training fold.
    return Pipeline([
        ("polynomial", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scaler", StandardScaler()),
        ("regressor", model),
    ])


def specifications():
    return [
        ("Mean baseline", DummyRegressor(strategy="mean"), None),
        ("Linear OLS", pipe(LinearRegression()), None),
        ("Linear SGD", pipe(SGDRegressor(
            random_state=SEED, loss="squared_error", penalty=None, max_iter=10000,
            tol=1e-7, learning_rate="adaptive", eta0=0.01)),
         {"regressor__eta0": [0.001, 0.01]}),
        ("Polynomial OLS", pipe(LinearRegression(), degree=2),
         {"polynomial__degree": [2, 3]}),
        ("Polynomial Ridge", pipe(Ridge(), degree=2),
         {"polynomial__degree": [2, 3],
          "regressor__alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}),
        ("Polynomial Lasso", pipe(Lasso(max_iter=50000, tol=1e-4), degree=2),
         {"polynomial__degree": [2, 3],
          "regressor__alpha": [0.0001, 0.0003, 0.001, 0.01, 0.1]}),
    ]


def write_csv(path, headers, data):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(data)


def evaluate(y, y_pred):
    return {
        "MAE": float(mean_absolute_error(y, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y, y_pred))),
        "R2": float(r2_score(y, y_pred)),
    }


def train():
    X, y = read_dataset()
    ids = np.arange(len(y), dtype=int)
    X_dev, X_test, y_dev, y_test, dev_ids, test_ids = train_test_split(
        X, y, ids, test_size=0.20, random_state=SEED, shuffle=True
    )
    assert len(dev_ids) == 112 and len(test_ids) == 28
    assert set(dev_ids).isdisjoint(set(test_ids))
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    summary = []
    fitted = {}
    grid_details = []

    for name, estimator, params in specifications():
        if params is None:
            cross = cross_validate(estimator, X_dev, y_dev, cv=cv,
                scoring="neg_mean_absolute_error", return_train_score=True)
            fold_val = -cross["test_score"]
            fold_train = -cross["train_score"]
            best_params = {}
            chosen = estimator.fit(X_dev, y_dev)
        else:
            search = GridSearchCV(estimator, params, cv=cv,
                scoring="neg_mean_absolute_error", return_train_score=True,
                refit=True, n_jobs=1, error_score="raise")
            search.fit(X_dev, y_dev)
            pos = search.best_index_
            fold_val = np.array([ -search.cv_results_[f"split{i}_test_score"][pos]
                                  for i in range(5)])
            fold_train = np.array([ -search.cv_results_[f"split{i}_train_score"][pos]
                                    for i in range(5)])
            best_params = search.best_params_
            chosen = search.best_estimator_
            for cv_par, cv_score in zip(search.cv_results_["params"],
                                        search.cv_results_["mean_test_score"]):
                grid_details.append([name, json.dumps(cv_par, sort_keys=True), -float(cv_score)])
        summary.append(dict(model=name,
                            train_cv_mae=float(fold_train.mean()),
                            validation_cv_mae=float(fold_val.mean()),
                            validation_cv_sd=float(fold_val.std(ddof=1)),
                            best_parameters=best_params,
                            validation_folds=fold_val.tolist()))
        fitted[name] = chosen
        print(f"{name:20s} train-CV MAE={fold_train.mean():.5f}, "
              f"validation-CV MAE={fold_val.mean():.5f} +/- {fold_val.std(ddof=1):.5f}, "
              f"params={best_params}", flush=True)

    # Model choice uses only development data (including cross-validation).
    selected = min(summary, key=lambda row: row["validation_cv_mae"])
    best = fitted[selected["model"]]
    # This is the first and only test-set evaluation (after all selection decisions).
    test_predictions = best.predict(X_test)
    test_metrics = evaluate(y_test, test_predictions)
    dev_metrics = evaluate(y_dev, best.predict(X_dev))

    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    joblib.dump(best, OUT / "fitted_model.joblib")
    write_csv(OUT / "cv_model_comparison.csv",
              ["model", "CV_train_MAE", "CV_validation_MAE", "CV_validation_SD", "best_parameters"],
              [[r["model"], r["train_cv_mae"], r["validation_cv_mae"],
                r["validation_cv_sd"], json.dumps(r["best_parameters"], sort_keys=True)]
               for r in summary])
    write_csv(OUT / "hyperparameter_search.csv", ["model", "parameters", "CV_validation_MAE"], grid_details)
    write_csv(OUT / "held_out_predictions.csv",
              ["CSV_row_number", "beta", "gamma", "tau", "FEA_SCF", "ML_predicted_SCF", "error"],
              [[int(i)+2, *xs, float(target), float(pred), float(pred-target)]
               for i,xs,target,pred in zip(test_ids, X_test, y_test, test_predictions)])
    info = {
        "dataset_rows": 140, "training_rows": 112, "test_rows": 28,
        "random_seed": SEED, "cv_folds": 5,
        "selected_model": selected["model"],
        "selected_hyperparameters": selected["best_parameters"],
        "selected_cv_validation_mae": selected["validation_cv_mae"],
        "selected_cv_training_mae": selected["train_cv_mae"],
        "training_MAE_final_refit": dev_metrics["MAE"],
        "test": test_metrics,
        "max_test_absolute_error": float(np.max(np.abs(test_predictions-y_test))),
        "max_error_source_CSV_row": int(test_ids[np.argmax(np.abs(test_predictions-y_test))]+2),
        "python": platform.python_version(), "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
    }
    (OUT / "metrics.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    (OUT / "split_indices.json").write_text(json.dumps({
        "seed": SEED, "data_row_indices_zero_based": {
            "development": dev_ids.tolist(), "test": test_ids.tolist()}
    }, indent=2), encoding="utf-8")

    plt.rcParams.update({"font.size":9})
    # Figure 1: validation and training errors by model.
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    labels = [r["model"] for r in summary]
    pos = np.arange(len(labels))
    width=0.37
    ax.bar(pos-width/2,[r["train_cv_mae"] for r in summary],width,label="CV training MAE")
    ax.bar(pos+width/2,[r["validation_cv_mae"] for r in summary],width,label="CV validation MAE")
    ax.set_xticks(pos, labels, rotation=21, ha="right")
    ax.set_ylabel("MAE of SCF (lower is better)")
    ax.set_title("Training vs validation performance (5-fold CV)")
    ax.legend(frameon=False)
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG/"cv_model_comparison.png",dpi=200); plt.close(fig)
    # Figure 2: independent testing, 28 unseen parameter combinations.
    fig, ax = plt.subplots(figsize=(4.3, 3.6))
    ax.scatter(y_test,test_predictions,s=31,alpha=.85)
    lo = min(np.min(y_test),np.min(test_predictions))-.025
    hi = max(np.max(y_test),np.max(test_predictions))+.025
    ax.plot([lo,hi],[lo,hi],ls="--",lw=1,c="0.3")
    ax.set(xlabel="FEA SCF",ylabel="Predicted SCF",xlim=(lo,hi),ylim=(lo,hi))
    ax.set_title(f"Held-out test (n=28), $R^2$={test_metrics['R2']:.3f}")
    ax.grid(alpha=.15)
    fig.tight_layout(); fig.savefig(FIG/"test_prediction_scatter.png",dpi=220); plt.close(fig)
    # Figure 3: error as a function of chord geometry ratio, for discussion.
    fig, ax = plt.subplots(figsize=(4.3, 3.2))
    ax.scatter(X_test[:, 1], test_predictions-y_test, s=32,alpha=.8)
    ax.axhline(0,linestyle="--",lw=1,c="0.3")
    ax.set(xlabel="Gamma",ylabel="Prediction minus FEA SCF",title="Held-out residuals")
    ax.grid(alpha=.15)
    fig.tight_layout(); fig.savefig(FIG/"test_residuals.png",dpi=200);plt.close(fig)

    print("\nSELECTED BY TRAIN-ONLY CV:",selected["model"],selected["best_parameters"])
    print("HELD-OUT TEST:",test_metrics)
    print("REPORT DATA:", json.dumps(info, indent=2))


def predict(beta,gamma,tau):
    values=(beta,gamma,tau)
    for value,(low,high),name in zip(values,BOUNDS,COLUMNS):
        if not (low <= value <= high):
            raise ValueError(f"{name} must be within [{low}, {high}]; extrapolation not supported")
    model_path = OUT / "fitted_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError("Train first: python scf_project.py train")
    # Only load artifacts built by this script, never unknown joblib files.
    model = joblib.load(model_path)
    print(f"Predicted FE-derived brace-saddle SCF: {model.predict(np.array([values]))[0]:.6f}")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest="cmd",required=True)
    sub.add_parser("train",help="Full CV, select model, refit, independent test")
    sub.add_parser("check-data",help="Check shape, duplicates, finite values, parameter grid")
    p=sub.add_parser("predict",help="Predict SCF for one valid input combination")
    for feature in COLUMNS: p.add_argument("--"+feature,required=True,type=float)
    args=ap.parse_args()
    if args.cmd=="train": train()
    elif args.cmd=="check-data":
        X,y=read_dataset()
        print(f"PASS: {len(y)} rows, 3 numerical inputs, 140 unique grid combinations, "
              f"FEA SCF range [{y.min():.3f}, {y.max():.3f}].")
    else: predict(args.beta,args.gamma,args.tau)

if __name__=="__main__": main()