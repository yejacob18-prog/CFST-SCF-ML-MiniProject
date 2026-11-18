# Machine Learning Mini-Project — CFST T-joint SCF prediction

**Title:** Machine Learning-Based Prediction of Brace-Saddle Stress Concentration Factors in CFST T-Joints under Axial Compression

**GitHub repository:** https://github.com/yejacob18-prog/CFST-SCF-ML-MiniProject

This repository contains an independently implemented, reproducible regression comparison. The prediction target is the **FEA stress concentration factor (SCF) at the brace saddle of a concrete-filled steel tubular (CFST) T-joint under brace compression**. It is a supervised regression problem, not a classification problem.

## 1. Data and provenance

Original data: S. Bajracharya, S. Nakamura, T. Nishikawa, **Dataset for SCF Prediction in CFST T- and K-Joints Using ANN**, Mendeley Data (2025), Version 1, [DOI: 10.17632/rp7vm5627z.1](https://doi.org/10.17632/rp7vm5627z.1) (CC BY 4.0). Related peer-reviewed article: *Prediction of stress concentration factors for concrete-filled steel tubular (CFST) T-joints and K-joints using artificial neural network*, *Results in Engineering* 27 (2025), article 106313, [DOI: 10.1016/j.rineng.2025.106313](https://doi.org/10.1016/j.rineng.2025.106313).

- Original workbook: `Compile data of SCF FEM and ANN.xlsx`, downloadable from the cited Mendeley Data release. The public repository includes the cleaned CSV; the complete local project ZIP also contains a copy of the original workbook.
- Exact sheet used: `T-JOINT BS IN COMP`, **140** numerical FE-derived cases.
- Analysis data: `data/scf_bs_compression.csv`, with 3 input ratios: `beta` (4 levels, 0.3-0.6), `gamma` (5 levels, 20-40), `tau` (7 levels, 0.4-1.0). Target: `SCF_FEA` (range 1.60-2.14).
- The original sheet also includes authors' `ANN` predictions. Those predictions are **deliberately excluded** from all model inputs, all targets, and the model-selection procedure; otherwise evaluation would be contaminated.
- 140 unique combinations = 4 × 5 × 7, no missing or nonfinite values. A source-to-CSV, row-for-row audit (0 mismatches) is in `data/DATA_AUDIT.json`. The original workbook is available at the Mendeley Data DOI for auditing; the training script uses the included CSV and does not need Excel.

The numerical FEA samples are **not new physical tests** and do not establish validated performance on unseen structures or geometries outside the sampled domain.

## 2. Setup

Install Python 3.11+ (results were generated using Python 3.13.5, NumPy 2.3.5, scikit-learn 1.8.0; the pinned dependencies in `requirements-lock.txt` require Python 3.11 or later) and run, from the project directory:

```bash
python -m pip install -r requirements.txt
python scf_project.py check-data
python scf_project.py train
python -m unittest discover -s tests -v
```

For exact library versions matching the executed experiment, install `requirements-lock.txt` instead of `requirements.txt` (see `environment.json`). No GPU, TensorFlow, external data download, spreadsheet app, or web UI is required.

## 3. Train / validate / test — exactly how

- A fixed **80%/20%** shuffle split (seed **123**) produces 112 development observations and **28 untouched test** observations. Actual row indices are saved in `results/split_indices.json`.
- On the **112 development cases only**, perform shuffled **5-fold cross-validation** (same seed and folds for fair comparison). In each fold, four portions fit parameters and preprocessing; one portion evaluates validation MAE. Grid search tunes polynomial degree (2 or 3), Ridge `alpha`, Lasso `alpha`, and SGD step size.
- Feature expansion and `StandardScaler` are *inside each scikit-learn Pipeline*, so each fold learns its scaling only from its training portion. This avoids information leakage.
- Compare **mean baseline**, **linear OLS**, **linear SGD**, **polynomial OLS**, **Ridge**, and **Lasso** on mean held-out fold MAE. An ordinary least-squares model and an iterative SGD fit also illustrate the two optimization approaches taught in class.
- Select the model/hyperparameters that have the **lowest mean cross-validation MAE** (never use test performance to choose model complexity). Refit the chosen model on all 112 development rows. Evaluate it **once** on the 28 held-out test rows using MAE, RMSE, and R².
- Overfitting is examined by comparing *fold training MAE* with *fold validation MAE*, rather than claiming training error alone proves generalization.

### Why certain lecture methods are not used

- Chapter 7 Gaussian-noise maximum likelihood *explains* squared-error regression, but we do not additionally run a probabilistic MLE model.
- Dropout, neural-network early stopping, and weight-decay schedules are **not required** for the main closed-form / coordinate-descent regressors. SGD uses a convergence tolerance and an adaptive learning rate. The original author's TensorFlow ANN code was examined for context but **was not reused or reproduced**.
- Bias-variance *interpretation* is supported by training-versus-validation gaps. A true numerical bias-variance decomposition would require repeated independent samples from a known data-generating distribution and is **not claimed**.

## 4. Results from running the supplied scripts

All metrics below are produced on the fixed split; the saved CSV and JSON are the sources of truth. Run `python scf_project.py train` to reproduce or regenerate the output artifacts listed below.

| Model family | Training-fold MAE | Validation-fold MAE | Best setting |
|:--|--:|--:|:--|
| Mean baseline | 0.10615 | 0.10819 | constant mean |
| Linear OLS | 0.03525 | 0.03607 | linear |
| Linear SGD | 0.03525 | 0.03608 | eta0=0.01 |
| Polynomial OLS | 0.01160 | **0.01722** | degree=3 |
| Polynomial Ridge | 0.01185 | 0.01723 | degree=3, alpha=0.001 |
| Polynomial Lasso | 0.01545 | 0.01853 | degree=3, alpha=0.0001 |

Selected by CV: **degree-3 polynomial OLS**, refit on all 112 training rows. Independent **28-point test:** **MAE 0.01973**, **RMSE 0.02755**, **R² 0.96693**; maximum test absolute error **0.09047** SCF. Unlike the earlier starter, this version evaluates polynomial degree 3, so its selected model is different.

These errors are on a small random subset of a finite parametric FE grid. They describe interpolation over input combinations from this particular distribution, not extrapolation or experimental accuracy. Ridge is nearly tied with OLS within CV, so the small numerical gap must not be overstated.

## 5. Example prediction / inference

```bash
python scf_project.py predict --beta 0.45 --gamma 30 --tau 0.7
```

This generates a numerical prediction from `results/fitted_model.joblib`. Inputs must stay within the original parameter ranges. **Do not** use predictions in engineering design without separate validation. Never load third-party `joblib` files from untrusted sources.

## 6. Generated artifacts

- `results/metrics.json` — final test metrics, selected model, versions and random seed
- `results/split_indices.json` — generated exact training and test rows
- `results/cv_model_comparison.csv` — 5-fold training and validation MAEs
- `results/hyperparameter_search.csv` — generated all validation candidate scores
- `results/held_out_predictions.csv` — generated 28 observed and predicted FE SCFs
- `results/fitted_model.joblib` — generated locally trained estimator (created by `train`)
- `figures/cv_model_comparison.png` — generated training-versus-validation comparison
- `figures/test_prediction_scatter.png` — generated held-out FEA versus ML predictions
- `figures/test_residuals.png` — generated held-out prediction residuals
- `report/SCF_MiniProject_Report.tex` — LaTeX report source (compile after `train` generates figures); the locally compiled two-page PDF is included in the complete project ZIP
- `COURSE_METHODS_CHECKLIST.md` — chapter-by-chapter verification of teaching methods (included in the full project ZIP)
- `requirements-lock.txt` and `environment.json` — exact dependency versions tested

## 7. Reproducibility checks

Run `python scf_project.py train` a second time and compare `results/metrics.json` for numerical consistency; the random seed and folds are fixed. Use `python -m unittest discover -s tests -v` to check expected data properties, no target leakage and the prediction pipeline. The project works from any working directory because paths are relative to the script location. **For a clean checkout**, first run the training command to generate the model weights, all images, split indices, hyperparameter tables and final predictions. These generated binary artifacts are intentionally not checked into this lightweight repository. The submitted CSV and source scripts are sufficient to reproduce them.

## 8. Submitting the assignment

1. Review the generated code, results and limitations before submission.
2. Check the author line in `report/SCF_MiniProject_Report.tex` and replace it if the submitting author(s) differ. The GitHub URL is already included. Recompile using `cd report && pdflatex SCF_MiniProject_Report.tex` (or Overleaf).
3. Use this public repository URL for submission: https://github.com/yejacob18-prog/CFST-SCF-ML-MiniProject.
4. Run `python scf_project.py train` to regenerate the report figure, then compile the report and verify the PDF is at most **2 A4 pages** after the author details are added. Keep dataset attribution and AI disclosures.

## 9. Attribution and AI assistance

The FE dataset and the original ANN sample code were authored by Bajracharya, Nakamura, and Nishikawa and published at the Mendeley DOI above. This project implements **new scikit-learn regression scripts**, not their published TensorFlow ANN. **Generative AI disclosure:** ChatGPT assisted with experimental planning, writing the independent Python scripts and report drafting, and running local tests. A student should review the code, results, and text before submitting them as coursework. No pretrained machine-learning model was used.