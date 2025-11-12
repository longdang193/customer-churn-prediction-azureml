# Source Code (`/src`)

End-to-end churn ML pipeline: data prep → training → evaluation → scoring. All stages can be driven by YAML configs and log to MLflow.

## Quickstart

```bash
# 0) Create & activate a virtualenv
python -m venv .venv && source .venv/bin/activate

# 1) Install deps
pip install -r requirements.txt

# 2) Set MLflow tracking URI (file store for quick local use)
export MLFLOW_TRACKING_URI="file:./mlruns"

# 3) Prepare data (reads configs/data.yaml by default)
python src/data_prep.py

# 4) Train models (logs runs + artifacts to MLflow)
python src/train.py

# 5) Evaluate the best model (use the PARENT run id printed in step 4)
python src/evaluate.py --run-id <PARENT_RUN_ID>

# 6) (Optional) Persist best model locally + run id for downstream use
python src/train.py \
  --model-artifact-dir models/local \
  --parent-run-id-output models/local/parent_run_id.txt

# 7) Score new data (CSV) using the saved artifact
python src/score.py --model models/local/rf_model.pkl \
                    --data-dir data/processed \
                    --input data/new_customers.csv \
                    --output predictions/preds.csv
```

> Tip: Launch the MLflow UI locally  
> `mlflow ui --backend-store-uri "${MLFLOW_TRACKING_URI}" --port 5000`

---

## Pipeline Overview

```
Raw CSV
  └─► data_prep.py
         ├─► data/processed/X_train.csv, X_test.csv, y_train.csv, y_test.csv
         └─► data/processed/{encoders.pkl, scaler.pkl, metadata.json}
              ↑ schema/feature names live here
Local workflow
  └─► train.py
         └─► MLflow experiment runs (one parent + nested per model)
                 └─► artifacts: model_<name>, metrics
                      tags on parent: best_model, best_model_run_id
  └─► evaluate.py
         └─► evaluation/<run>/ (metrics JSON + plots)
  └─► score.py
         └─► predictions/*.csv or *.json

Azure ML workflow
  └─► run_pipeline.py (DSL pipeline)
         ├─► data_prep component → processed dataset
         ├─► train component → model_output + parent_run_id
         └─► evaluate component → evaluation artifacts
  └─► run_hpo.py (HyperDrive sweep)
         └─► train component.sweep() exploring configs/train.yaml::hpo.search_space
```

---

## Scripts

### `data_prep.py` — prepare raw data

- Drops uninformative columns, encodes categoricals, splits train/test, scales numeric features (fit on train; transform on test), and writes artifacts.
- Reads defaults from `configs/data.yaml`; CLI flags override.

**Examples**

```bash
# Default config
python src/data_prep.py

# Custom output + seed
python src/data_prep.py --output data/processed_custom --random-state 1337
```

**Inputs**

- CSV: `data/churn.csv` (override with `--input`)
- Config (optional): `configs/data.yaml`

**Outputs (default: `data/processed/`)**

- `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`
- `encoders.pkl`, `scaler.pkl`
- `metadata.json` (feature names, dropped/encoded/scaled columns, target)

> Note: We use label encoding for categoricals. Tree models handle this well; for linear models consider one-hot encoding.

---

### `train.py` — train and log models

- Trains one or more models (`logreg`, `rf`, `xgboost`), optional SMOTE on the **train** split, logs params/metrics/artifacts to MLflow as **nested runs**.
- Tags the **parent run** with `best_model` and `best_model_run_id` for downstream evaluation.
- Optional `--model-artifact-dir` saves the best model (and metadata JSON) alongside the MLflow logs; `--parent-run-id-output` writes the parent run ID to a file for automation (e.g. AML components).
- Supports hyperparameter overrides via `--set model.param=value` (useful for manual tuning or integration with Azure ML HyperDrive).

**Examples**

```bash
# Use defaults from configs/train.yaml + configs/mlflow.yaml
python src/train.py

# Train only RF and XGB in a custom experiment
python src/train.py --models rf xgboost --experiment-name churn-experiments

# Enable SMOTE for imbalanced data
python src/train.py --use-smote

# Save the best model artifact locally
python src/train.py --model-artifact-dir models/local --parent-run-id-output models/local/parent_run_id.txt

# Override hyperparameters manually (useful for testing or manual tuning)
python src/train.py --set rf.n_estimators=200 --set rf.max_depth=15

# Multiple overrides for different models
python src/train.py --set rf.n_estimators=200 --set logreg.C=10.0
```

> Implementation note: when `class_weight='balanced'` and SMOTE is **off**, XGBoost maps imbalance via `scale_pos_weight`.

---

### `evaluate.py` — metrics & plots for best model

- Requires the **parent** MLflow run id; resolves the nested **best** run via tags and loads its model artifact.
- Produces: metrics JSON, confusion matrix, ROC/PR curves, probability histogram, and feature importance/coefs.
- If you captured the parent run id in a file (e.g., via `--parent-run-id-output`), you can pass `--parent-run-id-file` instead of `--run-id`.

**Examples**

```bash
python src/evaluate.py \
  --run-id <PARENT_RUN_ID> \
  --data data/processed \
  --output evaluation/latest

# Using a saved run-id file
python src/evaluate.py \
  --parent-run-id-file models/local/parent_run_id.txt \
  --data data/processed \
  --output evaluation/latest
```

**Outputs**

- `evaluation_report.json`
- `confusion_matrix.png`
- `roc_curve.png`
- `precision_recall_curve.png`
- `proba_distribution.png`
- `feature_importance.png` (or coefficients for linear models)

---

### `score.py` — batch & single-row inference

- Loads a trained model (pickle) and the prep artifacts to ensure consistent preprocessing.
- **Batch (CSV)**: reads new data and writes predictions (and optional probabilities).
- **JSON mode**: scores a single record or list of records to JSON.

**Examples**

```bash
# CSV → CSV
python src/score.py \
  --model models/local/rf_model.pkl \
  --data-dir data/processed \
  --input data/new_data.csv \
  --output predictions/new_predictions.csv

# JSON → JSON
python src/score.py \
  --json \
  --model models/local/rf_model.pkl \
  --data-dir data/processed \
  --input payloads/customer.json \
  --output predictions/customer_response.json
```

**Requirements**

- The columns in the input must match training schema (see `metadata.json`).
- Unseen categorical values will raise with `LabelEncoder`; if you expect them, consider switching prep to `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)`.

---

## Configuration Reference

Configs live in `configs/`. CLI flags always override config values.

- **`configs/data.yaml`** — controls raw input path, output dir, test split, random seed, target column, columns to drop/encode.
- **`configs/train.yaml`** — lists models to train, class weighting, SMOTE flag, random seed, and optional per-model hyperparameters. Also includes an `hpo` section defining search spaces for Azure ML HyperDrive sweeps.
- **`configs/mlflow.yaml`** — sets the experiment name and tracking parameters.
- **`configs/score.yaml` / `configs/evaluate.yaml`** — default options for scoring (e.g., include probabilities) and evaluation (e.g., top-n features).

---

## Hyperparameter Optimization (Azure ML HyperDrive)

The project includes an optional sweep job powered by Azure ML HyperDrive. The sweep uses the search space defined under `hpo.search_space` in `configs/train.yaml` (currently tuned for the random forest model).

**Submit the sweep job**

```bash
# Ensure Azure credentials are configured (.env) and a raw data asset exists
# Optionally override the asset name via AZURE_RAW_DATA_ASSET
python run_hpo.py
```

The sweep job:

1. Runs `data_prep.py` as the first pipeline step to materialize processed data in the workspace.
2. Launches a sweep over the specified random forest hyperparameters, logging all trials to MLflow.
3. Surfaces the best-performing configuration via MLflow metrics/tags (`best_model_*`).

**Configuration**

The HPO behavior is controlled by the `hpo` section in `configs/train.yaml`:

- `metric`: Primary metric to optimize (e.g., `f1`, `roc_auc`)
- `mode`: Optimization direction (`max` or `min`)
- `budget.max_trials`: Maximum number of trials to run
- `budget.max_concurrent`: Maximum parallel trials
- `early_stopping`: Configuration for early termination policies
- `search_space`: Hyperparameter ranges per model (currently supports `rf`, `logreg`, `xgboost`)

Results and individual trials can be inspected in Azure ML Studio using the URL printed after submission.

---

## Tests & Smoke Validation

### End-to-end smoke test (prep → train → evaluate)

Runs the full pipeline on a small sample and checks that core artifacts are created. It exercises the standard happy path of the CLI scripts and validates MLflow logging & artifact generation (it does **not** cover hyperparameter optimisation).

```bash
# Optional: point MLflow to a local file store
export MLFLOW_TRACKING_URI="file:./mlruns"

# Run the smoke test
python tests/smoke_test.py
```

**What it does**

1. **Prep:** builds `data/processed_smoke/` with `X_train.csv`, `X_test.csv`, `y_*.csv`, and preprocessing artifacts.
2. **Train:** logs a parent MLflow run plus nested runs per model; tags the parent with `best_model` and `best_model_run_id`, and (if requested) persists the best model artifact + run id.
3. **Evaluate:** loads the best nested run and writes `evaluation/smoke/` outputs.

**Pass criteria (examples)**

- MLflow has a **parent** run named `Churn_Training_Pipeline` with tags `best_model` and `best_model_run_id`.
- Files exist in `evaluation/smoke/`:

  - `evaluation_report.json`
  - `roc_curve.png` and `precision_recall_curve.png` (and other plots)

> If the test fails, the script prints which artifact was missing or which command errored.

---

### HPO smoke test (HyperDrive sweep validation)

Validates the full Azure ML HyperDrive workflow by submitting a minimal sweep (2 trials). This test requires Azure ML credentials and is skipped if not configured.

```bash
# Ensure Azure credentials are configured (.env file)
# Run the HPO smoke test
pytest tests/hpo_smoke_test.py -v
```

**What it does**

1. **Loads HPO config** from `configs/train.yaml::hpo`
2. **Submits minimal sweep** (2 trials, 1 concurrent) to Azure ML
3. **Validates submission** by checking job status and structure
4. **Tests parameter space building** to ensure config parsing works

**Pass criteria**

- Sweep job submits successfully to Azure ML
- Job status is valid (NotStarted, Queued, Starting, etc.)
- Parameter space is built correctly from config
- Studio URL is generated for monitoring

> **Note**: This test does not wait for sweep completion (to keep it fast). For full validation, monitor the job in Azure ML Studio or extend the test to wait for completion and validate the best trial.

**Skipping the test**

The test automatically skips if:
- Azure ML credentials are not configured (`AZURE_SUBSCRIPTION_ID` missing)
- Data asset is not available
- Azure ML workspace is unreachable

---

### Unit tests

Run fast checks on individual modules and helpers.

```bash
pytest -q
```

**Tips**

- Use `-k <pattern>` to run a subset, e.g. `pytest -q -k prep`.
- Add `-s` to see print/log output during a failing test.

---

## Troubleshooting

- **MLflow “No run found” in evaluate:** Ensure you pass the **parent** run id from training (or supply the file written by `--parent-run-id-output`). Check the MLflow UI; the parent run will have `best_model_run_id` in tags.
- **Shape/column mismatch when scoring:** Compare your input columns to `data/processed/metadata.json`. The scorer drops/encodes/scales according to this file.
- **Unseen categories at inference:** Update prep to use `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)` or map unknowns before transform.
- **No models trained / "best model" missing:** Check training logs; if all models error, the run won't tag a best model.
- **Hyperparameter override syntax:** Use `--set model.param=value` format. For boolean values, use `true`/`false` (lowercase). For `None`, use `none` (lowercase). Numeric values are parsed automatically.
