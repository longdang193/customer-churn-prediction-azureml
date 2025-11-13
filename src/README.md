# Source Code (`/src`)

End-to-end churn ML pipeline: data prep → training. All stages can be driven by YAML configs and log to MLflow.

## Quickstart

### Azure ML Pipeline Execution

```bash
# 1) Ensure config.env is configured with Azure ML settings
set -a && source config.env && set +a

# 2) Submit the regular training pipeline
python run_pipeline.py

# 3) Submit the HyperDrive sweep (HPO)
python run_hpo.py

# 4) After HPO completes, extract best hyperparameters and update config
python src/extract_best_params.py --parent-run-id <PARENT_RUN_ID>

# 5) Train the best model with optimized hyperparameters
python run_pipeline.py
```

> Tip: Run `mlflow ui --backend-store-uri "${MLFLOW_TRACKING_URI}" --port 5000` to inspect runs if you mirror MLflow tracking locally.

---

## Models

The project supports three models:
- **Logistic Regression** (`logreg`) - Fast baseline model
- **Random Forest** (`rf`) - Tree-based ensemble model, hyperparameters optimized in HPO
- **XGBoost** (`xgboost`) - Gradient boosting model

**Model Selection:**
- During HPO, each trial trains one model type with sampled hyperparameters
- The best model (highest F1 score) is selected and logged as `model_type` tag in MLflow
- After HPO, only the best model is trained with optimized hyperparameters

## Pipeline Overview

### Azure ML Pipeline

```
Azure data asset (e.g. churn-data:3)
  └─► run_pipeline.py / run_hpo.py
         ├─► data_prep component → processed dataset (uri_folder output)
         └─► train component → trains models, logs to MLflow
                ├─► In Azure ML: saves models as pickle files to outputs directory
                ├─► Logs metrics, params, and tags to MLflow
                └─► HyperDrive sweep (run_hpo.py): multiple trials, picks best based on configs/hpo.yaml

After HPO:
  └─► extract_best_params.py → extracts best hyperparameters and updates configs/train.yaml
  └─► run_pipeline.py → trains best model with optimized hyperparameters
```

### Key Differences: Local vs Azure ML

| Feature | Local Execution | Azure ML Execution |
|---------|----------------|-------------------|
| MLflow Runs | Nested runs supported | Uses active run (no nesting) |
| Model Saving | `mlflow.sklearn.log_model()` | Pickle files to outputs directory |
| Model Loading | `mlflow.sklearn.load_model()` | `joblib.load()` from outputs |
| Artifact Logging | Full MLflow API support | Limited (uses outputs directory) |

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

**Outputs (default: `data/processed/`)**

- `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`
- `encoders.pkl`, `scaler.pkl`
- `metadata.json` (feature names, dropped/encoded/scaled columns, target)

> Note: We use label encoding for categoricals. Tree models handle this well; for linear models consider one-hot encoding.

---

### `train.py` — train and log models

- Trains one or more models (`logreg`, `rf`, `xgboost`), optional SMOTE on the **train** split, logs params/metrics/artifacts to MLflow.
- **Azure ML Compatibility**: When running in Azure ML, nested runs are automatically disabled and models are saved as pickle files to the outputs directory (Azure ML automatically captures these as artifacts). In local execution, nested runs are used as before.
- Supports hyperparameter overrides via `--set model.param=value` (useful for manual tuning or integration with Azure ML HyperDrive).
- **HPO Mode**: When `--model-type` and `--hyperparams-json` are provided, trains only the specified model with hyperparameters from the JSON file.

**Examples**

```bash
# Use defaults from configs/train.yaml + configs/mlflow.yaml
python src/train.py

# Train only RF and XGB
python src/train.py --models rf xgboost --experiment-name churn-experiments

# Enable SMOTE for imbalanced data
python src/train.py --use-smote

# Override hyperparameters manually
python src/train.py --set rf.n_estimators=200 --set rf.max_depth=15

# HPO mode: train single model with hyperparameters from JSON
python src/train.py --model-type rf --hyperparams-json hyperparams.json
```

> Implementation note: when `class_weight='balanced'` and SMOTE is **off**, XGBoost maps imbalance via `scale_pos_weight`.

> **Azure ML Note**: In Azure ML environments, the script automatically detects the Azure ML context and:
> - Uses the existing active MLflow run instead of creating nested runs
> - Saves models as pickle files to `AZUREML_ARTIFACTS_DIRECTORY` or `AZUREML_OUTPUT_DIRECTORY` (automatically captured by Azure ML)
> - Logs model paths as MLflow tags for reference
> - Handles MLflow API limitations gracefully (some artifact APIs have different signatures in Azure ML)

---

### `extract_best_params.py` — extract best hyperparameters from HPO

- Extracts best hyperparameters from MLflow HPO sweep
- Automatically updates `configs/train.yaml` with best model and hyperparameters
- Sets `models: [best_model]` to train only the best model found

**Examples**

```bash
# Extract best params and update config
python src/extract_best_params.py --parent-run-id <PARENT_RUN_ID>

# Save params to JSON file (optional)
python src/extract_best_params.py --parent-run-id <PARENT_RUN_ID> --output best_params.json

# Preview changes without updating
python src/extract_best_params.py --parent-run-id <PARENT_RUN_ID> --dry-run
```

---

## Configuration Reference

Configs live in `configs/`. CLI flags always override config values.

- **`configs/data.yaml`** — controls raw input path, output dir, test split, random seed, target column, columns to drop/encode.
- **`configs/train.yaml`** — lists models to train (should be only best model after HPO), class weighting, SMOTE flag, random seed, and per-model hyperparameters.
- **`configs/hpo.yaml`** — defines search spaces, budget, and early stopping for Azure ML HyperDrive sweeps.
- **`configs/mlflow.yaml`** — sets the experiment name and tracking parameters.

---

## Hyperparameter Optimization (Azure ML HyperDrive)

The project includes an optional sweep job powered by Azure ML HyperDrive. The sweep uses the search space defined in `configs/hpo.yaml` for the configured model(s).

**Submit the sweep job**

```bash
# Ensure config.env is configured with Azure ML settings
set -a && source config.env && set +a
python run_hpo.py
```

The sweep job:

1. Runs `data_prep.py` as the first pipeline step to materialize processed data in the workspace.
2. Launches a sweep over the specified hyperparameters (configured in `configs/hpo.yaml`), logging all trials to MLflow.
3. Each trial trains one model type with sampled hyperparameters.
4. Surfaces the best-performing configuration via MLflow metrics/tags (`model_type` tag).

**After HPO completes:**

1. Extract best hyperparameters and update config:
   ```bash
   python src/extract_best_params.py --parent-run-id <PARENT_RUN_ID>
   ```
2. Train the best model:
   ```bash
   python run_pipeline.py
   ```

**Configuration**

The HPO behavior is controlled by `configs/hpo.yaml`:

- `metric`: Primary metric to optimize (e.g., `f1`, `roc_auc`)
- `mode`: Optimization direction (`max` or `min`)
- `budget.max_trials`: Maximum number of trials to run
- `budget.max_concurrent`: Maximum parallel trials
- `early_stopping`: Configuration for early termination policies
- `search_space`: Hyperparameter ranges per model (currently supports `rf`, `xgboost`)

Results and individual trials can be inspected in Azure ML Studio using the URL printed after submission.

---

## Troubleshooting

### General Issues

- **Missing parent run ID:** When extracting best parameters from HPO, ensure you use the correct parent run ID from the sweep job. Check Azure ML Studio for the sweep job run ID.
- **No models trained:** Check training logs; if all models error, the run won't complete successfully.
- **Hyperparameter override syntax:** Use `--set model.param=value` format. For boolean values, use `true`/`false` (lowercase). For `None`, use `none` (lowercase). Numeric values are parsed automatically.

### Azure ML Specific Issues

- **MLflow nested runs error:** The script automatically detects Azure ML and disables nested runs. If you see errors about active runs, ensure you're using the latest version of `train.py` that includes Azure ML detection.

- **Model artifact logging errors:** In Azure ML, `mlflow.sklearn.log_model()` and some artifact APIs have limitations. The script automatically falls back to saving models as pickle files to the outputs directory, which Azure ML captures automatically.

- **Model loading errors:** When loading models in Azure ML, the script looks for models in the outputs directory (`AZUREML_ARTIFACTS_DIRECTORY` or `AZUREML_OUTPUT_DIRECTORY`). Ensure models were saved during training.

- **Python version compatibility:** The project requires **Python 3.9** due to `azureml-core` compatibility. Ensure your Docker image uses Python 3.9.

For more detailed troubleshooting information, see [`../docs/TROUBLESHOOTING.md`](../docs/TROUBLESHOOTING.md).
