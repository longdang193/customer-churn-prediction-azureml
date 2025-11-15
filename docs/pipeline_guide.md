# HyperDrive Pipeline Guide

This guide walks through the churn pipeline when driven entirely by Azure ML HyperDrive. Each script/component plays a specific role from data preparation through scoring.

## 1. Data Preparation (`src/data_prep.py`)

Used in two contexts:
- **Azure ML component** (`aml/components/data_prep.yaml`) – runs once per pipeline to materialize processed data in the workspace.
- **Local parity** – optional local execution keeps evaluation/scoring aligned with the pipeline output.

Key behaviours:
- Accepts `uri_folder` input (directory containing CSV file(s)) and automatically loads all CSV files in the folder
- Drops ID columns (`RowNumber`, `CustomerId`, `Surname`).
- Label-encodes categorical features and persists encoders.
- Scales numeric columns via `StandardScaler` (fit on train, transform on test).
- Produces `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`, plus `encoders.pkl`, `scaler.pkl`, and `metadata.json`.

Usage (local parity):
```bash
python src/data_prep.py --config configs/data.yaml --output data/processed
```

## 2. HyperDrive Sweep (`run_hpo.py` + `aml/components/hpo.yaml`)

`run_hpo.py` is the primary entry point. It:
1. Automatically loads Azure ML configuration from `config.env`.
2. Reads the HyperDrive configuration from `configs/hpo.yaml` (metric, mode, budget, search space, early stopping).
3. Builds a DSL pipeline:
   - `data_prep` component → processed dataset (uri_folder output).
   - `train` component (using `aml/components/hpo.yaml`) launched with `.sweep(...)` so HyperDrive explores the search space.
4. Submits the job and prints the Studio URL.

`aml/components/hpo.yaml` maps the search space to CLI overrides on `train.py`:
```yaml
command: >-
  python train.py
  --data ${{inputs.processed_data}}
  --model-artifact-dir ${{outputs.model_output}}
  --parent-run-id-output ${{outputs.parent_run_id}}
  --model-type ${{inputs.model_type}}
  $[[--set rf.max_depth=${{inputs.rf_max_depth}}]]
  ...
```

Each trial:
- Trains one model type (from search space) with sampled hyperparameters.
- Applies the proposed hyperparameters via `--set model.param=value`.
- Logs metrics, parameters, and tags to MLflow (uses active run, no nested runs in Azure ML).
- Saves model artifacts to the component output (`model_output`).

HyperDrive picks the best trial based on the configured metric.

## 3. Extract Best Hyperparameters

After the sweep completes, extract the best hyperparameters and update the training configuration:

```bash
python src/extract_best_params.py --parent-run-id <PARENT_RUN_ID>
```

This script:
- Extracts best hyperparameters from the HPO sweep
- Automatically updates `configs/train.yaml` with the best model and hyperparameters
- Sets `models: [best_model]` to train only the best model found

## 4. Configuration (`configs/hpo.yaml` and `configs/train.yaml`)

**`configs/hpo.yaml`** defines the HyperDrive sweep:
- `metric` / `mode` – e.g., maximise F1.
- `budget.max_trials` / `budget.max_concurrent` – sweep size and parallelism.
- `early_stopping` – enables `MedianStoppingPolicy` during sweeps.
- `search_space` – candidate values per model (Random Forest, XGBoost, etc.).

**`configs/train.yaml`** contains:
- `training.models` – list of models to train (should be only the best model after HPO)
- `training.hyperparameters` – hyperparameters for each model (updated with best values after HPO)

`run_hpo.py` converts the search space lists into Azure ML `Choice` distributions.

## 5. Typical Workflow

1. Update `configs/hpo.yaml` with the search space for hyperparameter optimization.
2. Run `python run_hpo.py` to submit the HyperDrive sweep.
3. Track progress in Azure ML Studio; capture the parent run ID when complete.
4. Extract best hyperparameters and update config:
   ```bash
   python src/extract_best_params.py --parent-run-id <PARENT_RUN_ID>
   ```
5. Train the best model with optimized hyperparameters:
   ```bash
   python run_pipeline.py
   ```

That's the entire HyperDrive-first loop: configuration-driven sweeps, automatic best parameter extraction, and production-ready training.
