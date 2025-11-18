# Pipeline Guide

This guide walks through the churn prediction pipeline workflow, from data preparation through hyperparameter optimization and production training.

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

## 2. Hyperparameter Optimization (Notebook-Driven Workflow)

The HPO workflow is managed through `notebooks/hpo_manual_trials.ipynb`, which provides:

- **Manual control**: Submit sweeps per model type with custom configurations
- **Result analysis**: Built-in cells to load previous sweeps and analyze best models
- **Config export**: Automatically export best model configuration to `configs/train.yaml`
- **Pipeline integration**: Run training pipeline directly from the notebook

### Workflow Steps

1. **Setup**: Configure Azure ML client and load HPO configuration from `configs/hpo.yaml`
2. **Data**: Set training data URI (from previous data prep job or environment variable)
3. **Configure Sweeps**: Build sweep jobs per model type from `configs/hpo.yaml`
4. **Submit Sweeps**: Submit sweep jobs to Azure ML (or load previous submissions)
5. **Analyze Results**: Find best model and export configuration to `configs/train.yaml`
6. **Run Pipeline**: Train the best model using the exported configuration

### How It Works

The notebook uses `src/run_sweep_trial.py` as a helper script that:

- Invokes `train.py` with sweep-managed hyperparameters
- Converts sweep parameter names (e.g., `rf_n_estimators`) to `train.py` format (e.g., `rf.n_estimators`)
- Handles training-level parameters (e.g., `use_smote`, `class_weight`, `random_state`)

Each trial:

- Trains one model type (from search space) with sampled hyperparameters
- Applies the proposed hyperparameters via `--set model.param=value`
- Logs metrics, parameters, and tags to MLflow (uses active run, no nested runs in Azure ML)
- Saves model artifacts to the component output (`model_output`)

Azure ML picks the best trial based on the configured metric in `configs/hpo.yaml`.

## 3. Configuration (`configs/hpo.yaml` and `configs/train.yaml`)

**`configs/hpo.yaml`** defines the HyperDrive sweep:

- `metric` / `mode` – e.g., maximise F1
- `budget.max_trials` / `budget.max_concurrent` – sweep size and parallelism
- `early_stopping` – enables `MedianStoppingPolicy` during sweeps
- `search_space` – candidate values per model (Random Forest, XGBoost, Logistic Regression)
- `timeouts` – total and per-trial timeout limits

**`configs/train.yaml`** contains:

- `training.models` – list of models to train (should be only the best model after HPO)
- `training.hyperparameters` – hyperparameters for each model (updated with best values after HPO)
- Training-level parameters: `use_smote`, `class_weight`, `random_state`

The notebook converts the search space lists into Azure ML `Choice` distributions when building sweep jobs.

## 4. Production Training Pipeline

After HPO completes and the best model configuration is exported:

**Option A: Run from notebook** (recommended):

1. Run the "Run Training Pipeline" cell in `notebooks/hpo_manual_trials.ipynb`
2. This automatically uses the exported `configs/train.yaml` configuration

**Option B: Run from command line**:

   ```bash
   python run_pipeline.py
   ```

The pipeline:

- Uses `aml/components/train.yaml` component for training
- Trains models specified in `configs/train.yaml` → `training.models`
- Uses hyperparameters from `configs/train.yaml` → `training.hyperparameters`
- Produces production-ready model artifacts

## 5. Typical Workflow

1. **Initial Setup** (first time or when retuning):
   - Open `notebooks/hpo_manual_trials.ipynb`
   - Configure and submit HPO sweeps
   - Analyze results and export best model configuration
   - Run production training pipeline

2. **Regular Retraining** (with fixed hyperparameters):
   - Update data asset if needed
   - Run `python run_pipeline.py` (skip HPO)

3. **Periodic Re-optimization**:
   - Run HPO again if data distribution changes or performance degrades
   - Export new best configuration
   - Run production training pipeline

That's the entire workflow: notebook-driven HPO sweeps, automatic config export, and production-ready training.
