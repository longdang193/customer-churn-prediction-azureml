# Configuration Files

Centralized YAML inputs keep the data prep, training pipeline, and HPO flows in sync. Only four files are active today:

- `data.yaml` – input/output locations, target column, split ratios, categorical columns used by `src/data_prep.py` and the `data_prep` Azure ML component.
- `train.yaml` – which models to train plus their hyperparameters, class weights, SMOTE toggles, etc. Consumed by `src/train.py`, `run_pipeline.py`, and the training component.
- `hpo.yaml` – search-space definitions, sweep budget, and early-stopping rules loaded by `notebooks/hpo_manual_trials.ipynb` via `hpo_utils.py`.
- `mlflow.yaml` – the MLflow experiment name referenced by both CLI scripts and pipeline jobs.

## How the configs are used

| Flow | Configs referenced | Notes |
| --- | --- | --- |
| Local/AML data prep (`src/data_prep.py`, `aml/components/data_prep.yaml`) | `data.yaml` | Controls source asset, drops, categorical encoding, splits. |
| Production training (`run_pipeline.py`, `src/train.py`, AML `train.yaml` component) | `train.yaml`, `mlflow.yaml` | `training.models` drives which estimators run; other keys map directly to CLI `--set` overrides. |
| Manual sweeps (`notebooks/hpo_manual_trials.ipynb`) | `hpo.yaml`, `train.yaml`, `mlflow.yaml` | Notebook builds search spaces from `hpo.yaml`, then exports the best settings back into `train.yaml`. |

## Overriding values

- Point a script at an alternate file: `python src/train.py --config configs/train_smoke.yaml`.
- Override individual keys without editing YAML: `python src/train.py --set training.use_smote=true`.
- Azure ML sweeps automatically merge `hpo.yaml` definitions with the CLI overrides that `hpo_utils.py` emits.

### Precedence (highest → lowest)

1. CLI / notebook overrides (`--config`, `--set`, or Azure ML sweep parameters)
2. Values stored in these YAML files
3. Hard-coded defaults inside the scripts

Keep any experimental variations under `configs/` so they remain versioned and easy to diff. Remove unused files once a workflow is decommissioned to avoid stale settings.
