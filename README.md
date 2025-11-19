# Bank Customer Churn Prediction with Azure ML

Configuration-driven Azure ML workflow for bank customer churn prediction. Hyperparameter tuning runs from `notebooks/hpo_manual_trials.ipynb` on an Azure ML compute instance; MLflow captures every trial so you can evaluate, promote, and deploy the best model.

## Project Overview

- **Objective**: Predict whether a bank customer will churn (binary classification) using structured tabular features.
- **Model zoo**: Logistic Regression (`logreg`), Random Forest (`rf`), and XGBoost (`xgboost`) with shared preprocessing and MLflow logging.
- **Optimization path**: Run `notebooks/hpo_manual_trials.ipynb` on an Azure ML compute instance to submit sweeps defined in `configs/hpo.yaml`; the notebook exports the best configuration back into `configs/train.yaml`.
- **Production retraining**: Execute `run_pipeline.py` (or its notebook cell) to run the data prep → train pipeline end to end with the optimized settings, producing both pickle and MLflow artifacts under `outputs/`.
- **Deployment**: `notebooks/deploy_online_endpoint.ipynb` registers the MLflow bundle and deploys it to a managed online endpoint, with `sample-data.json` providing an encoded smoke-test payload.

### Key Features

- **HPO** – submit and monitor Azure ML sweeps from `notebooks/hpo_manual_trials.ipynb`.
- **Production retraining pipeline** – `run_pipeline.py` reuses the best config for fast, fixed-hyperparameter runs (data prep → train).
- **Managed online endpoint playbook** – `notebooks/deploy_online_endpoint.ipynb` registers MLflow bundles, deploys, smoke-tests, and cleans up endpoints.
- **Centralized configuration** – `configs/*.yaml` govern data prep, training defaults, MLflow settings, and sweep budgets for every workflow.
- **MLflow-first logging** – each trial/run captures parameters, metrics (F1, ROC-AUC, etc.), signatures, and artifacts for reproducibility.
- **Reproducible environments** – Dockerfile + pinned requirements keeps local, Docker, and Azure ML environments aligned; `aml/environments/environment.yml` mirrors the same image.

## Project Structure

```text
.
├── Job_train_job_OutputsAndLogs/      # Downloaded AML job logs & artifacts
├── aml/
│   ├── components/
│   │   ├── data_prep.yaml
│   │   └── train.yaml                 # Regular training component (fixed hyperparameters)
│   └── environments/
│       └── environment.yml            # Azure ML environment definition (Docker image reference)
├── artifacts/
│   └── mlflow_online_model/           # Latest packaged model for endpoint deployments
├── architecture.canvas                # Architecture diagram (VS Code canvas)
├── configs/
│   ├── data.yaml
│   ├── hpo.yaml
│   ├── mlflow.yaml
│   └── train.yaml
├── data/
│   └── README.md
├── docs/
│   ├── MASTER_PLAN.md                 # This file - project plan and guide
│   ├── TROUBLESHOOTING.md             # Troubleshooting guide for common issues
│   ├── dependencies.md
│   ├── pipeline_guide.md
│   ├── python_setup.md
│   └── setup_guide.md
├── logs/
│   └── artifacts/                     # Local copies of AML job logs/artifacts
├── mlruns/                            # Local MLflow tracking store
├── notebooks/
│   ├── deploy_online_endpoint.ipynb   # Managed online endpoint deployment workflow
│   ├── eda.ipynb                      # Exploratory data analysis
│   └── hpo_manual_trials.ipynb        # Manual HPO sweep orchestration in Azure ML
├── outputs/
│   ├── model_output/                  # Latest pipeline output bundle(s)
│   └── xgboost_mlflow/                # Current MLflow model directory
├── sample-data.json                   # Request payload for online endpoint smoke tests
├── setup/
│   ├── create_data_asset.py           # Script to create Azure ML data assets
│   ├── setup.sh                       # Bash script for Azure ML resource setup
│   ├── setup.ps1                      # PowerShell script for Azure ML resource setup
│   └── README.md                      # Setup documentation
├── src/
│   ├── data/                          # Data processing utilities
│   ├── models/                        # Model definitions (logreg, rf, xgboost)
│   ├── training/                      # Training utilities
│   ├── utils/                         # Utility modules
│   │   ├── azure_config.py            # Azure ML configuration loading
│   │   ├── config_loader.py           # YAML configuration loading
│   │   ├── env_loader.py              # Environment variable loading
│   │   ├── mlflow_utils.py            # MLflow integration utilities
│   │   └── ...
│   ├── data_prep.py                   # Data preprocessing script
│   ├── run_sweep_trial.py             # Helper script for HPO sweep trials
│   ├── train.py                       # Model training script
│   └── README.md
├── config.env                         # Environment configuration (not in git)
├── config.env.example                 # Example environment configuration template
├── Dockerfile                         # Docker image definition
├── hpo_utils.py                       # Hyperparameter optimization utilities
├── run_pipeline.py                    # Regular training pipeline orchestration script
├── README.md                          # Project overview
├── requirements.in                    # Core dependencies (source)
├── requirements.txt                   # Core dependencies (pinned)
├── dev-requirements.in                # Development dependencies (source)
├── dev-requirements.txt               # Development dependencies (pinned)
└── venv/                              # Local virtual environment (gitignored)
```

## What’s Implemented So Far

### 1. Environment Setup

```bash
# Clone the repository
git clone <repo-url>
cd customer-churn-prediction-azureml

# Create a Python 3.9 virtual environment (matches docker + AML images)
python3.9 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -r dev-requirements.txt  # optional tooling (ruff, black, etc.)
```

### 2. Azure ML Configuration

`config.env` stores the workspace, resource group, ACR, and data asset names that all scripts/notebooks consume via `load_azure_config()`. Copy `config.env.example`, fill in your values, then authenticate with `az login`. No additional sourcing is required.

### 3. Configuration Files

`configs/data.yaml`, `train.yaml`, `hpo.yaml`, and `mlflow.yaml` drive data prep, training defaults, sweep budgets, and experiment naming. The notebook exports the winning HPO settings back into `train.yaml`, and `run_pipeline.py` consumes the same file for production retraining.

### 4. Workflow Delivered

1. **HPO** – `notebooks/hpo_manual_trials.ipynb` submits Azure ML sweeps defined in `configs/hpo.yaml`, logs every trial to MLflow, and exports the best config to `configs/train.yaml`.
2. **Production retraining** – `run_pipeline.py` (or its notebook cell) runs the data_prep → train pipeline with the exported settings, producing pickle + MLflow artifacts under `outputs/`.
3. **Managed online endpoint** – `notebooks/deploy_online_endpoint.ipynb` discovers the newest MLflow bundle, registers it, deploys to a managed endpoint, and verifies predictions with `sample-data.json`.

Together these steps complete the train → optimize → deploy loop already implemented in this repo.

### 5. Inspect Experiments with MLflow

If you mirror the MLflow tracking URI locally, launch the UI to explore runs:

```bash
mlflow ui --backend-store-uri "${MLFLOW_TRACKING_URI}" --port 5000
```

## Running on Azure ML

- **Notebook-driven HPO**: `notebooks/hpo_manual_trials.ipynb` builds sweeps from `configs/hpo.yaml`, registers `aml/components/data_prep.yaml` + `train.yaml`, and logs results to MLflow.
- **Fixed-hyperparameter pipeline**: `run_pipeline.py` submits the training component (`aml/components/train.yaml`) using settings from `configs/train.yaml`.
- **Online deployment**: `notebooks/deploy_online_endpoint.ipynb` registers the MLflow bundle and deploys it to a managed endpoint.

All flows source workspace/data settings from `config.env`.

## Documentation

- `docs/MLZoomcamp-Project1-ProjectPlan-v2.md` – project plan and history
- `docs/pipeline_guide.md` – deep dive into the scripts and components
- `docs/setup_guide.md` – step-by-step setup instructions
- `docs/dependencies.md` – guidance on dependency management and pinning
- `docs/TROUBLESHOOTING.md` – common errors and solutions

## Quick Reference

**HPO > Train > Deploy (summary):**

1. `notebooks/hpo_manual_trials.ipynb` → submit sweeps, export best config.
2. `python run_pipeline.py` → run production training.
3. `notebooks/deploy_online_endpoint.ipynb` → register + deploy MLflow bundle, smoke test predictions.

---

Need help or want to extend the workflow? Open an issue or start a discussion!
