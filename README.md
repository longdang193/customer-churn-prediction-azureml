# Bank Customer Churn Prediction with Azure ML

This repo delivers a configuration-driven Azure ML workflow for bank customer churn prediction. Hyperparameter tuning now runs straight from the `notebooks/hpo_manual_trials.ipynb` notebook on an Azure ML compute instance, while MLflow tracks every trial so you can evaluate, promote, and score the best model with confidence.

## Project Overview

- **Problem**: binary classification of bank customers into churn / retain classes.
- **Models**: Supports three models - Logistic Regression (`logreg`), Random Forest (`rf`), and XGBoost (`xgboost`)
- **Approach**: open `notebooks/hpo_manual_trials.ipynb` on your Azure ML compute instance and execute the notebook to iterate through the search space defined in `configs/hpo.yaml`, logging each trial to MLflow.
- **Follow-up**: after the notebook finishes, extract or copy the best hyperparameters and train the optimized model. Models are saved as pickle files and can be deployed.

### Key Features

- **Notebook-driven HPO** – orchestrate one trial per cell execution directly from `hpo_manual_trials.ipynb`
- **Pipeline for retraining** – keep `run_pipeline.py` for fast, single-shot retraining once the best configuration is known
- **Configuration driven** – YAML files in `configs/` centralize data prep, training defaults, MLflow settings, and the HyperDrive search space
- **MLflow integration** – each trial logs parameters, metrics (e.g. `f1`), and artifacts, enabling reproducible model promotion
- **Robust preprocessing** – shared utilities ensure the same feature engineering is applied during training and inference
- **Reproducible environment** – a Dockerfile and pinned `requirements.txt` make local, Docker, and AML environments consistent

## Project Structure

```
.
├── aml/
│   └── components/              # Azure ML component definitions
│       ├── data_prep.yaml      # Data preprocessing component
│       ├── extract_best_params.yaml  # Extract best params component
│       └── train.yaml          # Training component (regular training with fixed hyperparameters)
├── configs/                     # Configuration files
│   ├── data.yaml               # Data preparation settings
│   ├── hpo.yaml                # Hyperparameter optimization (test/production)
│   ├── mlflow.yaml             # MLflow tracking configuration
│   ├── train.yaml              # Training config (best model + hyperparameters)
│   └── README.md               # Config documentation
├── data/                       # Data directory (local artifacts, gitignored)
├── docs/                       # Documentation
│   ├── MLZoomcamp-Project1-ProjectPlan-v2.md
│   ├── dependencies.md
│   ├── pipeline_guide.md
│   ├── setup_guide.md
│   └── TROUBLESHOOTING.md
├── notebooks/                  # Jupyter notebooks
│   ├── hpo_manual_trials.ipynb  # Notebook-based HPO workflow for compute instances
│   └── eda.ipynb
├── src/                        # Source code
│   ├── config_loader.py        # Configuration loading utilities
│   ├── data_prep.py            # Data preprocessing script
│   ├── extract_best_params.py  # Extract best params from HPO and update config
│   ├── train.py                # Model training script
│   ├── models/                 # Model definitions (logreg, rf, xgboost)
│   └── README.md               # Source code documentation
├── setup/                      # Setup scripts
│   ├── setup.sh/.ps1          # Initial Azure ML setup
│   └── README.md               # Common Azure ML commands
├── hpo_utils.py                # HPO configuration utilities
├── run_pipeline.py             # Submit training pipeline
├── config.env.example          # Example environment configuration
├── Dockerfile                  # Docker image definition
├── requirements.in             # Python dependencies (source)
├── requirements.txt            # Python dependencies (pinned)
├── dev-requirements.in         # Development dependencies (source)
└── dev-requirements.txt        # Development dependencies (pinned)
```

**Note**: The following folders contain local artifacts and are gitignored:

- `data/processed*` - Processed data outputs
- `evaluation/` - Evaluation outputs
- `models/` - Trained model artifacts
- `predictions/` - Prediction outputs
- `logs/` - Log files
- `mlruns/` - MLflow local tracking data

## Getting Started

### 1. Environment Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd customer-churn-prediction-azureml

# (Optional) create a virtual environment
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Azure ML

Create a `config.env` file with your workspace details:

```bash
AZURE_SUBSCRIPTION_ID=<subscription>
AZURE_RESOURCE_GROUP=<resource-group>
AZURE_WORKSPACE_NAME=<workspace>
DATA_ASSET_FULL=<data-asset-name>
DATA_VERSION=<data-version>
```

Authenticate with Azure (`az login`) before submitting jobs.

**Note**:

- Data is loaded from Azure ML data assets. Configure the data asset name and version in `config.env`.
- `run_pipeline.py` automatically loads `config.env`, and the `hpo_manual_trials.ipynb` notebook calls `load_dotenv`, so you usually don't need to source it manually (though you can if you prefer).

### 3. Configuration Files

The project uses YAML configuration files in `configs/`:

- **`hpo.yaml`** – Hyperparameter optimization settings (test/production sections)
- **`train.yaml`** – Training configuration with best model and hyperparameters (test/production sections)
- **`data.yaml`** – Data preparation settings
- **`mlflow.yaml`** – MLflow tracking configuration

**Important**: Both `hpo.yaml` and `train.yaml` have test and production sections. Uncomment the section you want to use.

### 4. Typical Workflow

#### Step 1: Run HPO (Find Best Model and Hyperparameters)

```bash
python run_hpo.py
```

This will:

- Run a HyperDrive sweep using the `hpo.yaml` component
- Each trial trains one model type (from the search space in `configs/hpo.yaml`) with sampled hyperparameters
- Models to optimize are specified in `configs/hpo.yaml` → `search_space` (e.g., `rf`, `xgboost`)
- Selects the best trial based on the configured metric (default: F1 score)
- Logs all trials to MLflow

**After HPO completes:**

1. Get the parent run ID from Azure ML Studio (the sweep job run ID)
2. Extract best hyperparameters and update config:

   ```bash
   python src/extract_best_params.py --parent-run-id <PARENT_RUN_ID>
   ```

   This automatically:
   - Extracts best hyperparameters from the HPO sweep
   - Updates `configs/train.yaml` with the best model and hyperparameters
   - Sets `models: [best_model]` (only one model)

#### Step 2: Train Best Model (After HPO)

```bash
python run_pipeline.py
```

This will:

- Use the best model and hyperparameters from `configs/train.yaml` (updated in Step 1)
- Run data prep → train pipeline using the `train.yaml` component
- Train models specified in `configs/train.yaml` → `training.models` (should be only the best model after HPO)
- Save the trained model as a pickle file
- Fast single run (no HPO overhead)
- Perfect for production retraining

**Note**: The `train.yaml` config file should contain only the best model from HPO. The `extract_best_params.py` script automatically sets `models: [best_model]` in the config.

### 5. Configuration: Test vs Production

Both `configs/hpo.yaml` and `configs/train.yaml` have test and production sections:

**For Testing:**

- Smaller budgets (fewer trials)
- Limited search spaces
- Faster execution

**For Production:**

- Larger budgets (more trials)
- Comprehensive search spaces
- Better optimization

Switch between them by commenting/uncommenting the appropriate section in each file.

### 6. Deploy and Score Models

Models are saved as pickle files during training and can be deployed:

- **Model files**: Saved to Azure ML outputs directory during training
- **MLflow tracking**: All runs are tracked in MLflow for experiment management
- **Model artifacts**: Models and metadata are automatically captured by Azure ML

### 7. Inspect Experiments with MLflow

If you mirror the MLflow tracking URI locally, launch the UI to explore runs:

```bash
mlflow ui --backend-store-uri "${MLFLOW_TRACKING_URI}" --port 5000
```

## Running on Azure ML

All pipelines load component specs from `aml/components/` and read configuration from:

- **HPO Pipeline** (`run_hpo.py`): Uses `aml/components/hpo.yaml` component and `configs/hpo.yaml` for sweep configuration
- **Regular Pipeline** (`run_pipeline.py`): Uses `aml/components/train.yaml` component and `configs/train.yaml` for training configuration

Both pipelines automatically load `config.env` for Azure ML workspace and data asset configuration.

## Documentation

- `docs/MLZoomcamp-Project1-ProjectPlan-v2.md` – project plan and history
- `docs/pipeline_guide.md` – deep dive into the scripts and components
- `docs/setup_guide.md` – step-by-step setup instructions
- `docs/dependencies.md` – guidance on dependency management and pinning
- `docs/TROUBLESHOOTING.md` – common errors and solutions

## Quick Reference

**HPO Workflow:**

```bash
# 1. Run HPO (uses hpo.yaml component)
python run_hpo.py

# 2. Extract best params and update config (after HPO completes)
python src/extract_best_params.py --parent-run-id <PARENT_RUN_ID>

# 3. Train best model (uses train.yaml component)
python run_pipeline.py
```

**Note**: Both scripts automatically load `config.env` for Azure ML configuration. No need to source it manually.

**Optional flags for `extract_best_params.py`:**

- `--output <file>` – Save params to JSON file (optional)
- `--no-update-config` – Skip config update (extract only)
- `--dry-run` – Preview changes without updating
- `--config <path>` – Specify config file (default: `configs/train.yaml`)

---

Need help or want to extend the workflow? Open an issue or start a discussion!
