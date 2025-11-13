# Project Plan & Guide

This document outlines the plan and structure for building the Bank Customer Churn MLOps pipeline. It serves as both a historical record of development and a guide to setting up the project from scratch.

## Final Project Structure

```
.
├── aml/
│   └── components/
│       ├── data_prep.yaml
│       └── train.yaml
├── configs/
│   ├── data.yaml
│   ├── hpo.yaml
│   ├── mlflow.yaml
│   └── train.yaml
├── data/
├── docs/
│   ├── MLZoomcamp-Project1-ProjectPlan-v2.md
│   ├── dependencies.md
│   ├── pipeline_guide.md
│   └── setup_guide.md
├── run_hpo.py
├── run_pipeline.py
├── setup/
│   ├── create_data_asset.py
│   ├── setup.sh
│   ├── start_compute.sh
│   └── ...
├── src/
│   ├── models/
│   ├── data_prep.py
│   ├── extract_best_params.py
│   └── train.py
├── Dockerfile
├── README.md
├── requirements.in / requirements.txt
└── dev-requirements.in / dev-requirements.txt
```

---

## Project Setup: Step-by-Step Guide

### Step 1: Set up the Project (Repo + README)

1.  **Initialize Git**: Create a new repository.
2.  **Create `README.md`**: Add a basic project overview.
3.  **Create `.gitignore`**: Add entries for temporary files, data, models, and environment files (`.env`, `__pycache__`, `*.csv`, `models/`, `evaluation/`, etc.).

### Step 2: Add the Dataset

1.  **Create `data/` directory**.
2.  Upload the dataset to Azure ML as a data asset (configured in `config.env`).

### Step 3: Perform EDA

1.  **Create `notebooks/` directory**.
2.  Develop `notebooks/eda.ipynb` to analyze the sample data, documenting findings on distributions, correlations, and data quality.

### Step 4: Create Core Scripts

1.  **Create `src/` directory** and `src/__init__.py`.
2.  **Develop Core Scripts**:
    -   `src/data_prep.py`: Data preprocessing stage
    -   `src/train.py`: Model training with MLflow logging (supports `--model-type` for HPO, `--set model.param=value` for overrides)
    -   `src/extract_best_params.py`: Extract best hyperparameters and model type from MLflow sweep runs, automatically updates config
3.  **Create `src/models/` package**: Individual model definitions (logreg, rf, xgboost)

### Step 5: Centralize Configuration

1.  **Create `configs/` directory**.
2.  **Create YAML files**:
    -   `configs/data.yaml`: Data prep settings (input/output paths, target column, columns to remove, categorical columns, train/test split)
    -   `configs/train.yaml`: Training settings (models, hyperparameters, class weights)
    -   `configs/hpo.yaml`: HPO configuration (search space, metric, budget, early-stopping)
    -   `configs/mlflow.yaml`: MLflow experiment name
3.  **Create `src/config_loader.py`**: Utility functions to load YAML files and get nested config values using dot notation
4.  **Create `hpo_utils.py`**: Utility functions for HPO (loads configs, builds parameter space)
5.  **Update Scripts**: Ensure scripts pull defaults from config but remain overrideable via CLI arguments

### Step 6: Create Pipeline Orchestration Scripts

1.  **Create `run_hpo.py`**: HPO pipeline with HyperDrive sweep
    - Uses `hpo_utils.py` to load HPO config from `configs/train.yaml`
    - Each trial trains one model type (categorical hyperparameter)
    - Use for: First-time optimization, exploring search spaces

2.  **Create `run_pipeline.py`**: Fixed-hyperparameter training pipeline
    - Simple pipeline: data_prep → train
    - Use for: Quick retraining, production deployments, after HPO

### Step 7: Declare Dependencies

1.  **Create `requirements.in`**: Core libraries
    - Data & ML: pandas, numpy, scikit-learn, imbalanced-learn, xgboost
    - MLflow: mlflow (experiment tracking)
    - Azure ML: azure-ai-ml, azure-identity
    - Utilities: pyyaml, matplotlib, seaborn, scipy
    - Optional: fastapi (for serving API)
2.  **Create `dev-requirements.in`**: Development tools
    - Dependency management: pip-tools
    - Code quality: black, ruff, isort, pre-commit
3.  **Compile Pinned Versions** (optional but recommended):
```bash
    pip install pip-tools
pip-compile requirements.in -o requirements.txt
pip-compile dev-requirements.in -o dev-requirements.txt
```
4.  **Sync environments**: Docker image, local venv, and AML compute all install from compiled requirements


### Step 9: Define the Docker Image

1.  **Create `Dockerfile`** in the project root
2.  Start from slim Python base image, copy `requirements.txt` first (for layer caching), then application code
3.  **Build and Test**:
```bash
docker build -t bank-churn:latest .
docker run --rm -v "$PWD:/app" -w /app bank-churn:latest bash -lc "python -m src.data_prep --help"
```

### Step 10: Build AML Components

1.  **Create `aml/components/` directory**
2.  **Create component YAML files**:
    -   `aml/components/data_prep.yaml`: Data preprocessing component
    -   `aml/components/train.yaml`: Training component (supports sweep overrides, model_type, and model-specific hyperparameters)
    -   `aml/components/extract_best_params.yaml`: Component to extract best hyperparameters from MLflow sweep runs (used in HPO pipeline)

### Step 11: Set up Azure ML Workspace

1.  **Create Azure ML workspace** (via Azure Portal or CLI)
2.  **Create compute cluster**: `cpu-cluster` for training jobs
3.  **Create data asset**: Register raw dataset as `bank-churn-raw` (version 1)
4.  **Set up authentication**: Configure `.env` file with Azure credentials:
    - `AZURE_SUBSCRIPTION_ID`
    - `AZURE_RESOURCE_GROUP`
    - `AZURE_WORKSPACE_NAME`

### Step 12: Register Azure ML Environment

1.  **Create and register environment**: `bank-churn-env:1`
    - **Option A (Python SDK - Dockerfile)**: Register environment from Dockerfile:
      ```python
      from azure.ai.ml import MLClient
      from azure.ai.ml.entities import Environment, BuildContext
      from azure.identity import DefaultAzureCredential
      
      ml_client = MLClient(DefaultAzureCredential(), subscription_id, resource_group, workspace_name)
      env = Environment(
          name="bank-churn-env",
          version="1",
          build=BuildContext(path=".", dockerfile_path="Dockerfile"),
          description="Environment for churn prediction pipeline"
      )
      ml_client.environments.create_or_update(env)
      ```
    - **Option B (Python SDK - conda.yml)**: Create `conda.yml` with pip section, then register:
      ```python
      env = Environment(
          name="bank-churn-env",
          version="1",
          conda_file="conda.yml",  # Contains pip section with requirements.txt
          description="Environment for churn prediction pipeline"
      )
      ```
    - **Option C (Azure CLI)**: `az ml environment create --file environment.yml`
    - **Option D (Azure ML Studio)**: Create environment via UI from Dockerfile or conda file
2.  **Verify environment registration**:
    - **Azure CLI (recommended)**:
      ```bash
      az ml environment show --name bank-churn-env --version 1
      ```
      The command returns JSON describing the environment, including `image`, `conda_file`, and metadata. A successful response confirms that the environment is registered.
    - **Python SDK (optional)**:
      ```python
      env = ml_client.environments.get(name="bank-churn-env", version="1")
      print(f"Environment: {env.name}:{env.version}")
      print(f"Description: {env.description}")
      print(f"Image: {env.image or 'None'}")
      if env.build:
          print(f"Build context: {env.build.path}, Dockerfile: {env.build.dockerfile_path}")
      if env.conda_file:
          print(f"Conda file: {env.conda_file}")
      ```
    - **Azure ML Studio**: Navigate to Environments -> `bank-churn-env:1` and confirm details in the UI
3.  **Verify dependencies**: Ensure all dependencies from `requirements.txt` are available in the environment
4.  **Note**: All components reference `azureml:bank-churn-env:1` for consistent dependencies

### Step 13: Test Pipeline Execution

1.  **Test data prep component**: Run data preprocessing pipeline
2.  **Test training pipeline**: Run `run_pipeline.py` with sample data
3.  **Test HPO pipeline**: Run `run_hpo.py` with small budget (e.g., 2-3 trials)
4.  **Verify outputs**: Check MLflow runs, model artifacts, and metrics

---

## Pipeline Entry Points

### `run_hpo.py` - Hyperparameter Optimization

- Runs HyperDrive sweep to find best hyperparameters
- Each trial trains one model type (logreg, rf, or xgboost) with its hyperparameters
- Efficient: 1 model per trial (3x reduction in compute cost)
- **Use when**: First-time optimization, exploring new search spaces
- **Outputs**: Best hyperparameters and model type from sweep results

### `run_pipeline.py` - Fixed-Hyperparameter Training

- Trains models with known good hyperparameters from `configs/train.yaml`
- Fast execution (no sweep overhead)
- Simple pipeline: data_prep → train
- **Use when**: Quick retraining, production deployments, after HPO
- **Outputs**: Trained model artifacts and MLflow run ID

### Typical Workflow

1. Run `run_hpo.py` to find best hyperparameters
2. Extract best hyperparameters and update config: `src/extract_best_params.py`
3. Update `configs/train.yaml` with best hyperparameters
4. Run `run_pipeline.py` for quick training with optimized hyperparameters

---

## Hyperparameter Strategy

- **HPO Approach**: `model_type` is treated as a categorical hyperparameter
  - Each trial trains only one model type (logreg, rf, or xgboost)
  - Model-specific hyperparameters included conditionally based on `model_type`
  - Efficient: 1 model per trial (3x reduction in compute cost)
  - Best model type logged as `model_type` tag in MLflow

- **Model Logging**: Models logged with MLflow including:
  - `pip_requirements`: Embedded dependencies for portability
  - `signature`: Model input/output schema
  - `input_example`: Sample input for testing
  - Azure ML deployment uses registered environment (`azureml:bank-churn-env:1`)

---

## Implementation Details

### Model Training & Logging

- Three models supported: Logistic Regression (`logreg`), Random Forest (`rf`), XGBoost (`xgboost`)
- Models logged with `mlflow.sklearn.log_model()` including `pip_requirements`, `signature`, `input_example`
- Models support MLflow pyfunc deployment (no custom scoring script needed)
- `predict_probabilities()` raises explicit error if model lacks `predict_proba` (no misleading fallbacks)

### Deployment Strategy

- **MLflow pyfunc deployment**: Models deployed directly to Azure ML online endpoints
- **Dual environment support**:
  - Azure ML deployment: Uses registered environment (`azureml:bank-churn-env:1`)
  - Local serving: Uses embedded `pip_requirements` (e.g., `mlflow models serve`)

---

## Future Plans

- **Step 14: Add a Makefile** for common commands
- **Step 15: Implement staging/prod deployment** with promotion control
  - Create `deploy.py` with staging/prod support
  - Add `--stage` flag to `run_pipeline.py --deploy`
  - Implement smoke tests and promotion workflow
- **Step 16: Add scheduled pipeline execution**
  - Create `schedule_pipeline.py` for Azure ML Schedule management
  - Support daily, weekly, monthly, and cron-based schedules
