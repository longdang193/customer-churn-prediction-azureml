# Project Plan & Guide

This document outlines the plan and structure for building the Bank Customer Churn MLOps pipeline. It serves as both a historical record of development and a guide to setting up the project from scratch.

## Final Project Structure

```
.
├── aml/
│   └── components/
│       ├── data_prep.yaml
│       ├── extract_best_params.yaml
│       ├── train.yaml          # Regular training component (fixed hyperparameters)
│       └── hpo.yaml            # HPO training component (hyperparameter sweeps)
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
   - **Important**: The data asset must be registered as `uri_folder` (directory containing CSV file(s))
   - The `data_prep` component accepts `uri_folder` input and automatically loads all CSV files in the folder
   - If multiple CSV files are present, they will be concatenated together (useful for splitting large datasets across multiple files)
   - Example: Register a folder containing `churn.csv` (or multiple CSV files) as `uri_folder` type

### Step 3: Perform EDA

1.  **Create `notebooks/` directory**.
2.  Develop `notebooks/eda.ipynb` to analyze the sample data, documenting findings on distributions, correlations, and data quality.

### Step 4: Create Core Scripts

1.  **Create `src/` directory** and `src/__init__.py`.
2.  **Develop Core Scripts**:
    -   `src/data_prep.py`: Data preprocessing stage
    -   `src/train.py`: Model training with MLflow logging (supports `--model-type` for HPO mode, `--set model.param=value` for hyperparameter overrides). In regular mode, models are determined from `configs/train.yaml` → `training.models`.
    -   `src/extract_best_params.py`: Extract best hyperparameters and model type from MLflow sweep runs, automatically updates `configs/train.yaml`
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
    - Uses `hpo_utils.py` to load HPO config from `configs/hpo.yaml`
    - Uses `aml/components/hpo.yaml` component for training
    - Each trial trains one model type (categorical hyperparameter) with sampled hyperparameters
    - Automatically loads `config.env` for Azure ML configuration
    - Use for: First-time optimization, exploring search spaces

2.  **Create `run_pipeline.py`**: Fixed-hyperparameter training pipeline
    - Uses `aml/components/train.yaml` component for training
    - Simple pipeline: data_prep → train
    - Models determined from `configs/train.yaml` → `training.models`
    - Automatically loads `config.env` for Azure ML configuration
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
    -   `aml/components/train.yaml`: Training component for regular training (fixed hyperparameters from config)
    -   `aml/components/hpo.yaml`: Training component for HPO sweeps (accepts hyperparameters as inputs for sweep)
    -   `aml/components/extract_best_params.yaml`: Component to extract best hyperparameters from MLflow sweep runs (optional, can be run locally)

### Step 11: Set up Azure ML Workspace

1.  **Create Azure ML workspace** (via Azure Portal or CLI)
2.  **Create compute cluster**: `cpu-cluster` for training jobs
3.  **Create data asset**: Register raw dataset as `uri_folder` type (name and version configured in `config.env`)
4.  **Set up authentication**: Create `config.env` file with Azure credentials:
    - `AZURE_SUBSCRIPTION_ID`
    - `AZURE_RESOURCE_GROUP`
    - `AZURE_WORKSPACE_NAME`
    - `AZURE_RAW_DATA_ASSET`: Name of registered data asset
    - `AZURE_RAW_DATA_VERSION`: Version of data asset
    - **Note**: Both `run_pipeline.py` and `run_hpo.py` automatically load `config.env`

### Step 12: Build Docker Image and Register Azure ML Environment

1.  **Build Docker image locally**:
    ```bash
    # Build the Docker image
    docker build -t bank-churn:latest .
    
    # Test the image locally (optional)
    docker run --rm bank-churn:latest python -c "import pandas; print('OK')"
    ```

2.  **Push Docker image to Azure Container Registry (ACR)**:
    ```bash
    # Login to ACR (if not already logged in)
    az acr login --name <your-acr-name>
    
    # Tag the image for ACR
    docker tag bank-churn:latest <your-acr-name>.azurecr.io/bank-churn:latest
    
    # Push to ACR
    docker push <your-acr-name>.azurecr.io/bank-churn:latest
    
    # Verify image is pushed (optional)
    az acr repository show-tags --name <your-acr-name> --repository bank-churn
    ```

3.  **Register environment in Azure ML** pointing to the ACR image:
    
    Update the `aml/environments/environment.yml` file with your ACR name:
    ```yaml
    $schema: https://azuremlschemas.azureedge.net/latest/environment.schema.json
    name: bank-churn-env
    version: "1"
    image: <your-acr-name>.azurecr.io/bank-churn:latest
    description: Environment for churn prediction pipeline
    ```
    
    Replace `<your-acr-name>` with your actual Azure Container Registry name, then register the environment using Azure CLI:
      ```bash
    az ml environment create --file aml/environments/environment.yml \
      --resource-group <resource-group> \
      --workspace-name <workspace-name>
    ```

4.  **Verify environment registration**:
    ```bash
    az ml environment show --name bank-churn-env --version 1 \
      --resource-group <resource-group> \
      --workspace-name <workspace-name>
    ```
    The command returns JSON describing the environment, including `image` and metadata. A successful response confirms that the environment is registered.
    
    **Note**: All components reference `azureml:bank-churn-env:1` for consistent dependencies

### Step 13: Test Pipeline Execution

1.  **Test regular training pipeline**: 
    ```bash
    python run_pipeline.py
    ```
    - Uses data asset configured in `config.env` (`AZURE_RAW_DATA_ASSET` and `AZURE_RAW_DATA_VERSION`)
    - Runs data prep → train pipeline using `train.yaml` component
    - Trains models specified in `configs/train.yaml` → `training.models`
    - Verify job submission in Azure ML Studio

2.  **Test HPO pipeline** (optional, for hyperparameter optimization):
    ```bash
    python run_hpo.py
    ```
    - Uses data asset configured in `config.env`
    - Runs data prep → HPO sweep using `hpo.yaml` component
    - Configure budget in `configs/hpo.yaml` (e.g., `max_trials: 2-3` for testing)
    - Verify sweep job submission in Azure ML Studio

3.  **Verify outputs**:
    - **Azure ML Studio**: Check job status, logs, and outputs
    - **MLflow**: View runs, metrics, parameters, and artifacts (if MLflow tracking is configured)
    - **Model artifacts**: Check that models are saved to outputs directory

---

## Pipeline Entry Points

### `run_hpo.py` - Hyperparameter Optimization

- Runs HyperDrive sweep to find best hyperparameters
- Uses `aml/components/hpo.yaml` component for training
- Loads HPO configuration from `configs/hpo.yaml`
- Each trial trains one model type (from search space) with sampled hyperparameters
- Efficient: 1 model per trial (reduces compute cost)
- Automatically loads `config.env` for Azure ML configuration
- **Use when**: First-time optimization, exploring new search spaces
- **Outputs**: Best hyperparameters and model type from sweep results (logged to MLflow)

### `run_pipeline.py` - Fixed-Hyperparameter Training

- Uses `aml/components/train.yaml` component for training
- Trains models specified in `configs/train.yaml` → `training.models`
- Uses hyperparameters from `configs/train.yaml` → `training.hyperparameters`
- Fast execution (no sweep overhead)
- Simple pipeline: data_prep → train
- Automatically loads `config.env` for Azure ML configuration
- **Use when**: Quick retraining, production deployments, after HPO
- **Outputs**: Trained model artifacts and MLflow run ID

### Typical Workflow

1. Run `run_hpo.py` to find best hyperparameters (uses `hpo.yaml` component)
2. Extract best hyperparameters and update config: `python src/extract_best_params.py --parent-run-id <PARENT_RUN_ID>`
   - This automatically updates `configs/train.yaml` with best model and hyperparameters
   - Sets `models: [best_model]` to train only the best model
3. Run `run_pipeline.py` for quick training with optimized hyperparameters (uses `train.yaml` component)

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
- **Model Selection**: 
  - Regular mode: Models determined from `configs/train.yaml` → `training.models`
  - HPO mode: Single model type specified via `--model-type` CLI argument
- Models saved as pickle files in Azure ML (automatically captured as artifacts)
- Models logged to MLflow with metrics, parameters, and tags
- Models support MLflow pyfunc deployment (no custom scoring script needed)

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
