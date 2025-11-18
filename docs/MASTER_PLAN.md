# Project Plan & Guide

This document outlines the plan and structure for building the Bank Customer Churn MLOps pipeline. It serves as both a historical record of development and a guide to setting up the project from scratch.

## Final Project Structure

```text
.
├── aml/
│   ├── components/
│   │   ├── data_prep.yaml
│   │   └── train.yaml          # Regular training component (fixed hyperparameters)
│   └── environments/
│       └── environment.yml     # Azure ML environment definition (Docker image reference)
├── configs/
│   ├── data.yaml
│   ├── hpo.yaml
│   ├── mlflow.yaml
│   └── train.yaml
├── data/
│   └── README.md
├── docs/
│   ├── MASTER_PLAN.md          # This file - project plan and guide
│   ├── TROUBLESHOOTING.md      # Troubleshooting guide for common issues
│   ├── dependencies.md
│   ├── pipeline_guide.md
│   ├── python_setup.md
│   └── setup_guide.md
├── notebooks/
│   ├── eda.ipynb               # Exploratory data analysis
│   └── hpo_manual_trials.ipynb # Manual HPO sweep orchestration in Azure ML
├── setup/
│   ├── create_data_asset.py    # Script to create Azure ML data assets
│   ├── setup.sh                # Bash script for Azure ML resource setup
│   ├── setup.ps1               # PowerShell script for Azure ML resource setup
│   └── README.md               # Setup documentation
├── src/
│   ├── data/                   # Data processing utilities
│   ├── models/                 # Model definitions (logreg, rf, xgboost)
│   ├── training/               # Training utilities
│   ├── utils/                  # Utility modules
│   │   ├── azure_config.py     # Azure ML configuration loading
│   │   ├── config_loader.py    # YAML configuration loading
│   │   ├── env_loader.py       # Environment variable loading
│   │   ├── mlflow_utils.py     # MLflow integration utilities
│   │   └── ...
│   ├── data_prep.py            # Data preprocessing script
│   ├── run_sweep_trial.py      # Helper script for HPO sweep trials
│   ├── train.py                # Model training script
│   └── README.md
├── config.env                  # Environment configuration (not in git)
├── config.env.example          # Example environment configuration template
├── Dockerfile                  # Docker image definition
├── hpo_utils.py                # Hyperparameter optimization utilities
├── run_pipeline.py             # Regular training pipeline orchestration script
├── README.md                   # Project overview
├── requirements.in             # Core dependencies (source)
├── requirements.txt            # Core dependencies (pinned)
├── dev-requirements.in         # Development dependencies (source)
└── dev-requirements.txt        # Development dependencies (pinned)
```

---

## Project Setup: Step-by-Step Guide

**Prerequisites:**

- Python 3.9 (required - matches Dockerfile)
- Docker
- Azure CLI
- Azure subscription with ML workspace

### Step 1: Set up the Project (Repo + README)

1. **Initialize Git**: Create a new repository.
2. **Create `README.md`**: Add a basic project overview.
3. **Create `.gitignore`**: Add entries for temporary files, data, models, and environment files (`.env`, `__pycache__`, `*.csv`, `models/`, `evaluation/`, etc.).

### Step 2: Set up Azure ML Workspace

See [[setup/README.md]].

**Important Setup Order**: For proper ACR authentication with managed identity, create resources in this order:

1. Azure ML workspace
2. Azure Container Registry (ACR) - if using custom Docker images
3. Compute cluster with system-assigned managed identity (AcrPull role automatically granted if ACR exists)

**Detailed Steps**:

1. **Create Azure ML workspace** (via Azure Portal or CLI, or use setup script)

2. **Create Azure Container Registry (ACR)** (optional, but recommended if using custom Docker images):

   - Set `AZURE_ACR_NAME` in `config.env` before running setup script
   - Or create manually: `az acr create --resource-group <rg> --name <acr-name> --sku Basic`
   - **Important**: ACR should be created **before** compute cluster for automatic AcrPull role assignment

3. **Create compute cluster**: `cpu-cluster` with system-assigned managed identity

   - Setup script automatically creates with `--identity-type systemassigned`
   - If ACR exists, AcrPull role is automatically granted to compute's managed identity

4. **Create data asset**: Register raw dataset as `uri_folder` type (name and version configured in `config.env`)

5. **Set up authentication**: Create `config.env` file with Azure credentials:

   - `AZURE_SUBSCRIPTION_ID`
   - `AZURE_RESOURCE_GROUP`
   - `AZURE_WORKSPACE_NAME`
   - `AZURE_ACR_NAME`: Azure Container Registry name (optional, for Docker images)
   - `DATA_ASSET_FULL`: Name of registered data asset (used by pipeline scripts)
   - `DATA_VERSION`: Version of data asset
   - **Note**: `run_pipeline.py` and `notebooks/hpo_manual_trials.ipynb` both load `config.env` (directly or via helper functions) to resolve Azure ML credentials and data asset metadata.

**Quick Setup**: Use the setup script to create all resources in the correct order:

```bash
# Set AZURE_ACR_NAME in config.env first (optional)
./setup/setup.sh
```

### Step 3: Add the Dataset

1. **Create `data/` directory** (if it doesn't exist).
2. **Add dataset files**:
    - `sample.csv` - Sample dataset (1,000 rows) for local development (tracked in git)
    - `churn.csv` - Full dataset (10,000 rows) for Azure ML pipelines (excluded from git via `.gitignore`)
    - See `data/README.md` for dataset details and usage
3. **Upload dataset to Azure ML as a data asset**:
    - Use the setup script or Azure ML CLI to register the dataset
    - **Important**: The data asset must be registered as `uri_folder` (directory containing CSV file(s))
    - The `data_prep` component accepts `uri_folder` input and automatically loads all CSV files in the folder
    - If multiple CSV files are present, they will be concatenated together (useful for splitting large datasets across multiple files)
    - Configure the data asset name and version in `config.env`:
      - `DATA_ASSET_FULL`: Name of registered data asset
      - `DATA_VERSION`: Version of data asset
    - Example: Register a folder containing `churn.csv` (or multiple CSV files) as `uri_folder` type

### Step 4: Perform EDA

1. **Create `notebooks/` directory**.
2. Develop `notebooks/eda.ipynb` to analyze the sample data, documenting findings on distributions, correlations, and data quality.

### Step 5: Create Core Scripts

1. **Create `src/` directory** and `src/__init__.py`.
2. **Develop Core Scripts**:
    - `src/data_prep.py`: Data preprocessing stage
    - `src/train.py`: Model training with MLflow logging (supports `--model-type` for HPO mode, `--set model.param=value` for hyperparameter overrides). In regular mode, models are determined from `configs/train.yaml` → `training.models`.
    - `src/run_sweep_trial.py`: Helper script that invokes `train.py` with sweep-managed hyperparameters (used by notebook HPO workflow)
3. **Create `src/models/` package**: Individual model definitions (logreg, rf, xgboost)

> **HPO mode** (`--model-type`): Required for hyperparameter optimization sweeps. Each sweep trial trains a single model type (selected by the sweep algorithm) with specific hyperparameters. The `model_type` itself is treated as a categorical hyperparameter, allowing the sweep to explore different model types and their hyperparameters simultaneously.

### Step 6: Centralize Configuration

1. **Create `configs/` directory**.
2. **Create YAML files**:
    - `configs/data.yaml`: Data prep settings (input/output paths, target column, columns to remove, categorical columns, train/test split)
    - `configs/train.yaml`: Training settings (models, hyperparameters, class weights)
    - `configs/hpo.yaml`: HPO configuration (search space, metric, budget, early-stopping)
    - `configs/mlflow.yaml`: MLflow experiment name
3. **Create `src/config_loader.py`**: Utility functions to load YAML files and get nested config values using dot notation
4. **Create `hpo_utils.py`**: Utility functions for HPO (loads configs, builds parameter space)
5. **Update Scripts**: Ensure scripts pull defaults from config but remain overrideable via CLI arguments

### Step 7: Create Pipeline Orchestration Paths

1. **`notebooks/hpo_manual_trials.ipynb`**: Notebook-driven HPO orchestration
    - Imports `hpo_utils.py` to load `configs/hpo.yaml` and construct the search space
    - Builds Azure ML sweep jobs per model type and wires them to `src/run_sweep_trial.py`, which invokes `train.py` with `--set model.param=value`
    - Handles data asset inputs, compute selection, and early stopping policies directly inside the notebook
    - Preferred workflow for launching sweeps today (run cells sequentially to submit/monitor jobs)

2. **`run_pipeline.py`**: Fixed-hyperparameter training pipeline
    - Uses `aml/components/train.yaml` component for training
    - Simple pipeline: data_prep → train
    - Models determined from `configs/train.yaml` → `training.models`
    - Automatically loads `config.env` for Azure ML configuration
    - Use for: Quick retraining, production deployments, after HPO

### Step 8: Declare Dependencies

**Important**: Use Python 3.9 from the start to match the Dockerfile (`python:3.9-slim`). This ensures all compiled requirements are compatible with the production environment.

1. **Create `requirements.in`**: Core libraries
    - Data & ML: pandas, numpy, scikit-learn, imbalanced-learn, xgboost
    - MLflow: mlflow (experiment tracking)
    - Azure ML: azure-ai-ml, azure-identity
    - Utilities: pyyaml, matplotlib, seaborn, scipy
    - Optional: fastapi (for serving API)

2. **Create `dev-requirements.in`**: Development tools
    - Dependency management: pip-tools
    - Code quality: black, ruff, isort, pre-commit

3. **Set up Python 3.9 environment** (see [[docs/python_setup.md]] for detailed instructions):

    ```text
    # Create virtual environment with Python 3.9
    python3.9 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    
    # Install pip-tools
    pip install pip-tools
    ```

4. **Compile Pinned Versions** (using Python 3.9):

    ```bash
    # Compile requirements first (ensures compatibility with Python 3.9)
    pip-compile requirements.in -o requirements.txt
    
    # Compile dev-requirements with constraints from requirements.txt
    # This ensures shared dependencies use compatible versions
    pip-compile dev-requirements.in -o dev-requirements.txt --constraint requirements.txt
    ```

5. **Sync environments**: Docker image, local venv, and AML compute all install from compiled requirements

### Step 9: Define and Test the Docker Image

1. **Create `Dockerfile`** in the project root
2. Start from slim Python base image, copy `requirements.txt` first (for layer caching), then application code
3. **Build and test locally**:

```bash
# Build the Docker image
docker build -t bank-churn:1 .

# Test the image locally
docker run --rm bank-churn:1 bash -c "cd /app/src && python data_prep.py --help"
docker run --rm bank-churn:1 python -c "import pandas; print('OK')"
```

### Step 10: Build AML Components

1. **Create `aml/components/` directory**
2. **Create component YAML files**:
    - `aml/components/data_prep.yaml`: Data preprocessing component
    - `aml/components/train.yaml`: Training component for regular training (fixed hyperparameters from config)
    - *HPO note*: Sweep jobs are defined directly inside `notebooks/hpo_manual_trials.ipynb` and call `src/run_sweep_trial.py`, so no dedicated `hpo.yaml` component is required.

### Step 11: Push Docker Image to ACR and Register Azure ML Environment

**Prerequisites:**

- Docker image built (from Step 9): `bank-churn:1`
- Azure Container Registry (ACR) created and configured in `config.env` as `AZURE_ACR_NAME`
- Compute cluster created with system-assigned managed identity (setup script does this automatically)
- ACR name is separate from the Azure ML environment name (`bank-churn-env`)

**Note**:

- If you don't have an ACR, create one using the setup script (set `AZURE_ACR_NAME` in `config.env` and run `./setup/setup.sh`) or manually using `az acr create`
- **Important**: If ACR was created before compute cluster, AcrPull role is automatically granted. If compute was created before ACR, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md#acr-authentication-for-compute-cluster) for manual role assignment

#### 1. Prepare Docker Image for ACR

**1.1. Build Docker image** (if not already built in Step 9):

```bash
# Build the Docker image locally
docker build -t bank-churn:1 .
```

**1.2. Login to Azure Container Registry**:

```bash
# Get ACR name from config.env or use directly
ACR_NAME=$(grep AZURE_ACR_NAME config.env | cut -d'"' -f2)
az acr login --name $ACR_NAME

# Or manually:
# az acr login --name <your-acr-name>
```

**1.3. Tag image for ACR**:

```bash
# Use ACR name from config.env
ACR_NAME=$(grep AZURE_ACR_NAME config.env | cut -d'"' -f2)
docker tag bank-churn:1 $ACR_NAME.azurecr.io/bank-churn:1

# Or manually:
# docker tag bank-churn:1 <your-acr-name>.azurecr.io/bank-churn:1
```

**1.4. Push image to ACR**:

```bash
# Push the tagged image
ACR_NAME=$(grep AZURE_ACR_NAME config.env | cut -d'"' -f2)
docker push $ACR_NAME.azurecr.io/bank-churn:1

# Or manually:
# docker push <your-acr-name>.azurecr.io/bank-churn:1
```

**1.5. Verify image is in ACR**:

```bash
ACR_NAME=$(grep AZURE_ACR_NAME config.env | cut -d'"' -f2)
az acr repository show-tags --name $ACR_NAME --repository bank-churn --output table

# Should show tag "1" in the output
```

#### 2. Update Environment Configuration

**2.1. Update `aml/environments/environment.yml`**:

Replace the placeholder with your actual ACR name. The file should look like:

```yaml
$schema: https://azuremlschemas.azureedge.net/latest/environment.schema.json
name: bank-churn-env
version: "1"
image: <your-acr-name>.azurecr.io/bank-churn:1
description: Environment for churn prediction pipeline
```

Replace `<your-acr-name>` with your actual ACR name (same as `AZURE_ACR_NAME` in `config.env`).

**Example** (if `AZURE_ACR_NAME="churnmlacr2025"`):

```yaml
image: churnmlacr2025.azurecr.io/bank-churn:1
```

#### 3. Register Environment in Azure ML

**3.1. Register using YAML file** (recommended):

```bash
# Load values from config.env
source <(grep -E "AZURE_RESOURCE_GROUP|AZURE_WORKSPACE_NAME" config.env | sed 's/^/export /' | sed 's/"//g')

az ml environment create --file aml/environments/environment.yml \
  --resource-group $AZURE_RESOURCE_GROUP \
  --workspace-name $AZURE_WORKSPACE_NAME

# Or manually:
# az ml environment create --file aml/environments/environment.yml \
#   --resource-group <resource-group> \
#   --workspace-name <workspace-name>
```

**3.2. Alternative: Register with CLI override** (without modifying YAML):

```bash
ACR_NAME=$(grep AZURE_ACR_NAME config.env | cut -d'"' -f2)
source <(grep -E "AZURE_RESOURCE_GROUP|AZURE_WORKSPACE_NAME" config.env | sed 's/^/export /' | sed 's/"//g')

az ml environment create --file aml/environments/environment.yml \
  --image $ACR_NAME.azurecr.io/bank-churn:1 \
  --resource-group $AZURE_RESOURCE_GROUP \
  --workspace-name $AZURE_WORKSPACE_NAME
```

This overrides the `image` field in the YAML file without modifying it.

#### 4. Verify Environment Registration

**4.1. Check environment exists**:

```bash
source <(grep -E "AZURE_RESOURCE_GROUP|AZURE_WORKSPACE_NAME" config.env | sed 's/^/export /' | sed 's/"//g')

az ml environment show --name bank-churn-env --version 1 \
  --resource-group $AZURE_RESOURCE_GROUP \
  --workspace-name $AZURE_WORKSPACE_NAME \
  --query "{Name:name, Version:version, Image:image}" \
  -o table
```

**4.2. Verify image reference**:

The output should show:

- `Name`: `bank-churn-env`
- `Version`: `1`
- `Image`: `<your-acr-name>.azurecr.io/bank-churn:1`

**Important Notes:**

- The Docker image **must exist in ACR** before registering the environment, otherwise registration will fail
- All component YAML files (`aml/components/*.yaml`) reference `azureml:bank-churn-env:1` for consistent dependencies
- If you update the Docker image, push a new tag and create a new environment version, or update the existing environment

### Step 12: Test Pipeline Execution

1. **Test regular training pipeline**:

    ```bash
    python run_pipeline.py
    ```

    - Uses data asset configured in `config.env` (`DATA_ASSET_FULL` and `DATA_VERSION`) via `get_data_asset_config()`
    - Runs data prep → train pipeline using `train.yaml` component
    - Trains models specified in `configs/train.yaml` → `training.models`
    - Verify job submission in Azure ML Studio

2. **Test HPO pipeline** (optional, for hyperparameter optimization):

    1. Open `notebooks/hpo_manual_trials.ipynb` in VS Code / Jupyter.
    2. Ensure `configs/hpo.yaml` reflects the search space and budgets you want to test (e.g., `max_trials: 2-3` for smoke tests).
    3. Run the notebook cells sequentially to submit sweeps. The notebook prints job names and Studio URLs for tracking.
    4. Monitor Azure ML Studio to confirm trials start and finish successfully.

3. **Verify outputs**:
    - **Azure ML Studio**: Check job status, logs, and outputs
    - **MLflow**: View runs, metrics, parameters, and artifacts (if MLflow tracking is configured)
    - **Model artifacts**: Check that models are saved to outputs directory

### Step 13: Production Workflow - Run Pipeline with Optimized Hyperparameters

After successfully testing the pipeline, follow this workflow for production runs:

#### 1. Run Hyperparameter Optimization (First Time or When Retuning)

**1.1. Configure HPO budget** in `configs/hpo.yaml`:

```yaml
budget:
  max_trials: 20  # Increase for production (was 2-3 for testing)
  max_concurrent: 4  # Parallel trials
```

**1.2. Submit HPO sweeps via notebook**:

1. Open `notebooks/hpo_manual_trials.ipynb`.
2. Run the preparation cells (workspace auth, data discovery, config loading).
3. Execute the sweep configuration cell to build `search_space` definitions sourced from `configs/hpo.yaml`.
4. Run the submission cell to launch one sweep per `model_type`. Record the printed job names and Studio URLs.

**1.3. Monitor HPO progress**:

- Track sweep status in Azure ML Studio using the URLs provided by the notebook.
- View MLflow/Studio charts to compare child trials in real time.
- Wait for all trials to complete (duration depends on `max_trials` and compute availability).

**1.4. Analyze results and export best model**:

1. After sweeps complete, run the "Analyze Results" cell in the notebook to find the best model across all sweeps.
2. Run the "Export Best Model Configuration" cell to automatically export the best model's hyperparameters to `configs/train.yaml`.
   - The export cell preserves existing config structure
   - Updates `training.models` to include only the best model
   - Updates `training.hyperparameters.<model_type>` with best hyperparameters
   - Ensures string values are properly quoted (e.g., `use_smote: "true"`, `max_features: "sqrt"`)

**1.5. Verify exported config**:

```bash
# Check that configs/train.yaml was updated
cat configs/train.yaml
```

You should see:

- `models: [best_model_type]` (only the best model)
- Updated hyperparameters for that model
- Training-level parameters (e.g., `use_smote: "true"`) properly quoted

#### 2. Run Production Training Pipeline

**2.1. Run pipeline with optimized hyperparameters**:

You can either:

**Option A: Run from notebook** (recommended):

1. Run the "Run Training Pipeline" cell in `notebooks/hpo_manual_trials.ipynb`
2. This automatically uses the exported `configs/train.yaml` configuration

**Option B: Run from command line**:

```bash
python run_pipeline.py
```

Both options will:

- Use the updated `configs/train.yaml` with best hyperparameters
- Train only the best model type (faster, more efficient)
- Produce production-ready model artifacts

**2.2. Monitor training job**:

- Check Azure ML Studio for job status
- View MLflow run for metrics and artifacts
- Wait for job completion (typically 5-15 minutes)

**2.3. Get model artifacts**:

After completion, model artifacts are available:

- **Azure ML Studio**: Job outputs → `model_output` folder
- **MLflow**: Run artifacts (if MLflow tracking enabled)
- Models are saved as pickle files and can be downloaded or registered

#### 3. Use Trained Models

**3.1. Download models locally** (if needed):

```bash
# Get job name from Azure ML Studio or from the output of run_pipeline.py
JOB_NAME="<job-name>"
az ml job download --name $JOB_NAME --output-dir ./models
```

**3.2. Register model in Azure ML** (for deployment):

```bash
# Register the model from job outputs
az ml model create \
  --name churn-prediction-model \
  --version 1 \
  --path azureml://jobs/$JOB_NAME/outputs/model_output \
  --resource-group <resource-group> \
  --workspace-name <workspace-name>
```

**3.3. Use model for inference**:

- Models are logged with MLflow and support `mlflow.pyfunc` deployment
- Can be deployed to Azure ML online endpoints
- Can be loaded locally: `mlflow.pyfunc.load_model("runs:/<run-id>/model")`

#### 4. Iterate and Improve

**4.1. Monitor model performance**:

- Track metrics in MLflow over time
- Compare different training runs
- Monitor production model performance (if deployed)

**4.2. Retrain when needed**:

- **Regular retraining**: Run `python run_pipeline.py` periodically with updated data
- **Re-optimize**: Run HPO again if:
  - Data distribution changes significantly
  - Model performance degrades
  - New features are added
  - Business requirements change

**4.3. Update data asset** (if using new data):

```bash
# Create new version of data asset
python setup/create_data_asset.py

# Update DATA_VERSION in config.env
# Then run pipeline again
python run_pipeline.py
```

#### Typical Production Workflow Summary

1. **Initial Setup**: Run HPO in notebook → Export best config in notebook → Run production pipeline (from notebook or CLI)
2. **Regular Retraining**: Update data → Run production pipeline (skip HPO)
3. **Periodic Re-optimization**: Run HPO in notebook → Export best config in notebook → Run production pipeline

**Time Estimates**:

- HPO sweep: 30-60 minutes (depending on `max_trials`)
- Production training: 5-15 minutes
- Total initial setup: ~1-2 hours

---

## Pipeline Entry Points

### `notebooks/hpo_manual_trials.ipynb` - Hyperparameter Optimization

- End-to-end notebook workflow for building and submitting Azure ML sweeps
- Loads `configs/hpo.yaml` via `hpo_utils.py` to construct per-model search spaces
- Uses `src/run_sweep_trial.py` → `train.py` to run each sampled configuration with `--set model.param=value`
- Provides interactive monitoring output (job names, Studio links, chosen hyperparameters)
- **Analyze Results**: Built-in cell to find the best model across all sweeps
- **Export Config**: Automatically exports best model configuration to `configs/train.yaml` with proper YAML formatting
- **Run Pipeline**: Optional cell to run training pipeline directly from the notebook
- **Use when**: First-time optimization, tuning new search spaces, or validating sweep fixes
- **Outputs**: Azure ML sweep jobs with logged metrics/params per child run, plus exported `configs/train.yaml` for production training

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

See [Step 13: Production Workflow](#step-13-production-workflow---run-pipeline-with-optimized-hyperparameters) for detailed instructions.

**Quick Summary**:

1. Use `notebooks/hpo_manual_trials.ipynb` to find best hyperparameters (uses Azure ML sweeps + `run_sweep_trial.py`)
2. In the notebook, run the "Analyze Results" cell to find the best model, then run the "Export Best Model Configuration" cell
   - This automatically updates `configs/train.yaml` with best model and hyperparameters
   - Sets `models: [best_model]` to train only the best model
   - Ensures proper YAML formatting with quoted string values
3. Run the "Run Training Pipeline" cell in the notebook, or run `run_pipeline.py` from command line
   - Both use the exported `configs/train.yaml` configuration

**For subsequent retraining**: Simply run `run_pipeline.py` with updated data (skip HPO unless re-optimization is needed).

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
