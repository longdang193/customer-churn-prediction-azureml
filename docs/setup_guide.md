# Project Setup Guide

This guide covers the setup for both local development and Azure Machine Learning.

## Local Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/customer-churn-prediction-azureml.git
cd customer-churn-prediction-azureml
```

### 2. Create the Python 3.9 Environment and Install Dependencies

> Detailed OS-specific instructions live in `docs/python_setup.md`. The quick version is below.

```bash
# Create & activate a Python 3.9 virtual environment
python3.9 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install tooling and sync pinned requirements
pip install --upgrade pip pip-tools
pip install -r requirements.txt
pip install -r dev-requirements.txt  # optional for local development tooling
```

Only run `pip-compile` if you change the `.in` files; see `docs/dependencies.md` for the full workflow.

### 3. Running the EDA Notebook

The EDA notebook is configured to run locally without Azure dependencies.

- **Navigate to `notebooks/eda.ipynb`** in your IDE.
- **Select a Python kernel** and run the cells.
- The notebook will automatically use the local `data/churn.csv` file.

## Azure ML Setup

### 1. Create `config.env`

Copy the example configuration file and fill in your Azure service principal or user credentials.

```bash
cp config.env.example config.env
```

Edit `config.env` with your Azure details:

```bash
AZURE_SUBSCRIPTION_ID="your-subscription-id"
AZURE_TENANT_ID="your-tenant-id"
AZURE_RESOURCE_GROUP="rg-churn-ml-project"
AZURE_WORKSPACE_NAME="churn-ml-workspace"
DATA_ASSET_FULL="churn-data"
DATA_VERSION="3"
```

**Important**: The data asset must be registered as `uri_folder` type (directory containing CSV file(s)). The `data_prep` component will automatically load all CSV files in the folder.

The template also includes optional entries you can customize:

- `AZURE_LOCATION`: Azure region for the workspace (e.g., `southeastasia`)
- `DATA_ASSET_FULL`, `DATA_VERSION`: Data asset name/version used by pipeline scripts via `get_data_asset_config()`
- `AZURE_COMPUTE_CLUSTER_NAME`: Name used when provisioning compute cluster
- `COMPUTE_CLUSTER_SIZE`: VM size for compute cluster
- `MODEL_NAME`, `EXPERIMENT_NAME`: MLflow experiment and registered model identifiers
- `ENDPOINT_NAME`, `DEPLOYMENT_NAME`: Names used for online endpoint deployment

**Note**: `run_pipeline.py` automatically loads `config.env`, so you don't need to source it manually.

### 2. Authenticate with Azure

Log in to the Azure CLI:

```bash
az login
```

### 3. Create the Azure ML Data Asset

The churn pipeline expects the dataset to be registered as a `uri_folder` so the `data_prep` component can automatically load all CSV files. Use the provided helper script to register (reads names/versions from `config.env`):

```bash
python setup/create_data_asset.py \
  --data-path data/churn.csv \
  --name "$DATA_ASSET_FULL" \
  --version "$DATA_VERSION"
```

If you prefer the CLI, the equivalent command is:

```bash
az ml data create \
  --name churn-data \
  --version 3 \
  --path data/ \
  --type uri_folder \
  --resource-group $AZURE_RESOURCE_GROUP \
  --workspace-name $AZURE_WORKSPACE_NAME
```

Keep the `name`/`version` aligned with `DATA_ASSET_FULL` and `DATA_VERSION`; `run_pipeline.py` and the notebooks read these values via `get_data_asset_config()`.
