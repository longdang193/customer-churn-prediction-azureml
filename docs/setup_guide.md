# Project Setup Guide

This guide covers the setup for both local development and Azure Machine Learning.

## Local Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/customer-churn-prediction-azureml.git
cd customer-churn-prediction-azureml
```

### 2. Install Dependencies

Install the required Python packages using the provided `requirements.txt` file.

```bash
pip install -r requirements.txt
```

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
AZURE_STORAGE_ACCOUNT="yourstorageaccount"
AZURE_STORAGE_CONTAINER="data"
AZURE_RAW_DATA_ASSET="churn-data"
AZURE_RAW_DATA_VERSION="1"
```

The template also includes optional entries you can customize:
- `AZURE_LOCATION`: Azure region for the workspace (e.g., `southeastasia`)
- `AZURE_RAW_DATA_ASSET`, `AZURE_RAW_DATA_VERSION`: Default data asset name/version used by pipeline scripts
- `AZURE_COMPUTE_INSTANCE_NAME`, `AZURE_COMPUTE_CLUSTER_NAME`: Names used when provisioning compute
- `COMPUTE_INSTANCE_SIZE`, `COMPUTE_CLUSTER_SIZE`: VM sizes for compute resources
- `DATA_ASSET_FULL`, `DATA_ASSET_SAMPLE`, `DATA_VERSION`: Data asset names and version tags
- `MODEL_NAME`, `EXPERIMENT_NAME`: MLflow experiment and registered model identifiers
- `ENDPOINT_NAME`, `DEPLOYMENT_NAME`: Names used for online endpoint deployment

After filling the placeholders, load the variables in your shell before running any setup scripts:

```bash
set -a
source config.env
set +a
```

### 2. Authenticate with Azure

Log in to the Azure CLI:

```bash
az login
```

### 3. Create Azure ML Data Asset

If you have uploaded `churn.csv` to your Azure Blob Storage, you can create an Azure ML Data Asset.

**Option A: Python (Recommended)**
```bash
python setup/create_data_asset.py
```

**Option B: Bash (Azure CLI)**
```bash
./setup/create_data_asset.sh
```

This allows you to reference the dataset in Azure ML pipelines as `azureml:bank-churn-raw:1`.

