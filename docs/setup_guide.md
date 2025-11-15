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
AZURE_RAW_DATA_ASSET="churn-data"
AZURE_RAW_DATA_VERSION="3"
```

**Important**: The data asset must be registered as `uri_folder` type (directory containing CSV file(s)). The `data_prep` component will automatically load all CSV files in the folder.

The template also includes optional entries you can customize:
- `AZURE_LOCATION`: Azure region for the workspace (e.g., `southeastasia`)
- `AZURE_RAW_DATA_ASSET`, `AZURE_RAW_DATA_VERSION`: Data asset name/version used by pipeline scripts
- `AZURE_COMPUTE_CLUSTER_NAME`: Name used when provisioning compute cluster
- `COMPUTE_CLUSTER_SIZE`: VM size for compute cluster
- `MODEL_NAME`, `EXPERIMENT_NAME`: MLflow experiment and registered model identifiers
- `ENDPOINT_NAME`, `DEPLOYMENT_NAME`: Names used for online endpoint deployment

**Note**: Both `run_pipeline.py` and `run_hpo.py` automatically load `config.env`, so you don't need to source it manually.

### 2. Authenticate with Azure

Log in to the Azure CLI:

```bash
az login
```

### 3. Create Azure ML Data Asset

**Important**: The data asset must be registered as `uri_folder` type (directory containing CSV file(s)). The `data_prep` component will automatically load and concatenate all CSV files in the folder.

Register the data asset using Azure CLI:

```bash
az ml data upload \
  --name churn-data \
  --version 3 \
  --path data/ \
  --type uri_folder \
  --resource-group <resource-group> \
  --workspace-name <workspace-name>
```

Replace `<resource-group>` and `<workspace-name>` with your values, or use the values from `config.env`.

**Note**: The data asset name and version should match `AZURE_RAW_DATA_ASSET` and `AZURE_RAW_DATA_VERSION` in your `config.env` file. The pipeline scripts will automatically use these values.

