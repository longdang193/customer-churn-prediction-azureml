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
python scripts/create_data_asset.py
```

**Option B: Bash (Azure CLI)**
```bash
./scripts/create_data_asset.sh
```

This allows you to reference the dataset in Azure ML pipelines as `azureml:churn-data:1`.

