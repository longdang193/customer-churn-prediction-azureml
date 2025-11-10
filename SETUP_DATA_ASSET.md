# Setting Up Azure ML Data Asset

Create an Azure ML Data Asset for the bank churn dataset.

## Prerequisites

- `churn.csv` uploaded to Azure Blob Storage
- Azure ML workspace created
- Azure credentials available

## Step 1: Create config.env

```bash
cp config.env.example config.env
```

Edit `config.env` with your Azure credentials:

```bash
AZURE_SUBSCRIPTION_ID="your-subscription-id"       # az account show --query id -o tsv
AZURE_TENANT_ID="your-tenant-id"                  # az account show --query tenantId -o tsv
AZURE_RESOURCE_GROUP="rg-churn-ml-project"
AZURE_WORKSPACE_NAME="churn-ml-workspace"
AZURE_STORAGE_ACCOUNT="churnmlstorage123"         # Where you uploaded churn.csv
AZURE_STORAGE_CONTAINER="data"                    # Container with churn.csv
DATA_ASSET_FULL="churn-data"
DATA_VERSION="1"
```

## Step 2: Authenticate

```bash
az login
```

## Step 3: Create Data Asset

**Option A: Python (Recommended)**

```bash
pip install azure-ai-ml azure-identity
python scripts/create_data_asset.py
```

**Option B: Bash (Azure CLI)**

```bash
./scripts/create_data_asset.sh
```

## Verify

View in Azure ML Studio: https://ml.azure.com → Data → `churn-data`

Or via CLI:

```bash
az ml data show --resource-group <rg> --workspace-name <ws> --name churn-data --version 1
```

## Usage

Reference in pipelines:

```python
raw_data = "azureml:churn-data:1"
```

Or in YAML:

```yaml
inputs:
  raw_data:
    type: uri_file
    path: azureml:churn-data:1
```

## Troubleshooting

**Workspace not found:** Run `./setup.sh` first

**Authentication failed:** Run `az login` and verify subscription with `az account show`

**Data asset exists:** Increment `DATA_VERSION` in `config.env` or delete existing asset

