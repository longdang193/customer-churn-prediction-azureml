# Azure ML Setup Guide

This guide explains how to set up your Azure ML workspace and compute resources for the customer churn prediction project.

## Prerequisites

1. **Azure CLI** installed and configured
   - Install from: https://docs.microsoft.com/cli/azure/install-azure-cli
   - Verify installation: `az --version`

2. **Azure Subscription** with appropriate permissions
   - You need Contributor or Owner role on the subscription/resource group
   - Verify access: `az account list`

3. **Bash shell** (for setup.sh)
   - Windows: Use Git Bash, WSL, or Azure Cloud Shell
   - Mac/Linux: Use terminal
   - Or use the PowerShell version: `setup.ps1`

## Quick Start

### Option 1: Using Bash Script (setup.sh)

```bash
# Make script executable (Linux/Mac)
chmod +x setup.sh

# Run the script
./setup.sh
```

### Option 2: Using PowerShell (setup.ps1)

```powershell
# Run the PowerShell script
.\setup.ps1
```

### Option 3: Using Azure Cloud Shell

1. Go to https://shell.azure.com
2. Upload `setup.sh` to Cloud Shell
3. Run: `bash setup.sh`

## Configuration

### Environment Variables

You can customize the setup by setting environment variables before running the script:

```bash
# Set custom values
export AZURE_RESOURCE_GROUP="my-rg"
export AZURE_LOCATION="westus2"
export AZURE_WORKSPACE_NAME="my-workspace"
export AZURE_COMPUTE_INSTANCE_NAME="my-compute"
export AZURE_COMPUTE_CLUSTER_NAME="my-cluster"
export COMPUTE_INSTANCE_SIZE="Standard_DS3_v2"
export COMPUTE_CLUSTER_SIZE="Standard_DS3_v2"
export MIN_NODES="0"
export MAX_NODES="4"

# Run setup
./setup.sh
```

### Default Configuration

- **Resource Group**: `rg-churn-ml-project`
- **Location**: `eastus`
- **Workspace Name**: `churn-ml-workspace`
- **Compute Instance**: `compute-instance-1` (Standard_DS3_v2)
- **Compute Cluster**: `cpu-cluster` (Standard_DS3_v2, 0-4 nodes)

## What the Script Does

1. **Checks prerequisites**: Verifies Azure CLI is installed and you're logged in
2. **Creates resource group**: Creates a new resource group (or uses existing)
3. **Creates Azure ML workspace**: Sets up the ML workspace
4. **Creates compute instance**: For running notebooks and EDA (takes 5-10 minutes)
5. **Creates compute cluster**: For training pipelines (scales from 0 to 4 nodes)

## After Setup

### 1. Access Azure ML Studio

Go to https://ml.azure.com and select your workspace.

### 2. Create Notebooks

- Navigate to **Notebooks** in Azure ML Studio
- Create `notebooks/eda.ipynb` on your compute instance
- Start exploring the data!

### 3. Upload Data

You have two options:

**Option A: Upload to Azure Blob Storage**
```bash
# Create storage account and container
az storage account create \
    --name <storage-account-name> \
    --resource-group $RESOURCE_GROUP \
    --location $LOCATION

az storage container create \
    --name data \
    --account-name <storage-account-name>

# Upload data
az storage blob upload \
    --account-name <storage-account-name> \
    --container-name data \
    --name churn.csv \
    --file data/churn.csv
```

**Option B: Create Data Asset in Azure ML**
- Go to **Data** in Azure ML Studio
- Click **Create** → **From local files**
- Upload `data/churn.csv`
- Name it `churn-data`

### 4. Register Environment

Register your conda environment from `environment/conda.yml`:

```python
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="<your-subscription-id>",
    resource_group_name="rg-churn-ml-project",
    workspace_name="churn-ml-workspace",
)

# Register environment
ml_client.environments.create_or_update(environment="environment/conda.yml")
```

Or use Azure ML Studio:
- Go to **Environments** → **Create**
- Upload `environment/conda.yml`

### 5. Connect from Python

Create a config file or use DefaultAzureCredential:

```python
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="<your-subscription-id>",
    resource_group_name="rg-churn-ml-project",
    workspace_name="churn-ml-workspace",
)
```

## Troubleshooting

### Authentication Issues

```bash
# Login to Azure
az login

# Verify subscription
az account show

# Set subscription (if needed)
az account set --subscription "<subscription-id>"
```

### Permission Issues

Make sure you have Contributor or Owner role:
```bash
# Check your role
az role assignment list --assignee $(az account show --query user.name -o tsv)
```

### Compute Instance Not Available

- Compute instance creation takes 5-10 minutes
- Check status in Azure ML Studio → Compute → Compute Instances
- Wait for status to be "Running"

### Region Not Available

Some regions may not support all VM sizes. Try a different region:
```bash
export AZURE_LOCATION="westus2"  # or eastus, westeurope, etc.
./setup.sh
```

## Cost Management

### Compute Instance
- **Cost**: ~$0.27/hour for Standard_DS3_v2 (when running)
- **Tip**: Stop the compute instance when not in use to save costs
- **Action**: Azure ML Studio → Compute → Compute Instances → Stop

### Compute Cluster
- **Cost**: Only charged when nodes are running
- **Auto-scale**: Scales to 0 when idle (saves money)
- **Min nodes**: Set to 0 to minimize costs

### Workspace
- **Cost**: Free (only pay for compute and storage)

## Next Steps

After setup, follow the project plan:

1. ✅ Set up workspace and compute (this script)
2. ⬜ Prepare data (upload to Azure)
3. ⬜ Exploratory Data Analysis (notebooks/eda.ipynb)
4. ⬜ Create training scripts (src/)
5. ⬜ Set up environment (environment/conda.yml)
6. ⬜ Local testing (local_test.py)
7. ⬜ Create AML components
8. ⬜ Build Azure ML pipeline (azureml_pipeline.py)
9. ⬜ Deploy model
10. ⬜ Document everything

## References

- [Azure ML Documentation](https://docs.microsoft.com/azure/machine-learning/)
- [Azure CLI ML Extension](https://docs.microsoft.com/cli/azure/ml)
- [Azure ML Pricing](https://azure.microsoft.com/pricing/details/machine-learning/)


