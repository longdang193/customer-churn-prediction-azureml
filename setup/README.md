# Setup Scripts and Common Commands

This directory contains setup scripts for Azure ML workspace and compute resources.

## Prerequisites

- Azure CLI installed: [Install Azure CLI](https://docs.microsoft.com/cli/azure/install-azure-cli)
- Authenticated with Azure: `az login`
- Appropriate permissions to create resources in the subscription

## Configuration Setup

Before running the setup scripts, you need to create a `config.env` file with your Azure credentials. Copy `config.env.example` to `config.env` and fill in the values:

```bash
cp config.env.example config.env
```

### Get Azure Configuration Values

Use these commands to retrieve the values needed for `config.env`:

#### Get Subscription ID

**Linux/Mac:**

```bash
az account show --query id -o tsv
```

**Windows (PowerShell):**

```powershell
az account show --query id -o tsv
```

#### Get Tenant ID

**Linux/Mac:**

```bash
az account show --query tenantId -o tsv
```

**Windows (PowerShell):**

```powershell
az account show --query tenantId -o tsv
```

#### Get Current Subscription Information

**Linux/Mac:**

```bash
az account show --query "{SubscriptionId:id, TenantId:tenantId, Name:name}" -o table
```

**Windows (PowerShell):**

```powershell
az account show --query "{SubscriptionId:id, TenantId:tenantId, Name:name}" -o table
```

#### List Available Locations

**Linux/Mac:**

```bash
az account list-locations --query "[].{Name:name, DisplayName:displayName}" -o table
```

**Windows (PowerShell):**

```powershell
az account list-locations --query "[].{Name:name, DisplayName:displayName}" -o table
```

#### Get Workspace Information (After Creation)

**Linux/Mac:**

```bash
az ml workspace show \
    --resource-group "${AZURE_RESOURCE_GROUP:-rg-churn-ml-project}" \
    --name "${AZURE_WORKSPACE_NAME:-churn-ml-workspace}" \
    --query "{Name:name, ResourceGroup:resourceGroup, Location:location}" \
    -o table
```

**Windows (PowerShell):**

```powershell
az ml workspace show `
    --resource-group $env:AZURE_RESOURCE_GROUP `
    --name $env:AZURE_WORKSPACE_NAME `
    --query "{Name:name, ResourceGroup:resourceGroup, Location:location}" `
    -o table
```

#### Get Storage Account Information

**Linux/Mac:**

```bash
# List storage accounts in resource group
az storage account list \
    --resource-group "${AZURE_RESOURCE_GROUP:-rg-churn-ml-project}" \
    --query "[].{Name:name, Location:location}" \
    -o table
```

**Windows (PowerShell):**

```powershell
az storage account list `
    --resource-group $env:AZURE_RESOURCE_GROUP `
    --query "[].{Name:name, Location:location}" `
    -o table
```

#### Get Azure Container Registry (ACR) Information

**Check if you have an existing ACR:**

**Linux/Mac:**

```bash
# List all ACRs in subscription
az acr list --output table

# List ACRs in specific resource group
az acr list --resource-group "${AZURE_RESOURCE_GROUP:-rg-churn-ml-project}" --output table

# Check if specific ACR exists
az acr show --name <acr-name> --resource-group "${AZURE_RESOURCE_GROUP:-rg-churn-ml-project}"
```

**Windows (PowerShell):**

```powershell
# List all ACRs in subscription
az acr list --output table

# List ACRs in specific resource group
az acr list --resource-group $env:AZURE_RESOURCE_GROUP --output table

# Check if specific ACR exists
az acr show --name <acr-name> --resource-group $env:AZURE_RESOURCE_GROUP
```

**Note:** ACR names must be globally unique (3-50 characters, alphanumeric only). If you don't have an ACR, you can create one using the setup script (set `AZURE_ACR_NAME` in `config.env`) or manually (see below).

### Quick Setup Script

You can also use this one-liner to create `config.env` with your current Azure account information:

**Linux/Mac:**

```bash
cat > config.env << EOF
# Azure Subscription and Resource Configuration
AZURE_SUBSCRIPTION_ID="$(az account show --query id -o tsv)"
AZURE_TENANT_ID="$(az account show --query tenantId -o tsv)"
AZURE_RESOURCE_GROUP="rg-churn-ml-project"
AZURE_LOCATION="southeastasia"

# Azure ML Workspace Configuration
AZURE_WORKSPACE_NAME="churn-ml-workspace"

# Compute Resources
AZURE_COMPUTE_CLUSTER_NAME="cpu-cluster"
COMPUTE_CLUSTER_SIZE="Standard_DS2_v2"

# Storage Configuration (update after creating storage account)
AZURE_STORAGE_ACCOUNT="your-storage-account-name"
AZURE_STORAGE_CONTAINER="data"

# Azure Container Registry (ACR) Configuration
# ACR name must be globally unique (3-50 characters, alphanumeric only)
AZURE_ACR_NAME="your-acr-name"

# Data Assets
# Used by pipeline scripts (run_pipeline.py, run_hpo.py) via get_data_asset_config()
DATA_ASSET_FULL="churn-data"
DATA_ASSET_SAMPLE="churn-data-sample"
DATA_VERSION="1"

# Model Configuration
MODEL_NAME="churn-prediction-model"
EXPERIMENT_NAME="churn-prediction-experiment"

# Deployment Configuration
ENDPOINT_NAME="churn-endpoint"
DEPLOYMENT_NAME="churn-deployment"
EOF
```

**Windows (PowerShell):**

```powershell
$subscriptionId = az account show --query id -o tsv
$tenantId = az account show --query tenantId -o tsv

@"
# Azure Subscription and Resource Configuration
AZURE_SUBSCRIPTION_ID="$subscriptionId"
AZURE_TENANT_ID="$tenantId"
AZURE_RESOURCE_GROUP="rg-churn-ml-project"
AZURE_LOCATION="southeastasia"

# Azure ML Workspace Configuration
AZURE_WORKSPACE_NAME="churn-ml-workspace"

# Compute Resources
AZURE_COMPUTE_CLUSTER_NAME="cpu-cluster"
COMPUTE_CLUSTER_SIZE="Standard_DS2_v2"

# Storage Configuration (update after creating storage account)
AZURE_STORAGE_ACCOUNT="your-storage-account-name"
AZURE_STORAGE_CONTAINER="data"

# Azure Container Registry (ACR) Configuration
# ACR name must be globally unique (3-50 characters, alphanumeric only)
AZURE_ACR_NAME="your-acr-name"

# Data Assets
# Used by pipeline scripts (run_pipeline.py, run_hpo.py) via get_data_asset_config()
DATA_ASSET_FULL="churn-data"
DATA_ASSET_SAMPLE="churn-data-sample"
DATA_VERSION="1"

# Model Configuration
MODEL_NAME="churn-prediction-model"
EXPERIMENT_NAME="churn-prediction-experiment"

# Deployment Configuration
ENDPOINT_NAME="churn-endpoint"
DEPLOYMENT_NAME="churn-deployment"
"@ | Out-File -FilePath config.env -Encoding utf8
```

**Note:** After running the quick setup script, you may still need to update:

- `AZURE_STORAGE_ACCOUNT`: If you have an existing storage account, or leave as-is if creating new resources
- `AZURE_ACR_NAME`: Set to your desired ACR name (must be globally unique) if you want the setup script to create ACR
- Other values can be customized as needed

## Setup Scripts

### Initial Setup

Run the setup script to create Azure ML workspace and compute resources:

**Linux/Mac:**

```bash
./setup/setup.sh
```

**Note:** If you encounter a "Permission denied" error, add execute permissions to the script:

```bash
chmod +x setup/setup.sh
```

**Windows (PowerShell):**

```powershell
.\setup\setup.ps1
```

**What it does:**

- Creates resource group (if it doesn't exist)
- Creates Azure ML workspace
- Creates Azure Container Registry (ACR) if `AZURE_ACR_NAME` is set in `config.env` (**before** compute cluster)
- Creates compute cluster with system-assigned managed identity (AcrPull role automatically granted if ACR exists)

**Configuration:**

The scripts use environment variables (with defaults):

- `AZURE_RESOURCE_GROUP` (default: `rg-churn-ml-project`)
- `AZURE_LOCATION` (default: `southeastasia`)
- `AZURE_WORKSPACE_NAME` (default: `churn-ml-workspace`)
- `AZURE_COMPUTE_CLUSTER_NAME` (default: `cpu-cluster`)
- `COMPUTE_CLUSTER_SIZE` (default: `Standard_DS2_v2`)
- `MIN_NODES` (default: `0`)
- `MAX_NODES` (default: `2`)
- `AZURE_ACR_NAME` (optional: if set, creates ACR during setup)
- `ACR_SKU` (default: `Basic` - options: Basic, Standard, Premium)

### Create Data Asset

After setting up the workspace, create a data asset from your local dataset:

**Linux/Mac:**

```bash
python setup/create_data_asset.py --data-path data
```

**Windows (PowerShell):**

```powershell
python setup/create_data_asset.py --data-path data
```

**Options:**

- `--data-path`: Path to data directory or file (default: `data`)
- `--name`: Data asset name (default: from `DATA_ASSET_FULL` in `config.env`)
- `--version`: Data asset version (default: from `DATA_VERSION` in `config.env` or `1`)
- `--description`: Description of the data asset

**Example:**

```bash
# Use defaults from config.env
python setup/create_data_asset.py

# Specify custom name and version
python setup/create_data_asset.py --name churn-data --version 1

# Use a specific data directory
python setup/create_data_asset.py --data-path data --name churn-data --version 1
```

**Important:** The data asset is registered as `uri_folder` type, which is required for the `data_prep` component. The script automatically uses the parent directory if you specify a file path.

## Azure Container Registry (ACR) Setup

**Important**: For proper ACR authentication, ACR should be created **before** the compute cluster. The setup script follows this order automatically. If you create ACR after compute cluster, you'll need to manually grant AcrPull role to the compute's managed identity.

### Create ACR

If you didn't create ACR during initial setup, you can create it manually:

**Linux/Mac:**
  
```bash
az acr create \
  --resource-group "${AZURE_RESOURCE_GROUP:-rg-churn-ml-project-2025-11-15}" \
  --name <your-acr-name> \
  --sku Basic \
  --location "${AZURE_LOCATION:-southeastasia}"
```

**Windows (PowerShell):**

```powershell
az acr create `
  --resource-group $env:AZURE_RESOURCE_GROUP `
  --name <your-acr-name> `
  --sku Basic `
  --location $env:AZURE_LOCATION
```

**Important:**

- ACR name must be globally unique (3-50 characters, alphanumeric only)
- After creating ACR, update `AZURE_ACR_NAME` in `config.env`
- Update `aml/environments/environment.yml` to replace `<your-acr-name>` with your actual ACR name
- **If compute cluster was created before ACR**: You need to manually grant AcrPull role to compute's managed identity (see [TROUBLESHOOTING.md](../docs/TROUBLESHOOTING.md#acr-authentication-for-compute-cluster))
- **If ACR exists before compute cluster**: AcrPull role is automatically granted when compute is created with `--identity-type systemassigned`

### Verify ACR Creation

**Linux/Mac:**

```bash
az acr show \
  --name <your-acr-name> \
  --resource-group "${AZURE_RESOURCE_GROUP:-rg-churn-ml-project}" \
  --query "{Name:name, LoginServer:loginServer, Sku:sku.name}" \
  -o table
```

**Windows (PowerShell):**

```powershell
az acr show `
  --name <your-acr-name> `
  --resource-group $env:AZURE_RESOURCE_GROUP `
  --query "{Name:name, LoginServer:loginServer, Sku:sku.name}" `
  -o table
```

### Update Environment Configuration

After creating ACR, update the environment YAML file:

1. Edit `aml/environments/environment.yml`
2. Replace `<your-acr-name>` with your actual ACR name in the `image` field:

   ```yaml
   image: <your-acr-name>.azurecr.io/bank-churn:1
   ```

   Should become:

   ```yaml
   image: myregistry.azurecr.io/bank-churn:1
   ```

### Common ACR Commands

**List repositories in ACR:**

**Linux/Mac:**

```bash
az acr repository list --name <your-acr-name> --output table
```

**Windows (PowerShell):**

```powershell
az acr repository list --name <your-acr-name> --output table
```

**List tags for a repository:**

**Linux/Mac:**

```bash
az acr repository show-tags --name <your-acr-name> --repository bank-churn --output table
```

**Windows (PowerShell):**

```powershell
az acr repository show-tags --name <your-acr-name> --repository bank-churn --output table
```

**Login to ACR:**

**Linux/Mac:**

```bash
az acr login --name <your-acr-name>
```

**Windows (PowerShell):**

```powershell
az acr login --name <your-acr-name>
```

**Delete ACR (if needed):**

**Linux/Mac:**

```bash
az acr delete \
  --name <your-acr-name> \
  --resource-group "${AZURE_RESOURCE_GROUP:-rg-churn-ml-project}" \
  --yes
```

**Windows (PowerShell):**

```powershell
az acr delete `
  --name <your-acr-name> `
  --resource-group $env:AZURE_RESOURCE_GROUP `
  --yes
```

## Access Azure ML Studio

After setup, access your workspace at:

```text
https://ml.azure.com
```

Navigate to your workspace: `${AZURE_WORKSPACE_NAME:-churn-ml-workspace}`

## Common Azure ML Commands

### Check Compute Cluster Status

**Linux/Mac:**

```bash
az ml compute show \
    --name "${AZURE_COMPUTE_CLUSTER_NAME:-cpu-cluster}" \
    --resource-group "${AZURE_RESOURCE_GROUP:-rg-churn-ml-project}" \
    --workspace-name "${AZURE_WORKSPACE_NAME:-churn-ml-workspace}" \
    --query "{Name:name, State:provisioning_state, MinNodes:min_instances, MaxNodes:max_instances}" \
    --output table
```

**Windows (PowerShell):**

```powershell
az ml compute show `
    --name $env:AZURE_COMPUTE_CLUSTER_NAME `
    --resource-group $env:AZURE_RESOURCE_GROUP `
    --workspace-name $env:AZURE_WORKSPACE_NAME `
    --query "{Name:name, State:provisioning_state, MinNodes:min_instances, MaxNodes:max_instances}" `
    --output table
```

**Note:** Compute cluster auto-scales to 0 nodes when idle, so no charges when not in use.

### List All Compute Resources

**Linux/Mac:**

```bash
az ml compute list \
    --resource-group "${AZURE_RESOURCE_GROUP:-rg-churn-ml-project}" \
    --workspace-name "${AZURE_WORKSPACE_NAME:-churn-ml-workspace}" \
    --output table
```

**Windows (PowerShell):**

```powershell
az ml compute list `
    --resource-group $env:AZURE_RESOURCE_GROUP `
    --workspace-name $env:AZURE_WORKSPACE_NAME `
    --output table
```

### Delete Compute Cluster

**Linux/Mac:**

```bash
az ml compute delete \
    --name "${AZURE_COMPUTE_CLUSTER_NAME:-cpu-cluster}" \
    --resource-group "${AZURE_RESOURCE_GROUP:-rg-churn-ml-project}" \
    --workspace-name "${AZURE_WORKSPACE_NAME:-churn-ml-workspace}" \
    --yes
```

**Windows (PowerShell):**

```powershell
az ml compute delete `
    --name $env:AZURE_COMPUTE_CLUSTER_NAME `
    --resource-group $env:AZURE_RESOURCE_GROUP `
    --workspace-name $env:AZURE_WORKSPACE_NAME `
    --yes
```

### Delete Resource Group

**Warning:** This will delete the entire resource group and all resources within it (workspace, compute cluster, data assets, etc.). This action cannot be undone.

**Linux/Mac:**

```bash
az group delete \
    --name "${AZURE_RESOURCE_GROUP:-rg-churn-ml-project}" \
    --yes \
    --no-wait
```

**Or as a one-liner:**

```bash
az group delete --name "${AZURE_RESOURCE_GROUP:-rg-churn-ml-project}" --yes --no-wait
```

**Windows (PowerShell):**

```powershell
az group delete `
    --name $env:AZURE_RESOURCE_GROUP `
    --yes `
    --no-wait
```

**Note:** The `--no-wait` flag allows the deletion to proceed asynchronously. Remove it if you want to wait for confirmation of deletion.

## Cost Management Tips

- **Compute Cluster**: Only charges when nodes are active. Auto-scales to 0 when idle.
- Use `MIN_NODES=0` to ensure cluster scales down completely when idle.

## Troubleshooting

## Git Remote Setup

If you need to set up a git remote repository:

**Linux/Mac:**

```bash
git remote add origin <repository-url>
git branch -M main
git push -u origin main
```

**Windows (PowerShell):**

```powershell
git remote add origin <repository-url>
git branch -M main
git push -u origin main
```
