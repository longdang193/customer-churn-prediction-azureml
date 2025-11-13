# Setup Scripts and Common Commands

This directory contains setup scripts for Azure ML workspace and compute resources.

## Setup Scripts

### Initial Setup

Run the setup script to create Azure ML workspace and compute resources:

**Linux/Mac:**
```bash
./setup/setup.sh
```

**Windows (PowerShell):**
```powershell
.\setup\setup.ps1
```

**What it does:**
- Creates resource group (if it doesn't exist)
- Creates Azure ML workspace
- Creates compute instance
- Creates compute cluster

**Configuration:**
The scripts use environment variables (with defaults):
- `AZURE_RESOURCE_GROUP` (default: `rg-churn-ml-project`)
- `AZURE_LOCATION` (default: `southeastasia`)
- `AZURE_WORKSPACE_NAME` (default: `churn-ml-workspace`)
- `AZURE_COMPUTE_INSTANCE_NAME` (default: `churn-compute-inst`)
- `AZURE_COMPUTE_CLUSTER_NAME` (default: `cpu-cluster`)
- `COMPUTE_INSTANCE_SIZE` (default: `Standard_DS2_v2`)
- `COMPUTE_CLUSTER_SIZE` (default: `Standard_DS2_v2`)
- `MIN_NODES` (default: `0`)
- `MAX_NODES` (default: `2`)

## Common Azure ML Commands

### Check Compute Status

**Linux/Mac:**
```bash
# Check compute instance status
az ml compute show \
    --name "${AZURE_COMPUTE_INSTANCE_NAME:-churn-compute-inst}" \
    --resource-group "${AZURE_RESOURCE_GROUP:-rg-churn-ml-project}" \
    --workspace-name "${AZURE_WORKSPACE_NAME:-churn-ml-workspace}" \
    --query "{Name:name, State:provisioning_state, Status:state, Size:size}" \
    --output table

# Check compute cluster status
az ml compute show \
    --name "${AZURE_COMPUTE_CLUSTER_NAME:-cpu-cluster}" \
    --resource-group "${AZURE_RESOURCE_GROUP:-rg-churn-ml-project}" \
    --workspace-name "${AZURE_WORKSPACE_NAME:-churn-ml-workspace}" \
    --query "{Name:name, State:provisioning_state, MinNodes:min_instances, MaxNodes:max_instances}" \
    --output table
```

**Windows (PowerShell):**
```powershell
# Check compute instance status
az ml compute show `
    --name $env:AZURE_COMPUTE_INSTANCE_NAME `
    --resource-group $env:AZURE_RESOURCE_GROUP `
    --workspace-name $env:AZURE_WORKSPACE_NAME `
    --query "{Name:name, State:provisioning_state, Status:state, Size:size}" `
    --output table

# Check compute cluster status
az ml compute show `
    --name $env:AZURE_COMPUTE_CLUSTER_NAME `
    --resource-group $env:AZURE_RESOURCE_GROUP `
    --workspace-name $env:AZURE_WORKSPACE_NAME `
    --query "{Name:name, State:provisioning_state, MinNodes:min_instances, MaxNodes:max_instances}" `
    --output table
```

### Start Compute Instance

**Linux/Mac:**
```bash
az ml compute start \
    --name "${AZURE_COMPUTE_INSTANCE_NAME:-churn-compute-inst}" \
    --resource-group "${AZURE_RESOURCE_GROUP:-rg-churn-ml-project}" \
    --workspace-name "${AZURE_WORKSPACE_NAME:-churn-ml-workspace}"
```

**Windows (PowerShell):**
```powershell
az ml compute start `
    --name $env:AZURE_COMPUTE_INSTANCE_NAME `
    --resource-group $env:AZURE_RESOURCE_GROUP `
    --workspace-name $env:AZURE_WORKSPACE_NAME
```

**Note:** Compute instance takes 2-3 minutes to be ready. Check status in Azure ML Studio.

### Stop Compute Instance

**Linux/Mac:**
```bash
az ml compute stop \
    --name "${AZURE_COMPUTE_INSTANCE_NAME:-churn-compute-inst}" \
    --resource-group "${AZURE_RESOURCE_GROUP:-rg-churn-ml-project}" \
    --workspace-name "${AZURE_WORKSPACE_NAME:-churn-ml-workspace}"
```

**Windows (PowerShell):**
```powershell
az ml compute stop `
    --name $env:AZURE_COMPUTE_INSTANCE_NAME `
    --resource-group $env:AZURE_RESOURCE_GROUP `
    --workspace-name $env:AZURE_WORKSPACE_NAME
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

### Delete Compute Resources

**Linux/Mac:**
```bash
# Delete compute instance
az ml compute delete \
    --name "${AZURE_COMPUTE_INSTANCE_NAME:-churn-compute-inst}" \
    --resource-group "${AZURE_RESOURCE_GROUP:-rg-churn-ml-project}" \
    --workspace-name "${AZURE_WORKSPACE_NAME:-churn-ml-workspace}" \
    --yes

# Delete compute cluster
az ml compute delete \
    --name "${AZURE_COMPUTE_CLUSTER_NAME:-cpu-cluster}" \
    --resource-group "${AZURE_RESOURCE_GROUP:-rg-churn-ml-project}" \
    --workspace-name "${AZURE_WORKSPACE_NAME:-churn-ml-workspace}" \
    --yes
```

**Windows (PowerShell):**
```powershell
# Delete compute instance
az ml compute delete `
    --name $env:AZURE_COMPUTE_INSTANCE_NAME `
    --resource-group $env:AZURE_RESOURCE_GROUP `
    --workspace-name $env:AZURE_WORKSPACE_NAME `
    --yes

# Delete compute cluster
az ml compute delete `
    --name $env:AZURE_COMPUTE_CLUSTER_NAME `
    --resource-group $env:AZURE_RESOURCE_GROUP `
    --workspace-name $env:AZURE_WORKSPACE_NAME `
    --yes
```

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

## Prerequisites

- Azure CLI installed: [Install Azure CLI](https://docs.microsoft.com/cli/azure/install-azure-cli)
- Authenticated with Azure: `az login`
- Appropriate permissions to create resources in the subscription

## Cost Management Tips

- **Compute Instance**: Charges when running. Stop it when not in use.
- **Compute Cluster**: Only charges when nodes are active. Auto-scales to 0 when idle.
- Use `MIN_NODES=0` to ensure cluster scales down completely when idle.

## Access Azure ML Studio

After setup, access your workspace at:
```
https://ml.azure.com
```

Navigate to your workspace: `${AZURE_WORKSPACE_NAME:-churn-ml-workspace}`

