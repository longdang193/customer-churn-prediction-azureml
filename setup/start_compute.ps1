# Start Azure ML Compute Instance
$ErrorActionPreference = "Stop"

# Load configuration
$ResourceGroup = if ($env:AZURE_RESOURCE_GROUP) { $env:AZURE_RESOURCE_GROUP } else { "rg-churn-ml-project" }
$WorkspaceName = if ($env:AZURE_WORKSPACE_NAME) { $env:AZURE_WORKSPACE_NAME } else { "churn-ml-workspace" }
$ComputeInstanceName = if ($env:AZURE_COMPUTE_INSTANCE_NAME) { $env:AZURE_COMPUTE_INSTANCE_NAME } else { "churn-compute-inst" }

Write-Host "Starting compute instance: $ComputeInstanceName" -ForegroundColor Yellow
az ml compute start `
    --name $ComputeInstanceName `
    --resource-group $ResourceGroup `
    --workspace-name $WorkspaceName

Write-Host "✓ Compute instance starting..." -ForegroundColor Green
Write-Host "Note: It may take 2-3 minutes to be ready. Check status in Azure ML Studio."

