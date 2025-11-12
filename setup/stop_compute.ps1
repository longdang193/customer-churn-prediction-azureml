# Stop Azure ML Compute Instance to avoid charges
$ErrorActionPreference = "Stop"

# Load configuration
$ResourceGroup = if ($env:AZURE_RESOURCE_GROUP) { $env:AZURE_RESOURCE_GROUP } else { "rg-churn-ml-project" }
$WorkspaceName = if ($env:AZURE_WORKSPACE_NAME) { $env:AZURE_WORKSPACE_NAME } else { "churn-ml-workspace" }
$ComputeInstanceName = if ($env:AZURE_COMPUTE_INSTANCE_NAME) { $env:AZURE_COMPUTE_INSTANCE_NAME } else { "churn-compute-inst" }

Write-Host "Stopping compute instance: $ComputeInstanceName" -ForegroundColor Yellow
az ml compute stop `
    --name $ComputeInstanceName `
    --resource-group $ResourceGroup `
    --workspace-name $WorkspaceName

Write-Host "✓ Compute instance stopped successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "Note: The compute cluster (cpu-cluster) auto-scales to 0 nodes when idle, so no charges."

