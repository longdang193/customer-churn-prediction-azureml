# Check status of Azure ML compute resources
$ErrorActionPreference = "Stop"

# Load configuration
$ResourceGroup = if ($env:AZURE_RESOURCE_GROUP) { $env:AZURE_RESOURCE_GROUP } else { "rg-churn-ml-project" }
$WorkspaceName = if ($env:AZURE_WORKSPACE_NAME) { $env:AZURE_WORKSPACE_NAME } else { "churn-ml-workspace" }
$ComputeInstanceName = if ($env:AZURE_COMPUTE_INSTANCE_NAME) { $env:AZURE_COMPUTE_INSTANCE_NAME } else { "churn-compute-inst" }
$ComputeClusterName = if ($env:AZURE_COMPUTE_CLUSTER_NAME) { $env:AZURE_COMPUTE_CLUSTER_NAME } else { "cpu-cluster" }

Write-Host "=== Compute Instance Status ===" -ForegroundColor Cyan
az ml compute show `
    --name $ComputeInstanceName `
    --resource-group $ResourceGroup `
    --workspace-name $WorkspaceName `
    --query "{Name:name, State:provisioning_state, Status:state, Size:size}" `
    --output table

Write-Host ""
Write-Host "=== Compute Cluster Status ===" -ForegroundColor Cyan
az ml compute show `
    --name $ComputeClusterName `
    --resource-group $ResourceGroup `
    --workspace-name $WorkspaceName `
    --query "{Name:name, State:provisioning_state, MinNodes:min_instances, MaxNodes:max_instances}" `
    --output table

Write-Host ""
Write-Host "💡 Tip: Compute instance charges when running. Compute cluster only charges when nodes are active." -ForegroundColor Yellow

