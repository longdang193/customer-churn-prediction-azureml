# Azure ML Setup Script - Creates workspace and compute resources
$ErrorActionPreference = "Stop"

# Configuration
$ResourceGroup = if ($env:AZURE_RESOURCE_GROUP) { $env:AZURE_RESOURCE_GROUP } else { "rg-churn-ml-project" }
$Location = if ($env:AZURE_LOCATION) { $env:AZURE_LOCATION } else { "southeastasia" }
$WorkspaceName = if ($env:AZURE_WORKSPACE_NAME) { $env:AZURE_WORKSPACE_NAME } else { "churn-ml-workspace" }
$ComputeClusterName = if ($env:AZURE_COMPUTE_CLUSTER_NAME) { $env:AZURE_COMPUTE_CLUSTER_NAME } else { "cpu-cluster" }
$ComputeClusterSize = if ($env:COMPUTE_CLUSTER_SIZE) { $env:COMPUTE_CLUSTER_SIZE } else { "Standard_DS2_v2" }
$MinNodes = if ($env:MIN_NODES) { [int]$env:MIN_NODES } else { 0 }
$MaxNodes = if ($env:MAX_NODES) { [int]$env:MAX_NODES } else { 2 }

function Write-Info { param([string]$Message) Write-Host "[INFO] $Message" -ForegroundColor Green }
function Write-Warning-Custom { param([string]$Message) Write-Host "[WARNING] $Message" -ForegroundColor Yellow }
function Write-Error-Custom { param([string]$Message) Write-Host "[ERROR] $Message" -ForegroundColor Red }

function Test-AzureCLI {
    try {
        $null = az --version 2>&1 | Select-Object -First 1
    }
    catch {
        Write-Error-Custom "Azure CLI not installed. Install from: https://docs.microsoft.com/cli/azure/install-azure-cli"
        exit 1
    }
}

function Test-AzureLogin {
    try {
        $null = az account show 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Warning-Custom "Not logged in. Running: az login"
            az login
        }
        $subscriptionId = az account show --query id -o tsv
        Write-Info "Using subscription: $subscriptionId"
    }
    catch {
        Write-Error-Custom "Failed to authenticate with Azure"
        exit 1
    }
}

function New-ResourceGroup {
    Write-Info "Creating resource group: $ResourceGroup"
    $null = az group show --name $ResourceGroup 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Warning-Custom "Resource group already exists"
    }
    else {
        az group create --name $ResourceGroup --location $Location
        Write-Info "Resource group created"
    }
}

function New-Workspace {
    Write-Info "Creating workspace: $WorkspaceName"
    $null = az ml workspace show --resource-group $ResourceGroup --name $WorkspaceName 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Warning-Custom "Workspace already exists"
    }
    else {
        az ml workspace create --resource-group $ResourceGroup --name $WorkspaceName --location $Location
        Write-Info "Workspace created"
    }
}

function New-ComputeCluster {
    Write-Info "Creating compute cluster: $ComputeClusterName"
    $null = az ml compute show --resource-group $ResourceGroup --workspace-name $WorkspaceName --name $ComputeClusterName 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Warning-Custom "Compute cluster already exists"
        return
    }
    az ml compute create --resource-group $ResourceGroup --workspace-name $WorkspaceName `
        --name $ComputeClusterName --type AmlCompute --size $ComputeClusterSize `
        --min-instances $MinNodes --max-instances $MaxNodes --idle-time-before-scale-down 1800
    Write-Info "Compute cluster created"
}

function Show-Info {
    Write-Info "=== Setup Complete ==="
    Write-Host "Resource Group: $ResourceGroup"
    Write-Host "Workspace: $WorkspaceName"
    Write-Host "Location: $Location"
    Write-Host "Compute Cluster: $ComputeClusterName ($MinNodes-$MaxNodes nodes)"
    Write-Host ""
    Write-Host "Next: Access https://ml.azure.com and navigate to workspace: $WorkspaceName"
}

function Main {
    Write-Info "Starting Azure ML setup..."
    Test-AzureCLI
    Test-AzureLogin
    New-ResourceGroup
    New-Workspace
    New-ComputeCluster
    Write-Host ""
    Show-Info
    Write-Info "Setup completed successfully!"
}

Main
