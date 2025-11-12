#!/bin/bash

# Azure ML Setup Script - Creates workspace and compute resources
set -e

# Configuration
RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:-rg-churn-ml-project}"
LOCATION="${AZURE_LOCATION:-southeastasia}"
WORKSPACE_NAME="${AZURE_WORKSPACE_NAME:-churn-ml-workspace}"
COMPUTE_INSTANCE_NAME="${AZURE_COMPUTE_INSTANCE_NAME:-churn-compute-inst}"
COMPUTE_CLUSTER_NAME="${AZURE_COMPUTE_CLUSTER_NAME:-cpu-cluster}"
COMPUTE_INSTANCE_SIZE="${COMPUTE_INSTANCE_SIZE:-Standard_DS2_v2}"
COMPUTE_CLUSTER_SIZE="${COMPUTE_CLUSTER_SIZE:-Standard_DS2_v2}"
MIN_NODES="${MIN_NODES:-0}"
MAX_NODES="${MAX_NODES:-2}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

check_azure_cli() {
    if ! command -v az &> /dev/null; then
        print_error "Azure CLI not installed. Install from: https://docs.microsoft.com/cli/azure/install-azure-cli"
        exit 1
    fi
}

check_azure_login() {
    if ! az account show &> /dev/null; then
        print_warning "Not logged in. Running: az login"
        az login
    fi
    SUBSCRIPTION_ID=$(az account show --query id -o tsv)
    print_info "Using subscription: $SUBSCRIPTION_ID"
}

create_resource_group() {
    print_info "Creating resource group: $RESOURCE_GROUP"
    if az group show --name "$RESOURCE_GROUP" &> /dev/null; then
        print_warning "Resource group already exists"
    else
        az group create --name "$RESOURCE_GROUP" --location "$LOCATION"
        print_info "Resource group created"
    fi
}

create_workspace() {
    print_info "Creating workspace: $WORKSPACE_NAME"
    if az ml workspace show --resource-group "$RESOURCE_GROUP" --name "$WORKSPACE_NAME" &> /dev/null; then
        print_warning "Workspace already exists"
    else
        az ml workspace create --resource-group "$RESOURCE_GROUP" --name "$WORKSPACE_NAME" --location "$LOCATION"
        print_info "Workspace created"
    fi
}

create_compute_instance() {
    print_info "Creating compute instance: $COMPUTE_INSTANCE_NAME"
    if az ml compute show --resource-group "$RESOURCE_GROUP" --workspace-name "$WORKSPACE_NAME" --name "$COMPUTE_INSTANCE_NAME" &> /dev/null; then
        print_warning "Compute instance already exists"
        return
    fi
    az ml compute create --resource-group "$RESOURCE_GROUP" --workspace-name "$WORKSPACE_NAME" \
        --name "$COMPUTE_INSTANCE_NAME" --type ComputeInstance --size "$COMPUTE_INSTANCE_SIZE"
    print_warning "Compute instance creation takes 5-10 minutes. Check status in Azure ML Studio."
}

create_compute_cluster() {
    print_info "Creating compute cluster: $COMPUTE_CLUSTER_NAME"
    if az ml compute show --resource-group "$RESOURCE_GROUP" --workspace-name "$WORKSPACE_NAME" --name "$COMPUTE_CLUSTER_NAME" &> /dev/null; then
        print_warning "Compute cluster already exists"
        return
    fi
    az ml compute create --resource-group "$RESOURCE_GROUP" --workspace-name "$WORKSPACE_NAME" \
        --name "$COMPUTE_CLUSTER_NAME" --type AmlCompute --size "$COMPUTE_CLUSTER_SIZE" \
        --min-instances "$MIN_NODES" --max-instances "$MAX_NODES" --idle-time-before-scale-down 1800
    print_info "Compute cluster created"
}

display_info() {
    print_info "=== Setup Complete ==="
    echo "Resource Group: $RESOURCE_GROUP"
    echo "Workspace: $WORKSPACE_NAME"
    echo "Location: $LOCATION"
    echo "Compute Instance: $COMPUTE_INSTANCE_NAME"
    echo "Compute Cluster: $COMPUTE_CLUSTER_NAME ($MIN_NODES-$MAX_NODES nodes)"
    echo ""
    echo "Next: Access https://ml.azure.com and navigate to workspace: $WORKSPACE_NAME"
}

main() {
    print_info "Starting Azure ML setup..."
    check_azure_cli
    check_azure_login
    create_resource_group
    create_workspace
    create_compute_instance
    create_compute_cluster
    echo ""
    display_info
    print_info "Setup completed successfully!"
}

main
