#!/bin/bash

# Check status of Azure ML compute resources
set -e

# Load configuration
RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:-rg-churn-ml-project}"
WORKSPACE_NAME="${AZURE_WORKSPACE_NAME:-churn-ml-workspace}"
COMPUTE_INSTANCE_NAME="${AZURE_COMPUTE_INSTANCE_NAME:-churn-compute-inst}"
COMPUTE_CLUSTER_NAME="${AZURE_COMPUTE_CLUSTER_NAME:-cpu-cluster}"

echo "=== Compute Instance Status ==="
az ml compute show \
    --name "$COMPUTE_INSTANCE_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --workspace-name "$WORKSPACE_NAME" \
    --query "{Name:name, State:provisioning_state, Status:state, Size:size}" \
    --output table

echo ""
echo "=== Compute Cluster Status ==="
az ml compute show \
    --name "$COMPUTE_CLUSTER_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --workspace-name "$WORKSPACE_NAME" \
    --query "{Name:name, State:provisioning_state, MinNodes:min_instances, MaxNodes:max_instances}" \
    --output table

echo ""
echo "💡 Tip: Compute instance charges when running. Compute cluster only charges when nodes are active."

