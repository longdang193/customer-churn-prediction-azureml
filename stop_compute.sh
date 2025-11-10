#!/bin/bash

# Stop Azure ML Compute Instance to avoid charges
set -e

# Load configuration
RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:-rg-churn-ml-project}"
WORKSPACE_NAME="${AZURE_WORKSPACE_NAME:-churn-ml-workspace}"
COMPUTE_INSTANCE_NAME="${AZURE_COMPUTE_INSTANCE_NAME:-churn-compute-inst}"

echo "Stopping compute instance: $COMPUTE_INSTANCE_NAME"
az ml compute stop \
    --name "$COMPUTE_INSTANCE_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --workspace-name "$WORKSPACE_NAME"

echo "✓ Compute instance stopped successfully!"
echo ""
echo "Note: The compute cluster (cpu-cluster) auto-scales to 0 nodes when idle, so no charges."

