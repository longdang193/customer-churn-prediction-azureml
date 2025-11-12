#!/bin/bash

# Start Azure ML Compute Instance
set -e

# Load configuration
RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:-rg-churn-ml-project}"
WORKSPACE_NAME="${AZURE_WORKSPACE_NAME:-churn-ml-workspace}"
COMPUTE_INSTANCE_NAME="${AZURE_COMPUTE_INSTANCE_NAME:-churn-compute-inst}"

echo "Starting compute instance: $COMPUTE_INSTANCE_NAME"
az ml compute start \
    --name "$COMPUTE_INSTANCE_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --workspace-name "$WORKSPACE_NAME"

echo "✓ Compute instance starting..."
echo "Note: It may take 2-3 minutes to be ready. Check status in Azure ML Studio."

