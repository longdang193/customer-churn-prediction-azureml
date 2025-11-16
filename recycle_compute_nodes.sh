#!/bin/bash
# Helper script to recycle compute nodes after ACR authentication changes

set -e

# Load config
source <(grep -E "AZURE_RESOURCE_GROUP|AZURE_WORKSPACE_NAME|AZURE_COMPUTE_CLUSTER_NAME" config.env | sed 's/^/export /' | sed 's/"//g')

echo "=== Recycling Compute Nodes ==="
echo "Resource Group: $AZURE_RESOURCE_GROUP"
echo "Workspace: $AZURE_WORKSPACE_NAME"
echo "Compute Cluster: $AZURE_COMPUTE_CLUSTER_NAME"
echo ""

echo "IMPORTANT: You need to delete running nodes via Azure Portal:"
echo "1. Go to https://ml.azure.com"
echo "2. Navigate to: Compute -> $AZURE_COMPUTE_CLUSTER_NAME"
echo "3. Click on running nodes and DELETE them"
echo "4. New nodes will start with updated ACR credentials"
echo ""
echo "Alternatively, wait ~30 minutes for auto-scale down (if min_instances=0)"
echo ""
echo "Current cluster settings:"
az ml compute show --name $AZURE_COMPUTE_CLUSTER_NAME \
  --resource-group $AZURE_RESOURCE_GROUP \
  --workspace-name $AZURE_WORKSPACE_NAME \
  --query "{Name:name, MinInstances:scale_settings.min_node_count, MaxInstances:scale_settings.max_node_count}" \
  -o table 2>&1 | grep -v "Class DeploymentTemplateOperations" | grep -v "experimental" || true
