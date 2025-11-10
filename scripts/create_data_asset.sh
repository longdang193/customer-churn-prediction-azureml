#!/bin/bash
# Create Azure ML Data Asset using Azure CLI

set -e

if [ ! -f "config.env" ]; then
    echo "ERROR: config.env file not found!"
    echo "Please copy config.env.example to config.env and fill in your values."
    exit 1
fi

source config.env

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo "============================================================"
echo "Creating Azure ML Data Asset"
echo "============================================================"

echo -e "\n${GREEN}[1/4]${NC} Checking Azure CLI authentication..."
if ! az account show &> /dev/null; then
    echo -e "${RED}Not logged in. Please run: az login${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Authenticated"

echo -e "\n${GREEN}[2/4]${NC} Setting subscription..."
az account set --subscription "$AZURE_SUBSCRIPTION_ID"
echo -e "${GREEN}✓${NC} Using subscription: $AZURE_SUBSCRIPTION_ID"

echo -e "\n${GREEN}[3/4]${NC} Verifying workspace..."
if ! az ml workspace show \
    --resource-group "$AZURE_RESOURCE_GROUP" \
    --name "$AZURE_WORKSPACE_NAME" &> /dev/null; then
    echo -e "${RED}Workspace not found: $AZURE_WORKSPACE_NAME${NC}"
    echo "Please create the workspace first by running: ./setup.sh"
    exit 1
fi
echo -e "${GREEN}✓${NC} Workspace exists: $AZURE_WORKSPACE_NAME"

echo -e "\n${GREEN}[4/4]${NC} Creating data asset..."

DATASTORE=$(az ml datastore list \
    --resource-group "$AZURE_RESOURCE_GROUP" \
    --workspace-name "$AZURE_WORKSPACE_NAME" \
    --query "[?is_default].name" -o tsv)

if [ -z "$DATASTORE" ]; then
    DATASTORE="workspaceblobstore"
fi

echo "  Using datastore: $DATASTORE"
echo "  Creating asset: $DATA_ASSET_FULL (version $DATA_VERSION)"

az ml data create \
    --resource-group "$AZURE_RESOURCE_GROUP" \
    --workspace-name "$AZURE_WORKSPACE_NAME" \
    --name "$DATA_ASSET_FULL" \
    --version "$DATA_VERSION" \
    --type uri_file \
    --path "azureml://datastores/$DATASTORE/paths/churn.csv" \
    --description "Bank Customer Churn Dataset - Full dataset (10k rows)"

echo -e "\n${GREEN}✓${NC} Data asset created successfully!"

echo -e "\n${GREEN}Verifying data asset...${NC}"
az ml data show \
    --resource-group "$AZURE_RESOURCE_GROUP" \
    --workspace-name "$AZURE_WORKSPACE_NAME" \
    --name "$DATA_ASSET_FULL" \
    --version "$DATA_VERSION"

echo ""
echo "============================================================"
echo -e "${GREEN}SUCCESS!${NC} Data asset created."
echo "============================================================"
echo ""
echo "Reference in pipelines: azureml:$DATA_ASSET_FULL:$DATA_VERSION"
echo "View in Azure ML Studio: https://ml.azure.com"
echo "============================================================"

