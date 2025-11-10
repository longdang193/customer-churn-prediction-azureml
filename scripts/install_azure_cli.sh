#!/bin/bash
# Install Azure CLI in GitHub Codespace

set -e

echo "Installing Azure CLI..."

# Install dependencies
sudo apt-get update
sudo apt-get install -y ca-certificates curl apt-transport-https lsb-release gnupg

# Download and install Microsoft signing key
sudo mkdir -p /etc/apt/keyrings
curl -sLS https://packages.microsoft.com/keys/microsoft.asc |
    gpg --dearmor |
    sudo tee /etc/apt/keyrings/microsoft.gpg > /dev/null
sudo chmod go+r /etc/apt/keyrings/microsoft.gpg

# Add Azure CLI repository
AZ_DIST=$(lsb_release -cs)
echo "deb [arch=`dpkg --print-architecture` signed-by=/etc/apt/keyrings/microsoft.gpg] https://packages.microsoft.com/repos/azure-cli/ $AZ_DIST main" |
    sudo tee /etc/apt/sources.list.d/azure-cli.list

# Install Azure CLI
sudo apt-get update
sudo apt-get install -y azure-cli

echo ""
echo "✓ Azure CLI installed successfully!"
echo ""
echo "Verify installation:"
az --version

