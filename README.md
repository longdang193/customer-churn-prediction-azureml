# Customer Churn Prediction with Azure ML

MLOps pipeline for predicting bank customer churn using Azure Machine Learning.

## Project Overview

This project implements a complete MLOps pipeline for customer churn prediction, following best practices for machine learning operations on Azure.

## Project Structure

```
.
├── data/                    # Data files (sample dataset)
├── notebooks/               # Jupyter notebooks for EDA
├── src/                     # Source code (to be created)
├── environment/             # Environment configuration (to be created)
├── setup.sh                 # Azure ML setup script
├── config.env.example       # Configuration template
└── README.md               # This file
```

## Setup

### Prerequisites

- Azure CLI installed and configured
- Azure subscription with appropriate permissions
- Python 3.9+

### 1. Configure Environment

Copy the example configuration file:

```bash
cp config.env.example config.env
```

Edit `config.env` with your Azure subscription details.

### 2. Set Up Azure ML Workspace

Run the setup script to create workspace and compute resources:

```bash
# Bash
./setup.sh

# PowerShell
.\setup.ps1
```

This will create:
- Resource group
- Azure ML workspace
- Compute instance (for notebooks)
- Compute cluster (for training)

### 3. Upload Data

Data has already been uploaded to Azure Blob Storage and registered as data assets:
- `churn-data` - Full dataset (10,000 rows)
- `churn-data-sample` - Sample dataset (1,000 rows)

## Usage

### Start/Stop Compute Instance

To save costs, stop the compute instance when not in use:

```bash
# Stop compute
./stop_compute.sh

# Start compute
./start_compute.sh

# Check status
./compute_status.sh
```

### Run EDA

1. Start the compute instance
2. Open Azure ML Studio: https://ml.azure.com
3. Navigate to Notebooks
4. Upload and run `notebooks/eda.ipynb`

## Project Plan

This project follows the MLZoomcamp Project 1 plan:

1. ✅ Set up project (repo + README)
2. ✅ Prepare data (local + cloud)
3. ✅ Exploratory Data Analysis
4. ⬜ Move logic from notebook → scripts
5. ⬜ Define the environment
6. ⬜ Local testing before AML
7. ⬜ Create AML components
8. ⬜ Build Azure ML pipeline
9. ⬜ Containerization
10. ⬜ Deploy and test
11. ⬜ Document everything

## Cost Management

- **Compute Instance**: Charges when running. Stop when not in use.
- **Compute Cluster**: Auto-scales to 0 nodes when idle (no charge).
- **Workspace**: Free (only storage charges apply).

## Configuration

All Azure resources are configured in `config.env`:
- Subscription ID
- Resource group
- Workspace name
- Compute resources
- Data assets

## Next Steps

1. Complete EDA analysis
2. Create training scripts in `src/`
3. Set up conda environment
4. Build Azure ML pipeline
5. Deploy model as endpoint

## License

This project is part of the MLZoomcamp course.

