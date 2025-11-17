# Troubleshooting Guide

This document contains troubleshooting information and solutions for common issues when running the Azure ML pipeline for customer churn prediction.

## Environment Setup

### Python Version Requirements

**Critical**: The project requires **Python 3.9** due to compatibility with `azureml-core` 1.1.5.7.

- **Dockerfile**: Uses `python:3.9-slim` as base image
- **Local Development**: Use Python 3.9 virtual environment
- **Type Hints**: Use `Optional[Type]` from `typing` module, not `Type | None` (Python 3.10+ syntax)

See [[docs/python_setup.md]] for detailed Python 3.9 installation and setup instructions.

### Compiling Requirements

**Always compile requirements with Python 3.9** to ensure compatibility:

```bash
# Using Python 3.9 virtual environment (recommended)
python3.9 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel pip-tools

# Compile requirements first
pip-compile requirements.in -o requirements.txt

# Compile dev-requirements with constraints from requirements.txt
# This ensures shared dependencies (matplotlib, scipy, etc.) use compatible versions
pip-compile dev-requirements.in -o dev-requirements.txt --constraint requirements.txt
```

**Or using Docker**:

```bash
docker run --rm -v $(pwd):/workspace -w /workspace python:3.9-slim bash -c \
  "apt-get update -qq && apt-get install -y -qq gcc > /dev/null 2>&1 && \
   pip install --upgrade pip setuptools wheel pip-tools && \
   pip-compile --output-file requirements.txt requirements.in && \
   pip-compile --constraint requirements.txt --output-file dev-requirements.txt dev-requirements.in
```

### Package Version Constraints

For Python 3.9 compatibility, these packages have specific version constraints:

- `numpy`: 1.26.4 (not 2.x)
- `scipy`: 1.13.1 (not 1.16+)
- `matplotlib`: 3.9.4 (not 3.10+)
- `mlflow`: 3.1.4 (not 3.6+)
- `imbalanced-learn`: 0.12.4 (not 0.14+)
- `contourpy`: 1.3.0 (not 1.3.2+)
- `click`: 8.1.8 (not 8.3+)

---

## Azure ML Configuration

### Required Environment Variables

Create `config.env` with the following variables:

```bash
AZURE_SUBSCRIPTION_ID=<your-subscription-id>
AZURE_RESOURCE_GROUP=<your-resource-group>
AZURE_WORKSPACE_NAME=<your-workspace-name>
DATA_ASSET_FULL=churn-data
DATA_VERSION=3
```

**Note**: Both `run_pipeline.py` and `run_hpo.py` automatically load `config.env`. If the file is missing, they fall back to defaults.

### Compute Cluster Setup

**Requirements**:

- `max_instances >= 1` (cannot be 0 - Azure ML requirement)
- `min_instances` can be 0 (allows auto-scale down when idle)
- Managed identity with proper RBAC permissions OR ACR admin user enabled

**Check current settings**:

```bash
az ml compute show --name cpu-cluster --resource-group <rg> --workspace-name <ws> \
  --query "{Name:name, MinInstances:scale_settings.min_node_count, MaxInstances:scale_settings.max_node_count}" \
  -o table
```

**Update settings**:

```bash
# Set min_instances to 0 (allows auto-scale down)
az ml compute update --name cpu-cluster --resource-group <rg> --workspace-name <ws> --set min_instances=0

# Ensure max_instances is at least 1
az ml compute update --name cpu-cluster --resource-group <rg> --workspace-name <ws> --set max_instances=2
```

**Note**: You cannot set `max_instances=0`. To stop all nodes, set `min_instances=0` and wait for auto-scale down, or delete nodes via Azure Portal.

### ACR Authentication for Compute Cluster

The compute cluster needs access to pull Docker images from Azure Container Registry (ACR). Use managed identity with AcrPull for secure, production-ready authentication.

**Important**: When you create a compute cluster with system-assigned managed identity, Azure ML **automatically grants** the `AcrPull` role on the workspace ACR, **if the ACR exists before the compute is created**.

**Create compute cluster with system-assigned managed identity**:

```bash
az ml compute create \
  --name cpu-cluster \
  --type amlcompute \
  --size Standard_DS2_v2 \
  --min-instances 0 \
  --max-instances 2 \
  --identity-type systemassigned \
  --resource-group <rg> \
  --workspace-name <ws>
```

**Automatic role assignment**:

- If workspace ACR exists **before** creating compute: `AcrPull` role is **automatically granted** to compute's managed identity
- If compute is created **before** workspace ACR: You must **manually grant** `AcrPull` role

**Manual role assignment** (only needed if compute was created before ACR):

```bash
# Load values from config.env
source <(grep -E "AZURE_RESOURCE_GROUP|AZURE_WORKSPACE_NAME|AZURE_ACR_NAME|AZURE_SUBSCRIPTION_ID|AZURE_STORAGE_ACCOUNT|AZURE_COMPUTE_CLUSTER_NAME" config.env | sed 's/^/export /' | sed 's/"//g')

# Get compute identity principal ID
COMPUTE_ID=$(az ml compute show \
  --name $AZURE_COMPUTE_CLUSTER_NAME \
  --resource-group $AZURE_RESOURCE_GROUP \
  --workspace-name $AZURE_WORKSPACE_NAME \
  --query identity.principal_id -o tsv 2>/dev/null)

if [ -n "$COMPUTE_ID" ] && [ "$COMPUTE_ID" != "None" ]; then
  echo "Granting AcrPull to compute identity: $COMPUTE_ID"
  
  # Grant AcrPull on ACR (only needed if compute was created before ACR)
  az role assignment create \
    --assignee $COMPUTE_ID \
    --role AcrPull \
    --scope /subscriptions/$AZURE_SUBSCRIPTION_ID/resourceGroups/$AZURE_RESOURCE_GROUP/providers/Microsoft.ContainerRegistry/registries/$AZURE_ACR_NAME

  # Grant Storage Blob Data Reader on storage account
  az role assignment create \
    --assignee $COMPUTE_ID \
    --role "Storage Blob Data Reader" \
    --scope /subscriptions/$AZURE_SUBSCRIPTION_ID/resourceGroups/$AZURE_RESOURCE_GROUP/providers/Microsoft.Storage/storageAccounts/$AZURE_STORAGE_ACCOUNT
else
  echo "Compute cluster doesn't have managed identity enabled"
  echo "Recreate compute with --identity-type systemassigned"
fi
```

**Proper Setup Order**:

1. **Create workspace** (may auto-create ACR)
2. **Create or configure ACR** (ensure it exists for managed identity)
3. **Create compute cluster** with `--identity-type systemassigned` (AcrPull automatically granted if ACR exists)
4. **Push Docker images** to ACR
5. **Register environment** pointing to ACR image

**Reference**: [Azure ML Identity-based Service Authentication](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-identity-based-service-authentication?view=azureml-api-2&tabs=cli#scenario-azure-container-registry-without-admin-user)

---

## Docker & Environment

### Building Docker Image

**Base Image**: `python:3.9-slim` (for compatibility with `azureml-core` 1.1.5.7)

**Important**: The Dockerfile sets `ENV PYTHONPATH=/app/src` to enable imports like `from data import ...` and `from utils import ...`. This matches how Azure ML components run the code (from the `src/` directory).

**Build and push**:

```bash
# Build locally
docker build -t bank-churn:1 .

# Tag for ACR
docker tag bank-churn:1 <your-acr-name>.azurecr.io/bank-churn:1

# Push to ACR
docker push <your-acr-name>.azurecr.io/bank-churn:1

# Verify
az acr repository show-tags --name <your-acr-name> --repository bank-churn
```

### Testing Docker Image

**Test commands** (match how Azure ML runs components):

```bash
# Test data_prep component
docker run --rm bank-churn:1 bash -c "cd /app/src && python data_prep.py --help"

# Test package imports
docker run --rm bank-churn:1 python -c "import pandas; print('OK')"
```

### Registering Azure ML Environment

1. **Update `aml/environments/environment.yml`**:

   ```yaml
   $schema: https://azuremlschemas.azureedge.net/latest/environment.schema.json
   name: bank-churn-env
   version: "1"
   image: <your-acr-name>.azurecr.io/bank-churn:1
   description: Environment for churn prediction pipeline
   ```

2. **Register the environment**:

   ```bash
   az ml environment create --file aml/environments/environment.yml \
     --resource-group <resource-group> \
     --workspace-name <workspace-name>
   ```

3. **Verify**:

   ```bash
   az ml environment show --name bank-churn-env --version 1 \
     --resource-group <resource-group> \
     --workspace-name <workspace-name>
   ```

---

## Data Access

### Registering Data Asset

**Issue**: Pipeline fails with `ResourceNotFoundError` for data asset

**Solution**: Register data asset as `uri_folder` type:

```bash
az ml data upload \
  --name churn-data \
  --version 3 \
  --path data/ \
  --type uri_folder \
  --resource-group <rg> \
  --workspace-name <ws>
```

**Important**: The data asset must be `uri_folder` type. The `data_prep` component expects a folder input and automatically loads all CSV files in the folder.

**Update `config.env`**:

```bash
DATA_ASSET_FULL="churn-data"
DATA_VERSION="1"
```

### Storage Access Issues

**Issue**: `ScriptExecution.StreamAccess.NotFound` when accessing data

**Solution**:

1. Verify managed identity has `Storage Blob Data Reader` role on storage account
2. Ensure compute cluster has system-assigned managed identity enabled
3. Re-upload data to ensure it's accessible

---

## MLflow Integration

### Azure ML MLflow Limitations

Azure ML's MLflow integration has limitations compared to standard MLflow:

#### 1. No Nested Runs

**Issue**: `mlflow.start_run(nested=True)` causes API errors

**Solution**: Code detects Azure ML environment and uses active run directly instead of creating nested runs.

#### 2. No Model Registry API

**Issue**: `mlflow.sklearn.log_model()` returns 404

**Solution**: Models are saved as pickle files to outputs directory. Azure ML automatically captures outputs as artifacts.

#### 3. Artifact API Differences

**Issue**: `mlflow.log_artifact(file_path, artifact_path)` fails with signature error

**Solution**: Use `mlflow.log_artifact(file_path)` without artifact_path parameter, or rely on Azure ML's native artifact capture.

#### 4. Active Run Context

**Issue**: Azure ML automatically creates an active MLflow run, causing conflicts

**Solution**: Code checks for active run before starting a new one:

```python
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "")
is_azure_ml = "azureml" in tracking_uri.lower() if tracking_uri else False

if not is_azure_ml:
    parent_run = mlflow.start_run(run_name="Churn_Training_Pipeline")
else:
    parent_run = mlflow.active_run()
```

---

## Hyperparameter Optimization

### Search Space Configuration

**Best Practices**:

1. **Always include `model_type` in search space**:

   ```yaml
   search_space:
     rf:
       n_estimators: [100, 200]
       max_depth: [6, 10]
     xgboost:
       n_estimators: [100, 200]
       learning_rate: [0.1, 0.2]
     # logreg: {}  # Comment out if no hyperparameters to optimize
   ```

2. **Validate hyperparameter ranges**:
   - Random Forest: `min_samples_split >= 2`, `min_samples_leaf >= 1`
   - XGBoost: `n_estimators > 0`, `max_depth > 0`, `learning_rate > 0`

3. **Test with small budget first**:

   ```yaml
   budget:
     max_trials: 2
   ```

### Model Type Filtering

The code automatically filters hyperparameters based on `model_type`:

- `hpo_utils.py` builds parameter space with `model_type` as categorical choice
- `train.py` receives `--model-type` argument in HPO mode
- `train_all_models()` filters hyperparameters to only include the current model type

**Verification**: Check that each trial in Azure ML Studio has a single `model_type` tag and only relevant hyperparameters logged.

### Notebook-Driven HPO Workflow

The project includes a notebook-driven HPO workflow (`notebooks/hpo_manual_trials.ipynb`) that provides:

1. **Manual control**: Submit sweeps per model type with custom configurations
2. **Result analysis**: Built-in cells to load previous sweeps and analyze best models
3. **Flexible loading**: Load specific sweep jobs by name or auto-discover from experiment

**Key Features**:

- `load_previous_sweeps()` function to retrieve previous sweep submissions
- Best model analysis cell that extracts the winning configuration
- Support for both explicit job names and auto-discovery

**Usage**:

- To submit new sweeps: Configure and run the submission cell
- To load previous sweeps: Use `load_previous_sweeps()` with `SPECIFIC_SWEEP_JOBS` dictionary or enable `auto_discovery=True`
- To analyze results: Run the best model analysis cell

**See**: [HPO Sweep Job Errors (Notebook-Driven Workflow)](#hpo-sweep-job-errors-notebook-driven-workflow) for common issues and solutions.

---

## Common Errors

### Python & Environment Errors

#### `ImportError: cannot import name 'Iterable' from 'collections'`

**Cause**: Python 3.10+ compatibility issue with `azureml-core` 1.1.5.7

**Solution**: Use Python 3.9 in Dockerfile

#### `TypeError: '<' not supported between instances of 'str' and 'int'` when compiling requirements

**Cause**: Outdated setuptools version causing matplotlib build failures

**Solution**: Upgrade setuptools before compiling:

```bash
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip-compile requirements.in -o requirements.txt
```

#### `TypeError: unsupported operand type(s) for |: '_GenericAlias' and 'NoneType'`

**Cause**: Code uses Python 3.10+ union syntax (`JSONDict | None`) but project requires Python 3.9

**Solution**: Replace `|` union syntax with `Optional[Type]` from `typing` module

#### `ImportError: cannot import name 'encode_categoricals' from 'data'` in Docker

**Cause**: Python cannot find the `data` module when running scripts in Docker container

**Error Details**:

- Error occurs when running `python -m src.data_prep` or `python data_prep.py` in Docker
- Code uses absolute imports like `from data import ...` which expect `src/` to be in Python path
- Azure ML components run from `src/` directory (via `code: ../../src`), but Docker test commands may not

**Solution**: The Dockerfile includes `ENV PYTHONPATH=/app/src` to add the `src/` directory to Python's module search path. This allows imports to work correctly.

**Verification**: Test the Docker image:

```bash
docker run --rm bank-churn:1 bash -c "cd /app/src && python data_prep.py --help"
```

**Note**: When testing locally, run commands from `/app/src` directory to match Azure ML's execution environment.

### Azure ML Errors

#### `Requested 1 nodes but AzureMLCompute cluster only has 0 maximum nodes`

**Cause**: Compute cluster `max_instances` set to 0

**Solution**: `az ml compute update --name cpu-cluster --set max_instances=2`

#### `Failed to pull Docker image ... This error may occur because the compute could not authenticate`

**Error Message**:

```text
Failed to pull Docker image churnmlacr2025.azurecr.io/bank-churn:1. 
This error may occur because the compute could not authenticate with the Docker registry 
to pull the image. If using ACR please ensure the ACR has Admin user enabled or a Managed 
Identity with `AcrPull` access to the ACR is assigned to the compute.
```

**Causes**:

1. Compute managed identity lacks `AcrPull` permission (compute was created before ACR, so automatic role assignment didn't occur)
2. Compute cluster doesn't have system-assigned managed identity enabled
3. Docker image doesn't exist in ACR

**Solutions**:

**Solution 1: Use Managed Identity** (Recommended):

**If compute was created with managed identity but before ACR existed**:

```bash
# Manually grant AcrPull role (see [ACR Authentication for Compute Cluster](#acr-authentication-for-compute-cluster))
# Then recycle compute nodes
```

**If compute doesn't have managed identity**:

```bash
# Recreate compute with system-assigned managed identity
# Azure ML will automatically grant AcrPull if ACR exists
az ml compute create \
  --name cpu-cluster \
  --type amlcompute \
  --size Standard_DS2_v2 \
  --min-instances 0 \
  --max-instances 2 \
  --identity-type systemassigned \
  --resource-group <rg> \
  --workspace-name <ws>
```

**Solution 2: Verify Image Exists in ACR**:

```bash
ACR_NAME=$(grep AZURE_ACR_NAME config.env | cut -d'"' -f2)
az acr repository show-tags --name $ACR_NAME --repository bank-churn --output table

# If image doesn't exist, build and push:
docker build -t bank-churn:1 .
docker tag bank-churn:1 $ACR_NAME.azurecr.io/bank-churn:1
docker push $ACR_NAME.azurecr.io/bank-churn:1
```

**Troubleshooting if still failing**:

1. **Verify compute has managed identity**:

   ```bash
   source <(grep -E "AZURE_RESOURCE_GROUP|AZURE_WORKSPACE_NAME|AZURE_COMPUTE_CLUSTER_NAME" config.env | sed 's/^/export /' | sed 's/"//g')
   az ml compute show --name $AZURE_COMPUTE_CLUSTER_NAME \
     --resource-group $AZURE_RESOURCE_GROUP \
     --workspace-name $AZURE_WORKSPACE_NAME \
     --query identity -o json
   ```

2. **Verify AcrPull role is assigned**:

   ```bash
   source <(grep -E "AZURE_RESOURCE_GROUP|AZURE_WORKSPACE_NAME|AZURE_ACR_NAME|AZURE_SUBSCRIPTION_ID|AZURE_COMPUTE_CLUSTER_NAME" config.env | sed 's/^/export /' | sed 's/"//g')
   COMPUTE_ID=$(az ml compute show \
     --name $AZURE_COMPUTE_CLUSTER_NAME \
     --resource-group $AZURE_RESOURCE_GROUP \
     --workspace-name $AZURE_WORKSPACE_NAME \
     --query identity.principal_id -o tsv 2>/dev/null)
   
   az role assignment list \
     --assignee $COMPUTE_ID \
     --scope /subscriptions/$AZURE_SUBSCRIPTION_ID/resourceGroups/$AZURE_RESOURCE_GROUP/providers/Microsoft.ContainerRegistry/registries/$AZURE_ACR_NAME \
     --query "[?roleDefinitionName=='AcrPull']" -o table
   ```

3. **Verify image exists in ACR**:

   ```bash
   ACR_NAME=$(grep AZURE_ACR_NAME config.env | cut -d'"' -f2)
   az acr repository show-tags --name $ACR_NAME --repository bank-churn --output table
   ```

4. **If compute was created before ACR, manually grant AcrPull**:

   See [ACR Authentication for Compute Cluster](#acr-authentication-for-compute-cluster) for manual role assignment instructions.

#### `Could not resolve uris of type data for assets azureml://.../bank-churn-raw/versions/1`

**Cause**: Data asset not found or incorrect configuration in `config.env`

**Solution**:

1. Verify `config.env` has correct `DATA_ASSET_FULL` and `DATA_VERSION`
2. Verify data asset exists: `az ml data show --name <name> --version <version>`
3. Register data asset if missing (see [Registering Data Asset](#registering-data-asset))

### MLflow Errors

#### `mlflow.exceptions.MlflowException: Cannot start run with ID ...`

**Cause**: Trying to start a new MLflow run when Azure ML already has one active

**Solution**: Code automatically handles this by checking for active run (see [MLflow Integration](#mlflow-integration))

#### `API request to endpoint /api/2.0/mlflow/logged-models failed with error code 404`

**Cause**: Azure ML doesn't support MLflow model registry API

**Solution**: Models are saved as pickle files to outputs directory (handled automatically)

#### `azureml_artifacts_builder() takes from 0 to 1 positional arguments but 2 were given`

**Cause**: Azure ML's MLflow artifact APIs have different signatures

**Solution**: Use `mlflow.log_artifact(file_path)` without artifact_path (handled automatically)

### HPO Errors

#### `sklearn.utils._param_validation.InvalidParameterError: The 'min_samples_split' parameter must be an int in the range [2, inf)`

**Cause**: Random Forest search space includes `min_samples_split: [1, 2, ...]` which violates sklearn's requirement

**Solution**: Ensure `min_samples_split >= 2` in `configs/hpo.yaml`:

```yaml
search_space:
  rf:
    min_samples_split: [2, 5, 10]  # Never use 1
```

#### XGBoost models not being trained in HPO

**Cause**: XGBoost not included in search space or `model_type` filtering not working

**Solution**: Ensure `xgboost` is included in `configs/hpo.yaml::search_space`

#### Logistic Regression being trained multiple times with default configurations

**Cause**: Logistic Regression included in search space but has no hyperparameters (`logreg: {}`)

**Solution**: Remove from search space or add hyperparameters to optimize:

```yaml
search_space:
  # logreg: {}  # Commented out - no hyperparameters to optimize
  rf: ...
  xgboost: ...
```

#### Models receiving mismatching hyperparameters

**Cause**: Hyperparameters not filtered by `model_type`

**Solution**: Code automatically filters hyperparameters (see [Model Type Filtering](#model-type-filtering))

### HPO Sweep Job Errors (Notebook-Driven Workflow)

#### `run_sweep_trial.py: error: argument --xgboost_max_depth: expected one argument`

**Error Message**:

```text
Execution failed. User process 'python' exited with status code 2.
Error: usage: run_sweep_trial.py [-h] --data DATA --model-type MODEL_TYPE ...
run_sweep_trial.py: error: argument --xgboost_max_depth: expected one argument
run_sweep_trial.py: error: argument --rf_max_depth: expected one argument
```

**Cause**: The sweep command in `hpo_manual_trials.ipynb` was incorrectly passing hyperparameter flags as `${{inputs.<param>}}` without values. Azure ML sweep jobs need to reference the search space directly.

**Solution**: In the notebook cell that builds the sweep command, change from:

```python
command_segments.append(f"--{prefixed_name} ${{{{inputs.{prefixed_name}}}}}")
```

To:

```python
command_segments.append(f"--{prefixed_name} ${{{{search_space.{prefixed_name}}}}}")
```

This allows Azure ML to inject the sampled values from the search space directly into the command.

#### `ValueError: Invalid override format 'rf_n_estimators=100'`

**Error Message**:

```text
Execution failed. User process 'python' exited with status code 1.
Error: ValueError: Invalid override format 'rf_n_estimators=100'
```

**Cause**: The `train.py` script expects hyperparameter overrides in `model.param=value` format (e.g., `rf.n_estimators=100`), but `run_sweep_trial.py` was passing them as `param=value` format (e.g., `rf_n_estimators=100`).

**Solution**: The `run_sweep_trial.py` script includes a `_format_override_key()` function that converts sweep parameter names to the format expected by `train.py`:

- `rf_n_estimators` → `rf.n_estimators`
- `xgboost_max_depth` → `xgboost.max_depth`
- `logreg_C` → `logreg.C`

This conversion happens automatically when building the CLI arguments for `train.py`.

**Verification**: Check that `run_sweep_trial.py` includes the formatting function and uses it when building the `--set` arguments.

#### Best model analysis cell returns "No completed trials yet" despite jobs being completed

**Issue**: The analysis cell in `hpo_manual_trials.ipynb` shows "No completed trials yet" even when sweep jobs have finished.

**Cause**: The original implementation relied on `sweep_job.best_trial`, which might not be immediately populated or reliable. The sweep job properties need to be accessed directly.

**Solution**: The cell should access sweep job properties directly:

```python
best_child_run_id = sweep_job.properties.get("best_child_run_id")
raw_score = sweep_job.properties.get("score")
```

Then fetch the child job to retrieve its parameters:

```python
child_job = ml_client.jobs.get(best_child_run_id)
params = {k: _coerce(v) for k, v in (getattr(child_job, "parameters", {}) or {}).items()}
```

**Note**: The `best_child_run_id` and `score` properties are populated by Azure ML when the sweep completes. If these are `None`, the sweep may still be running or failed.

#### `ml_client.jobs.list(experiment_name=experiment_name)` fails with TypeError

**Error Message**:

```text
TypeError: list() got an unexpected keyword argument 'experiment_name'
```

**Cause**: The `ml_client.jobs.list()` method does not accept `experiment_name` as a direct argument in the Azure ML SDK v2.

**Solution**: Call `ml_client.jobs.list()` without arguments and filter the results:

```python
for job in ml_client.jobs.list():
    if (job.type == "sweep" 
        and getattr(job, "experiment_name", None) == experiment_name
        and getattr(job, "display_name", "").startswith(prefix)):
        # Process matching sweep job
```

**Note**: The `load_previous_sweeps()` function in `hpo_manual_trials.ipynb` handles this correctly by iterating through all jobs and filtering by attributes.

---

## Best Practices

### 1. Test Locally First

Before running on Azure ML, test locally:

```bash
python src/train.py --data data/processed --experiment-name test
```

### 2. Monitor Pipeline Jobs

Use Azure ML Studio to:

- Check each step's logs
- Verify data access
- Monitor MLflow metrics

### 3. Version Control

- Tag Docker images with versions (not `latest`)
- Use versioned data assets
- Keep track of environment configurations

### 4. Resource Management

- Scale compute clusters appropriately
- Use early stopping in HPO to save resources
- Clean up old jobs and artifacts periodically

### 5. Debugging Tips

**Check compute node logs**:

```bash
az ml job download --name <job-name> --download-path logs/
```

**Verify environment variables**:

```bash
az ml job show --name <job-name> --query environment_variables
```

**Test MLflow connection**:

```python
import mlflow
print(mlflow.get_tracking_uri())
print(mlflow.active_run())
```

---

## Quick Reference

### Run Pipelines

```bash
# Regular training pipeline
python run_pipeline.py

# HPO pipeline (script-based)
python run_hpo.py

# HPO pipeline (notebook-driven - recommended)
# Open notebooks/hpo_manual_trials.ipynb and run cells sequentially
```

**Note**: Both scripts automatically load `config.env` for Azure ML configuration. The notebook-driven workflow provides more control and better visibility into sweep results.

### Check Job Status

```bash
az ml job show --name <job-name> --resource-group <rg> --workspace-name <ws> --query status
az ml job list --parent-job-name <parent-job> --resource-group <rg> --workspace-name <ws>
az ml job download --name <job-name> --resource-group <rg> --workspace-name <ws> --download-path logs/
```

### Docker Commands

```bash
# Build
docker build -t bank-churn:1 .

# Tag and push
docker tag bank-churn:1 <your-acr-name>.azurecr.io/bank-churn:1
docker push <your-acr-name>.azurecr.io/bank-churn:1
```
