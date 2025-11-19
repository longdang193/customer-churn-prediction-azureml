# Troubleshooting Guide

This document contains troubleshooting information and solutions for common issues when running the Azure ML pipeline for customer churn prediction.

## Table of Contents

- [Environment Setup](#environment-setup)
- [Azure ML Workspace Preparation](#azure-ml-workspace-preparation)
- [Docker & Environment](#docker--environment)
- [Data Access](#data-access)
- [MLflow Integration](#mlflow-integration)
- [Online Endpoint Deployment Issues](#online-endpoint-deployment-issues)
- [Notebook Runtime Errors](#notebook-runtime-errors)
- [Azure ML Service Errors](#azure-ml-service-errors)
  - [HPO (Hyperparameter Optimization) Errors](#hpo-hyperparameter-optimization-errors)

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

## Azure ML Workspace Preparation

### Required Environment Variables

Create `config.env` with the following variables:

```bash
AZURE_SUBSCRIPTION_ID=<your-subscription-id>
AZURE_RESOURCE_GROUP=<your-resource-group>
AZURE_WORKSPACE_NAME=<your-workspace-name>
DATA_ASSET_FULL=churn-data
DATA_VERSION=3
```

**Note**: `run_pipeline.py` automatically loads `config.env`. If the file is missing, it falls back to defaults.

### Resource Provider Registration Errors

**Symptom**: `ResourceOperationFailure: Resource provider [N/A] isn't registered with Subscription [N/A]`

**What it means**: The subscription hasn’t registered the Azure resource providers needed by managed online endpoints (e.g., `Microsoft.MachineLearningServices`, `Microsoft.PolicyInsights`, `Microsoft.Cdn`, etc.). Azure won’t provision environments or endpoints until the registration is complete.

**Fix**:

1. Open Azure Portal → **Subscriptions** → select the subscription you deploy into.
2. Under **Settings**, choose **Resource providers**.
3. Register every provider that shows `NotRegistered`. At a minimum confirm these are registered:
   - `Microsoft.MachineLearningServices`
   - `Microsoft.PolicyInsights`
   - `Microsoft.Cdn`
   - `Microsoft.ContainerRegistry`
   - `Microsoft.Storage`
   - `Microsoft.KeyVault`
   - `Microsoft.ManagedIdentity`
4. Some teams only resolved the error after registering *all* providers surfaced in the list, so if the issue persists, continue registering until none remain unregistered.
5. Wait a few minutes for the registration to propagate, then retry the deployment (`deploy_online_endpoint.ipynb`, CLI, or Studio).

Reference: [Resource provider isn’t registered with subscription](https://learn.microsoft.com/en-us/answers/questions/2129910/resource-provider-(n-a)-isnt-registered-with-subsc).

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
  --version 1 \
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

## Online Endpoint Deployment Issues

The following failures surfaced while running `notebooks/deploy_online_endpoint.ipynb`.

### `FileNotFoundError: No MLflow bundles (*_mlflow/) were found`

**When it happens**: Section 2 of the notebook looks for `outputs/*_mlflow/` (or `AML_MLFLOW_BUNDLE_PATH`) but only finds legacy pickle artifacts.

**Fix**:

- Run `python run_pipeline.py` (or download artifacts from Azure ML) so that `outputs/<model>_mlflow/` exists locally.
- Set `AML_MLFLOW_BUNDLE_PATH=/abs/path/to/<bundle>_mlflow` before running the notebook if the bundle is elsewhere.
- The deployment path expects the entire MLflow directory, not just `_model.pkl`.

### `ValidationException: We could not find config.json in: .`

**Cause**: `MLClient.from_config()` expects `config.json`, but this repo stores credentials in `config.env`.

**Fix**:

```python
from src.utils import load_azure_config
azure_cfg = load_azure_config()
ml_client = MLClient(
    credential,
    subscription_id=azure_cfg["subscription_id"],
    resource_group_name=azure_cfg["resource_group"],
    workspace_name=azure_cfg["workspace_name"],
)
```

Ensure `config.env` exists at the repo root and contains the workspace IDs.

### `BadRequest: Specified deployment [...] is in an unrecoverable state`

**Cause**: Reusing an endpoint name that already has a failed deployment slot (`blue`, `green`, etc.).

**Fix**:

- Before creating the endpoint, call `ml_client.online_endpoints.get()` and delete it if `provisioning_state in ["Failed", "Canceled"]`.
- If the SDK delete hangs, use CLI:

```bash
az ml online-endpoint delete --name <endpoint> --yes
az ml online-endpoint show --name <endpoint> --query provisioning_state
```

Or unset `AML_ONLINE_ENDPOINT_NAME` so the notebook generates a new name.

### `ResourceNotFoundError: ...onlineEndpoints/... could not be found`

**Cause**: Deployment succeeded on a hard-coded endpoint, but the invoke cell used a different timestamp-based name.

**Fix**: Ensure every section uses the same `ENDPOINT_NAME` variable. If you override the env var, update both deployment (Section 6) and invocation (Section 8).

### `BadRequest: Not enough quota available for Standard_DS3_v2`

**Cause**: Default SKU exceeds subscription quota.

**Fix**:

```bash
export AML_ONLINE_INSTANCE_TYPE=Standard_D2as_v4
```

Try again with a smaller SKU or open an Azure quota request.

### `ImageBuildFailure: Deployment failed due to no Environment Image available`

**Observed causes**:

- MLflow bundle `conda.yaml` pins Python 3.12 while the Docker image uses Python 3.9.
- Dependency conflicts (e.g., `mlflow==3.1.4` forcing incompatible `pyarrow`).

**Fix**:

- Re-save/log the model with `conda_env` pinned to Python 3.9 (see `aml/environments/mlflow_conda.yaml`).
- Update the bundle’s `conda.yaml` + `MLmodel` so `python_version` is 3.9.
- Download the referenced build log to inspect the exact failure (`az storage blob download ... image_build_log.txt`).

### `(None) POST body should be a JSON dictionary`

**Cause**: Sending a raw JSON array instead of the MLflow pandas structure.

**Fix**: Invoke with `{"input_data": {"columns": [...], "data": [[...]]}}`. The shipped `sample-data.json` already follows this format; Section 8 simply forwards it via `request_file`.

### `(None) A value is not provided for the 'input_data' parameter.` / `TypeError: expected str, bytes or os.PathLike object, not NoneType`

**Cause**: Mixing `input_data` and `request_file` or skipping the setup cell so `PROJECT_ROOT` is undefined.

**Fix**:

- Run the notebook from the top so Section 1 defines `PROJECT_ROOT`.
- Stick to one invocation style. If you use `request_file`, pass the file path; if you switch to `input_data`, send a dict (e.g., `json.dumps(payload)`), not a path.

### `ValueError: invalid literal for int() with base 10: 'France'`

**Cause**: Payload contains human-readable categorical values, but the model signature expects integer-encoded columns.

**Fix**:

- Encode categorical columns the same way as training (`Geography: France=0, Germany=1, Spain=2`; `Gender: Female=0, Male=1`).
- Use the curated `sample-data.json` or map values before invocation.

## Notebook Runtime Errors

### `NameError: name 'PROJECT_ROOT' is not defined` in notebook

**Error Message**:

```text
NameError: name 'PROJECT_ROOT' is not defined
```

**Cause**: In `notebooks/main.ipynb`, Cell 2 uses `PROJECT_ROOT` which is defined in Cell 1. If Cell 2 is executed before Cell 1, or if Cell 1 fails, `PROJECT_ROOT` will not be defined.

**Solution**: Always execute cells in order. Cell 1 must be executed before Cell 2. The notebook includes a check in Cell 2 that will raise a clear error message if `PROJECT_ROOT` is not defined:

```python
# Ensure PROJECT_ROOT is defined (from Cell 1)
if 'PROJECT_ROOT' not in globals():
    raise RuntimeError("Cell 1 must be executed first to define PROJECT_ROOT")
```

**Best Practice**: Use "Run All" or execute cells sequentially from top to bottom to ensure all dependencies are defined.

### `ModuleNotFoundError: No module named 'src'` in notebooks

**Error Message**:

```text
ModuleNotFoundError: No module named 'src'
```

**Cause**: The notebook is being executed from a directory that is not the project root, so Python cannot resolve the `src/` package.

**Solutions**:

1. **Run the setup cell** at the top of each notebook. It changes the working directory to the repo root and appends it to `sys.path`.
1. **Manually set the working directory** before launching Jupyter:

```bash
cd /workspaces/customer-churn-prediction-azureml
jupyter lab
```

1. **Add the repo root to `PYTHONPATH`** when running in ad-hoc environments:

```bash
export PYTHONPATH="/workspaces/customer-churn-prediction-azureml:${PYTHONPATH}"
```

Once the interpreter can see the `src` package, `from src.utils import ...` imports succeed.

### `TypeError: '<' not supported between instances of 'str' and 'int'` when compiling requirements

**Cause**: Outdated setuptools version causing matplotlib build failures

**Solution**: Upgrade setuptools before compiling:

```bash
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip-compile requirements.in -o requirements.txt
```

## Azure ML Service Errors

### `Requested 1 nodes but AzureMLCompute cluster only has 0 maximum nodes`

**Cause**: Compute cluster `max_instances` set to 0

**Solution**: `az ml compute update --name cpu-cluster --set max_instances=2`

### `Failed to pull Docker image ... This error may occur because the compute could not authenticate`

**Error Message**:

```text
Failed to pull Docker image churnmlacr2025.azurecr.io/bank-churn:1. 
This error may occur because the compute could not authenticate with the Docker registry 
to pull the image. If using ACR please ensure the ACR has Admin user enabled or a Managed 
Identity with `AcrPull` access to the ACR is assigned to the compute.
```

**Causes**:

1. Compute managed identity lacks `AcrPull` permission (compute was created before ACR)
1. Compute cluster doesn't have system-assigned managed identity enabled
1. Docker image doesn't exist in ACR

**Solutions**:

1. **Verify image exists in ACR**:

```bash
ACR_NAME=$(grep AZURE_ACR_NAME config.env | cut -d'"' -f2)
az acr repository show-tags --name $ACR_NAME --repository bank-churn --output table

# If image doesn't exist, build and push:
docker build -t bank-churn:1 .
docker tag bank-churn:1 $ACR_NAME.azurecr.io/bank-churn:1
docker push $ACR_NAME.azurecr.io/bank-churn:1
```

1. **Fix ACR authentication**:

   - If compute was created before ACR: See [ACR Authentication for Compute Cluster](#acr-authentication-for-compute-cluster) for manual role assignment
   - If compute doesn't have managed identity: Recreate with `--identity-type systemassigned` (see [ACR Authentication for Compute Cluster](#acr-authentication-for-compute-cluster))

### `Could not resolve uris of type data for assets azureml://.../bank-churn-raw/versions/1`

**Cause**: Data asset not found or incorrect configuration in `config.env`

**Solution**:

1. Verify `config.env` has correct `DATA_ASSET_FULL` and `DATA_VERSION`
1. Verify data asset exists: `az ml data show --name <name> --version <version>`
1. Register data asset if missing (see [Registering Data Asset](#registering-data-asset))

## HPO (Hyperparameter Optimization) Errors

### `sklearn.utils._param_validation.InvalidParameterError: The 'min_samples_split' parameter must be an int in the range [2, inf)`

**Cause**: Random Forest search space includes `min_samples_split: [1, 2, ...]` which violates sklearn's requirement

**Solution**: Ensure `min_samples_split >= 2` in `configs/hpo.yaml`:

```yaml
search_space:
  rf:
    min_samples_split: [2, 5, 10]  # Never use 1
```

### `run_sweep_trial.py: error: argument --xgboost_max_depth: expected one argument`

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

### `ValueError: Invalid override format 'rf_n_estimators=100'`

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

### Best model analysis cell returns "No completed trials yet" despite jobs being completed

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

### `ml_client.jobs.list(experiment_name=experiment_name)` fails with TypeError

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

### `max_depth` or other hyperparameters with `null` values cause sweep job failures

**Error Message**:

```text
Validation failed. Error: Type: Boolean is not supported in choice
# OR
Invalid parameter value: null is not a valid choice
```

**Cause**: Azure ML's `Choice` search space does not accept `None`/`null` values. If your `configs/hpo.yaml` includes `null` in any hyperparameter list (e.g., `max_depth: [1, 2, null, 4]`), the sweep job will fail during validation.

**Solution**: The `hpo_utils.build_parameter_space()` function automatically filters out `null`/`None` values from all lists in the search space. However, it's best practice to avoid `null` values in your YAML configuration:

```yaml
search_space:
  rf:
    max_depth: [4, 6, 8, 10]  # ✅ Correct - no null values
    # max_depth: [4, 6, null, 10]  # ❌ Avoid - null will cause errors
```

**Note**: If you need to represent "unlimited depth" for Random Forest or XGBoost, use a large number instead of `null`:

```yaml
search_space:
  rf:
    max_depth: [4, 6, 8, 10, 100]  # Use 100 instead of null for unlimited depth
```

**Automatic Filtering**: The `_filter_nulls()` function in `hpo_utils.py` recursively removes any `None`/`null` values from lists before building the search space, so even if you accidentally include them, they will be filtered out automatically.
