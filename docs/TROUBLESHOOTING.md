# Troubleshooting and Important Considerations

This document contains important considerations and troubleshooting information for running the Azure ML pipeline for customer churn prediction.

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Azure ML Configuration](#azure-ml-configuration)
3. [MLflow Integration Issues](#mlflow-integration-issues)
4. [Docker Image Compatibility](#docker-image-compatibility)
5. [Data Access Issues](#data-access-issues)
6. [Hyperparameter Optimization (HPO) Issues](#hyperparameter-optimization-hpo-issues)
7. [Common Errors and Solutions](#common-errors-and-solutions)
8. [Best Practices](#best-practices)

---

## Environment Setup

### Python Version Compatibility

**Critical**: The project uses **Python 3.9** due to compatibility requirements with `azureml-core` 1.1.5.7.

- **Issue**: `azureml-core` 1.1.5.7 is not compatible with Python 3.10+ due to `collections.Iterable` import issues
- **Solution**: Dockerfile uses `python:3.9-slim` as base image
- **Note**: Requirements files (`requirements.txt` and `dev-requirements.txt`) must be compiled with Python 3.9

**Type Hint Compatibility**: 
- **Do NOT use** Python 3.10+ union syntax: `Type | None` (causes `TypeError`)
- **Use** Python 3.9 compatible syntax: `Optional[Type]` from `typing` module
- The `tuple[...]` syntax is valid in Python 3.9, only the `|` union operator requires Python 3.10+

**To recompile requirements**:
```bash
docker run --rm -v $(pwd):/workspace -w /workspace python:3.9-slim bash -c \
  "apt-get update -qq && apt-get install -y -qq gcc > /dev/null 2>&1 && \
   pip install -q pip-tools && \
   pip-compile --output-file requirements.txt requirements.in && \
   pip-compile --output-file dev-requirements.txt dev-requirements.in"
```

### Package Version Compatibility

Several packages have version constraints for Python 3.9:
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

Ensure `config.env` is properly configured with:
- `AZURE_SUBSCRIPTION_ID`
- `AZURE_RESOURCE_GROUP`
- `AZURE_WORKSPACE_NAME`
- `AZURE_RAW_DATA_ASSET`: Name of registered data asset (default: "churn-data")
- `AZURE_RAW_DATA_VERSION`: Version of data asset (default: "3")

### Compute Cluster Configuration

**Important**: The compute cluster must have:
- `max_instances >= 1` (not 0)
- Managed identity with proper RBAC permissions

**To check and update**:
```bash
az ml compute show --name cpu-cluster --resource-group <rg> --workspace-name <ws>
az ml compute update --name cpu-cluster --resource-group <rg> --workspace-name <ws> --set max_instances=2
```

### Managed Identity Permissions

The compute cluster's managed identity requires:

1. **Azure Container Registry (ACR)**: `AcrPull` role
   ```bash
   # Get compute identity principal ID
   COMPUTE_ID=$(az ml compute show --name cpu-cluster --resource-group <rg> --workspace-name <ws> --query identity.principal_id -o tsv)
   
   # Grant AcrPull on ACR
   az role assignment create \
     --assignee $COMPUTE_ID \
     --role AcrPull \
     --scope /subscriptions/<sub>/resourceGroups/<rg>/providers/Microsoft.ContainerRegistry/registries/<acr-name>
   ```

2. **Storage Account**: `Storage Blob Data Reader` role
   ```bash
   # Grant Storage Blob Data Reader on storage account
   az role assignment create \
     --assignee $COMPUTE_ID \
     --role "Storage Blob Data Reader" \
     --scope /subscriptions/<sub>/resourceGroups/<rg>/providers/Microsoft.Storage/storageAccounts/<storage-name>
   ```

3. **Workspace Managed Identity**: Also needs `Storage Blob Data Reader` on the storage account

**Critical**: After granting permissions, **recycle the compute node** to pick up new credentials:
- Delete the node from Azure ML Studio, OR
- Scale cluster to 0 and back up

---

## MLflow Integration Issues

### Azure ML MLflow Limitations

Azure ML's MLflow integration has several limitations compared to standard MLflow:

#### 1. No Nested Runs Support

**Issue**: `mlflow.start_run(nested=True)` causes API errors in Azure ML

**Solution**: In `src/train.py`, nested runs are disabled when running in Azure ML:
- Detects Azure ML environment via `MLFLOW_TRACKING_URI`
- Uses active run directly instead of creating nested runs
- Logs model name as tag to identify different models

#### 2. No Model Registry API

**Issue**: `mlflow.sklearn.log_model()` calls `/api/2.0/mlflow/logged-models` which returns 404

**Solution**: In Azure ML, models are saved as pickle files to outputs directory:
- Saves model to `AZUREML_ARTIFACTS_DIRECTORY` or `AZUREML_OUTPUT_DIRECTORY`
- Attempts `mlflow.log_artifact()` without artifact_path (may fail gracefully)
- Azure ML automatically captures outputs directory as artifacts

#### 3. Artifact API Signature Differences

**Issue**: `mlflow.log_artifact(file_path, artifact_path)` and `mlflow.log_artifacts(dir_path, artifact_path)` have different signatures in Azure ML

**Error**: `azureml_artifacts_builder() takes from 0 to 1 positional arguments but 2 were given`

**Solution**: 
- Use `mlflow.log_artifact(file_path)` without artifact_path parameter
- If that fails, gracefully handle exception and rely on Azure ML's native artifact capture
- Never use `mlflow.artifacts.download_artifacts()` in Azure ML - use direct file access instead

#### 4. Active Run Context

**Issue**: Azure ML automatically creates an active MLflow run, causing conflicts when trying to start a new one

**Error**: `Cannot start run with ID ... because active run ID does not match environment run ID`

**Solution**: Check for active run before starting a new one:
```python
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "")
is_azure_ml = "azureml" in tracking_uri.lower() if tracking_uri else False

if not is_azure_ml:
    # Local execution: start new run
    mlflow.set_experiment(experiment_name)
    parent_run = mlflow.start_run(run_name="Churn_Training_Pipeline")
    started_run = True
else:
    # Azure ML: use existing active run
    parent_run = mlflow.active_run()
    started_run = False
```

---

## Docker Image Compatibility

### Base Image

**Current**: `python:3.9-slim`

**Why**: Compatibility with `azureml-core` 1.1.5.7

### Building and Pushing

**Always rebuild after dependency changes**:
```bash
# Build the Docker image locally
docker build -t bank-churn:latest .

# Tag for ACR
docker tag bank-churn:latest <your-acr-name>.azurecr.io/bank-churn:latest

# Push to ACR
docker push <your-acr-name>.azurecr.io/bank-churn:latest
```

**Verify image is pushed**:
```bash
az acr repository show-tags --name <your-acr-name> --repository bank-churn
```

### Environment Registration

The Docker image should be registered as an Azure ML environment using Azure CLI:

1. **Update `aml/environments/environment.yml`** with your ACR name:
   ```yaml
   $schema: https://azuremlschemas.azureedge.net/latest/environment.schema.json
   name: bank-churn-env
   version: "1"
   image: <your-acr-name>.azurecr.io/bank-churn:latest
   description: Environment for churn prediction pipeline
   ```

2. **Register the environment**:
   ```bash
   az ml environment create --file aml/environments/environment.yml \
     --resource-group <resource-group> \
     --workspace-name <workspace-name>
   ```

3. **Verify registration**:
   ```bash
   az ml environment show --name bank-churn-env --version 1 \
     --resource-group <resource-group> \
     --workspace-name <workspace-name>
   ```

---

## Data Access Issues

### Data Asset Registration

**Issue**: Pipeline fails with `ResourceNotFoundError` for data asset

**Solution**: Ensure data asset is registered as `uri_folder` type:
```bash
# List existing data assets
az ml data list --resource-group <rg> --workspace-name <ws>

# Upload and register data as uri_folder (directory containing CSV file(s))
az ml data upload \
  --name churn-data \
  --version 3 \
  --path data/ \
  --type uri_folder \
  --resource-group <rg> \
  --workspace-name <ws>
```

**Note**: The data asset must be registered as `uri_folder` type. The `data_prep` component expects a folder input and will automatically load all CSV files in the folder.

**Update config.env**:
```bash
AZURE_RAW_DATA_ASSET="churn-data"
AZURE_RAW_DATA_VERSION="3"
```

### Storage Access

**Issue**: `ScriptExecution.StreamAccess.NotFound` when accessing data

**Symptoms**: 
- Error: `URI: azureml://.../workspaceblobstore/paths/data/churn.csv`
- Data prep step fails

**Solution**:
1. Verify managed identity has `Storage Blob Data Reader` role on storage account
2. Recycle compute node after granting permissions
3. Re-upload data to ensure it's accessible

### Data Asset Configuration Loading

**Issue**: Pipeline fails with `Could not resolve uris of type data for assets azureml://.../bank-churn-raw/versions/1` even though `config.env` specifies a different data asset

**Cause**: Pipeline scripts not loading `config.env` explicitly, falling back to hardcoded defaults

**Solution**: Both `run_pipeline.py` and `run_hpo.py` now automatically load `config.env`. Ensure your `config.env` file has:
```bash
AZURE_RAW_DATA_ASSET="churn-data"
AZURE_RAW_DATA_VERSION="3"
```

**Note**: The scripts will fall back to defaults (`bank-churn-raw` version 1) if `config.env` is not found or variables are not set. See the "Common Errors and Solutions" section below for detailed troubleshooting.

---

## Hyperparameter Optimization (HPO) Issues

### Search Space Configuration

**Issue**: Models not being trained or receiving incorrect hyperparameters

**Root Causes**:
1. Model type not included in search space
2. Hyperparameters not filtered by model type
3. Invalid hyperparameter ranges (e.g., `min_samples_split < 2` for Random Forest)
4. Models with no hyperparameters causing redundant trials

**Best Practices**:

1. **Always include `model_type` in search space**:
   ```yaml
   search_space:
     rf: ...
     xgboost: ...
     # logreg: {}  # Comment out if no hyperparameters to optimize
   ```

2. **Validate hyperparameter ranges**:
   - Random Forest: `min_samples_split >= 2`, `min_samples_leaf >= 1`
   - XGBoost: `n_estimators > 0`, `max_depth > 0`, `learning_rate > 0`
   - Logistic Regression: If included, add hyperparameters like `C`, `penalty`, `solver`

3. **Test configuration before full HPO**:
   ```bash
   # Test with single model and small budget
   # In configs/hpo.yaml, set:
   # max_trials: 2
   # search_space: { rf: { n_estimators: [100, 200] } }
   ```

### Model Type Filtering

**Issue**: Each trial should train only one model type, but hyperparameters from other models are being passed

**Solution**: The code automatically filters hyperparameters based on `model_type`:
- `hpo_utils.py` builds parameter space with `model_type` as categorical choice
- `train.py` receives `--model-type` argument in HPO mode
- `train_all_models()` filters `hyperparams_by_model` to only include the current model type

**Verification**: Check that each trial in Azure ML Studio has a single `model_type` tag and only relevant hyperparameters logged.

### Trial Failures

**Common failure patterns**:

1. **All trials fail immediately**: Check Docker image, environment variables, data access
2. **Specific model types fail**: Check hyperparameter validation (e.g., `min_samples_split >= 2`)
3. **Intermittent failures**: Check compute resources, storage access, MLflow connectivity

**Debugging steps**:
1. Check individual trial logs in Azure ML Studio
2. Verify hyperparameters logged match the model type
3. Test the model type locally with the same hyperparameters
4. Check for validation errors in `train.py` logs

---

## Common Errors and Solutions

### Error: `ImportError: cannot import name 'Iterable' from 'collections'`

**Cause**: Python 3.10+ compatibility issue with `azureml-core` 1.1.5.7

**Solution**: Use Python 3.9 in Dockerfile

### Error: `AttributeError: 'CommandComponent' object has no attribute 'sweep'`

**Cause**: Incorrect usage of sweep API in Azure ML v2 SDK

**Solution**: Call `.sweep()` on the Command builder, not the component:
```python
base_train_job = components["train"](processed_data=data_prep_job.outputs.processed_data)
sweep_job = base_train_job.sweep(...)
```

### Error: `Requested 1 nodes but AzureMLCompute cluster only has 0 maximum nodes`

**Cause**: Compute cluster `max_instances` set to 0

**Solution**: Update compute cluster:
```bash
az ml compute update --name cpu-cluster --set max_instances=2
```

### Error: `Failed to pull Docker image ... This error may occur because the compute could not authenticate`

**Cause**: Compute managed identity lacks `AcrPull` permission

**Solution**: Grant `AcrPull` role and recycle compute node

### Error: `mlflow.exceptions.MlflowException: Cannot start run with ID ...`

**Cause**: Trying to start a new MLflow run when Azure ML already has one active

**Solution**: Check for active run and use it instead of starting a new one (see MLflow Integration Issues section)

### Error: `API request to endpoint /api/2.0/mlflow/logged-models failed with error code 404`

**Cause**: Azure ML doesn't support MLflow model registry API

**Solution**: Save models as pickle files to outputs directory instead of using `mlflow.sklearn.log_model()`

### Error: `azureml_artifacts_builder() takes from 0 to 1 positional arguments but 2 were given`

**Cause**: Azure ML's MLflow artifact APIs have different signatures

**Solution**: 
- Use `mlflow.log_artifact(file_path)` without artifact_path
- Or save to Azure ML outputs directory and let Azure ML capture it automatically

### Error: `sklearn.utils._param_validation.InvalidParameterError: The 'min_samples_split' parameter of RandomForestClassifier must be an int in the range [2, inf)`

**Cause**: Random Forest hyperparameter search space included `min_samples_split: [1, 2, ...]` which violates sklearn's requirement that `min_samples_split >= 2`

**Solution**: 
- Ensure `min_samples_split` in HPO search space is always `>= 2`
- The code now validates and corrects this automatically, but best practice is to set correct ranges in `configs/hpo.yaml`
- Example fix in `configs/hpo.yaml`:
  ```yaml
  search_space:
    rf:
      min_samples_split: [2, 5, 10]  # Never use 1
  ```

### Error: XGBoost models not being trained in HPO

**Cause**: XGBoost not included in the HPO search space, or `model_type` filtering not working correctly

**Solution**:
- Ensure `xgboost` is included in `configs/hpo.yaml::search_space`
- Verify `model_type` is correctly passed to `train.py` and hyperparameters are filtered by model type
- Check that `hpo_utils.py` correctly builds the parameter space with `model_type` as a categorical choice

### Error: Logistic Regression being trained multiple times with default configurations

**Cause**: Logistic Regression included in HPO search space but has no hyperparameters to optimize (`logreg: {}`), causing redundant trials with identical default parameters

**Solution**:
- Remove Logistic Regression from HPO search space if it has no hyperparameters:
  ```yaml
  search_space:
    # logreg: {}  # Commented out - no hyperparameters to optimize
    rf: ...
    xgboost: ...
  ```
- If you want to include Logistic Regression, add hyperparameters to optimize (e.g., `C`, `penalty`, `solver`)

### Error: Models receiving mismatching hyperparameters (e.g., XGBoost receiving Random Forest parameters)

**Cause**: Hyperparameters not filtered by `model_type` before being passed to `train_model()`, causing all hyperparameters from the search space to be passed regardless of which model is being trained

**Solution**:
- The code now filters hyperparameters in `train_all_models()` based on the current `model_type`
- In HPO mode, `train.py` receives `--model-type` argument and only passes relevant hyperparameters
- Ensure `hpo_utils.py` correctly structures the parameter space with `model_type` as a categorical hyperparameter

### Error: All HPO trials failing after code changes

**Cause**: Over-aggressive error handling or validation logic that rejects valid configurations

**Solution**:
- Keep validation minimal and focused on critical constraints (e.g., `min_samples_split >= 2`)
- Ensure error handling doesn't prevent valid trials from running
- Test with a single model type first before running full HPO sweep

### Error: `TypeError: unsupported operand type(s) for |: '_GenericAlias' and 'NoneType'`

**Cause**: Code uses Python 3.10+ union type syntax (`JSONDict | None`) but the project requires Python 3.9

**Error Location**: 
- `src/training/model_utils.py`: `def apply_hyperparameters(model: Any, hyperparams: JSONDict | None)`
- `src/training/training.py`: `model_hyperparams: JSONDict | None = None`

**Solution**: Replace `|` union syntax with `Optional` from `typing` module:
- Change `JSONDict | None` to `Optional[JSONDict]`
- Ensure `Optional` is imported: `from typing import Optional`

**Fixed Files**:
- `src/training/model_utils.py`: Line 36
- `src/training/training.py`: Line 53

**Note**: The `tuple[...]` syntax is valid in Python 3.9, only the `|` union operator requires Python 3.10+.

### Error: `Could not resolve uris of type data for assets azureml://.../bank-churn-raw/versions/1`

**Cause**: Data asset not found or incorrect configuration in `config.env`

**Solution**: 
1. Verify `config.env` exists and contains correct values:
   ```bash
   AZURE_RAW_DATA_ASSET="churn-data"
   AZURE_RAW_DATA_VERSION="3"
   ```

2. Verify the data asset exists and is registered as `uri_folder` type:
   ```bash
   az ml data show --name churn-data --version 3 --resource-group <rg> --workspace-name <ws>
   ```

3. If the data asset doesn't exist, register it:
   ```bash
   az ml data upload \
     --name churn-data \
     --version 3 \
     --path data/ \
     --type uri_folder \
     --resource-group <rg> \
     --workspace-name <ws>
   ```

**Note**: Both `run_pipeline.py` and `run_hpo.py` automatically load `config.env`. If the file is missing or variables are not set, they will fall back to defaults (`bank-churn-raw` version 1).

---

## Best Practices

### 1. Always Test Locally First

Before running on Azure ML, test the training script locally:
```bash
python src/train.py --data data/processed --experiment-name test
# Models are determined from configs/train.yaml
```

### 2. Monitor Pipeline Jobs

Use Azure ML Studio to monitor pipeline execution:
- Check each step's logs
- Verify data access
- Monitor MLflow metrics

### 3. Version Control

- Tag Docker images with versions
- Use versioned data assets
- Keep track of environment configurations

### 4. Resource Management

- Scale compute clusters appropriately
- Use early stopping in HPO to save resources
- Clean up old jobs and artifacts periodically

### 5. Error Handling

The code includes graceful error handling for Azure ML MLflow limitations:
- Models are saved even if MLflow artifact logging fails
- Active run detection prevents conflicts
- Fallback mechanisms for model loading

### 6. Debugging Tips

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

### Rebuild Docker Image
```bash
# Build locally
docker build -t bank-churn:latest .

# Tag and push to ACR
docker tag bank-churn:latest <your-acr-name>.azurecr.io/bank-churn:latest
docker push <your-acr-name>.azurecr.io/bank-churn:latest

# Update environment.yml with new image tag if needed, then re-register
az ml environment create --file aml/environments/environment.yml \
  --resource-group <resource-group> \
  --workspace-name <workspace-name>
```

### Run Regular Pipeline
```bash
python run_pipeline.py
```
**Note**: The script automatically loads `config.env` for Azure ML configuration.

### Run HPO Pipeline
```bash
python run_hpo.py
```
**Note**: The script automatically loads `config.env` for Azure ML configuration.

### Check Job Status
```bash
az ml job show --name <job-name> --resource-group <rg> --workspace-name <ws> --query status
```

### List Child Jobs
```bash
az ml job list --parent-job-name <parent-job> --resource-group <rg> --workspace-name <ws>
```

### Download Job Logs
```bash
az ml job download --name <job-name> --resource-group <rg> --workspace-name <ws> --download-path logs/
```

---

## Additional Resources

- [Azure ML Documentation](https://docs.microsoft.com/azure/machine-learning/)
- [MLflow with Azure ML](https://docs.microsoft.com/azure/machine-learning/how-to-use-mlflow)
- [Azure ML Python SDK v2](https://docs.microsoft.com/python/api/azure-ai-ml/azure.ai.ml)
- [Troubleshooting Azure ML](https://docs.microsoft.com/azure/machine-learning/how-to-troubleshoot-jobs)

---

**Last Updated**: 2025-01-13
**Maintained By**: MLOps Team

---

## Recent Fixes (2025-01-13)

1. **Data Asset Configuration**: Fixed `run_pipeline.py` and `run_hpo.py` to explicitly load `config.env` instead of relying on default `.env` file
2. **Python 3.9 Type Hints**: Replaced `|` union syntax with `Optional[...]` for Python 3.9 compatibility in `model_utils.py` and `training.py`

