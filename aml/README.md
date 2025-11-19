# Azure ML Assets

This directory contains the Azure ML (v2) assets referenced by the training pipeline and notebooks. Keep these YAMLs aligned with the scripts in `src/` so local runs and remote jobs stay in sync.

## Structure

```
aml/
├── components/
│   ├── data_prep.yaml   # data preparation component
│   └── train.yaml       # model training component
└── environments/
    └── environment.yml  # Docker image reference for the pipeline
```

### `components/`

| File | Purpose | Notes |
| --- | --- | --- |
| `data_prep.yaml` | Defines the Azure ML component that wraps `src/data_prep.py`. | Expects a `uri_folder` input. Mirrors the CLI arguments from `configs/data.yaml`. |
| `train.yaml` | Defines the component that runs `src/train.py` with fixed hyperparameters. | Loads models/hyperparameters from `configs/train.yaml`; outputs both pickle and MLflow artifacts. |

Both component YAMLs assume the environment below is registered as `azureml:bank-churn-env:1`. Update the `environment` reference if you publish a new version.

### `environments/`

`environment.yml` points to the Docker image (usually stored in your ACR) that contains the project dependencies:

```yaml
$schema: https://azuremlschemas.azureedge.net/latest/environment.schema.json
name: bank-churn-env
version: "1"
image: <your-acr-name>.azurecr.io/bank-churn:1
```

After building/pushing a new image, update the `image:` tag and re-register the environment:

```bash
az ml environment create --file aml/environments/environment.yml \
  --resource-group $AZURE_RESOURCE_GROUP \
  --workspace-name $AZURE_WORKSPACE_NAME
```

## Updating components and environments

1. Edit the YAML file.
2. Re-register via CLI (examples below).
3. Update pipeline definitions or scripts to reference the new version if needed.

```bash
# Register data_prep component
az ml component create --file aml/components/data_prep.yaml \
  --resource-group $AZURE_RESOURCE_GROUP \
  --workspace-name $AZURE_WORKSPACE_NAME

# Register train component
az ml component create --file aml/components/train.yaml \
  --resource-group $AZURE_RESOURCE_GROUP \
  --workspace-name $AZURE_WORKSPACE_NAME
```

## Tips

- Keep component inputs/outputs in sync with the CLI scripts (`src/data_prep.py`, `src/train.py`). Changes to argument names must be reflected in both places.
- When bumping Python or dependency versions, rebuild the Docker image first, then update `environments/environment.yml`, and finally rerun `az ml environment create`.
- For quick smoke tests, you can run the same commands locally (e.g., `python src/train.py --config configs/train.yaml`) before publishing the component.
