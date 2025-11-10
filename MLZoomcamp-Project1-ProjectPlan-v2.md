## Data

Data: [Churn for Bank Customers](https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers). USE THIS!

## 1) Set up the project (repo + README)

1. Create a new GitHub repo — e.g., `bank-churn-mlops-pipeline`.
2. Add a `README.md` with:
	* Problem statement: Predict bank customer churn
	* Why it matters: Helps reduce customer loss
	* Example Input → Output
	* Tools: “Azure Machine Learning, MLflow, Docker, Scikit-learn”

## 2) Add the dataset

* Put a **small sample** at `data/sample.csv` (a few hundred–few thousand rows).
* Upload the **full dataset** to Azure Blob/Data Lake.

Create an AML **Data asset** (once you know your datastore path):

```bash
az ml data create \
  --name bank-churn-raw --version 1 \
  --type uri_folder \
  --path "azureml://datastores/<YourDatastore>/paths/bank_churn/full/"
```

## 3) Do EDA

* Create `notebooks/eda.ipynb` (locally: Small data → local EDA (cheap & fast))
2. Do EDA: distributions, correlations, churn rates, missing values.
3. Document key findings in the notebook and summarize in README.

## 4) Move logic from notebook → scripts

Create the following files:

```
src/
├── data_prep.py
├── train.py  # baseline (LogReg) + Random Forest + XGBoost, log metrics
├── evaluate.py
└── score.py  # Inference logic for deployment
```

> Keep each script runnable by CLI args (e.g., `--input`, `--output`).

## 5) Centralize configuration (simple YAMLs)

Create:

- **`configs/data.yaml`** – controls data input/output and target info
- **`configs/train.yaml`** – controls training behavior and reproducibility

These let you reuse the same settings locally and in AML.

## 6) Declare dependencies (editable lists)

**Runtime** → `requirements.in`

```
pandas
scikit-learn
xgboost
mlflow
azureml-mlflow
fastapi
uvicorn
hydra-core
pandera[io]
joblib
```

**Dev** → `dev-requirements.in`

```
pytest
black
ruff
isort
pre-commit
```

Pin them once (use either tool):

**pip-tools**

```bash
python -m pip install pip-tools
pip-compile requirements.in -o requirements.txt
pip-compile dev-requirements.in -o dev-requirements.txt
```

## 7) Add a fast smoke test

**`scripts/smoke.py`**

```python
import os, subprocess
os.makedirs("data/processed", exist_ok=True)
subprocess.check_call("python src/data_prep.py --input data/sample.csv --output data/processed", shell=True)
subprocess.check_call("python src/train.py --data data/processed --out models/local", shell=True)
subprocess.check_call("python src/evaluate.py --model models/local --data data/processed", shell=True)
```

This verifies prep → train → eval end-to-end on the sample.

## 8) Define the Docker image

**`Dockerfile`**

```dockerfile
FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest
RUN apt-get update && apt-get install -y --no-install-recommends python3-pip libgomp1 && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --no-cache-dir --upgrade pip
WORKDIR /app
# install deps first for layer caching
COPY requirements.txt dev-requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r dev-requirements.txt
# then copy source
COPY . /app
```

Build and test **inside** the container:

```bash
docker build -t bank-churn:latest .
docker run --rm -v "$PWD:/app" -w /app bank-churn:latest \
  bash -lc "pytest -q && python scripts/smoke.py"
```

> If imports fail, add the missing package to `.in`, re-pin, rebuild, re-run.

## 9) Create AML command components (YAML v2)

Example: **`aml/components/data_prep.yaml`**

```yaml
$schema: https://azuremlschemas.azureedge.net/latest/commandComponent.schema.json
name: data_prep
version: 1
type: command
inputs:
  raw_data: {type: uri_file}
outputs:
  processed: {type: uri_folder}
code: ../../
environment: azureml:bank-churn-env:1   # register this in step 11
command: >
  python src/data_prep.py
  --input ${{inputs.raw_data}}
  --output ${{outputs.processed}}
```

Create similar files for `train`, `evaluate`, and (optionally) `register`.

## 10) Wire the AML pipeline

**`azureml_pipeline.py`** (sketch)

```python
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, dsl, load_component

ml = MLClient(DefaultAzureCredential(), "<SUBSCRIPTION_ID>", "<RESOURCE_GROUP>", "<WORKSPACE>")
data_prep = load_component("aml/components/data_prep.yaml")
train = load_component("aml/components/train.yaml")
evaluate = load_component("aml/components/evaluate.yaml")

@dsl.pipeline(compute="cpu-cluster")
def pipeline(raw_data):
    prep = data_prep(raw_data=raw_data)
    tr = train(processed=prep.outputs.processed)
    ev = evaluate(model=tr.outputs.model, processed=prep.outputs.processed)
    return {"metrics": ev.outputs.metrics}

job = pipeline(raw_data="azureml:bank-churn-raw:1")
ml.jobs.create_or_update(job)
```

## 11) Register and reuse the exact image in AML

Push the tested image to **ACR** and create an AML **Environment** that references it.

```bash
az acr build --registry <acr_name> --image bank-churn:latest .
az ml environment create --name bank-churn-env --image <acr_name>.azurecr.io/bank-churn:latest
```

Now all AML jobs use the same runtime you tested locally.

## 12) Run the pipeline in Azure ML

```bash
python azureml_pipeline.py
```

Confirm the run completes and metrics are logged in MLflow.

## 13) Add a Makefile for muscle-memory commands (optional)

Creating a simple "shortcut" menu for the most common project command: **`Makefile`**

```makefile
.PHONY: build test train-local aml
build: ; docker build -t bank-churn:latest .
test:  ; docker run --rm -v "$$(pwd):/app" -w /app bank-churn:latest bash -lc "pytest -q && python scripts/smoke.py"
train-local:
	python src/data_prep.py --input data/sample.csv --output data/processed
	python src/train.py --data data/processed --out models/local
aml: ; python azureml_pipeline.py
```

## 14) Deploy safely (staging → prod)

* Create a **staging** endpoint, invoke with a small payload, then promote to **prod**.
* Keep a **lean serving image** (runtime deps only; copy `score.py` + model artifact).
