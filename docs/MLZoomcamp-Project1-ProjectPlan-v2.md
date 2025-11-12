# Project Plan & Guide

This document outlines the plan and structure for building the Bank Customer Churn MLOps pipeline. It serves as both a historical record of development and a guide to setting up the project from scratch.

## Final Project Structure

The final, organized structure of the project is as follows:

```
.github/
└── workflows/          # CI/CD workflows (e.g., GitHub Actions)
configs/
├── data.yaml
├── evaluate.yaml
├── score.yaml
└── train.yaml
data/
├── README.md
└── sample.csv
docs/
├── README.md
├── pipeline_guide.md
├── setup_guide.md
└── ...
models/                 # (Git-ignored) Trained model artifacts
notebooks/
└── eda.ipynb
setup/
├── setup.sh
├── start_compute.sh
└── ...
src/
├── models/
│   ├── __init__.py
│   ├── logistic_regression.py
│   ├── random_forest.py
│   └── xgboost_model.py
├── __init__.py
├── config_loader.py
├── data_prep.py
├── evaluate.py
├── score.py
└── train.py
tests/
├── __init__.py
├── conftest.py
├── smoke_test.py
└── test_data_prep.py

.gitignore
Dockerfile
README.md
requirements.in
dev-requirements.in
```

---

## Project Setup: Step-by-Step Guide

This guide outlines how to construct the project from a blank slate.

### Step 1: Set up the Project (Repo + README)

1.  **Initialize Git**: Create a new repository.
2.  **Create `README.md`**: Add a basic project overview.
3.  **Create `.gitignore`**: Add entries for temporary files, data, models, and environment files (`.env`, `__pycache__`, `*.csv`, `models/`, `evaluation/`, etc.).

### Step 2: Add the Dataset

1.  **Create `data/` directory**.
2.  Add a **small sample** of the dataset to `data/sample.csv`.
3.  Ensure the full dataset (`data/churn.csv`) is included in `.gitignore`.

### Step 3: Perform EDA

1.  **Create `notebooks/` directory**.
2.  Develop `notebooks/eda.ipynb` to analyze the sample data, documenting findings on distributions, correlations, and data quality.

### Step 4: Create Pipeline Scripts

1.  **Create `src/` directory** and `src/__init__.py`.
2.  **Develop Core Scripts** with AML/HyperDrive in mind:
    -   `src/data_prep.py`: deterministic prep stage whose outputs can be reused by CLI and AML components.
    -   `src/train.py`: trains models, logs to MLflow, supports nested runs and `--set model.param=value` overrides (used by HyperDrive).
    -   `src/evaluate.py`: evaluates the parent MLflow run and exports plots/metrics.
    -   `src/score.py`: batch and JSON inference entry point.
3.  **Refactor Models**: Create a `src/models/` package to hold individual model definitions so HyperDrive sweeps can instantiate clean base estimators.

### Step 5: Centralize Configuration

1.  **Create `configs/` directory**.
2.  **Create YAML files** for each stage:
    -   `configs/data.yaml`: For data prep settings.
    -   `configs/train.yaml`: For training settings plus an `hpo` block describing HyperDrive search space, metric, budget, and early-stopping policy.
    -   `configs/mlflow.yaml`, `configs/evaluate.yaml`, `configs/score.yaml`: Track experiment names and CLI defaults.
3.  **Create `src/config_loader.py`**: Load YAML files, merge with CLI arguments, and expose helpers used by both CLI scripts and AML components.
4.  **Update Scripts**: Ensure `data_prep.py` and `train.py` pull defaults from config but remain overrideable via CLI/`--set`.

### Step 6: Declare Dependencies

1.  **Create `requirements.in`**: include core libraries (e.g., `pandas`, `scikit-learn`, `mlflow`, `azure-ai-ml`, `azure-identity`, `xgboost`, `imbalanced-learn`).
2.  **Create `dev-requirements.in`**: add tooling for tests, formatting, and local development (e.g., `pytest`, `pip-tools`).
3.  **Compile Pinned Versions** (optional but recommended) to keep AML + local environments reproducible:
```bash
    pip install pip-tools
pip-compile requirements.in -o requirements.txt
pip-compile dev-requirements.in -o dev-requirements.txt
```
4.  **Sync environments**: ensure Docker image, local venv, and AML compute all install from the compiled requirements.

### Step 7: Add Tests

1.  **Create `tests/` directory** and `tests/__init__.py`.
2.  **Write Unit Tests**: e.g., `tests/test_data_prep.py` covering parsing, scaling, metadata generation.
3.  **Create CLI Smoke Test**: `tests/smoke_test.py` exercises the CLI pipeline (prep → train → evaluate) with deterministic hyperparameters for fast local validation.
4.  **Create HPO Smoke Test**: `tests/hpo_smoke_test.py` submits a minimal HyperDrive sweep (2-3 trials) to validate the full AML workflow. This test requires Azure ML credentials and is skipped if not configured.
5.  **Test Strategy**: Use CLI smoke test for quick checks; use HPO smoke test to validate AML integration before full production sweeps.

### Step 8: Define the Docker Image

1.  **Create `Dockerfile`** in the project root.
2.  The Dockerfile should:
    -   Start from a slim Python base image.
    -   Copy `requirements.txt` and install dependencies first (for layer caching).
    -   Copy the rest of the application source code.
3.  **Build and Test** the container to ensure the environment is correct:
```bash
docker build -t bank-churn:latest .
docker run --rm -v "$PWD:/app" -w /app bank-churn:latest \
      bash -lc "pytest -q && python tests/smoke_test.py"
```

---

## Hyperparameter Strategy

We adopt an Azure ML HyperDrive-first pipeline. The single-run CLI/AML jobs remain for quick validation, but the primary orchestration will be:

1. Use the `train` HyperDrive sweep to explore the search space defined in `configs/train.yaml` (data_prep runs once, train.sweep evaluates candidates).
2. Promote the best trial's parameters into `configs/train.yaml` (training.hyperparameters) for any follow-up single-run jobs.
3. Evaluate and package the selected run for downstream deployment/monitoring.

## Future Plans

-   **Step 9: Build AML command components (data_prep, train) ready for sweep overrides**
-   **Step 10: Create and submit the HyperDrive sweep pipeline (data_prep → train.sweep)**
-   **Step 11: Capture best trial metrics/params, update configs/train.yaml**
-   **Step 12: Register best model artifact + document promotion workflow**
-   **Step 13: Add a Makefile for common commands**
-   **Step 14: Deploy safely (staging → prod)**
