# Bank Customer Churn Prediction with Azure ML

This project demonstrates an end-to-end MLOps pipeline for predicting bank customer churn using Azure Machine Learning. It covers data preparation, model training, evaluation, and scoring, all organized into a clean, reproducible structure.

## Project Overview

The goal is to predict whether a bank customer will churn (exit) based on their attributes. This is a binary classification problem. The pipeline is designed to be run both locally for development and on Azure ML for production-scale training and deployment.

### Key Features

- **End-to-End Pipeline**: Scripts for data prep, training, evaluation, and scoring.
- **Configuration Driven**: YAML files in `configs/` control the pipeline's behavior.
- **MLflow Integrated**: Automated experiment tracking for parameters, metrics, and model artifacts.
- **Multiple Models**: Supports Logistic Regression, Random Forest, and XGBoost.
- **Testing**: Includes unit tests (`pytest`) and a robust end-to-end smoke test.
- **Dockerized Environment**: A `Dockerfile` ensures a reproducible environment for local and cloud execution.
- **Azure ML Ready**: Includes component definitions (`aml/`) and a pipeline script (`run_pipeline.py`) for easy deployment on Azure.

## Project Structure

```
.
├── aml/                  # Azure ML component and pipeline definitions
├── configs/              # YAML configuration files for the pipeline
├── data/                 # Raw and sample data
├── docs/                 # Detailed project documentation
├── evaluation/           # (Git-ignored) Evaluation reports and plots
├── models/               # (Git-ignored) Trained model artifacts
├── notebooks/            # Jupyter notebooks for EDA
├── predictions/          # (Git-ignored) Scored model outputs
├── setup/                # Scripts for setting up local and Azure environments
├── src/                  # Python source code for the ML pipeline
│   ├── models/           # Model definitions
│   ├── config_loader.py  # Utility for loading YAML configs
│   ├── data_prep.py      # Data preparation script
│   ├── train.py          # Model training script
│   ├── evaluate.py       # Model evaluation script
│   └── score.py          # Model scoring/inference script
├── tests/                # Test suite for the project
│   ├── test_data_prep.py # Unit tests for data preparation
│   └── smoke_test.py     # End-to-end smoke test
├── Dockerfile            # Defines the containerized environment
├── README.md             # This file
├── requirements.in       # Application dependencies
└── dev-requirements.in   # Development dependencies
```

## Getting Started

### 1. Setup

Clone the repository and install the required dependencies.

```bash
# Clone the repository
git clone <your-repo-url>
cd <your-repo-name>

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Full Pipeline (Local)

The following commands will run the entire pipeline using the sample data and default configurations.

```bash
# 1. Prepare the data (uses configs/data.yaml)
python src/data_prep.py

# 2. Train the models (uses configs/train.yaml and mlflow.yaml)
python src/train.py

# 3. Evaluate the best model (e.g., Random Forest)
python src/evaluate.py --model models/local/rf_model.pkl --data data/processed --output evaluation/rf

# 4. Score new data
python src/score.py --model models/local/rf_model.pkl --data-dir data/processed --input data/sample.csv --output predictions/sample_predictions.csv
```

### 3. View Experiments with MLflow UI

After running the training script, you can inspect the results using the MLflow UI.

```bash
#low UI
mlflow ui
```

This will start a local server, typically at `http://127.0.0.1:5000`, where you can view your experiment runs, compare metrics, and see the logged artifacts.

### 4. Build and Test with Docker

To ensure a consistent and reproducible environment, you can build the Docker image and run the tests inside the container.

```bash
# 1. Build the Docker image
docker build -t bank-churn:latest .

# 2. Run the test suite inside the container
docker run --rm -v "$PWD:/app" -w /app bank-churn:latest \
  bash -lc "pytest -q && python tests/smoke_test.py"
```

### 5. Run Tests

To ensure everything is working correctly, run the test suite.

```bash
# Run unit tests
pytest

# Run the end-to-end smoke test
python tests/smoke_test.py

# Run the smoke test with HPO enabled
SMOKE_HPO=1 SMOKE_HPO_MODEL=rf python tests/smoke_test.py
```

## Running the Pipeline on Azure ML

To run the full training pipeline on Azure Machine Learning, ensure your `.env` file is correctly configured and that you have authenticated with Azure (`az login`). Then, execute the pipeline script:

```bash
python run_pipeline.py
```

This will submit a new pipeline job to your Azure ML workspace and print a link to view the run in the studio.

## Documentation

For more detailed information, please refer to the guides in the `docs/` directory:

- **[Project Plan](docs/MLZoomcamp-Project1-ProjectPlan-v2.md)**: The development plan and history of the project.
- **[Pipeline Guide](docs/pipeline_guide.md)**: Detailed explanation of each script in the ML pipeline.
- **[Setup Guide](docs/setup_guide.md)**: Instructions for local and Azure ML setup.
- **[SMOTE Comparison](docs/smote_comparison.md)**: Analysis of using SMOTE for class imbalance.
- **[Dependencies Guide](docs/dependencies.md)**: Information on managing Python dependencies.
