# Pipeline Guide

This guide details the steps and scripts involved in the end-to-end machine learning pipeline, from data preparation to model scoring.

## Step 1: Data Preparation (`src/data_prep.py`)

This script prepares the raw data for model training.

**Features:**
- Removes uninformative columns (e.g., `RowNumber`, `CustomerId`).
- Encodes categorical features (`Geography`, `Gender`).
- Scales numerical features using `StandardScaler`.
- Performs a stratified train/test split to maintain class distribution.
- Saves the processed data (`X_train`, `X_test`, `y_train`, `y_test`) and preprocessing artifacts (`encoders.pkl`, `scaler.pkl`).

**Usage:**
```bash
python src/data_prep.py --config configs/data.yaml
```

## Step 2: Model Training (`src/train.py`)

This script trains multiple machine learning models and compares their performance.

**Features:**
- Trains Logistic Regression, Random Forest, and XGBoost (if installed).
- Handles class imbalance using `class_weight='balanced'` or SMOTE (`--use-smote`).
- Logs comprehensive metrics (Accuracy, Precision, Recall, F1, ROC-AUC).
- Compares models and identifies the best performer based on F1 score.
- Supports MLflow for experiment tracking.

**Usage:**
```bash
python src/train.py --config configs/train.yaml
```

## Step 3: Model Evaluation (`src/evaluate.py`)

This script provides a comprehensive evaluation of a trained model.

**Features:**
- Calculates a full suite of classification metrics.
- Generates and saves key visualizations:
  - Confusion Matrix
  - ROC Curve
  - Precision-Recall Curve
  - Prediction Distribution
  - Feature Importance
- Saves a detailed `evaluation_report.json`.

**Usage:**
```bash
python src/evaluate.py \
  --model models/local/rf_model.pkl \
  --data data/processed \
  --output evaluation/rf
```

## Step 4: Model Scoring (`src/score.py`)

This script uses a trained model to make predictions on new data.

**Features:**
- Supports batch scoring from CSV files.
- Supports single predictions from JSON (ideal for APIs).
- Automatically applies the same preprocessing steps used during training.

**Usage (Batch):**
```bash
python src/score.py \
  --model models/local/rf_model.pkl \
  --data-dir data/processed \
  --input data/new_customers.csv \
  --output predictions/predictions.csv
```

**Usage (Single):**
```bash
python src/score.py \
  --model models/local/rf_model.pkl \
  --data-dir data/processed \
  --input customer.json \
  --output result.json \
  --json
```

