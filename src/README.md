# Source Code

Python scripts for the bank customer churn prediction pipeline.

## Scripts

### 1. `data_prep.py` - Data Preparation

Prepares raw data for model training:
- Removes uninformative columns (RowNumber, CustomerId, Surname)
- Encodes categorical features (Geography, Gender)
- Scales numerical features using StandardScaler
- Splits data into train/test sets (stratified)
- Saves processed data and preprocessing artifacts

**Usage:**

```bash
python src/data_prep.py \
  --input data/churn.csv \
  --output data/processed \
  --test-size 0.2 \
  --random-state 42
```

**Arguments:**
- `--input`: Path to raw CSV file (required)
- `--output`: Directory to save processed data (required)
- `--test-size`: Test set proportion (default: 0.2)
- `--random-state`: Random seed (default: 42)
- `--target`: Target column name (default: 'Exited')

**Output Files:**
- `X_train.csv`, `X_test.csv` - Feature matrices
- `y_train.csv`, `y_test.csv` - Target vectors
- `encoders.pkl` - Label encoders for categorical features
- `scaler.pkl` - StandardScaler for numerical features
- `metadata.json` - Dataset metadata

**Example:**

```bash
# Prepare data with default settings
python src/data_prep.py --input data/churn.csv --output data/processed

# Custom train/test split
python src/data_prep.py \
  --input data/churn.csv \
  --output data/processed \
  --test-size 0.3 \
  --random-state 123
```

### 2. `train.py` - Model Training

Trains multiple machine learning models and compares performance:
- Logistic Regression (baseline)
- Random Forest
- XGBoost (if installed)

**Usage:**

```bash
python src/train.py \
  --data data/processed \
  --out models/local \
  --models logreg rf xgboost \
  --class-weight balanced
```

**Arguments:**
- `--data`: Directory with preprocessed data (required)
- `--out`: Directory to save trained models (required)
- `--models`: Models to train (default: all available)
- `--class-weight`: Class weight strategy (default: 'balanced')
- `--random-state`: Random seed (default: 42)
- `--use-mlflow`: Enable MLflow tracking
- `--experiment-name`: MLflow experiment name (default: 'churn-prediction')

**Output Files:**
- `{model}_model.pkl` - Trained model for each model type
- `{model}_metrics.json` - Evaluation metrics for each model
- `model_comparison.json` - Comparison of all models

**Features:**
- Handles class imbalance with balanced weights
- Comprehensive metrics (accuracy, precision, recall, F1, ROC-AUC)
- Feature importance analysis
- Confusion matrix
- Classification report
- Optional MLflow tracking

**Example:**

```bash
# Train all available models
python src/train.py --data data/processed --out models/local

# Train specific models
python src/train.py --data data/processed --out models/local --models logreg rf

# With SMOTE (balance training data)
python src/train.py --data data/processed --out models/with_smote --use-smote

# With MLflow tracking
python src/train.py --data data/processed --out models/local --use-mlflow

# With SMOTE + MLflow
python src/train.py --data data/processed --out models/smote_exp --use-smote --use-mlflow
```

### 3. `evaluate.py` - Model Evaluation

Comprehensive model evaluation with metrics and visualizations:
- Calculates all classification metrics
- Generates confusion matrix
- Plots ROC and Precision-Recall curves
- Shows prediction distribution
- Displays feature importance
- Saves detailed JSON report

**Usage:**

```bash
python src/evaluate.py \
  --model models/local/rf_model.pkl \
  --data data/processed \
  --output evaluation/rf
```

**Arguments:**
- `--model`: Path to trained model pickle file (required)
- `--data`: Directory with preprocessed test data (required)
- `--output`: Directory to save evaluation results (required)
- `--model-name`: Custom model name for report (default: inferred from filename)

**Output Files:**
- `evaluation_report.json` - Comprehensive metrics and classification report
- `confusion_matrix.png` - Heatmap of confusion matrix
- `roc_curve.png` - ROC curve with AUC score
- `precision_recall_curve.png` - PR curve with average precision
- `prediction_distribution.png` - Distribution of predicted probabilities
- `feature_importance.png` - Top 10 most important features

**Example:**

```bash
# Evaluate Random Forest model
python src/evaluate.py --model models/local/rf_model.pkl --data data/processed --output evaluation/rf

# Evaluate Logistic Regression
python src/evaluate.py --model models/local/logreg_model.pkl --data data/processed --output evaluation/logreg

# Evaluate with custom name
python src/evaluate.py --model models/exp/model.pkl --data data/processed --output evaluation/custom --model-name my_model
```

### 4. `score.py` - Inference/Scoring

Model scoring script for batch and single predictions:
- Batch scoring from CSV files
- Single prediction from JSON
- Automatic preprocessing using saved artifacts
- Outputs predictions with probabilities

**Usage:**

```bash
# Batch scoring from CSV
python src/score.py \
  --model models/local/rf_model.pkl \
  --data-dir data/processed \
  --input data/new_customers.csv \
  --output predictions/predictions.csv

# JSON scoring (single prediction)
python src/score.py \
  --model models/local/rf_model.pkl \
  --data-dir data/processed \
  --input input.json \
  --output output.json \
  --json
```

**Arguments:**
- `--model`: Path to trained model pickle file (required)
- `--data-dir`: Directory with preprocessing artifacts (encoders, scaler, metadata) (required)
- `--input`: Input CSV or JSON file (required)
- `--output`: Output file for predictions (required)
- `--threshold`: Classification threshold (default: 0.5)
- `--no-proba`: Do not include probability scores
- `--json`: Use JSON format for input/output

**Output:**
- CSV: Adds `predicted_churn` and `churn_probability` columns
- JSON: Returns prediction dictionary with `prediction`, `churn_probability`, `predicted_class`

**Example:**

```bash
# Batch scoring
python src/score.py --model models/local/rf_model.pkl \
  --data-dir data/processed --input data/test.csv --output predictions.csv

# JSON scoring (for API deployment)
python src/score.py --model models/local/rf_model.pkl \
  --data-dir data/processed --input customer.json --output result.json --json
```

## Data Flow

```
Raw Data (churn.csv)
    ↓
data_prep.py
    ↓
Processed Data (data/processed/)
    ↓
train.py → Model + Metrics
    ↓
evaluate.py → Evaluation Report
    ↓
score.py → Predictions
```

## Features

After preprocessing, the dataset contains:

**Numerical Features (8):**
- CreditScore
- Age
- Tenure
- Balance
- NumOfProducts
- EstimatedSalary
- HasCrCard (binary)
- IsActiveMember (binary)

**Categorical Features (2 - encoded):**
- Geography (France=0, Germany=1, Spain=2)
- Gender (Female=0, Male=1)

**Target:**
- Exited (0=Retained, 1=Churned)

## Dependencies

```
pandas
scikit-learn
```

Install with:
```bash
pip install pandas scikit-learn
```

