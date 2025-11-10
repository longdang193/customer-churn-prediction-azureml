# Step 4b: Model Training - Complete ✅

Model training script has been created and tested successfully.

## What Was Created

### `src/train.py` (472 lines)
Complete model training pipeline with:
- ✅ Multiple models (Logistic Regression, Random Forest, XGBoost)
- ✅ Class imbalance handling
- ✅ Comprehensive metrics
- ✅ Feature importance analysis
- ✅ Model comparison
- ✅ Optional MLflow tracking
- ✅ CLI interface

## Quick Start

### Train All Models

```bash
python src/train.py --data data/processed --out models/local
```

### Train Specific Models

```bash
python src/train.py --data data/processed --out models/local --models logreg rf
```

## Training Results

### Model Performance (on test set):

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Random Forest** | **83.05%** | **57.05%** | **67.57%** | **61.87%** | **85.81%** |
| Logistic Regression | 70.75% | 38.32% | 71.74% | 49.96% | 77.41% |

🏆 **Best Model:** Random Forest (F1 = 0.6187)

### Key Findings:

**Random Forest:**
- ✅ Best overall performance
- ✅ High accuracy (83%)
- ✅ Balanced precision/recall
- ✅ Excellent ROC-AUC (86%)
- Top features: Age, NumOfProducts, Balance

**Logistic Regression:**
- ✅ Good baseline
- ⚠️ Lower precision (38%)
- ✅ High recall (72%) - catches most churners
- Top features: Age, IsActiveMember, Balance

### Confusion Matrix (Random Forest):

```
                 Predicted
                 Retained  Churned
Actual Retained    1386      207
       Churned      132      275
```

- **True Negatives (TN):** 1,386 - Correctly predicted retained
- **False Positives (FP):** 207 - Incorrectly predicted churned
- **False Negatives (FN):** 132 - Incorrectly predicted retained
- **True Positives (TP):** 275 - Correctly predicted churned

## Output Files

```
models/local/
├── logreg_model.pkl          # Logistic Regression model
├── logreg_metrics.json       # Metrics for LogReg
├── rf_model.pkl              # Random Forest model (4.2MB)
├── rf_metrics.json           # Metrics for RF
└── model_comparison.json     # Comparison of all models
```

## CLI Arguments

```bash
--data           # Directory with preprocessed data (required)
--out            # Directory to save models (required)
--models         # Models to train: logreg, rf, xgboost (default: all)
--class-weight   # Class weight strategy (default: 'balanced')
--random-state   # Random seed (default: 42)
--use-mlflow     # Enable MLflow tracking
--experiment-name # MLflow experiment name
```

## Features Implemented

### 1. Multiple Models
- Logistic Regression (baseline)
- Random Forest (best performer)
- XGBoost (optional, requires installation)

### 2. Class Imbalance Handling
- Uses `class_weight='balanced'` for LogReg and RF
- Automatically adjusts for 80/20 class distribution

### 3. Comprehensive Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix
- Classification Report

### 4. Feature Importance
- Tree-based: feature_importances_
- Linear models: coefficient magnitudes
- Top 5 features displayed

### 5. Model Comparison
- Side-by-side metrics table
- Automatic best model selection (by F1 score)
- Comparison saved to JSON

### 6. MLflow Integration (Optional)
- Experiment tracking
- Parameter logging
- Metric logging
- Model artifact logging
- Feature importance logging

## Top 5 Important Features

### Random Forest:
1. **Age** (33.85%) - Older customers more likely to churn
2. **NumOfProducts** (21.39%) - 3-4 products = higher churn
3. **Balance** (11.39%) - Higher balances = higher churn
4. **EstimatedSalary** (7.80%)
5. **CreditScore** (7.75%)

### Logistic Regression:
1. **Age** (0.8050) - Strong positive coefficient
2. **IsActiveMember** (0.4395) - Inactive = higher churn
3. **Balance** (0.3262)
4. **Gender** (0.2737) - Female = higher churn
5. **CreditScore** (0.0851)

## Installation

### Required:
```bash
pip install pandas scikit-learn numpy
```

### Optional (for XGBoost):
```bash
pip install xgboost
```

### Optional (for MLflow):
```bash
pip install mlflow
```

## Example Usage

### Basic Training
```bash
python src/train.py --data data/processed --out models/local
```

### With XGBoost
```bash
pip install xgboost
python src/train.py --data data/processed --out models/local --models logreg rf xgboost
```

### With MLflow Tracking
```bash
pip install mlflow
python src/train.py --data data/processed --out models/local --use-mlflow
mlflow ui  # View results
```

### Custom Settings
```bash
python src/train.py \
  --data data/processed \
  --out models/experiment1 \
  --models rf \
  --random-state 123
```

## Next Steps

✅ Data preparation (data_prep.py) - DONE  
✅ Model training (train.py) - DONE  
🔄 Model evaluation (evaluate.py) - NEXT  
⏳ Scoring/inference (score.py) - TODO  

## Validation

✅ Trains successfully on preprocessed data  
✅ Handles class imbalance correctly  
✅ Generates all output files  
✅ Comparison works correctly  
✅ Feature importance calculated  
✅ CLI arguments work  
✅ Gracefully handles missing dependencies (XGBoost, MLflow)  

## Status

✅ **COMPLETE** - Model training script working and validated

**Best Model:** Random Forest with 61.87% F1 score and 85.81% ROC-AUC

