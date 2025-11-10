# Step 4: Data Preparation - Complete ✅

Data preparation script has been created and tested successfully.

## What Was Created

### 1. `src/data_prep.py`
Complete data preprocessing pipeline with:
- ✅ Remove uninformative columns (RowNumber, CustomerId, Surname)
- ✅ Encode categorical features (Geography, Gender)
- ✅ Scale numerical features (StandardScaler)
- ✅ Stratified train/test split
- ✅ Save preprocessing artifacts (encoders, scaler)
- ✅ CLI interface with argparse

### 2. `src/validate_data.py`
Validation script to verify prepared data quality.

### 3. `src/README.md`
Documentation for all source scripts.

## Quick Start

### Run Data Preparation

```bash
python src/data_prep.py \
  --input data/churn.csv \
  --output data/processed \
  --test-size 0.2 \
  --random-state 42
```

### Validate Prepared Data

```bash
python src/validate_data.py --data-dir data/processed
```

## Output

The script creates in `data/processed/`:

```
data/processed/
├── X_train.csv         # Training features (8,000 × 10)
├── X_test.csv          # Test features (2,000 × 10)
├── y_train.csv         # Training labels (8,000)
├── y_test.csv          # Test labels (2,000)
├── encoders.pkl        # LabelEncoders for categorical features
├── scaler.pkl          # StandardScaler for numerical features
└── metadata.json       # Dataset metadata
```

## Features

**10 features** after preprocessing:

| Feature | Type | Description |
|---------|------|-------------|
| CreditScore | Numerical (scaled) | Credit score |
| Geography | Categorical (encoded) | France=0, Germany=1, Spain=2 |
| Gender | Categorical (encoded) | Female=0, Male=1 |
| Age | Numerical (scaled) | Customer age |
| Tenure | Numerical (scaled) | Years with bank |
| Balance | Numerical (scaled) | Account balance |
| NumOfProducts | Numerical (scaled) | Number of products |
| HasCrCard | Binary | Has credit card |
| IsActiveMember | Binary | Active member status |
| EstimatedSalary | Numerical (scaled) | Estimated salary |

**Target:** Exited (0=Retained, 1=Churned)

## Validation Results ✅

```
✓ All required files present
✓ Train set: 8,000 samples (80%)
✓ Test set: 2,000 samples (20%)
✓ No missing values
✓ Stratified sampling maintained (20.38% vs 20.35% churn)
✓ All preprocessing artifacts saved
```

## Next Steps

Now you can proceed to:
- **Step 4b:** Create `train.py` (train models)
- **Step 4c:** Create `evaluate.py` (evaluate models)
- **Step 4d:** Create `score.py` (inference)

## Example Usage

### Default settings
```bash
python src/data_prep.py --input data/churn.csv --output data/processed
```

### Custom test split (30%)
```bash
python src/data_prep.py \
  --input data/churn.csv \
  --output data/processed \
  --test-size 0.3
```

### With sample data
```bash
python src/data_prep.py \
  --input data/sample.csv \
  --output data/processed_sample \
  --test-size 0.2
```

## Key Design Decisions

1. **Stratified Split:** Maintains class distribution in train/test
2. **StandardScaler:** Chosen over MinMaxScaler for better handling of outliers
3. **Label Encoding:** Used for categorical features (suitable for tree-based models)
4. **Removed Columns:** RowNumber, CustomerId, Surname (high cardinality, no predictive value)
5. **Artifacts Saved:** Encoders and scaler saved for consistent preprocessing at inference time

## Script Architecture

```python
load_data()
    ↓
remove_uninformative_columns()
    ↓
train_test_split() [stratified]
    ↓
encode_categorical_features() [fit on train]
    ↓
scale_numerical_features() [fit on train]
    ↓
save_processed_data() + save_artifacts()
```

## Status

✅ **COMPLETE** - Data preparation script working and validated

