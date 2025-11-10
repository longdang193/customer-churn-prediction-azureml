# Step 5: Centralize Configuration - Complete ✅

Configuration centralization with YAML files has been implemented successfully.

## What Was Created

### Configuration Files

1. **`configs/data.yaml`** - Data preparation configuration
2. **`configs/train.yaml`** - Training configuration
3. **`configs/evaluate.yaml`** - Evaluation configuration (optional)
4. **`configs/score.yaml`** - Scoring configuration (optional)

### Configuration Loader

**`src/config_loader.py`** - Utility for loading and parsing YAML configs:
- `load_config()` - Load YAML file
- `get_config_value()` - Get nested values with dot notation
- `merge_configs()` - Merge CLI args with config file
- `validate_config()` - Validate required keys

## Configuration Structure

### `configs/data.yaml`

```yaml
data:
  input_path: "data/churn.csv"
  output_dir: "data/processed"
  target_column: "Exited"
  columns_to_remove:
    - "RowNumber"
    - "CustomerId"
    - "Surname"
  categorical_columns:
    - "Geography"
    - "Gender"
  test_size: 0.2
  random_state: 42
  stratify: true
```

### `configs/train.yaml`

```yaml
training:
  models:
    - "logreg"
    - "rf"
  class_weight: "balanced"
  random_state: 42
  use_smote: false

mlflow:
  enabled: false
  experiment_name: "churn-prediction"

output:
  model_dir: "models/local"
  save_metrics: true
```

## Updated Scripts

### `data_prep.py`
- ✅ Reads from `configs/data.yaml` by default
- ✅ CLI arguments override config values
- ✅ Falls back to defaults if config not found

### `train.py`
- ✅ Reads from `configs/train.yaml` by default
- ✅ CLI arguments override config values
- ✅ Supports MLflow configuration from YAML

## Usage Examples

### Using Config Files (Recommended)

```bash
# Data preparation - uses configs/data.yaml
python src/data_prep.py

# Training - uses configs/train.yaml
python src/train.py

# Evaluation - uses configs/evaluate.yaml (if implemented)
python src/evaluate.py --config configs/evaluate.yaml
```

### Overriding Config Values

```bash
# Override output directory
python src/data_prep.py --output data/custom_output

# Override models to train
python src/train.py --models rf xgboost

# Override and enable SMOTE
python src/train.py --use-smote

# Override MLflow experiment name
python src/train.py --experiment-name my-experiment
```

### Custom Config Files

```bash
# Use custom config file
python src/data_prep.py --config configs/data_azure.yaml
python src/train.py --config configs/train_prod.yaml
```

## Configuration Precedence

1. **CLI Arguments** (highest priority)
2. **Config File Values**
3. **Default Values** (lowest priority)

## Benefits

✅ **Centralized:** All settings in config files  
✅ **Reproducible:** Same config = same results  
✅ **Flexible:** CLI overrides when needed  
✅ **Version Control:** Track config changes in git  
✅ **Azure ML Ready:** Same configs work locally and in cloud  
✅ **Easy Updates:** Change config, not code  

## Test Results

### Data Prep with Config:
```
Loaded configuration from: configs/data.yaml
✓ Uses input_path: data/churn.csv
✓ Uses output_dir: data/processed
✓ Uses test_size: 0.2
✓ Uses random_state: 42
```

### Training with Config:
```
Loaded configuration from: configs/train.yaml
✓ Uses models: ['logreg', 'rf']
✓ Uses class_weight: balanced
✓ Uses random_state: 42
✓ Uses use_smote: false
```

## Complete Workflow with Configs

```bash
# 1. Prepare data (uses configs/data.yaml)
python src/data_prep.py

# 2. Train models (uses configs/train.yaml)
python src/train.py

# 3. Evaluate (CLI args)
python src/evaluate.py --model models/local/rf_model.pkl \
  --data data/processed --output evaluation/rf

# 4. Score (CLI args)
python src/score.py --model models/local/rf_model.pkl \
  --data-dir data/processed --input data/new_customers.csv \
  --output predictions.csv
```

## Environment-Specific Configs

Create different configs for different environments:

```bash
# Local development
configs/data_local.yaml
configs/train_local.yaml

# Azure ML
configs/data_azure.yaml
configs/train_azure.yaml

# Production
configs/data_prod.yaml
configs/train_prod.yaml
```

Then use:
```bash
python src/data_prep.py --config configs/data_azure.yaml
python src/train.py --config configs/train_azure.yaml
```

## Configuration Files Summary

| File | Purpose | Used By |
|------|---------|---------|
| `data.yaml` | Data preprocessing | `data_prep.py` |
| `train.yaml` | Model training | `train.py` |
| `evaluate.yaml` | Evaluation settings | `evaluate.py` (optional) |
| `score.yaml` | Scoring settings | `score.py` (optional) |

## Next Steps

✅ Step 2: Data asset creation  
✅ Step 3: EDA notebook  
✅ Step 4: Scripts (data_prep, train, evaluate, score)  
✅ Step 5: Configuration (YAMLs)  
⏳ Step 6: Dependencies (requirements.txt)  
⏳ Step 7: Smoke test  
⏳ Step 8: Docker image  

## Status

✅ **COMPLETE** - Configuration centralization working with YAML files

**Features:**
- ✅ YAML config files for all scripts
- ✅ Config loader utility
- ✅ CLI argument override support
- ✅ Default config file paths
- ✅ Validation and error handling
- ✅ Ready for Azure ML deployment

