# Configuration Files

Centralized YAML configuration files for the ML pipeline.

## Configuration Files

### 1. `data.yaml` - Data Preparation

Controls data preprocessing settings:
- Input/output paths
- Target variable
- Columns to remove
- Categorical columns
- Train/test split parameters

### 2. `train.yaml` - Model Training

Controls training behavior:
- Models to train
- Class imbalance handling
- Hyperparameters
- MLflow settings
- Output configuration

### 3. `evaluate.yaml` - Model Evaluation

Controls evaluation settings:
- Visualization options
- Feature importance settings
- Output format

### 4. `score.yaml` - Model Scoring

Controls scoring/inference:
- Classification threshold
- Output format
- Batch processing

## Usage

### Using Config Files (Recommended)

```bash
# Data preparation (uses configs/data.yaml)
python src/data_prep.py

# Training (uses configs/train.yaml)
python src/train.py

# With custom config
python src/data_prep.py --config configs/custom_data.yaml
```

### Overriding Config Values

CLI arguments override config file values:

```bash
# Use config but override output directory
python src/data_prep.py --output data/custom_output

# Use config but override models
python src/train.py --models rf xgboost

# Use config but enable SMOTE
python src/train.py --use-smote
```

### Config File Structure

```yaml
# Example: data.yaml
data:
  input_path: "data/churn.csv"
  output_dir: "data/processed"
  target_column: "Exited"
  test_size: 0.2
  random_state: 42
```

## Benefits

✅ **Centralized Configuration:** All settings in one place  
✅ **Reproducibility:** Same config = same results  
✅ **Easy Updates:** Change config, not code  
✅ **CLI Override:** Flexibility when needed  
✅ **Version Control:** Track config changes in git  
✅ **Azure ML Ready:** Same configs work locally and in cloud  

## Configuration Precedence

1. **CLI Arguments** (highest priority)
2. **Config File Values**
3. **Default Values** (lowest priority)

## Example Workflows

### Local Development

```bash
# Use default configs
python src/data_prep.py
python src/train.py
```

### Azure ML Pipeline

```bash
# Use same configs in Azure ML
python src/data_prep.py --config configs/data.yaml
python src/train.py --config configs/train.yaml
```

### Experimentation

```bash
# Create custom config for experiment
cp configs/train.yaml configs/experiment1.yaml
# Edit experiment1.yaml
python src/train.py --config configs/experiment1.yaml
```

## Config File Locations

- `configs/data.yaml` - Data preparation
- `configs/train.yaml` - Model training
- `configs/evaluate.yaml` - Model evaluation
- `configs/score.yaml` - Model scoring

## Environment-Specific Configs

Create environment-specific configs:

- `configs/data_local.yaml` - Local development
- `configs/data_azure.yaml` - Azure ML
- `configs/train_prod.yaml` - Production training

Then use:
```bash
python src/data_prep.py --config configs/data_azure.yaml
```

