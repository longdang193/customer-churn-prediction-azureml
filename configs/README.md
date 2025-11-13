# Configuration Files

This directory contains centralized YAML configuration files for the ML pipeline.

## Configuration Files

- **`data.yaml`**: Data preprocessing settings (input/output paths, split ratios, target column, etc.)
- **`train.yaml`**: Model training settings (models to train, hyperparameters)
- **`hpo.yaml`**: Hyperparameter optimization settings (search space, budget, early stopping)
- **`mlflow.yaml`**: MLflow experiment tracking (experiment name)
- **`evaluate.yaml`**: Model evaluation settings (number of top features to display)

## Usage

Each script in `src/` loads its corresponding configuration file by default:

```bash
# Uses configs/data.yaml
python src/data_prep.py

# Uses configs/train.yaml and configs/mlflow.yaml
python src/train.py

# Uses configs/evaluate.yaml
python src/evaluate.py --run-id <run_id> --data data/processed --output evaluation/
```

### Overriding Configuration

Override the default config file using `--config`, or override specific parameters with CLI arguments:

```bash
# Use a custom training configuration
python src/train.py --config configs/train_custom.yaml

# Override models to train
python src/train.py --models rf
```

### Configuration Precedence

1. **CLI Arguments** (highest priority)
2. **Config File Values**
3. **Script Defaults** (lowest priority)

## Benefits

- **Centralized**: All settings in one place
- **Reproducible**: Same configuration produces same results
- **Version Controlled**: Track configuration changes in Git
- **Flexible**: Easy to switch between different configurations
