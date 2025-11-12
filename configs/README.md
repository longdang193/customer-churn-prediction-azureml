# Configuration Files

This directory contains the centralized YAML configuration files for the ML pipeline.

## Configuration Files

- **`data.yaml`**: Controls data preprocessing settings (paths, split ratios, etc.).
- **`train.yaml`**: Controls model training behavior (models to train, hyperparameters, HPO search spaces).
- **`evaluate.yaml`**: Controls model evaluation settings (e.g., number of features to plot).
- **`score.yaml`**: Controls scoring/inference behavior (e.g., whether to include probabilities).
- **`mlflow.yaml`**: Configures MLflow for experiment tracking (e.g., experiment name, tracking URI).

## Usage

Each script in the `src/` directory is designed to load its corresponding configuration file by default. For example, `src/train.py` will automatically load `configs/train.yaml` and `configs/mlflow.yaml`.

### Default Usage

```bash
# Uses configs/data.yaml
python src/data_prep.py

# Uses configs/train.yaml and configs/mlflow.yaml
python src/train.py
```

### Overriding Configuration

You can override the default configuration file using the `--config` flag, or override specific parameters with other CLI arguments.

```bash
# Use a custom training configuration
python src/train.py --config configs/train_hpo_experiment.yaml

# Use the default config but override the models to train
python src/train.py --models rf xgboost
```

### Configuration Precedence

1.  **CLI Arguments** (highest priority)
2.  **Config File Values**
3.  **Script Defaults** (lowest priority)

## Benefits

-   **Centralized**: All settings are in one place.
-   **Reproducible**: The same configuration produces the same results.
-   **Version Controlled**: Configuration changes can be tracked in Git.
-   **Flexible**: Easily switch between local, development, and production settings.
