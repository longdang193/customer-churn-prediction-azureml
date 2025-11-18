# Utils Package

This package provides atomic, reusable utility functions organized by domain. All utilities are designed to be independent, testable, and follow the single responsibility principle.

## Structure

The package is organized into two categories:

### Core Utilities (Atomic, Reusable)

These are low-level utilities that can be used independently across the project:

- **`config_loader.py`** - YAML configuration file loading
- **`env_loader.py`** - Environment variable loading from `.env` files
- **`path_utils.py`** - Path resolution utilities
- **`type_utils.py`** - Type conversion and parsing utilities

### Domain-Specific Utilities

These utilities build on core utilities to provide domain-specific functionality:

- **`azure_config.py`** - Azure ML configuration loading
- **`config_utils.py`** - Data preparation configuration utilities
- **`mlflow_utils.py`** - MLflow run management and Azure ML detection
- **`metrics.py`** - Model evaluation metrics calculation

## Core Utilities

### `config_loader.py`

YAML configuration file loading utilities.

**Functions:**

- `load_config(config_path: str) -> Dict[str, Any]` - Load configuration from YAML file
- `get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any` - Get nested configuration value using dot notation

**Example:**

```python
from utils import load_config, get_config_value

# Load YAML config
config = load_config("configs/data.yaml")

# Get nested value
test_size = get_config_value(config, "data.test_size", default=0.2)
```

### `env_loader.py`

Environment variable loading utilities.

**Functions:**

- `load_env_file(config_path: Optional[str] = None) -> None` - Load environment variables from config.env file
- `get_env_var(key: str, default: Optional[str] = None, required: bool = False) -> Optional[str]` - Get environment variable with validation

**Example:**

```python
from utils import load_env_file, get_env_var

# Load config.env
load_env_file()

# Get environment variable
subscription_id = get_env_var("AZURE_SUBSCRIPTION_ID", required=True)
workspace_name = get_env_var("AZURE_WORKSPACE_NAME", default="default-workspace")
```

### `path_utils.py`

Path resolution utilities.

**Functions:**

- `get_project_root() -> Path` - Get the project root directory
- `get_config_env_path(config_path: Optional[str] = None) -> Path` - Get the path to config.env file

**Example:**

```python
from utils import get_project_root, get_config_env_path

# Get project root
root = get_project_root()

# Get config.env path
config_path = get_config_env_path()
```

### `type_utils.py`

Type conversion and parsing utilities.

**Functions:**

- `parse_bool(value: Any, *, default: bool) -> bool` - Parse loose truthy/falsey values

**Example:**

```python
from utils import parse_bool

# Parse boolean from various formats
value1 = parse_bool("true", default=False)  # True
value2 = parse_bool("yes", default=False)   # True
value3 = parse_bool("0", default=True)      # False
value4 = parse_bool(None, default=True)     # True (default)
```

## Domain-Specific Utilities

### `azure_config.py`

Azure ML configuration loading utilities. Uses `env_loader` and `path_utils` internally.

**Functions:**

- `load_azure_config(config_path: Optional[str] = None) -> Dict[str, str]` - Load Azure ML workspace configuration
- `get_data_asset_config(config_path: Optional[str] = None) -> Dict[str, str]` - Get data asset configuration (requires `DATA_ASSET_FULL` and `DATA_VERSION` in your `config.env`)

**Example:**

```python
from utils import load_azure_config, get_data_asset_config

# Load Azure ML config
azure_config = load_azure_config()
# Returns: {
#     "subscription_id": "...",
#     "resource_group": "...",
#     "workspace_name": "..."
# }

# Get data asset config (values must be defined in config.env)
data_config = get_data_asset_config()
# Example returns: {
#     "data_asset_name": "churn-data",
#     "data_asset_version": "1"
# }
```

### `config_utils.py`

Data preparation configuration utilities. Uses `config_loader`, `path_utils`, and `type_utils` internally.

**Functions:**

- `get_data_prep_config(args: argparse.Namespace) -> Dict[str, Any]` - Load data prep config from file and merge with CLI arguments

**Constants:**

- `DEFAULT_CONFIG` - Default path to data.yaml config file
- `DEFAULT_COLUMNS_TO_REMOVE` - Default columns to remove during data prep
- `DEFAULT_CATEGORICAL` - Default categorical columns

**Example:**

```python
from utils import get_data_prep_config
import argparse

args = argparse.Namespace(
    input="data/churn.csv",
    output="data/processed",
    test_size=0.2
)

config = get_data_prep_config(args)
# Returns configuration dictionary with input_path, output_dir, test_size, etc.
```

### `mlflow_utils.py`

MLflow run management and Azure ML environment detection.

**Functions:**

- `is_azure_ml() -> bool` - Check if running in Azure ML environment
- `get_run_id(run_obj: Any) -> str` - Extract run ID from MLflow run object
- `get_active_run()` - Get the active MLflow run
- `start_parent_run(experiment_name: str, run_name: str = "Churn_Training_Pipeline")` - Start a parent MLflow run
- `start_nested_run(run_name: str)` - Start a nested MLflow run

**Example:**

```python
from utils import is_azure_ml, start_parent_run, start_nested_run

# Check environment
if is_azure_ml():
    print("Running in Azure ML")

# Start parent run
with start_parent_run("churn-prediction-experiment"):
    # Start nested run
    nested_run, run_id = start_nested_run("data_prep")
    # ... your code ...
```

### `metrics.py`

Model evaluation metrics calculation.

**Functions:**

- `calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]` - Calculate core evaluation metrics

**Example:**

```python
from utils import calculate_metrics
import numpy as np

y_true = np.array([0, 1, 1, 0])
y_pred = np.array([0, 1, 0, 0])
y_pred_proba = np.array([0.1, 0.9, 0.4, 0.2])

metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
# Returns: {
#     "accuracy": 0.75,
#     "precision": 1.0,
#     "recall": 0.5,
#     "f1": 0.67,
#     "roc_auc": 0.75
# }
```

## Usage

### Importing Utilities

All utilities can be imported from the `utils` package:

```python
# Import specific functions
from utils import load_azure_config, get_data_asset_config

# Import multiple utilities
from utils import (
    load_config,
    get_config_value,
    parse_bool,
    calculate_metrics
)
```

### Module Dependencies

The utilities are organized to minimize dependencies:

```
Core Utilities (no dependencies on other utils):
├── config_loader.py
├── env_loader.py (depends on path_utils)
├── path_utils.py
└── type_utils.py

Domain-Specific Utilities:
├── azure_config.py (depends on env_loader, path_utils)
├── config_utils.py (depends on config_loader, path_utils, type_utils)
├── mlflow_utils.py (no dependencies on other utils)
└── metrics.py (no dependencies on other utils)
```

## Design Principles

1. **Atomic Functions**: Each function has a single, well-defined responsibility
2. **Reusability**: Core utilities can be used independently across the project
3. **No Duplication**: Shared logic is extracted to common modules
4. **Type Hints**: All functions include type hints for better IDE support
5. **Documentation**: All functions include docstrings with examples
6. **Error Handling**: Functions provide clear error messages when validation fails

## Testing

Each utility module should have corresponding tests. The atomic nature of these utilities makes them easy to test in isolation.

## Contributing

When adding new utilities:

1. **Determine if it's core or domain-specific**: Core utilities should be generic and reusable. Domain-specific utilities can depend on core utilities.
2. **Follow naming conventions**: Use descriptive function names and follow existing patterns.
3. **Add type hints**: All functions should include type hints.
4. **Write docstrings**: Include docstrings with Args, Returns, Raises, and Examples.
5. **Update `__init__.py`**: Export new functions from the package `__init__.py`.
6. **Update this README**: Document new utilities in the appropriate section.
