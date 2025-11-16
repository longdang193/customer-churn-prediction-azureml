"""Utility modules for configuration, MLflow, and metrics.

This package provides atomic, reusable utilities organized by domain:
- config_loader: YAML configuration file loading
- env_loader: Environment variable loading
- path_utils: Path resolution utilities
- type_utils: Type conversion utilities
- azure_config: Azure ML specific configuration
- config_utils: Data preparation specific configuration
- mlflow_utils: MLflow run management
- metrics: Model evaluation metrics
"""

# Core utilities (atomic, reusable)
from .config_loader import get_config_value, load_config
from .env_loader import get_env_var, load_env_file
from .path_utils import get_config_env_path, get_project_root
from .type_utils import parse_bool

# Domain-specific utilities
from .azure_config import get_data_asset_config, load_azure_config
from .config_utils import (
    DEFAULT_CONFIG as DATA_PREP_DEFAULT_CONFIG,
    get_data_prep_config,
)

# Optional dependencies - import with fallback
try:
    from .mlflow_utils import (
        get_active_run,
        get_run_id,
        is_azure_ml,
        start_nested_run,
        start_parent_run,
    )
except ImportError:
    # mlflow not available - define stubs or skip
    get_active_run = None
    get_run_id = None
    is_azure_ml = None
    start_nested_run = None
    start_parent_run = None

try:
    from .metrics import calculate_metrics
except ImportError:
    calculate_metrics = None

__all__ = [
    # Core utilities
    "get_config_value",
    "load_config",
    "get_env_var",
    "load_env_file",
    "get_config_env_path",
    "get_project_root",
    "parse_bool",
    # Domain-specific utilities
    "DATA_PREP_DEFAULT_CONFIG",
    "DEFAULT_CONFIG",  # Alias for backward compatibility
    "get_data_prep_config",
    "get_data_asset_config",
    "load_azure_config",
    # Optional utilities (may be None if dependencies not installed)
    "get_active_run",
    "get_run_id",
    "is_azure_ml",
    "start_nested_run",
    "start_parent_run",
    "calculate_metrics",
]

# Alias for backward compatibility
DEFAULT_CONFIG = DATA_PREP_DEFAULT_CONFIG
