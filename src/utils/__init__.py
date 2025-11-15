"""Utility modules for configuration, MLflow, and metrics."""

from .config_loader import get_config_value, load_config
from .config_utils import (
    DEFAULT_CONFIG as DATA_PREP_DEFAULT_CONFIG,
    get_data_prep_config,
    parse_bool,
)
from .mlflow_utils import (
    get_active_run,
    get_run_id,
    is_azure_ml,
    start_nested_run,
    start_parent_run,
)
from .metrics import calculate_metrics

__all__ = [
    # Config loader
    "get_config_value",
    "load_config",
    # Config utils
    "DATA_PREP_DEFAULT_CONFIG",
    "DEFAULT_CONFIG",  # Alias for backward compatibility
    "get_data_prep_config",
    "parse_bool",
    # MLflow utils
    "get_active_run",
    "get_run_id",
    "is_azure_ml",
    "start_nested_run",
    "start_parent_run",
    # Metrics
    "calculate_metrics",
]

# Alias for backward compatibility
DEFAULT_CONFIG = DATA_PREP_DEFAULT_CONFIG

