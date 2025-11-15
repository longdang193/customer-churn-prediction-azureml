"""Configuration parsing and loading utilities."""

import argparse
from pathlib import Path
from typing import Any, Dict

from .config_loader import get_config_value, load_config

DEFAULT_CONFIG = Path(__file__).parents[2] / "configs" / "data.yaml"
DEFAULT_COLUMNS_TO_REMOVE = ("RowNumber", "CustomerId", "Surname")
DEFAULT_CATEGORICAL = ("Geography", "Gender")


def parse_bool(value: Any, *, default: bool) -> bool:
    """Parse loose truthy/falsey values without relying on distutils.
    
    Args:
        value: Value to parse as boolean
        default: Default value if value is None
        
    Returns:
        Boolean value
        
    Raises:
        ValueError: If value cannot be interpreted as boolean
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False

    raise ValueError(f"Cannot interpret value '{value}' as boolean.")


def get_data_prep_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Load config from file and merge with CLI arguments.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Dictionary of configuration values for data preparation
    """
    config_path = Path(args.config or DEFAULT_CONFIG)
    config = load_config(str(config_path)) if config_path.exists() else {}
    cfg = get_config_value(config, "data", {}) or {}

    stratify_raw = get_config_value(cfg, "stratify", True)
    stratify = parse_bool(stratify_raw, default=True)

    return {
        "input_path": Path(args.input or get_config_value(cfg, "input_path", "data/churn.csv")),
        "output_dir": Path(args.output or get_config_value(cfg, "output_dir", "data/processed")),
        "test_size": float(args.test_size or get_config_value(cfg, "test_size", 0.2)),
        "random_state": int(args.random_state or get_config_value(cfg, "random_state", 42)),
        "target_col": args.target or get_config_value(cfg, "target_column", "Exited"),
        "columns_to_remove": get_config_value(cfg, "columns_to_remove", DEFAULT_COLUMNS_TO_REMOVE),
        "categorical_cols": get_config_value(cfg, "categorical_columns", DEFAULT_CATEGORICAL),
        "stratify": stratify,
    }

