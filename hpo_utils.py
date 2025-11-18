"""Shared utilities for HPO pipelines."""

from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml
from azure.ai.ml.sweep import Choice

# Resolve relative to project root so the module works from any CWD (e.g., notebooks/)
PROJECT_ROOT = Path(__file__).resolve().parents[0]
CONFIG_PATH = PROJECT_ROOT / "configs" / "hpo.yaml"


def load_hpo_config() -> Dict[str, Any]:
    """Load HPO configuration from hpo.yaml."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"HPO config not found at {CONFIG_PATH}")
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}
    return config


def _resolve_model_types(search_space: Dict[str, Any]) -> List[str]:
    configured_types = search_space.get("model_types")
    if configured_types:
        return list(configured_types)
    inferred: List[str] = []
    for candidate in ("logreg", "rf", "xgboost"):
        if candidate in search_space:
            inferred.append(candidate)
    return inferred


def _filter_nulls(obj: Any) -> Any:
    """Recursively filter out None/null values from lists in the search space.
    
    Azure ML sweep Choice does not accept None/null values.
    """
    if isinstance(obj, dict):
        return {key: _filter_nulls(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        # Filter out None values from lists
        filtered = [item for item in obj if item is not None]
        return [_filter_nulls(item) for item in filtered]
    else:
        return obj


def build_parameter_space(search_space: Dict[str, Any]) -> Dict[str, Any]:
    """Build the sweep search space from YAML config, filtering out null values.
    
    Azure ML sweep Choice does not accept None/null values, so they are removed.
    """
    return _filter_nulls(search_space)