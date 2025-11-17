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


def build_parameter_space(search_space: Dict[str, Any]) -> Dict[str, Any]:
    """Return the sweep search space exactly as defined in the YAML config."""
    return search_space