"""Shared utilities for HPO pipelines."""

from pathlib import Path
from typing import Any, Dict

import yaml
from azure.ai.ml.sweep import Choice

CONFIG_PATH = Path("configs/hpo.yaml")


def load_hpo_config() -> Dict[str, Any]:
    """Load HPO configuration from hpo.yaml."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"HPO config not found at {CONFIG_PATH}")
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}
    return config


def build_parameter_space(search_space: Dict[str, Any]) -> Dict[str, Choice]:
    """Build parameter space for sweep from config.
    
    Treats model_type as a categorical hyperparameter, so each trial trains only one model.
    Model-specific hyperparameters are included conditionally based on model_type.
    """
    parameter_space: Dict[str, Choice] = {}
    
    # Add model_type as categorical hyperparameter
    available_models = []
    if "logreg" in search_space:
        available_models.append("logreg")
    if "rf" in search_space:
        available_models.append("rf")
    if "xgboost" in search_space:
        available_models.append("xgboost")
    
    if not available_models:
        raise ValueError("No models found in search space. Check configs/hpo.yaml::search_space")
    
    parameter_space["model_type"] = Choice(values=available_models)

    def to_choice(values):
        cleaned = [None if v is None else v for v in values]
        return Choice(values=cleaned)

    # Add Random Forest hyperparameters
    rf_space = search_space.get("rf", {})
    if rf_space:
        if "n_estimators" in rf_space:
            parameter_space["rf_n_estimators"] = to_choice(rf_space["n_estimators"])
        if "max_depth" in rf_space:
            parameter_space["rf_max_depth"] = to_choice(rf_space["max_depth"])
        if "min_samples_split" in rf_space:
            parameter_space["rf_min_samples_split"] = to_choice(rf_space["min_samples_split"])
        if "min_samples_leaf" in rf_space:
            parameter_space["rf_min_samples_leaf"] = to_choice(rf_space["min_samples_leaf"])

    # Add Logistic Regression hyperparameters
    logreg_space = search_space.get("logreg", {})
    if logreg_space:
        if "C" in logreg_space:
            parameter_space["logreg_C"] = to_choice(logreg_space["C"])
        if "solver" in logreg_space:
            parameter_space["logreg_solver"] = to_choice(logreg_space["solver"])

    # Add XGBoost hyperparameters
    xgboost_space = search_space.get("xgboost", {})
    if xgboost_space:
        if "n_estimators" in xgboost_space:
            parameter_space["xgboost_n_estimators"] = to_choice(xgboost_space["n_estimators"])
        if "max_depth" in xgboost_space:
            parameter_space["xgboost_max_depth"] = to_choice(xgboost_space["max_depth"])
        if "learning_rate" in xgboost_space:
            parameter_space["xgboost_learning_rate"] = to_choice(xgboost_space["learning_rate"])
        if "subsample" in xgboost_space:
            parameter_space["xgboost_subsample"] = to_choice(xgboost_space["subsample"])
        if "colsample_bytree" in xgboost_space:
            parameter_space["xgboost_colsample_bytree"] = to_choice(xgboost_space["colsample_bytree"])

    if len(parameter_space) == 1:  # Only model_type, no hyperparameters
        raise ValueError("No hyperparameters found in search space. Check configs/hpo.yaml::search_space")
    
    return parameter_space

