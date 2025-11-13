#!/usr/bin/env python3
"""Extract best hyperparameters from HPO sweep and update training config.

This script:
1. Extracts best hyperparameters from MLflow HPO sweep
2. Automatically updates configs/train.yaml with best model and hyperparameters
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import yaml

# Import azureml.mlflow before mlflow to register Azure ML tracking store
try:
    import azureml.mlflow  # noqa: F401
except ImportError:
    pass  # Not in Azure ML environment

import mlflow
from mlflow.tracking import MlflowClient


def extract_best_params_from_sweep(parent_run_id: str, metric_name: str = "f1") -> Dict[str, Any]:
    """Extract best hyperparameters and model type from a sweep run."""
    client = MlflowClient()
    
    # Get the parent run
    try:
        parent_run = client.get_run(parent_run_id)
    except Exception as e:
        raise ValueError(f"Could not get parent run {parent_run_id}: {e}")
    
    # Find best child run (trial)
    experiment_id = parent_run.info.experiment_id
    search_filter = f"tags.mlflow.parentRunId = '{parent_run_id}'"
    child_runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=search_filter,
        order_by=[f"metrics.{metric_name} DESC"],
        max_results=1
    )
    
    if not child_runs:
        all_runs = client.search_runs(
            experiment_ids=[experiment_id],
            order_by=[f"metrics.{metric_name} DESC"],
            max_results=100
        )
        child_runs = [run for run in all_runs if run.data.tags.get("mlflow.parentRunId") == parent_run_id]
    
    if not child_runs:
        raise ValueError(f"No child runs found for parent run {parent_run_id}")
    
    best_run = child_runs[0]
    
    # Extract model_type from best run
    model_type = best_run.data.tags.get("model_type")
    if not model_type:
        # Infer from hyperparameters
        params_keys = list(best_run.data.params.keys())
        if any(k.startswith("rf_") for k in params_keys):
            model_type = "rf"
        elif any(k.startswith("xgboost_") for k in params_keys):
            model_type = "xgboost"
        elif any(k.startswith("logreg_") for k in params_keys):
            model_type = "logreg"
    
    # Extract hyperparameters
    param_mapping = {
        "rf_max_depth": "rf_max_depth",
        "rf_n_estimators": "rf_n_estimators",
        "rf_min_samples_split": "rf_min_samples_split",
        "rf_min_samples_leaf": "rf_min_samples_leaf",
        "xgboost_max_depth": "xgboost_max_depth",
        "xgboost_n_estimators": "xgboost_n_estimators",
        "xgboost_learning_rate": "xgboost_learning_rate",
        "xgboost_subsample": "xgboost_subsample",
        "xgboost_colsample_bytree": "xgboost_colsample_bytree",
        "logreg_C": "logreg_C",
        "logreg_solver": "logreg_solver",
    }
    
    best_params = {}
    for mlflow_key, component_key in param_mapping.items():
        if mlflow_key in best_run.data.params:
            value = best_run.data.params[mlflow_key]
            # Convert to appropriate type
            try:
                if isinstance(value, str) and value.lower() in ("none", "null", ""):
                    best_params[component_key] = None
                else:
                    try:
                        best_params[component_key] = int(value)
                    except (ValueError, TypeError):
                        try:
                            best_params[component_key] = float(value)
                        except (ValueError, TypeError):
                            best_params[component_key] = value
            except (ValueError, AttributeError, TypeError):
                best_params[component_key] = value
    
    if not best_params:
        raise ValueError(
            f"No hyperparameters found in best run {best_run.info.run_id}.\n"
            f"Available params: {list(best_run.data.params.keys())}\n"
            f"Available tags: {list(best_run.data.tags.keys())}"
        )
    
    # Add model_type for config update
    if model_type:
        best_params["_model_type"] = model_type
    
    return best_params


def update_config(config_path: str, best_params: Dict[str, Any], dry_run: bool = False) -> None:
    """Update configs/train.yaml with best hyperparameters and model."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load config
    with open(config_file, "r") as f:
        config = yaml.safe_load(f) or {}
    
    # Convert flat params to nested format and extract model_type
    nested_params: Dict[str, Dict[str, Any]] = {}
    model_type = best_params.pop("_model_type", None)
    
    for key, value in best_params.items():
        if key.startswith("rf_"):
            nested_params.setdefault("rf", {})[key[3:]] = value
        elif key.startswith("xgboost_"):
            nested_params.setdefault("xgboost", {})[key[8:]] = value
        elif key.startswith("logreg_"):
            nested_params.setdefault("logreg", {})[key[7:]] = value
    
    # Update config
    if "training" not in config:
        config["training"] = {}
    if "hyperparameters" not in config["training"]:
        config["training"]["hyperparameters"] = {}
    
    # Update hyperparameters
    for model_name, params in nested_params.items():
        config["training"]["hyperparameters"].setdefault(model_name, {}).update(params)
    
    # Update models list to only include best model
    if model_type:
        config["training"]["models"] = [model_type]
    elif nested_params:
        config["training"]["models"] = [list(nested_params.keys())[0]]
    
    # Write updated config
    if not dry_run:
        with open(config_file, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"✓ Updated {config_path}")
        if model_type:
            print(f"  Best model: {model_type}")
    else:
        print(f"[DRY RUN] Would update {config_path}:")
        print(yaml.dump(config, default_flow_style=False, sort_keys=False))


def main() -> None:
    """CLI entry-point."""
    parser = argparse.ArgumentParser(
        description="Extract best hyperparameters from HPO sweep and update training config"
    )
    parser.add_argument(
        "--parent-run-id",
        type=str,
        help="Parent run ID from HPO sweep (alternative to --parent-run-id-file)"
    )
    parser.add_argument(
        "--parent-run-id-file",
        type=str,
        help="File containing the parent run ID from sweep job"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="f1",
        help="Metric to optimize (default: f1)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Config file to update (default: configs/train.yaml)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Optional: Save best params to JSON file"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without modifying files"
    )
    parser.add_argument(
        "--no-update-config",
        action="store_true",
        help="Skip config update (only extract params)"
    )
    args = parser.parse_args()
    
    # Get parent run ID
    if args.parent_run_id:
        parent_run_id = args.parent_run_id
    elif args.parent_run_id_file:
        run_id_file = Path(args.parent_run_id_file)
        if not run_id_file.exists():
            raise FileNotFoundError(f"Parent run ID file not found: {run_id_file}")
        parent_run_id = run_id_file.read_text().strip()
    else:
        parser.error("Either --parent-run-id or --parent-run-id-file must be provided")
    
    # Extract best parameters
    print(f"Extracting best hyperparameters from HPO sweep (run: {parent_run_id})...")
    best_params = extract_best_params_from_sweep(parent_run_id, args.metric)
    
    # Save to JSON if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(best_params, f, indent=2)
        print(f"✓ Saved best params to {output_path}")
    
    # Update config (default behavior)
    if not args.no_update_config:
        update_config(args.config, best_params.copy(), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
