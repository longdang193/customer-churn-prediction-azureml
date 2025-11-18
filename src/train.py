#!/usr/bin/env python3
"""Training entry-point for the churn prediction models."""

import argparse
from pathlib import Path

from training import (
    determine_models_to_train,
    is_hpo_mode,
    prepare_regular_hyperparams,
    train_pipeline_stage,
)
from utils import get_config_value, load_config

DEFAULT_CONFIG = Path(__file__).parents[1] / "configs" / "train.yaml"


def main() -> None:
    """CLI entry-point."""
    parser = argparse.ArgumentParser(
        description='Train churn prediction models.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data", type=str, default=None, help="Directory with preprocessed data")
    parser.add_argument("--config", type=str, default=None, help=f"Config file (default: {DEFAULT_CONFIG})")
    parser.add_argument(
        "--model-type", type=str, default=None, choices=['logreg', 'rf', 'xgboost'],
        help='Single model type for HPO mode'
    )
    parser.add_argument("--class-weight", type=str, default=None, help='Class weight strategy')
    parser.add_argument("--random-state", type=int, default=None, help='Random seed')
    parser.add_argument("--experiment-name", type=str, default=None, help='MLflow experiment name')
    parser.add_argument("--use-smote", action='store_true', help='Apply SMOTE')
    parser.add_argument("--model-artifact-dir", type=str, default=None, help='Directory to save best model')
    parser.add_argument("--parent-run-id-output", type=str, default=None, help='File to write parent run ID')
    parser.add_argument(
        "--set", action="append", default=[], metavar="model.param=value",
        help="Override hyperparameters (can be used multiple times)"
    )
    args = parser.parse_args()
    
    config_path = Path(args.config or DEFAULT_CONFIG)
    config = load_config(str(config_path)) if config_path.exists() else {}
    mlflow_config = load_config(str(config_path.parent / "mlflow.yaml")) if (config_path.parent / "mlflow.yaml").exists() else {}
    
    training_config = get_config_value(config, 'training', {})
    mlflow_config = get_config_value(mlflow_config, 'mlflow', {})
    
    # Apply training-level parameter overrides from --set (e.g., use_smote, class_weight, random_state)
    # These are parameters that don't have a model prefix (unlike model.param=value)
    from training.hyperparams import parse_override_value
    training_level_params = ['use_smote', 'class_weight', 'random_state']
    model_param_overrides = []  # Filter out training-level params for model hyperparams
    for override in args.set:
        if '=' in override:
            try:
                key, value_str = override.split('=', 1)
                if key in training_level_params:
                    # Parse the value (handles booleans, ints, strings, etc.)
                    parsed_value = parse_override_value(value_str)
                    training_config[key] = parsed_value
                else:
                    # Keep model-specific overrides (model.param=value format)
                    model_param_overrides.append(override)
            except (ValueError, AttributeError):
                # Keep invalid overrides - they'll be handled by apply_param_overrides
                model_param_overrides.append(override)
        else:
            # Keep overrides without '=' (shouldn't happen, but be safe)
            model_param_overrides.append(override)
    
    hpo_mode = is_hpo_mode(args.model_type)
    hyperparams_by_model = prepare_regular_hyperparams(training_config, model_param_overrides)
    
    models_to_train = determine_models_to_train(hpo_mode, args.model_type, training_config)
    
    train_pipeline_stage(
        data_dir=args.data or 'data/processed',
        models=models_to_train,
        class_weight=args.class_weight or get_config_value(training_config, 'class_weight', 'balanced'),
        random_state=args.random_state or get_config_value(training_config, 'random_state', 42),
        experiment_name=args.experiment_name or get_config_value(mlflow_config, 'experiment_name', 'churn-prediction'),
        use_smote=args.use_smote or get_config_value(training_config, 'use_smote', True),
        hyperparams_by_model=hyperparams_by_model,
        model_artifact_dir=args.model_artifact_dir,
        parent_run_id_output=args.parent_run_id_output,
    )


if __name__ == '__main__':
    main()
