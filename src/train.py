#!/usr/bin/env python3
"""Training entry-point for the churn prediction models."""

from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from config_loader import get_config_value, load_config
from models import get_logistic_regression, get_random_forest, get_xgboost

# Import azureml.mlflow before mlflow to register Azure ML tracking store
import azureml.mlflow  # noqa: F401
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

JSONDict = Dict[str, Any]
DEFAULT_CONFIG = Path(__file__).parents[1] / "configs" / "train.yaml"


def _is_azure_ml() -> bool:
    """Check if running in Azure ML environment."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "")
    return "azureml" in tracking_uri.lower() if tracking_uri else False


def _get_run_id(run_obj: Any) -> str:
    """Extract run ID from MLflow run object."""
    if hasattr(run_obj, 'info'):
        return run_obj.info.run_id
    elif hasattr(run_obj, 'run_id'):
        return run_obj.run_id
    else:
        return os.getenv("MLFLOW_RUN_ID", "unknown")


def load_prepared_data(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load preprocessed training and test data."""
    data_path = Path(data_dir)
    return (
        pd.read_csv(data_path / 'X_train.csv'),
        pd.read_csv(data_path / 'X_test.csv'),
        pd.read_csv(data_path / 'y_train.csv').squeeze(),
        pd.read_csv(data_path / 'y_test.csv').squeeze(),
    )


def get_model(model_name: str, class_weight: Optional[str] = 'balanced', random_state: int = 42) -> Any:
    """Return a configured estimator instance."""
    model_map = {
        'logreg': get_logistic_regression,
        'rf': get_random_forest,
        'xgboost': get_xgboost,
    }
    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {list(model_map.keys())}")
    return model_map[model_name](class_weight=class_weight, random_state=random_state)


def apply_hyperparameters(model: Any, hyperparams: JSONDict | None) -> tuple[Any, JSONDict]:
    """Apply hyperparameters via set_params where supported."""
    applied: JSONDict = {}
    if not hyperparams:
        return model, applied
    
    try:
        valid_params = model.get_params(deep=True)
        applied = {k: v for k, v in hyperparams.items() if k in valid_params}
        
        # Validate RandomForest min_samples_split (sklearn requirement)
        if "min_samples_split" in applied and applied["min_samples_split"] < 2:
            applied["min_samples_split"] = 2
        
        if applied:
            model.set_params(**applied)
    except Exception:
        applied = {}
    
    return model, applied


def predict_probabilities(model: Any, X: pd.DataFrame, model_name: str = "model") -> np.ndarray:
    """Return positive class probabilities."""
    if not hasattr(model, "predict_proba"):
        raise ValueError(
            f"{model_name} lacks predict_proba method. "
            "Use CalibratedClassifierCV wrapper or a model that supports probability prediction."
        )
    
    proba = np.asarray(model.predict_proba(X), dtype=float)
    if proba.ndim == 1:
        return proba
    if proba.shape[1] >= 2:
        return proba[:, 1]
    return proba.ravel()


def maybe_apply_class_weight_adjustments(
    model_name: str,
    model: Any,
    y_train: pd.Series,
    class_weight: Optional[str],
    tuned_params: JSONDict,
) -> JSONDict:
    """Adjust XGBoost scale_pos_weight when class_weight='balanced'."""
    if model_name != "xgboost" or class_weight != "balanced":
        return {}
    if "scale_pos_weight" in (tuned_params or {}):
        return {}
    if not hasattr(model, "get_params"):
        return {}
    
    params = model.get_params()
    if "scale_pos_weight" not in params:
        return {}
    
    positives = (y_train == 1).sum()
    negatives = (y_train == 0).sum()
    if positives == 0 or negatives == 0:
        return {}
    
    current = params.get("scale_pos_weight")
    if current not in (None, 0, 1):
        return {}
    
    scale_pos_weight = float(negatives / positives)
    try:
        model.set_params(scale_pos_weight=scale_pos_weight)
        return {"scale_pos_weight": scale_pos_weight}
    except Exception:
        return {}


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> JSONDict:
    """Return core evaluation metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_pred_proba),
    }


def get_pip_requirements() -> list[str]:
    """Generate pip requirements for MLflow model deployment (local only)."""
    return ["mlflow", "scikit-learn", "pandas", "numpy", "xgboost"]


def train_model(
    model_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    class_weight: Optional[str] = "balanced",
    random_state: int = 42,
    model_hyperparams: JSONDict | None = None,
) -> JSONDict:
    """Train and evaluate a single model, logging to MLflow."""
    is_azure_ml = _is_azure_ml()
    
    # Handle MLflow run context
    if is_azure_ml:
        nested_run = mlflow.active_run()
        if nested_run is None:
            run_id = os.getenv("MLFLOW_RUN_ID")
            if run_id:
                nested_run = mlflow.tracking.MlflowClient().get_run(run_id)
            else:
                raise RuntimeError("Could not determine MLflow run context in Azure ML")
        run_id = _get_run_id(nested_run)
        mlflow.set_tag("model_name", model_name)
    else:
        nested_run = mlflow.start_run(run_name=model_name, nested=True)
        run_id = nested_run.info.run_id
    
    try:
        model = get_model(model_name, class_weight=class_weight, random_state=random_state)
        model, tuned_params = apply_hyperparameters(model, model_hyperparams)
        
        extra_params = maybe_apply_class_weight_adjustments(
            model_name, model, y_train, class_weight, tuned_params
        )
        
        # Log parameters
        params_to_log = {**tuned_params, **extra_params}
        if params_to_log:
            mlflow.log_params(params_to_log)
        
        # Train and evaluate
        model.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)
        y_test_proba = predict_probabilities(model, X_test, model_name)
        test_metrics = calculate_metrics(y_test, y_test_pred, y_test_proba)
        
        for metric, value in test_metrics.items():
            mlflow.log_metric(f"test_{metric}", value)
        
        artifact_path = f"model_{model_name}"
        
        # Save model
        if is_azure_ml:
            outputs_dir = os.getenv("AZUREML_ARTIFACTS_DIRECTORY", os.getenv("AZUREML_OUTPUT_DIRECTORY", "/tmp"))
            model_path = Path(outputs_dir) / f"{model_name}_model.pkl"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, model_path)
            mlflow.set_tag(f"{model_name}_model_file", str(model_path))
        else:
            signature = infer_signature(X_test.head(10), model.predict(X_test.head(10)))
            mlflow.sklearn.log_model(
                model,
                artifact_path,
                pip_requirements=get_pip_requirements(),
                signature=signature,
                input_example=X_test.head(1).to_dict(orient="records")[0]
            )
        
        return {
            'test_metrics': test_metrics,
            'run_id': run_id,
            'artifact_path': artifact_path,
        }
    finally:
        if not is_azure_ml:
            mlflow.end_run()


def apply_smote(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """Apply SMOTE to balance training labels."""
    if not SMOTE_AVAILABLE:
        raise ImportError("SMOTE not available. Install with: pip install imbalanced-learn")
    
    smote = SMOTE(random_state=random_state)
    X_bal, y_bal = smote.fit_resample(X_train, y_train)
    return pd.DataFrame(X_bal, columns=X_train.columns), pd.Series(y_bal)


def train_all_models(
    data_dir: str,
    models: Iterable[str],
    class_weight: Optional[str],
    random_state: int,
    experiment_name: str,
    use_smote: bool,
    hyperparams_by_model: Optional[Dict[str, JSONDict]],
    model_artifact_dir: Optional[str] = None,
    parent_run_id_output: Optional[str] = None,
) -> Dict[str, JSONDict]:
    """Train a collection of models within a parent MLflow run."""
    is_azure_ml = _is_azure_ml()
    started_run = False
    parent_run = None
    
    if not is_azure_ml:
        mlflow.set_experiment(experiment_name)
        parent_run = mlflow.start_run(run_name="Churn_Training_Pipeline")
        started_run = True
    else:
        parent_run = mlflow.active_run()
        if parent_run is None:
            run_id = os.getenv("MLFLOW_RUN_ID")
            if run_id:
                parent_run = mlflow.tracking.MlflowClient().get_run(run_id)
    
    try:
        mlflow.log_params({"use_smote": use_smote, "class_weight": class_weight, "random_state": random_state})
        
        X_train, X_test, y_train, y_test = load_prepared_data(data_dir)
        if use_smote:
            X_train, y_train = apply_smote(X_train, y_train, random_state)
            class_weight = None
        
        results = {}
        for model_name in models:
            try:
                model_hps = (hyperparams_by_model or {}).get(model_name, {})
                result = train_model(model_name, X_train, X_test, y_train, y_test, class_weight, random_state, model_hps)
                results[model_name] = result
            except Exception as e:
                raise RuntimeError(f"Error training {model_name}: {e}") from e
        
        if not results:
            raise RuntimeError("No models were successfully trained. Check earlier errors/logs.")
        
        # Log metrics based on mode
        best_model_name = list(results.keys())[0] if len(results) == 1 else max(results, key=lambda m: results[m]['test_metrics']['f1'])
        best_result = results[best_model_name]
        best_metrics = best_result['test_metrics']
        best_run_id = best_result['run_id']
        
        if len(results) == 1:
            # HPO mode: single model
            mlflow.log_metric(f"{best_model_name}_f1", best_metrics['f1'])
            mlflow.log_metric(f"{best_model_name}_roc_auc", best_metrics['roc_auc'])
            mlflow.log_metric("f1", best_metrics['f1'])
            mlflow.log_metric("roc_auc", best_metrics['roc_auc'])
            mlflow.set_tag("model_type", best_model_name)
        else:
            # Regular mode: multiple models
            mlflow.log_metric("best_model_f1", best_metrics['f1'])
            mlflow.log_metric("best_model_roc_auc", best_metrics['roc_auc'])
            mlflow.set_tag("best_model", best_model_name)
            mlflow.set_tag("best_model_run_id", best_run_id)
        
        # Write parent run ID if requested
        if parent_run_id_output:
            run_id_path = Path(parent_run_id_output)
            run_id_path.parent.mkdir(parents=True, exist_ok=True)
            run_id = _get_run_id(parent_run) if parent_run else os.getenv("MLFLOW_RUN_ID", "unknown")
            run_id_path.write_text(run_id)
        
        # Save model artifact if requested
        if model_artifact_dir:
            output_dir = Path(model_artifact_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if is_azure_ml:
                outputs_dir = os.getenv("AZUREML_ARTIFACTS_DIRECTORY", os.getenv("AZUREML_OUTPUT_DIRECTORY", "/tmp"))
                source_file = Path(outputs_dir) / f"{best_model_name}_model.pkl"
                dest_file = output_dir / f"{best_model_name}_model.pkl"
                if source_file.exists():
                    shutil.copy2(source_file, dest_file)
                else:
                    raise FileNotFoundError(f"Model file not found: {source_file}")
            else:
                model_uri = f"runs:/{best_run_id}/{best_result['artifact_path']}"
                best_model = mlflow.sklearn.load_model(model_uri)
                joblib.dump(best_model, output_dir / f"{best_model_name}_model.pkl")
            
            # Save metadata
            parent_run_id = _get_run_id(parent_run) if parent_run else os.getenv("MLFLOW_RUN_ID", "unknown")
            metadata = {
                "parent_run_id": parent_run_id,
                "best_model": best_model_name,
                "best_model_run_id": best_run_id,
                "artifact_path": best_result['artifact_path'],
                "metrics": best_metrics,
            }
            with open(output_dir / "model_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
        
        return results
    finally:
        if started_run:
            mlflow.end_run()


def parse_override_value(value: str) -> Any:
    """Parse scalar override values."""
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        lowered = value.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        if lowered == "none":
            return None
        return value


def load_hyperparams_from_json(json_path: str, model_type: str | None = None) -> Dict[str, JSONDict]:
    """Load hyperparameters from JSON file, optionally filtered by model_type."""
    json_file = Path(json_path)
    if not json_file.exists():
        raise FileNotFoundError(f"Hyperparameters JSON file not found: {json_path}")
    
    with open(json_file, "r") as f:
        params_dict = json.load(f)
    
    nested_params: Dict[str, JSONDict] = {}
    for key, value in params_dict.items():
        if model_type and not key.startswith(f"{model_type}_"):
            continue
        
        if key.startswith("rf_"):
            nested_params.setdefault("rf", {})[key[3:]] = value
        elif key.startswith("logreg_"):
            nested_params.setdefault("logreg", {})[key[7:]] = value
        elif key.startswith("xgboost_"):
            nested_params.setdefault("xgboost", {})[key[8:]] = value
    
    return nested_params


def apply_param_overrides(overrides: list[str], hyperparams: Dict[str, JSONDict]) -> Dict[str, JSONDict]:
    """Apply CLI overrides to hyperparameter dictionary."""
    if not overrides:
        return hyperparams or {}
    
    updated: Dict[str, JSONDict] = {model: dict(params) for model, params in (hyperparams or {}).items()}
    for assignment in overrides:
        try:
            model_part, value_part = assignment.split("=", 1)
            model_name, param_name = model_part.split(".", 1)
            value = parse_override_value(value_part)
            updated.setdefault(model_name, {})[param_name] = value
        except ValueError as exc:
            raise ValueError(f"Invalid override format '{assignment}'. Use model.param=value") from exc
    
    return updated


def is_hpo_mode(model_type: Optional[str] = None, hyperparams_json: Optional[str] = None) -> bool:
    """Check if running in HPO mode."""
    return model_type is not None or hyperparams_json is not None


def prepare_hpo_hyperparams(hyperparams_json: str, model_type: Optional[str]) -> Dict[str, JSONDict]:
    """Prepare hyperparameters for HPO mode (filtered by model_type)."""
    hyperparams_by_model = load_hyperparams_from_json(hyperparams_json, model_type)
    if model_type:
        return {model_type: hyperparams_by_model.get(model_type, {})}
    return hyperparams_by_model


def prepare_regular_hyperparams(training_config: Dict[str, Any], param_overrides: list[str]) -> Dict[str, JSONDict]:
    """Prepare hyperparameters for regular mode (from config + CLI overrides)."""
    hyperparams_by_model = get_config_value(training_config, 'hyperparameters', {})
    if param_overrides:
        hyperparams_by_model = apply_param_overrides(param_overrides, hyperparams_by_model)
    return hyperparams_by_model


def determine_models_to_train(
    is_hpo: bool,
    model_type: Optional[str],
    models_arg: Optional[list[str]],
    training_config: Dict[str, Any]
) -> list[str]:
    """Determine which models to train based on mode."""
    if is_hpo and model_type:
        return [model_type]
    return models_arg or get_config_value(training_config, 'models', ['logreg', 'rf'])


def main() -> None:
    """CLI entry-point."""
    parser = argparse.ArgumentParser(
        description='Train churn prediction models.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data", type=str, default=None, help="Directory with preprocessed data")
    parser.add_argument("--config", type=str, default=None, help=f"Config file (default: {DEFAULT_CONFIG})")
    parser.add_argument(
        "--models", type=str, nargs='+', default=None, choices=['logreg', 'rf', 'xgboost'],
        help='Models to train. Overrides config.'
    )
    parser.add_argument(
        "--model-type", type=str, default=None, choices=['logreg', 'rf', 'xgboost'],
        help='Single model type for HPO (overrides --models)'
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
    parser.add_argument(
        "--hyperparams-json", type=str, default=None,
        help="JSON file with hyperparameters (overrides config and --set)"
    )
    args = parser.parse_args()
    
    config_path = Path(args.config or DEFAULT_CONFIG)
    config = load_config(str(config_path)) if config_path.exists() else {}
    mlflow_config = load_config(str(config_path.parent / "mlflow.yaml")) if (config_path.parent / "mlflow.yaml").exists() else {}
    
    training_config = get_config_value(config, 'training', {})
    mlflow_config = get_config_value(mlflow_config, 'mlflow', {})
    
    hpo_mode = is_hpo_mode(args.model_type, args.hyperparams_json)
    
    if hpo_mode and args.hyperparams_json:
        hyperparams_by_model = prepare_hpo_hyperparams(args.hyperparams_json, args.model_type)
    else:
        hyperparams_by_model = prepare_regular_hyperparams(training_config, args.set)
    
    models_to_train = determine_models_to_train(hpo_mode, args.model_type, args.models, training_config)
    
    train_all_models(
        data_dir=args.data or 'data/processed',
        models=models_to_train,
        class_weight=args.class_weight or get_config_value(training_config, 'class_weight', 'balanced'),
        random_state=args.random_state or get_config_value(training_config, 'random_state', 42),
        experiment_name=args.experiment_name or get_config_value(mlflow_config, 'experiment_name', 'churn-prediction'),
        use_smote=args.use_smote or get_config_value(training_config, 'use_smote', False),
        hyperparams_by_model=hyperparams_by_model,
        model_artifact_dir=args.model_artifact_dir,
        parent_run_id_output=args.parent_run_id_output,
    )


if __name__ == '__main__':
    main()
