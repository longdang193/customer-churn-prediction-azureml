#!/usr/bin/env python3
"""Training entry-point for the churn prediction models."""

from __future__ import annotations

import argparse
import ast
import json
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

import mlflow
import mlflow.sklearn

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

JSONDict = Dict[str, Any]
DEFAULT_CONFIG = Path(__file__).parents[1] / "configs" / "train.yaml"


def load_prepared_data(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load preprocessed training and test data."""
    data_path = Path(data_dir)
    X_train = pd.read_csv(data_path / 'X_train.csv')
    X_test = pd.read_csv(data_path / 'X_test.csv')
    y_train = pd.read_csv(data_path / 'y_train.csv').squeeze()
    y_test = pd.read_csv(data_path / 'y_test.csv').squeeze()
    print(f"Loaded data: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    return X_train, X_test, y_train, y_test

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
    """Apply the provided hyperparameters via ``set_params`` where supported."""
    applied: JSONDict = {}
    if hyperparams:
        try:
            valid_params = model.get_params(deep=True)
            applied = {k: v for k, v in hyperparams.items() if k in valid_params}
            if applied:
                model.set_params(**applied)
        except Exception:
            applied = {}
    return model, applied


def predict_probabilities(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Return positive class probabilities with sensible fallbacks."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        proba = np.asarray(proba, dtype=float)
        if proba.ndim == 1:
            return proba
        if proba.shape[1] >= 2:
            return proba[:, 1]
        return proba.ravel()

    if hasattr(model, "decision_function"):
        decision = model.decision_function(X)
        decision = np.asarray(decision, dtype=float)
        if decision.ndim > 1:
            decision = decision[:, 0]
        min_val = np.min(decision)
        max_val = np.max(decision)
        if np.isclose(max_val, min_val):
            return np.full(decision.shape, 0.5, dtype=float)
        return (decision - min_val) / (max_val - min_val)

    return np.full(shape=(len(X),), fill_value=0.5, dtype=float)


def maybe_apply_class_weight_adjustments(
    model_name: str,
    model: Any,
    y_train: pd.Series,
    class_weight: Optional[str],
    tuned_params: JSONDict,
) -> JSONDict:
    """Adjust model-specific weighting parameters when class weights are requested."""
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
    except Exception:
        return {}

    return {"scale_pos_weight": scale_pos_weight}

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> JSONDict:
    """Return a dictionary of core evaluation metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_pred_proba),
    }

def train_model(
    model_name: str, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series,
    class_weight: Optional[str] = "balanced", random_state: int = 42,
    model_hyperparams: JSONDict | None = None
) -> JSONDict:
    """Train and evaluate a single estimator, logging to a nested MLflow run."""
    with mlflow.start_run(run_name=model_name, nested=True) as nested_run:
        print(f"\n{'=' * 70}\nTRAINING: {model_name.upper()} (Run ID: {nested_run.info.run_id})\n{'=' * 70}")

        model = get_model(model_name, class_weight=class_weight, random_state=random_state)
        model, tuned_params = apply_hyperparameters(model, model_hyperparams)

        extra_params = maybe_apply_class_weight_adjustments(
            model_name=model_name,
            model=model,
            y_train=y_train,
            class_weight=class_weight,
            tuned_params=tuned_params,
        )

        params_to_log: JSONDict = {}
        if tuned_params:
            params_to_log.update(tuned_params)
        if extra_params:
            params_to_log.update(extra_params)
        if params_to_log:
            mlflow.log_params(params_to_log)

        model.fit(X_train, y_train)

        y_test_pred = model.predict(X_test)
        y_test_proba = predict_probabilities(model, X_test)
        test_metrics = calculate_metrics(y_test, y_test_pred, y_test_proba)

        print(f"\nTest Metrics: Acc={test_metrics['accuracy']:.3f}, F1={test_metrics['f1']:.3f}, AUC={test_metrics['roc_auc']:.3f}")


        for metric, value in test_metrics.items():
            mlflow.log_metric(f"test_{metric}", value)
        
        mlflow.sklearn.log_model(model, f"model_{model_name}")

        return {
            'test_metrics': test_metrics,
            'run_id': nested_run.info.run_id,
            'artifact_path': f"model_{model_name}",
        }

def apply_smote(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """Apply SMOTE to balance the training labels."""
    if not SMOTE_AVAILABLE:
        raise ImportError("SMOTE not available. Install with: pip install imbalanced-learn")

    print("\nApplying SMOTE to training data...")
    smote = SMOTE(random_state=random_state)
    X_bal, y_bal = smote.fit_resample(X_train, y_train)
    print(f"  Before: {y_train.value_counts().to_dict()}")
    print(f"  After : {pd.Series(y_bal).value_counts().to_dict()}")
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
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="Churn_Training_Pipeline") as parent_run:
        print(f"Parent run ID: {parent_run.info.run_id}")
        mlflow.log_params({"use_smote": use_smote, "class_weight": class_weight, "random_state": random_state})

        X_train, X_test, y_train, y_test = load_prepared_data(data_dir)
        if use_smote:
            X_train, y_train = apply_smote(X_train, y_train, random_state)
            class_weight = None

        results = {}
        for model_name in models:
            try:
                model_hps = (hyperparams_by_model or {}).get(model_name, {})
                result = train_model(
                    model_name, X_train, X_test, y_train, y_test, 
                    class_weight, random_state, model_hyperparams=model_hps
                )
                results[model_name] = result
            except Exception as e:
                print(f"✗ Error training {model_name}: {e}")

        if not results:
            raise RuntimeError("No models were successfully trained. Check earlier errors/logs.")

        best_model_name = max(results, key=lambda m: results[m]['test_metrics']['f1'])
        best_result = results[best_model_name]
        best_metrics = best_result['test_metrics']
        best_run_id = best_result['run_id']
        mlflow.log_metric("best_model_f1", best_metrics['f1'])
        mlflow.log_metric("best_model_roc_auc", best_metrics['roc_auc'])
        mlflow.set_tag("best_model", best_model_name)
        mlflow.set_tag("best_model_run_id", best_run_id)

        if parent_run_id_output:
            run_id_path = Path(parent_run_id_output)
            run_id_path.parent.mkdir(parents=True, exist_ok=True)
            run_id_path.write_text(parent_run.info.run_id)

        if model_artifact_dir:
            output_dir = Path(model_artifact_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            model_uri = f"runs:/{best_run_id}/{best_result['artifact_path']}"
            best_model = mlflow.sklearn.load_model(model_uri)
            model_filename = f"{best_model_name}_model.pkl"
            joblib.dump(best_model, output_dir / model_filename)
            metadata = {
                "parent_run_id": parent_run.info.run_id,
                "best_model": best_model_name,
                "best_model_run_id": best_run_id,
                "artifact_path": best_result['artifact_path'],
                "metrics": best_metrics,
            }
            with open(output_dir / "model_metadata.json", "w") as meta_file:
                json.dump(metadata, meta_file, indent=2)
        
        print(f"\n{'=' * 70}\n✓ BEST MODEL: {best_model_name.upper()} (F1={best_metrics['f1']:.3f})\n{'=' * 70}")
        return results

def parse_override_value(value: str) -> Any:
    """Best-effort parsing of scalar override values."""
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        lowered = value.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        if lowered == "none":
            return None
        return value


def apply_param_overrides(overrides: list[str], hyperparams: Dict[str, JSONDict]) -> Dict[str, JSONDict]:
    """Apply CLI overrides to the hyperparameter dictionary."""
    if not overrides:
        return hyperparams or {}

    base_params = hyperparams or {}
    updated: Dict[str, JSONDict] = {model: dict(params) for model, params in base_params.items()}
    for assignment in overrides:
        try:
            model_part, value_part = assignment.split("=", 1)
            model_name, param_name = model_part.split(".", 1)
        except ValueError as exc:
            raise ValueError(f"Invalid override format '{assignment}'. Use model.param=value") from exc

        value = parse_override_value(value_part)
        updated.setdefault(model_name, {})[param_name] = value
    return updated

def main() -> None:
    """CLI entry-point."""
    parser = argparse.ArgumentParser(description='Train churn prediction models.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data", type=str, default=None, help="Directory with preprocessed data")
    parser.add_argument("--config", type=str, default=None, help=f"Config file (default: {DEFAULT_CONFIG})")
    parser.add_argument("--models", type=str, nargs='+', default=None, choices=['logreg', 'rf', 'xgboost'], help='Models to train (overrides config)')
    parser.add_argument("--class-weight", type=str, default=None, help='Class weight strategy (overrides config)')
    parser.add_argument("--random-state", type=int, default=None, help='Random seed (overrides config)')
    parser.add_argument("--experiment-name", type=str, default=None, help='MLflow experiment name (overrides config)')
    parser.add_argument("--use-smote", action='store_true', help='Apply SMOTE (overrides config)')
    parser.add_argument("--model-artifact-dir", type=str, default=None, help='Optional directory to save the best model artifact')
    parser.add_argument("--parent-run-id-output", type=str, default=None, help='Optional file path to write the parent MLflow run ID')
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="model.param=value",
        help="Override hyperparameters (can be used multiple times)",
    )
    args = parser.parse_args()

    config_path = Path(args.config or DEFAULT_CONFIG)
    config = load_config(str(config_path)) if config_path.exists() else {}
    
    mlflow_config_path = config_path.parent / "mlflow.yaml"
    mlflow_config = load_config(str(mlflow_config_path)) if mlflow_config_path.exists() else {}

    training_config = get_config_value(config, 'training', {})
    mlflow_config = get_config_value(mlflow_config, 'mlflow', {})

    hyperparams_by_model = get_config_value(training_config, 'hyperparameters', {})
    hyperparams_by_model = apply_param_overrides(args.set, hyperparams_by_model)

    train_all_models(
        data_dir=args.data or 'data/processed',

        models=args.models or get_config_value(training_config, 'models', ['logreg', 'rf']),
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
