"""Core training logic for model training and evaluation."""

# Import azureml.mlflow before mlflow to register Azure ML tracking store
import azureml.mlflow  # noqa: F401

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
 
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from data import apply_smote, load_prepared_data
from utils import (
    calculate_metrics,
    get_active_run,
    get_run_id,
    is_azure_ml,
    start_nested_run,
    start_parent_run,
)

from .model_utils import (
    apply_class_weight_adjustments,
    apply_hyperparameters,
    get_model,
)

JSONDict = Dict[str, Any]


def get_pip_requirements() -> list[str]:
    """Generate pip requirements for MLflow model deployment (local only).
    
    Returns:
        List of package names required for model deployment
    """
    return ["mlflow", "scikit-learn", "pandas", "numpy", "xgboost"]


def train_model(
    model_name: str,
    X_train: Any,
    X_test: Any,
    y_train: Any,
    y_test: Any,
    class_weight: Optional[str] = "balanced",
    random_state: int = 42,
    model_hyperparams: Optional[JSONDict] = None,
) -> JSONDict:
    """Train and evaluate a single model, logging to MLflow.
    
    Args:
        model_name: Model identifier ('logreg', 'rf', or 'xgboost')
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        class_weight: Class weight strategy
        random_state: Random seed
        model_hyperparams: Optional hyperparameters for the model
        
    Returns:
        Dictionary with test_metrics, run_id, and artifact_path
    """
    nested_run, run_id = start_nested_run(model_name)
    
    try:
        model = get_model(model_name, class_weight=class_weight, random_state=random_state)
        model, tuned_params = apply_hyperparameters(model, model_hyperparams)
        
        extra_params = apply_class_weight_adjustments(
            model_name, model, y_train, class_weight, tuned_params
        )
        
        # Log parameters
        params_to_log = {**tuned_params, **extra_params}
        if params_to_log:
            mlflow.log_params(params_to_log)
        
        # Train and evaluate
        model.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)[:, 1]
        test_metrics = calculate_metrics(y_test, y_test_pred, y_test_proba)
        
        for metric, value in test_metrics.items():
            mlflow.log_metric(f"test_{metric}", value)
        
        artifact_path = f"model_{model_name}"
        
        # Save model
        if is_azure_ml():
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
        if not is_azure_ml():
            mlflow.end_run()


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
    """Train a collection of models within a parent MLflow run.
    
    Args:
        data_dir: Directory containing preprocessed data
        models: Iterable of model names to train
        class_weight: Class weight strategy
        random_state: Random seed
        experiment_name: MLflow experiment name
        use_smote: Whether to apply SMOTE
        hyperparams_by_model: Optional dictionary mapping model names to hyperparameters
        model_artifact_dir: Optional directory to save best model
        parent_run_id_output: Optional file path to write parent run ID
        
    Returns:
        Dictionary mapping model names to their training results
        
    Raises:
        RuntimeError: If no models are successfully trained
    """
    is_azure = is_azure_ml()
    started_run = False
    parent_run = None
    
    if not is_azure:
        parent_run = start_parent_run(experiment_name)
        started_run = True
    else:
        parent_run = get_active_run()
    
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
            run_id = get_run_id(parent_run) if parent_run else os.getenv("MLFLOW_RUN_ID", "unknown")
            run_id_path.write_text(run_id)
        
        # Save model artifact if requested
        if model_artifact_dir:
            output_dir = Path(model_artifact_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if is_azure:
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
            parent_run_id = get_run_id(parent_run) if parent_run else os.getenv("MLFLOW_RUN_ID", "unknown")
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

