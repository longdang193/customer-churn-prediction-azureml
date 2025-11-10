#!/usr/bin/env python3
"""Model Training Script for Bank Customer Churn Prediction."""

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

sys.path.insert(0, str(Path(__file__).parent))
from config_loader import load_config, get_config_value
from models import get_logistic_regression, get_random_forest, get_xgboost

warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False


def load_prepared_data(data_dir: str) -> tuple:
    """Load preprocessed training and test data."""
    data_path = Path(data_dir)
    X_train = pd.read_csv(data_path / 'X_train.csv')
    X_test = pd.read_csv(data_path / 'X_test.csv')
    y_train = pd.read_csv(data_path / 'y_train.csv').squeeze()
    y_test = pd.read_csv(data_path / 'y_test.csv').squeeze()
    
    print(f"Loaded data: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    return X_train, X_test, y_train, y_test


def get_model(model_name: str, class_weight: str = 'balanced', random_state: int = 42) -> Any:
    """Get model instance by name from models package."""
    if model_name == 'logreg':
        return get_logistic_regression(class_weight=class_weight, random_state=random_state)
    elif model_name == 'rf':
        return get_random_forest(class_weight=class_weight, random_state=random_state)
    elif model_name == 'xgboost':
        return get_xgboost(class_weight=class_weight, random_state=random_state)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from: logreg, rf, xgboost")


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive evaluation metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_pred_proba)
    }
    return metrics


def train_model(
    model_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: str,
    class_weight: str = 'balanced',
    random_state: int = 42,
    use_mlflow: bool = False
) -> Dict[str, Any]:
    """Train a single model and evaluate it."""
    print(f"\n{'='*70}\nTRAINING: {model_name.upper()}\n{'='*70}")
    
    model = get_model(model_name, class_weight=class_weight, random_state=random_state)
    
    if use_mlflow and MLFLOW_AVAILABLE:
        mlflow.start_run(run_name=model_name)
        if hasattr(model, 'get_params'):
            mlflow.log_params(model.get_params())
    
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    train_metrics = calculate_metrics(y_train, y_train_pred, y_train_proba)
    test_metrics = calculate_metrics(y_test, y_test_pred, y_test_proba)
    
    cm = confusion_matrix(y_test, y_test_pred)
    
    print(f"\nTest Metrics: Acc={test_metrics['accuracy']:.3f} "
          f"F1={test_metrics['f1']:.3f} AUC={test_metrics['roc_auc']:.3f}")
    print(classification_report(y_test, y_test_pred, target_names=['Retained', 'Churned']))
    
    feature_importance = None
    if hasattr(model, 'feature_importances_'):
        feature_importance = dict(zip(X_train.columns, model.feature_importances_))
    elif hasattr(model, 'coef_'):
        feature_importance = dict(zip(X_train.columns, np.abs(model.coef_[0])))
    
    if use_mlflow and MLFLOW_AVAILABLE:
        for metric, value in test_metrics.items():
            mlflow.log_metric(f"test_{metric}", value)
        for metric, value in train_metrics.items():
            mlflow.log_metric(f"train_{metric}", value)
        if feature_importance:
            for feat, imp in feature_importance.items():
                mlflow.log_metric(f"feature_{feat}", float(imp))
        mlflow.sklearn.log_model(model, f"model_{model_name}")
        mlflow.end_run()
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model_file = output_path / f'{model_name}_model.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    
    metrics_data = {
        'model_name': model_name,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'confusion_matrix': cm.tolist(),
        'feature_importance': feature_importance
    }
    
    with open(output_path / f'{model_name}_metrics.json', 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    print(f"✓ Saved: {model_file}")
    
    return {
        'model': model,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'feature_importance': feature_importance
    }


def apply_smote(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42) -> tuple:
    """Apply SMOTE to balance training data."""
    if not SMOTE_AVAILABLE:
        raise ImportError("SMOTE not available. Install with: pip install imbalanced-learn")
    
    print("\nApplying SMOTE to training data...")
    original_counts = y_train.value_counts().to_dict()
    print(f"  Before SMOTE: {original_counts} (total: {len(y_train)})")
    
    smote = SMOTE(random_state=random_state)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    balanced_counts = pd.Series(y_train_balanced).value_counts().to_dict()
    print(f"  After SMOTE:  {balanced_counts} (total: {len(y_train_balanced)})")
    print(f"  ✓ Generated {len(y_train_balanced) - len(y_train)} synthetic samples")
    
    return pd.DataFrame(X_train_balanced, columns=X_train.columns), pd.Series(y_train_balanced)


def train_all_models(
    data_dir: str,
    output_dir: str,
    models: list = None,
    class_weight: str = 'balanced',
    random_state: int = 42,
    use_mlflow: bool = False,
    experiment_name: str = 'churn-prediction',
    use_smote: bool = False
) -> Dict[str, Dict]:
    """Train multiple models and compare results."""
    print(f"{'='*70}\nBANK CHURN PREDICTION - MODEL TRAINING\n{'='*70}")
    
    if models is None:
        models = ['logreg', 'rf']
        if XGBOOST_AVAILABLE:
            models.append('xgboost')
    
    if use_mlflow and MLFLOW_AVAILABLE:
        mlflow.set_experiment(experiment_name)
        print(f"MLflow experiment: {experiment_name}")
    
    X_train, X_test, y_train, y_test = load_prepared_data(data_dir)
    
    if use_smote:
        X_train, y_train = apply_smote(X_train, y_train, random_state)
        class_weight = None  # Disable class_weight when using SMOTE
    
    results = {}
    for model_name in models:
        try:
            result = train_model(
                model_name, X_train, X_test, y_train, y_test,
                output_dir, class_weight, random_state, use_mlflow
            )
            results[model_name] = result
        except Exception as e:
            print(f"✗ Error training {model_name}: {e}")
            continue
    
    print(f"\n{'='*70}\nMODEL COMPARISON\n{'-'*70}")
    print(f"{'Model':<12} | {'Acc':<6} | {'Prec':<6} | {'Rec':<6} | {'F1':<6} | {'AUC':<6}")
    print('-'*70)
    for model_name, result in results.items():
        m = result['test_metrics']
        print(f"{model_name:<12} | {m['accuracy']:.3f}  | {m['precision']:.3f}  | "
              f"{m['recall']:.3f}  | {m['f1']:.3f}  | {m['roc_auc']:.3f}")
    
    best_model = max(results.items(), key=lambda x: x[1]['test_metrics']['f1'])
    print(f"\n{'='*70}\n✓ BEST: {best_model[0].upper()} "
          f"(F1={best_model[1]['test_metrics']['f1']:.3f})\n{'='*70}")
    
    comparison_data = {
        model_name: {
            'test_metrics': result['test_metrics'],
            'train_metrics': result['train_metrics']
        }
        for model_name, result in results.items()
    }
    comparison_data['best_model'] = best_model[0]
    
    with open(Path(output_dir) / 'model_comparison.json', 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    return results


def main():
    """Main function with CLI."""
    parser = argparse.ArgumentParser(
        description='Train bank churn prediction models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--data',
        type=str,
        default=None,
        help='Directory with preprocessed data'
    )
    
    parser.add_argument(
        '--out',
        type=str,
        default=None,
        help='Directory to save trained models'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file (default: configs/train.yaml)'
    )
    
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=None,
        choices=['logreg', 'rf', 'xgboost'],
        help='Models to train (overrides config)'
    )
    
    parser.add_argument(
        '--class-weight',
        type=str,
        default=None,
        help='Class weight strategy (overrides config)'
    )
    
    parser.add_argument(
        '--random-state',
        type=int,
        default=None,
        help='Random seed (overrides config)'
    )
    
    parser.add_argument(
        '--use-mlflow',
        action='store_true',
        help='Enable MLflow tracking (overrides config)'
    )
    
    parser.add_argument(
        '--experiment-name',
        type=str,
        default=None,
        help='MLflow experiment name (overrides config)'
    )
    
    parser.add_argument(
        '--use-smote',
        action='store_true',
        help='Apply SMOTE (overrides config)'
    )
    
    args = parser.parse_args()
    
    # Determine config path (relative to project root)
    if args.config is None:
        project_root = Path(__file__).parent.parent
        config_path = project_root / 'configs' / 'train.yaml'
    else:
        config_path = Path(args.config)
    
    # Load configuration from YAML
    config = {}
    if config_path.exists():
        config = load_config(str(config_path))
        print(f"Loaded configuration from: {config_path}")
    else:
        print(f"Config file not found: {config_path}, using defaults")
    
    # Get values from config or CLI args (CLI takes precedence)
    training_config = get_config_value(config, 'training', {})
    output_config = get_config_value(config, 'output', {})
    mlflow_config = get_config_value(config, 'mlflow', {})
    
    data_dir = args.data or 'data/processed'
    output_dir = args.out or get_config_value(output_config, 'model_dir', 'models/local')
    models = args.models or get_config_value(training_config, 'models', ['logreg', 'rf'])
    class_weight = args.class_weight or get_config_value(training_config, 'class_weight', 'balanced')
    random_state = args.random_state or get_config_value(training_config, 'random_state', 42)
    use_smote = args.use_smote or get_config_value(training_config, 'use_smote', False)
    
    # MLflow settings
    if args.use_mlflow:
        use_mlflow = True
    else:
        use_mlflow = get_config_value(mlflow_config, 'enabled', False)
    
    experiment_name = args.experiment_name or get_config_value(mlflow_config, 'experiment_name', 'churn-prediction')
    
    train_all_models(
        data_dir=data_dir,
        output_dir=output_dir,
        models=models,
        class_weight=class_weight,
        random_state=random_state,
        use_mlflow=use_mlflow,
        experiment_name=experiment_name,
        use_smote=use_smote
    )


if __name__ == '__main__':
    main()

