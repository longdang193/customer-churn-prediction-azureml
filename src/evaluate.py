#!/usr/bin/env python3
"""Evaluate a trained model using metrics and plots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

import mlflow
import mlflow.sklearn

sys.path.insert(0, str(Path(__file__).parent))
from config_loader import get_config_value, load_config

sns.set_style("whitegrid")

Metrics = Dict[str, float]
DEFAULT_CONFIG = Path(__file__).parents[1] / "configs" / "evaluate.yaml"


def load_artifacts_from_mlflow(
    run_id: str, data_dir: Path
) -> Tuple[Any, pd.DataFrame, pd.Series, str]:
    """Load model from MLflow run and test data from a specified path."""
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    best_model_name = run.data.tags.get("best_model", "rf")
    best_model_run_id = run.data.tags.get("best_model_run_id", run_id)
    
    model_uri = f"runs:/{best_model_run_id}/model_{best_model_name}"
    print(f"Loading model from: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)
    
    X_test = pd.read_csv(data_dir / "X_test.csv")
    y_test = pd.read_csv(data_dir / "y_test.csv").squeeze()
    return model, X_test, y_test, best_model_name


def predict_probabilities(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Return positive class probabilities with fallbacks for compatible estimators."""
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


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Metrics:
    """Return a dictionary of the primary evaluation metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "average_precision": average_precision_score(y_true, y_proba),
    }


def generate_visualizations(
    model: Any, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, 
    feature_names: List[str], output_path: Path, top_n_features: int
) -> None:
    """Create and save all evaluation plots."""
    print("\n[4/5] Generating visualizations...")
    plot_confusion_matrix(y_true, y_pred, output_path)
    plot_roc(y_true, y_proba, output_path)
    plot_precision_recall(y_true, y_proba, output_path)
    plot_prediction_distribution(y_proba, y_true, output_path)
    plot_feature_importance(model, feature_names, output_path, top_n_features)


def _save_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Retained", "Churned"], yticklabels=["Retained", "Churned"])
    plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.ylabel("Actual", fontsize=12)
    plt.xlabel("Predicted", fontsize=12)
    _save_plot(output / "confusion_matrix.png")
    print("  ✓ Confusion matrix saved")


def plot_roc(y_true: np.ndarray, y_proba: np.ndarray, output: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
    plt.xlim([0, 1]); plt.ylim([0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right"); plt.grid(alpha=0.3)
    _save_plot(output / "roc_curve.png")
    print("  ✓ ROC curve saved")


def plot_precision_recall(y_true: np.ndarray, y_proba: np.ndarray, output: Path) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="darkorange", lw=2, label=f"PR curve (AP = {ap:.3f})")
    plt.xlabel("Recall", fontsize=12); plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    plt.legend(loc="upper right"); plt.grid(alpha=0.3)
    _save_plot(output / "precision_recall_curve.png")
    print("  ✓ Precision-Recall curve saved")


def plot_feature_importance(model: Any, feature_names: List[str], output: Path, top_n: int) -> None:
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        title = 'Feature Importance'
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
        title = 'Feature Importance (|Coefficient|)'
    else:
        print("  ⚠ Model has no feature importance")
        return
    
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(top_features)), top_importances, color="steelblue")
    ax.set_yticks(range(len(top_features)), top_features)
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    for idx, val in enumerate(top_importances):
        ax.text(val, idx, f" {val:.3f}", va="center", fontsize=10)
    _save_plot(output / "feature_importance.png")
    print("  ✓ Feature importance saved")


def plot_prediction_distribution(y_proba: np.ndarray, y_true: np.ndarray, output: Path) -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(y_proba[y_true == 0], bins=50, alpha=0.6, label="Retained", color="green", density=True)
    plt.hist(y_proba[y_true == 1], bins=50, alpha=0.6, label="Churned", color="red", density=True)
    plt.xlabel("Predicted Churn Probability", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title("Prediction Probability Distribution", fontsize=14, fontweight="bold")
    plt.legend(); plt.axvline(x=0.5, color="black", linestyle="--", label="Threshold"); plt.grid(alpha=0.3)
    _save_plot(output / "prediction_distribution.png")
    print("  ✓ Prediction distribution saved")


def evaluate_model(
    run_id: str, data_dir: str, output_dir: str, top_n_features: int = 10
) -> Dict[str, Any]:
    """Comprehensive model evaluation, loading the model from an MLflow run."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"{'='*70}\nMODEL EVALUATION (MLflow Run ID: {run_id})\n{'='*70}")

    print("\n[1/5] Loading model and data...")
    model, X_test, y_test, model_name = load_artifacts_from_mlflow(run_id, Path(data_dir))
    print(f"  Model: {type(model).__name__}, Test samples: {len(X_test)}")

    print("\n[2/5] Making predictions...")
    y_pred = model.predict(X_test)
    y_proba = predict_probabilities(model, X_test)

    print("\n[3/5] Calculating metrics...")
    metrics = calculate_metrics(y_test, y_pred, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, target_names=['Retained', 'Churned'], output_dict=True)

    print(f"\nTest Metrics: Acc={metrics['accuracy']:.3f}, P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}, AUC={metrics['roc_auc']:.3f}")
    print(f"Confusion Matrix: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")

    generate_visualizations(model, y_test, y_pred, y_proba, X_test.columns.tolist(), output_path, top_n_features)

    print("\n[5/5] Saving evaluation report...")
    report = {
        'model_name': model_name, 'model_type': type(model).__name__, 'test_samples': len(X_test),
        'metrics': metrics, 'confusion_matrix': cm.tolist(), 'classification_report': report_dict
    }
    with open(output_path / 'evaluation_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    with mlflow.start_run(run_id=run_id):
        print("\nLogging evaluation artifacts to MLflow...")
        mlflow.log_artifact(output_path / 'evaluation_report.json', artifact_path="evaluation")
        for plot_file in output_path.glob('*.png'):
            mlflow.log_artifact(plot_file, artifact_path="evaluation/plots")
        print("  ✓ Logged evaluation artifacts.")

    print(f"\n{'='*70}\n✓ EVALUATION COMPLETE\n{'='*70}")
    return report


def main() -> None:
    """CLI entry-point."""
    parser = argparse.ArgumentParser(description='Evaluate a model from an MLflow run.')
    parser.add_argument("--run-id", type=str, required=True, help="MLflow run ID to load the model from")
    parser.add_argument("--data", type=str, required=True, help="Directory with test data")
    parser.add_argument("--output", type=str, required=True, help="Directory to save evaluation results")
    parser.add_argument("--config", type=str, help=f"Config file (default: {DEFAULT_CONFIG})")
    parser.add_argument("--parent-run-id-file", type=str, help="Optional file containing the parent MLflow run ID")
    args = parser.parse_args()

    config_path = Path(args.config or DEFAULT_CONFIG)
    config = load_config(str(config_path)) if config_path.exists() else {}
    eval_config = get_config_value(config, 'evaluation', {})

    run_id = args.run_id
    if not run_id and args.parent_run_id_file:
        run_id = Path(args.parent_run_id_file).read_text().strip()
    if not run_id:
        parser.error("Either --run-id or --parent-run-id-file must be provided.")

    evaluate_model(
        run_id=run_id,
        data_dir=args.data,
        output_dir=args.output,
        top_n_features=get_config_value(eval_config, 'top_n_features', 10)
    )

if __name__ == '__main__':
    main()
