#!/usr/bin/env python3
"""Model Evaluation Script for Bank Customer Churn Prediction."""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report, precision_recall_curve, average_precision_score
)

sns.set_style('whitegrid')


def load_model(model_path: str) -> Any:
    """Load trained model from pickle file."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def load_test_data(data_dir: str) -> tuple:
    """Load test data."""
    data_path = Path(data_dir)
    X_test = pd.read_csv(data_path / 'X_test.csv')
    y_test = pd.read_csv(data_path / 'y_test.csv').squeeze()
    return X_test, y_test


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive evaluation metrics."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
        'average_precision': average_precision_score(y_true, y_pred_proba)
    }


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Retained', 'Churned'],
                yticklabels=['Retained', 'Churned'])
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Confusion matrix saved")


def plot_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray, output_path: Path) -> None:
    """Plot and save ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ ROC curve saved")


def plot_precision_recall_curve(y_true: np.ndarray, y_pred_proba: np.ndarray, output_path: Path) -> None:
    """Plot and save precision-recall curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    avg_precision = average_precision_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2,
             label=f'PR curve (AP = {avg_precision:.3f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'precision_recall_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Precision-Recall curve saved")


def plot_feature_importance(model: Any, feature_names: list, output_path: Path, top_n: int = 10) -> None:
    """Plot and save feature importance."""
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
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(range(len(top_features)), top_importances, color='steelblue')
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('Importance', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    
    for i, (bar, val) in enumerate(zip(bars, top_importances)):
        plt.text(val, i, f' {val:.3f}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path / 'feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Feature importance saved")


def plot_prediction_distribution(y_pred_proba: np.ndarray, y_true: np.ndarray, output_path: Path) -> None:
    """Plot distribution of prediction probabilities."""
    plt.figure(figsize=(10, 6))
    
    plt.hist(y_pred_proba[y_true == 0], bins=50, alpha=0.6, 
             label='Retained (Actual)', color='green', density=True)
    plt.hist(y_pred_proba[y_true == 1], bins=50, alpha=0.6, 
             label='Churned (Actual)', color='red', density=True)
    
    plt.xlabel('Predicted Churn Probability', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Prediction Probability Distribution', fontsize=14, fontweight='bold')
    plt.legend()
    plt.axvline(x=0.5, color='black', linestyle='--', label='Threshold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'prediction_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Prediction distribution saved")


def evaluate_model(
    model_path: str,
    data_dir: str,
    output_dir: str,
    model_name: str = None
) -> Dict[str, Any]:
    """Comprehensive model evaluation."""
    
    if model_name is None:
        model_name = Path(model_path).stem.replace('_model', '')
    
    print(f"{'='*70}\nMODEL EVALUATION: {model_name.upper()}\n{'='*70}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n[1/5] Loading model and data...")
    model = load_model(model_path)
    X_test, y_test = load_test_data(data_dir)
    print(f"  Model: {type(model).__name__}")
    print(f"  Test samples: {len(X_test)}")
    
    print("\n[2/5] Making predictions...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print("\n[3/5] Calculating metrics...")
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    
    print(f"\nTest Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall:    {metrics['recall']:.3f}")
    print(f"  F1 Score:  {metrics['f1']:.3f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.3f}")
    print(f"  Avg Prec:  {metrics['average_precision']:.3f}")
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"  TN: {cm[0,0]:>6}  |  FP: {cm[0,1]:>6}")
    print(f"  FN: {cm[1,0]:>6}  |  TP: {cm[1,1]:>6}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Retained', 'Churned']))
    
    print("\n[4/5] Generating visualizations...")
    plot_confusion_matrix(y_test, y_pred, output_path)
    plot_roc_curve(y_test, y_pred_proba, output_path)
    plot_precision_recall_curve(y_test, y_pred_proba, output_path)
    plot_prediction_distribution(y_pred_proba, y_test, output_path)
    plot_feature_importance(model, X_test.columns.tolist(), output_path)
    
    print("\n[5/5] Saving evaluation report...")
    
    report = {
        'model_name': model_name,
        'model_type': type(model).__name__,
        'test_samples': len(X_test),
        'metrics': metrics,
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(y_test, y_pred, 
                                                       target_names=['Retained', 'Churned'],
                                                       output_dict=True)
    }
    
    with open(output_path / 'evaluation_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n{'='*70}\n✓ EVALUATION COMPLETE\n{'='*70}")
    print(f"Results saved to: {output_dir}")
    print(f"  - evaluation_report.json")
    print(f"  - confusion_matrix.png")
    print(f"  - roc_curve.png")
    print(f"  - precision_recall_curve.png")
    print(f"  - prediction_distribution.png")
    print(f"  - feature_importance.png")
    
    return report


def main():
    """Main function with CLI."""
    parser = argparse.ArgumentParser(
        description='Evaluate trained churn prediction model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model (.pkl file)'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Directory with preprocessed test data'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Directory to save evaluation results'
    )
    
    parser.add_argument(
        '--model-name',
        type=str,
        default=None,
        help='Model name for report (default: inferred from filename)'
    )
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model,
        data_dir=args.data,
        output_dir=args.output,
        model_name=args.model_name
    )


if __name__ == '__main__':
    main()

