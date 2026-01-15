"""Evaluation metrics and visualization for multiclass classification."""

from typing import Dict, List, Any, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import matplotlib
matplotlib.use('Agg')
from .utils.display_utils import print_file_saved
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, 
                   class_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """Compute comprehensive evaluation metrics."""
    # Get unique labels present in both y_true and y_pred
    all_labels = np.unique(np.concatenate([y_true, y_pred]))
    
    # Filter class_names to only include labels present in the data
    if class_names is not None:
        # Create filtered class names based on labels actually present
        filtered_class_names = [class_names[i] for i in all_labels if i < len(class_names)]
        labels_param = all_labels[all_labels < len(class_names)]
    else:
        filtered_class_names = None
        labels_param = None
    
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=labels_param),
        'classification_report_str': classification_report(
            y_true, y_pred, target_names=filtered_class_names, labels=labels_param, zero_division=0
        ),
        'classification_report_dict': classification_report(
            y_true, y_pred, target_names=filtered_class_names, labels=labels_param, output_dict=True, zero_division=0
        ),
        'class_names': filtered_class_names
    }


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], 
                         save_path: Optional[str] = None, figsize: tuple = (14, 12),
                         title: str = 'Confusion Matrix') -> None:
    """Plot confusion matrix heatmap."""
    plt.figure(figsize=figsize)
    
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                square=True, cbar_kws={'shrink': 0.8})
    
    plt.title(title, fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print_file_saved(save_path, "Confusion matrix")
    
    plt.close()


def plot_class_distribution(labels: np.ndarray, class_names: List[str],
                           save_path: Optional[str] = None,
                           title: str = 'Class Distribution') -> None:
    """Plot class distribution bar chart."""
    plt.figure(figsize=(12, 6))
    
    unique, counts = np.unique(labels, return_counts=True)
    
    bars = plt.bar(range(len(unique)), counts, color='steelblue', edgecolor='black')
    
    plt.title(title, fontsize=14)
    plt.xlabel('Category', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(range(len(unique)), [class_names[i] for i in unique], 
               rotation=45, ha='right', fontsize=8)
    
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print_file_saved(save_path, "Class distribution")
    
    plt.close()


def print_evaluation_summary(results: Dict[str, Any]) -> None:
    """Print formatted evaluation summary."""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print("\nOverall Metrics:")
    print("-"*40)
    print(f"Accuracy:             {results['accuracy']:.4f}")
    print(f"Precision (weighted): {results['precision_weighted']:.4f}")
    print(f"Recall (weighted):    {results['recall_weighted']:.4f}")
    print(f"F1-Score (weighted):  {results['f1_weighted']:.4f}")
    
    print("\nMacro-Averaged Metrics:")
    print("-"*40)
    print(f"Precision (macro):    {results['precision_macro']:.4f}")
    print(f"Recall (macro):       {results['recall_macro']:.4f}")
    print(f"F1-Score (macro):     {results['f1_macro']:.4f}")
    
    print("\nPer-Class Report:")
    print("-"*40)
    print(results['classification_report_str'])
