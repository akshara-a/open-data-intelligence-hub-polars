"""
Evaluation module for Customer Feedback Analysis System.

Provides reusable functions for computing metrics, confusion matrices,
classification reports, and error analysis across all project stages.
"""

from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)


def evaluate_single_label(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    task_name: str = "Classification",
    plot: bool = True,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate a single-label classification task.

    Prints accuracy, precision, recall, F1-score, and optionally
    displays a confusion matrix heatmap.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted labels.
    labels : list of str, optional
        Class names for the confusion matrix axes.
    task_name : str
        Name used in print output (e.g. "Sentiment").
    plot : bool
        Whether to display a confusion matrix plot.
    save_path : str, optional
        If given, the confusion matrix figure is saved to this path
        instead of being shown (useful for CLI scripts).

    Returns
    -------
    dict
        Dictionary containing accuracy, precision, recall, f1, and
        the full classification report string.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    report = classification_report(y_true, y_pred, target_names=labels, zero_division=0)

    print(f"\n{'=' * 50}")
    print(f"  {task_name} Evaluation")
    print(f"{'=' * 50}")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"\n{report}")

    if plot and labels is not None:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels,
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"{task_name} - Confusion Matrix")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=120)
            print(f"  Confusion matrix saved to: {save_path}")
        else:
            plt.show()

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "report": report,
    }


def evaluate_multilabel(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    task_name: str = "Multi-Label Classification",
) -> Dict[str, Any]:
    """
    Evaluate a multi-label classification task.

    Reports micro and macro F1 scores, hamming loss, and a
    per-class classification report.
    """
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    hamming = hamming_loss(y_true, y_pred)
    samples_f1 = f1_score(y_true, y_pred, average="samples", zero_division=0)

    report = classification_report(
        y_true, y_pred, target_names=labels, zero_division=0
    )

    print(f"\n{'=' * 50}")
    print(f"  {task_name} Evaluation")
    print(f"{'=' * 50}")
    print(f"  Micro F1:     {micro_f1:.4f}")
    print(f"  Macro F1:     {macro_f1:.4f}")
    print(f"  Weighted F1:  {weighted_f1:.4f}")
    print(f"  Samples F1:   {samples_f1:.4f}")
    print(f"  Hamming Loss: {hamming:.4f}")
    print(f"\n{report}")

    return {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "samples_f1": samples_f1,
        "hamming_loss": hamming,
        "report": report,
    }


def error_analysis(
    texts: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    max_examples: int = 10,
) -> pd.DataFrame:
    """
    Show examples where the model's prediction differs from the ground truth.

    This helps identify patterns in errors such as:
    - Misclassified sentiment (sarcasm, negation, ambiguity)
    - Overlapping categories
    - Short or ambiguous feedback

    Parameters
    ----------
    texts : array-like
        Original text samples.
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted labels.
    labels : list of str, optional
        Human-readable label names.
    max_examples : int
        Maximum number of error examples to display.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: text, actual, predicted.
    """
    mask = y_true != y_pred
    error_texts = texts[mask]
    error_true = y_true[mask]
    error_pred = y_pred[mask]

    n_errors = mask.sum()
    print(f"\n{'=' * 50}")
    print(f"  Error Analysis")
    print(f"{'=' * 50}")
    print(f"  Total errors: {n_errors} out of {len(y_true)} samples "
          f"({n_errors / len(y_true) * 100:.1f}%)")

    if labels is not None:
        # Map indices back to labels for readability
        error_true_labels = [labels[i] for i in error_true]
        error_pred_labels = [labels[i] for i in error_pred]
    else:
        error_true_labels = list(error_true)
        error_pred_labels = list(error_pred)

    df_errors = pd.DataFrame({
        "text": error_texts,
        "actual": error_true_labels,
        "predicted": error_pred_labels,
    })

    n_show = min(max_examples, len(df_errors))
    print(f"\n  Showing {n_show} misclassified examples:\n")
    for i in range(n_show):
        row = df_errors.iloc[i]
        print(f"  [{i+1}] \"{row['text'][:100]}...\"")
        print(f"      Actual: {row['actual']}  |  Predicted: {row['predicted']}")
        print()

    return df_errors


def compare_models(
    results_dict: Dict[str, Dict[str, float]],
    metrics: Optional[List[str]] = None,
) -> None:
    """
    Display a side-by-side comparison table of multiple models.

    Parameters
    ----------
    results_dict : dict
        Mapping from model name -> {metric: value}.
    metrics : list of str, optional
        Which metrics to display. Defaults to accuracy, precision, recall, f1.
    """
    if metrics is None:
        metrics = ["accuracy", "precision", "recall", "f1"]

    print(f"\n{'=' * 60}")
    print("  Model Comparison")
    print(f"{'=' * 60}")

    # Header
    header = f"  {'Metric':<15}"
    for name in results_dict:
        header += f"{name:>15}"
    print(header)
    print("  " + "-" * (15 + 15 * len(results_dict)))

    # Rows
    for metric in metrics:
        row = f"  {metric:<15}"
        for name, res in results_dict.items():
            val = res.get(metric, float("nan"))
            row += f"{val:>15.4f}"
        print(row)
    print()
