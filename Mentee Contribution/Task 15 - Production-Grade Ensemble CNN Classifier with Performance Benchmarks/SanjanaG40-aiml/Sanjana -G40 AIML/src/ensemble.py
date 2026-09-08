"""
WeatherNet-05
Validation-Based Weighted CNN Ensemble

CNN1 and CNN2 are evaluated on the validation set to determine
the best probability-averaging weight.

The selected weight is then evaluated ONCE on the test set.

IMPORTANT:
The test set is never used to select ensemble weights.
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

import matplotlib.pyplot as plt


# ============================================================
# Project root
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# Imports
# ============================================================

from src.preprocessing import load_weathernet_datasets


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 32
NUM_CLASSES = 5

CNN1_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "cnn_baseline.keras"
)

CNN2_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "cnn2.keras"
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# Ensemble weights
# ============================================================

# CNN1 gets weights from 0.0 -> 1.0.
# CNN2 receives the remaining weight.
#
# Example:
# CNN1 = 0.8
# CNN2 = 0.2

WEIGHT_GRID = np.arange(
    0.0,
    1.01,
    0.05
)


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):

    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred
        ),

        "macro_precision": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "macro_recall": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "weighted_f1": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )
    }


# ============================================================
# Predict
# ============================================================

def generate_predictions(
    model,
    dataset,
    model_name
):

    print(
        f"\nGenerating predictions: {model_name}"
    )

    start = time.perf_counter()

    probabilities = model.predict(
        dataset,
        verbose=1
    )

    elapsed = time.perf_counter() - start

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    print(
        f"{model_name} inference time: "
        f"{elapsed:.3f} seconds"
    )

    print(
        f"{model_name} predictions: "
        f"{len(predictions)}"
    )

    return (
        probabilities,
        predictions,
        elapsed
    )


# ============================================================
# Extract labels
# ============================================================

def extract_labels(dataset):

    labels = []

    for _, batch_labels in dataset:

        labels.extend(
            batch_labels.numpy()
        )

    return np.asarray(
        labels,
        dtype=np.int32
    )


# ============================================================
# Weighted probability averaging
# ============================================================

def weighted_ensemble(
    cnn1_probabilities,
    cnn2_probabilities,
    cnn1_weight
):

    cnn2_weight = 1.0 - cnn1_weight

    probabilities = (
        cnn1_weight * cnn1_probabilities
        +
        cnn2_weight * cnn2_probabilities
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    return predictions


# ============================================================
# Validation weight search
# ============================================================

def search_best_weight(
    y_val,
    cnn1_probabilities,
    cnn2_probabilities
):

    print("\n")
    print("=" * 70)
    print("VALIDATION-BASED ENSEMBLE WEIGHT SEARCH")
    print("=" * 70)

    print(
        "\nTesting CNN1/CNN2 probability weights..."
    )

    rows = []

    for cnn1_weight in WEIGHT_GRID:

        cnn2_weight = 1.0 - cnn1_weight

        predictions = weighted_ensemble(
            cnn1_probabilities,
            cnn2_probabilities,
            cnn1_weight
        )

        metrics = calculate_metrics(
            y_val,
            predictions
        )

        rows.append({
            "cnn1_weight": cnn1_weight,
            "cnn2_weight": cnn2_weight,
            **metrics
        })

    dataframe = pd.DataFrame(
        rows
    )

    # Primary selection criterion:
    # validation accuracy.
    #
    # If accuracy is tied, use macro F1.

    dataframe = dataframe.sort_values(
        by=[
            "accuracy",
            "macro_f1"
        ],
        ascending=False
    ).reset_index(
        drop=True
    )

    best = dataframe.iloc[0]

    print("\nValidation weight results:")
    print("-" * 70)

    print(
        dataframe.to_string(
            index=False,
            formatters={
                "cnn1_weight": "{:.2f}".format,
                "cnn2_weight": "{:.2f}".format,
                "accuracy": "{:.4f}".format,
                "macro_precision": "{:.4f}".format,
                "macro_recall": "{:.4f}".format,
                "macro_f1": "{:.4f}".format,
                "weighted_f1": "{:.4f}".format,
            }
        )
    )

    print("\n")
    print("=" * 70)
    print("BEST VALIDATION WEIGHTS")
    print("=" * 70)

    print(
        f"CNN 1 weight : {best['cnn1_weight']:.2f}"
    )

    print(
        f"CNN 2 weight : {best['cnn2_weight']:.2f}"
    )

    print(
        f"Validation accuracy : "
        f"{best['accuracy']:.4%}"
    )

    print(
        f"Validation macro F1  : "
        f"{best['macro_f1']:.4f}"
    )

    path = os.path.join(
        RESULTS_DIR,
        "ensemble_weight_search.csv"
    )

    dataframe.to_csv(
        path,
        index=False
    )

    print(
        f"\nSaved: {path}"
    )

    return (
        float(best["cnn1_weight"]),
        float(best["cnn2_weight"]),
        dataframe
    )


# ============================================================
# Confusion matrix
# ============================================================

def save_confusion_matrix(
    y_true,
    y_pred,
    filename,
    title,
    normalize=False
):

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    if normalize:

        cm = cm.astype(
            np.float64
        )

        row_sums = cm.sum(
            axis=1,
            keepdims=True
        )

        cm = np.divide(
            cm,
            row_sums,
            out=np.zeros_like(cm),
            where=row_sums != 0
        )

    plt.figure(
        figsize=(8, 6)
    )

    plt.imshow(
        cm,
        interpolation="nearest"
    )

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(
        NUM_CLASSES
    )

    plt.xticks(
        tick_marks,
        [f"Class {i}" for i in tick_marks]
    )

    plt.yticks(
        tick_marks,
        [f"Class {i}" for i in tick_marks]
    )

    threshold = (
        cm.max() / 2.0
        if cm.size
        else 0
    )

    for i in range(
        cm.shape[0]
    ):

        for j in range(
            cm.shape[1]
        ):

            if normalize:

                text = f"{cm[i, j]:.2f}"

            else:

                text = str(
                    cm[i, j]
                )

            plt.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white"
                if cm[i, j] > threshold
                else "black"
            )

    plt.ylabel(
        "True Label"
    )

    plt.xlabel(
        "Predicted Label"
    )

    plt.tight_layout()

    path = os.path.join(
        RESULTS_DIR,
        filename
    )

    plt.savefig(
        path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {path}"
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("WEATHERNET-05 — VALIDATION-BASED CNN ENSEMBLE")
    print("=" * 70)

    print(
        "\nEnsemble strategy:"
    )

    print(
        "Validation-selected weighted probability averaging"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Weights are selected using validation data only."
    )

    print(
        "The test set is used only for final evaluation."
    )

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    print(
        "\nLoading WeatherNet-05 dataset..."
    )

    (
        train_ds,
        val_ds,
        test_ds
    ) = load_weathernet_datasets(
        batch_size=BATCH_SIZE
    )

    print(
        "\nDataset loaded successfully! ✅"
    )

    # --------------------------------------------------------
    # Extract labels
    # --------------------------------------------------------

    print(
        "\nExtracting validation labels..."
    )

    y_val = extract_labels(
        val_ds
    )

    print(
        f"Validation labels: {len(y_val)}"
    )

    print(
        "\nExtracting test labels..."
    )

    y_test = extract_labels(
        test_ds
    )

    print(
        f"Test labels: {len(y_test)}"
    )

    # --------------------------------------------------------
    # Load CNN1
    # --------------------------------------------------------

    print(
        "\nLoading CNN 1..."
    )

    if not os.path.exists(
        CNN1_PATH
    ):

        raise FileNotFoundError(
            f"CNN1 model not found:\n{CNN1_PATH}"
        )

    cnn1 = keras.models.load_model(
        CNN1_PATH
    )

    print(
        "CNN 1 loaded successfully! ✅"
    )

    print(
        f"Parameters: {cnn1.count_params():,}"
    )

    # --------------------------------------------------------
    # Load CNN2
    # --------------------------------------------------------

    print(
        "\nLoading CNN 2..."
    )

    if not os.path.exists(
        CNN2_PATH
    ):

        raise FileNotFoundError(
            f"CNN2 model not found:\n{CNN2_PATH}"
        )

    cnn2 = keras.models.load_model(
        CNN2_PATH
    )

    print(
        "CNN 2 loaded successfully! ✅"
    )

    print(
        f"Parameters: {cnn2.count_params():,}"
    )

    # --------------------------------------------------------
    # Validation predictions
    # --------------------------------------------------------

    (
        cnn1_val_prob,
        cnn1_val_pred,
        cnn1_val_time
    ) = generate_predictions(
        cnn1,
        val_ds,
        "CNN 1 validation"
    )

    (
        cnn2_val_prob,
        cnn2_val_pred,
        cnn2_val_time
    ) = generate_predictions(
        cnn2,
        val_ds,
        "CNN 2 validation"
    )

    # --------------------------------------------------------
    # Validation metrics
    # --------------------------------------------------------

    cnn1_val_metrics = calculate_metrics(
        y_val,
        cnn1_val_pred
    )

    cnn2_val_metrics = calculate_metrics(
        y_val,
        cnn2_val_pred
    )

    print("\n")
    print("=" * 70)
    print("VALIDATION MODEL PERFORMANCE")
    print("=" * 70)

    print("\nCNN 1")
    print("-" * 50)

    print(
        f"Accuracy        : "
        f"{cnn1_val_metrics['accuracy']:.4%}"
    )

    print(
        f"Macro F1        : "
        f"{cnn1_val_metrics['macro_f1']:.4f}"
    )

    print("\nCNN 2")
    print("-" * 50)

    print(
        f"Accuracy        : "
        f"{cnn2_val_metrics['accuracy']:.4%}"
    )

    print(
        f"Macro F1        : "
        f"{cnn2_val_metrics['macro_f1']:.4f}"
    )

    # --------------------------------------------------------
    # Find best ensemble weight
    # --------------------------------------------------------

    (
        best_cnn1_weight,
        best_cnn2_weight,
        weight_results
    ) = search_best_weight(
        y_val,
        cnn1_val_prob,
        cnn2_val_prob
    )

    # --------------------------------------------------------
    # Release validation predictions
    # --------------------------------------------------------

    del cnn1_val_prob
    del cnn2_val_prob

    # --------------------------------------------------------
    # Test predictions
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)

    (
        cnn1_test_prob,
        cnn1_test_pred,
        cnn1_test_time
    ) = generate_predictions(
        cnn1,
        test_ds,
        "CNN 1 test"
    )

    (
        cnn2_test_prob,
        cnn2_test_pred,
        cnn2_test_time
    ) = generate_predictions(
        cnn2,
        test_ds,
        "CNN 2 test"
    )

    # --------------------------------------------------------
    # Individual metrics
    # --------------------------------------------------------

    cnn1_test_metrics = calculate_metrics(
        y_test,
        cnn1_test_pred
    )

    cnn2_test_metrics = calculate_metrics(
        y_test,
        cnn2_test_pred
    )

    # --------------------------------------------------------
    # Final weighted ensemble
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FINAL WEIGHTED ENSEMBLE")
    print("=" * 70)

    print(
        f"\nCNN 1 weight: "
        f"{best_cnn1_weight:.2f}"
    )

    print(
        f"CNN 2 weight: "
        f"{best_cnn2_weight:.2f}"
    )

    ensemble_test_pred = weighted_ensemble(
        cnn1_test_prob,
        cnn2_test_prob,
        best_cnn1_weight
    )

    ensemble_metrics = calculate_metrics(
        y_test,
        ensemble_test_pred
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results = pd.DataFrame([
        {
            "model": "CNN1",
            "test_accuracy":
                cnn1_test_metrics["accuracy"],
            "macro_precision":
                cnn1_test_metrics["macro_precision"],
            "macro_recall":
                cnn1_test_metrics["macro_recall"],
            "macro_f1":
                cnn1_test_metrics["macro_f1"],
            "weighted_f1":
                cnn1_test_metrics["weighted_f1"],
            "parameters":
                cnn1.count_params(),
            "inference_time_seconds":
                cnn1_test_time,
            "ensemble_weight":
                1.0
        },

        {
            "model": "CNN2",
            "test_accuracy":
                cnn2_test_metrics["accuracy"],
            "macro_precision":
                cnn2_test_metrics["macro_precision"],
            "macro_recall":
                cnn2_test_metrics["macro_recall"],
            "macro_f1":
                cnn2_test_metrics["macro_f1"],
            "weighted_f1":
                cnn2_test_metrics["weighted_f1"],
            "parameters":
                cnn2.count_params(),
            "inference_time_seconds":
                cnn2_test_time,
            "ensemble_weight":
                0.0
        },

        {
            "model": "Weighted Ensemble",
            "test_accuracy":
                ensemble_metrics["accuracy"],
            "macro_precision":
                ensemble_metrics["macro_precision"],
            "macro_recall":
                ensemble_metrics["macro_recall"],
            "macro_f1":
                ensemble_metrics["macro_f1"],
            "weighted_f1":
                ensemble_metrics["weighted_f1"],
            "parameters":
                cnn1.count_params()
                + cnn2.count_params(),
            "inference_time_seconds":
                cnn1_test_time
                + cnn2_test_time,
            "ensemble_weight":
                1.0
        }
    ])

    print("\n")
    print("=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)

    print(
        results.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Improvement
    # --------------------------------------------------------

    cnn1_accuracy = (
        cnn1_test_metrics["accuracy"]
    )

    ensemble_accuracy = (
        ensemble_metrics["accuracy"]
    )

    absolute_change = (
        ensemble_accuracy
        - cnn1_accuracy
    )

    relative_change = (
        absolute_change
        / cnn1_accuracy
        * 100
    )

    print("\n")
    print("-" * 70)
    print("ENSEMBLE IMPROVEMENT OVER CNN 1")
    print("-" * 70)

    print(
        f"CNN 1 accuracy : "
        f"{cnn1_accuracy:.4%}"
    )

    print(
        f"Ensemble       : "
        f"{ensemble_accuracy:.4%}"
    )

    print(
        f"Absolute change: "
        f"{absolute_change:+.4%}"
    )

    print(
        f"Relative change: "
        f"{relative_change:+.2f}%"
    )

    if absolute_change > 0:

        print(
            "\n✅ Weighted ensemble improves over CNN 1!"
        )

    elif abs(absolute_change) < 1e-9:

        print(
            "\nℹ️ Weighted ensemble matches CNN 1."
        )

    else:

        print(
            "\n⚠️ Weighted ensemble is still worse "
            "than CNN 1."
        )

        print(
            "CNN 1 remains the recommended final model."
        )

    # --------------------------------------------------------
    # Save comparison
    # --------------------------------------------------------

    comparison_path = os.path.join(
        RESULTS_DIR,
        "weighted_ensemble_comparison.csv"
    )

    results.to_csv(
        comparison_path,
        index=False
    )

    print(
        f"\nSaved: {comparison_path}"
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report = classification_report(
        y_test,
        ensemble_test_pred,
        target_names=[
            "Class 0",
            "Class 1",
            "Class 2",
            "Class 3",
            "Class 4"
        ],
        digits=4,
        zero_division=0
    )

    report_path = os.path.join(
        RESULTS_DIR,
        "weighted_ensemble_classification_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "WeatherNet-05 Weighted Ensemble\n"
        )

        file.write(
            "=" * 65
            + "\n\n"
        )

        file.write(
            f"CNN1 weight: "
            f"{best_cnn1_weight:.2f}\n"
        )

        file.write(
            f"CNN2 weight: "
            f"{best_cnn2_weight:.2f}\n\n"
        )

        file.write(
            report
        )

    print(
        f"Saved: {report_path}"
    )

    # --------------------------------------------------------
    # Confusion matrices
    # --------------------------------------------------------

    print(
        "\nGenerating confusion matrices..."
    )

    save_confusion_matrix(
        y_test,
        ensemble_test_pred,
        "weighted_ensemble_confusion_matrix.png",
        "Weighted Ensemble Confusion Matrix",
        normalize=False
    )

    save_confusion_matrix(
        y_test,
        ensemble_test_pred,
        "weighted_ensemble_confusion_matrix_normalized.png",
        "Weighted Ensemble Normalized Confusion Matrix",
        normalize=True
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    prediction_dataframe = pd.DataFrame({
        "true_label": y_test,
        "cnn1_prediction": cnn1_test_pred,
        "cnn2_prediction": cnn2_test_pred,
        "ensemble_prediction": ensemble_test_pred
    })

    prediction_path = os.path.join(
        RESULTS_DIR,
        "weighted_ensemble_predictions.csv"
    )

    prediction_dataframe.to_csv(
        prediction_path,
        index=False
    )

    print(
        f"Saved: {prediction_path}"
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("WEIGHTED ENSEMBLE EVALUATION COMPLETE! ✅")
    print("=" * 70)

    print(
        "\nFinal Results:"
    )

    print(
        f"  CNN 1           : "
        f"{cnn1_accuracy:.4%}"
    )

    print(
        f"  CNN 2           : "
        f"{cnn2_test_metrics['accuracy']:.4%}"
    )

    print(
        f"  Weighted Ensemble: "
        f"{ensemble_accuracy:.4%}"
    )

    print(
        "\nSelected weights:"
    )

    print(
        f"  CNN 1 = {best_cnn1_weight:.2f}"
    )

    print(
        f"  CNN 2 = {best_cnn2_weight:.2f}"
    )

    print(
        "\nResults saved to:"
    )

    print(
        RESULTS_DIR
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()