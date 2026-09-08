"""
WeatherNet-05
Model Evaluation Pipeline

Evaluates:
    CNN 1 baseline
    CNN 2

Outputs:
    - Test accuracy
    - Test loss
    - Classification report
    - Confusion matrix
    - Normalized confusion matrix
    - Evaluation summary CSV
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

import matplotlib.pyplot as plt


# ============================================================
# Project Root
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# Project Imports
# ============================================================

from src.preprocessing import build_datasets


# ============================================================
# Configuration
# ============================================================

NUM_CLASSES = 5

CLASS_NAMES = [
    "Class 0",
    "Class 1",
    "Class 2",
    "Class 3",
    "Class 4"
]

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results"
)

CNN1_PATH = os.path.join(
    MODEL_DIR,
    "cnn_baseline.keras"
)

CNN2_PATH = os.path.join(
    MODEL_DIR,
    "cnn2.keras"
)


# ============================================================
# Create Results Directory
# ============================================================

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# Plot Confusion Matrix
# ============================================================

def save_confusion_matrix(
    cm,
    model_name,
    normalized=False
):

    if normalized:

        row_sums = cm.sum(
            axis=1,
            keepdims=True
        )

        row_sums[row_sums == 0] = 1

        matrix = cm.astype(
            np.float32
        ) / row_sums

        title = (
            f"{model_name} - "
            "Normalized Confusion Matrix"
        )

        filename = (
            f"{model_name.lower()}_"
            "confusion_matrix_normalized.png"
        )

    else:

        matrix = cm

        title = (
            f"{model_name} - "
            "Confusion Matrix"
        )

        filename = (
            f"{model_name.lower()}_"
            "confusion_matrix.png"
        )

    plt.figure(
        figsize=(8, 6)
    )

    plt.imshow(
        matrix,
        interpolation="nearest"
    )

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(
        NUM_CLASSES
    )

    plt.xticks(
        tick_marks,
        CLASS_NAMES,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        tick_marks,
        CLASS_NAMES
    )

    threshold = (
        matrix.max() / 2.0
    )

    for i in range(NUM_CLASSES):

        for j in range(NUM_CLASSES):

            if normalized:

                text = f"{matrix[i, j]:.2f}"

            else:

                text = str(
                    matrix[i, j]
                )

            plt.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color=(
                    "white"
                    if matrix[i, j] > threshold
                    else "black"
                )
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
# Evaluate Model
# ============================================================

def evaluate_model(
    model_path,
    model_name,
    test_ds
):

    print("\n")
    print("=" * 65)
    print(f"{model_name} EVALUATION")
    print("=" * 65)

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not os.path.exists(
        model_path
    ):

        print(
            f"\nERROR: Model not found:"
        )

        print(
            model_path
        )

        return None

    print(
        f"\nLoading model:"
    )

    print(
        model_path
    )

    model = keras.models.load_model(
        model_path
    )

    print(
        "Model loaded successfully! ✅"
    )

    # --------------------------------------------------------
    # Evaluate loss / accuracy
    # --------------------------------------------------------

    print(
        "\nEvaluating test dataset..."
    )

    test_loss, test_accuracy = (
        model.evaluate(
            test_ds,
            verbose=1
        )
    )

    print(
        f"\nTest Loss     : {test_loss:.4f}"
    )

    print(
        f"Test Accuracy : {test_accuracy:.4%}"
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    print(
        "\nGenerating predictions..."
    )

    probabilities = model.predict(
        test_ds,
        verbose=1
    )

    predicted_labels = np.argmax(
        probabilities,
        axis=1
    )

    # --------------------------------------------------------
    # Extract true labels
    # --------------------------------------------------------

    true_labels = []

    for _, labels in test_ds:

        true_labels.extend(
            labels.numpy()
        )

    true_labels = np.asarray(
        true_labels
    )

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if len(true_labels) != len(
        predicted_labels
    ):

        raise ValueError(
            "Number of predictions does not "
            "match number of test labels."
        )

    # --------------------------------------------------------
    # Accuracy verification
    # --------------------------------------------------------

    calculated_accuracy = (
        accuracy_score(
            true_labels,
            predicted_labels
        )
    )

    print(
        f"\nVerified Accuracy : "
        f"{calculated_accuracy:.4%}"
    )

    # --------------------------------------------------------
    # Classification Report
    # --------------------------------------------------------

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=np.arange(NUM_CLASSES),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )

    print(
        "\nClassification Report"
    )

    print(
        "-" * 65
    )

    print(
        report
    )

    # --------------------------------------------------------
    # Save classification report
    # --------------------------------------------------------

    report_path = os.path.join(
        RESULTS_DIR,
        f"{model_name.lower()}_classification_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            f"{model_name} "
            "Classification Report\n"
        )

        file.write(
            "=" * 65
            + "\n\n"
        )

        file.write(
            report
        )

    print(
        f"Saved: {report_path}"
    )

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=np.arange(NUM_CLASSES)
    )

    print(
        "\nConfusion Matrix"
    )

    print(
        cm
    )

    save_confusion_matrix(
        cm,
        model_name,
        normalized=False
    )

    save_confusion_matrix(
        cm,
        model_name,
        normalized=True
    )

    # --------------------------------------------------------
    # Return metrics
    # --------------------------------------------------------

    return {
        "model": model_name,
        "test_loss": float(test_loss),
        "test_accuracy": float(
            test_accuracy
        ),
        "verified_accuracy": float(
            calculated_accuracy
        )
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 65)
    print("WEATHERNET-05 — MODEL EVALUATION")
    print("=" * 65)

    print(
        "\nLoading WeatherNet-05 test dataset..."
    )

    (
        train_ds,
        val_ds,
        test_ds,
        dataframe,
        train_indices,
        val_indices,
        test_indices
    ) = build_datasets()

    print(
        "\nTest dataset ready! ✅"
    )

    print(
        f"Test samples: {len(test_indices)}"
    )

    # --------------------------------------------------------
    # Evaluate CNN 1
    # --------------------------------------------------------

    cnn1_results = evaluate_model(
        CNN1_PATH,
        "CNN1",
        test_ds
    )

    # --------------------------------------------------------
    # Evaluate CNN 2
    # --------------------------------------------------------

    cnn2_results = evaluate_model(
        CNN2_PATH,
        "CNN2",
        test_ds
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    results = []

    if cnn1_results is not None:
        results.append(
            cnn1_results
        )

    if cnn2_results is not None:
        results.append(
            cnn2_results
        )

    if results:

        results_df = pd.DataFrame(
            results
        )

        summary_path = os.path.join(
            RESULTS_DIR,
            "model_evaluation_summary.csv"
        )

        results_df.to_csv(
            summary_path,
            index=False
        )

        print("\n")
        print("=" * 65)
        print("MODEL COMPARISON")
        print("=" * 65)

        print(
            results_df.to_string(
                index=False
            )
        )

        print(
            f"\nSaved: {summary_path}"
        )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print("\n")
    print("=" * 65)
    print("MODEL EVALUATION COMPLETE! ✅")
    print("=" * 65)

    print(
        "\nResults directory:"
    )

    print(
        RESULTS_DIR
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    main()