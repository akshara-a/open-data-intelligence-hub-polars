"""
WeatherNet-05
Production Performance Benchmark

Benchmarks CNN 1, CNN 2, and the selected weighted ensemble.

No model training is performed.
"""

import os
import sys
import time
import platform

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, f1_score


# ============================================================
# Project root
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# Project imports
# ============================================================

from src.preprocessing import load_weathernet_datasets


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 32
NUM_CLASSES = 5

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

OUTPUT_PATH = os.path.join(
    RESULTS_DIR,
    "performance_benchmark.csv"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# Reproducibility
# ============================================================

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# Utility functions
# ============================================================

def get_model_size_mb(path):
    """Return model file size in MB."""

    if not os.path.exists(path):
        return 0.0

    return os.path.getsize(path) / (
        1024 * 1024
    )


def extract_labels(dataset):
    """Extract labels from tf.data.Dataset."""

    labels = []

    for _, batch_labels in dataset:

        labels.extend(
            batch_labels.numpy().tolist()
        )

    return np.asarray(
        labels,
        dtype=np.int32
    )


def benchmark_model(
    model,
    dataset,
    true_labels,
    model_name
):
    """
    Benchmark one model.

    Measures:
    - inference time
    - accuracy
    - macro F1
    - weighted F1
    - throughput
    """

    print("\n" + "=" * 65)
    print(f"BENCHMARKING {model_name}")
    print("=" * 65)

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    print("\nWarm-up inference...")

    for images, _ in dataset.take(1):
        model.predict(
            images,
            verbose=0
        )

    # --------------------------------------------------------
    # Timed inference
    # --------------------------------------------------------

    print("Running timed inference...")

    start_time = time.perf_counter()

    probabilities = model.predict(
        dataset,
        verbose=1
    )

    end_time = time.perf_counter()

    inference_time = (
        end_time - start_time
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    sample_count = len(
        true_labels
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        true_labels,
        predictions
    )

    macro_f1 = f1_score(
        true_labels,
        predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        true_labels,
        predictions,
        average="weighted",
        zero_division=0
    )

    throughput = (
        sample_count /
        inference_time
    )

    latency_ms = (
        inference_time /
        sample_count
    ) * 1000

    parameters = model.count_params()

    print("\nResults")
    print("-" * 65)

    print(
        f"Samples             : {sample_count}"
    )

    print(
        f"Inference time      : "
        f"{inference_time:.4f} seconds"
    )

    print(
        f"Average latency     : "
        f"{latency_ms:.4f} ms/image"
    )

    print(
        f"Throughput          : "
        f"{throughput:.2f} images/sec"
    )

    print(
        f"Parameters          : "
        f"{parameters:,}"
    )

    print(
        f"Accuracy            : "
        f"{accuracy:.4%}"
    )

    print(
        f"Macro F1            : "
        f"{macro_f1:.4f}"
    )

    print(
        f"Weighted F1         : "
        f"{weighted_f1:.4f}"
    )

    return {
        "model": model_name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "parameters": parameters,
        "inference_time_seconds": inference_time,
        "latency_ms_per_image": latency_ms,
        "throughput_images_per_second": throughput
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("WEATHERNET-05 — PRODUCTION PERFORMANCE BENCHMARK")
    print("=" * 70)

    print("\nEnvironment")
    print("-" * 70)

    print(
        f"Platform       : {platform.platform()}"
    )

    print(
        f"Python         : {platform.python_version()}"
    )

    print(
        f"TensorFlow     : {tf.__version__}"
    )

    print(
        f"Batch size     : {BATCH_SIZE}"
    )

    # --------------------------------------------------------
    # Check models
    # --------------------------------------------------------

    print("\nChecking trained models...")

    if not os.path.exists(CNN1_PATH):
        raise FileNotFoundError(
            f"CNN 1 model not found:\n{CNN1_PATH}"
        )

    if not os.path.exists(CNN2_PATH):
        raise FileNotFoundError(
            f"CNN 2 model not found:\n{CNN2_PATH}"
        )

    print(
        f"CNN 1 size: "
        f"{get_model_size_mb(CNN1_PATH):.2f} MB"
    )

    print(
        f"CNN 2 size: "
        f"{get_model_size_mb(CNN2_PATH):.2f} MB"
    )

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    print("\nLoading WeatherNet-05 test dataset...")

    (
        train_ds,
        val_ds,
        test_ds
    ) = load_weathernet_datasets(
        batch_size=BATCH_SIZE
    )

    print(
        "Dataset loaded successfully! ✅"
    )

    # --------------------------------------------------------
    # Extract test labels
    # --------------------------------------------------------

    print("\nExtracting test labels...")

    true_labels = extract_labels(
        test_ds
    )

    print(
        f"Test samples: {len(true_labels)}"
    )

    # --------------------------------------------------------
    # Load CNN 1
    # --------------------------------------------------------

    print("\nLoading CNN 1...")

    cnn1 = keras.models.load_model(
        CNN1_PATH
    )

    print(
        "CNN 1 loaded successfully! ✅"
    )

    # --------------------------------------------------------
    # Load CNN 2
    # --------------------------------------------------------

    print("\nLoading CNN 2...")

    cnn2 = keras.models.load_model(
        CNN2_PATH
    )

    print(
        "CNN 2 loaded successfully! ✅"
    )

    # --------------------------------------------------------
    # Benchmark CNN 1
    # --------------------------------------------------------

    cnn1_result = benchmark_model(
        cnn1,
        test_ds,
        true_labels,
        "CNN1"
    )

    # --------------------------------------------------------
    # Benchmark CNN 2
    # --------------------------------------------------------

    cnn2_result = benchmark_model(
        cnn2,
        test_ds,
        true_labels,
        "CNN2"
    )

    # --------------------------------------------------------
    # Build comparison
    # --------------------------------------------------------

    results = [
        cnn1_result,
        cnn2_result
    ]

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Add model sizes
    # --------------------------------------------------------

    results_df[
        "model_size_mb"
    ] = [
        get_model_size_mb(CNN1_PATH),
        get_model_size_mb(CNN2_PATH)
    ]

    # --------------------------------------------------------
    # Calculate relative performance
    # --------------------------------------------------------

    baseline_accuracy = (
        cnn1_result["accuracy"]
    )

    baseline_time = (
        cnn1_result[
            "inference_time_seconds"
        ]
    )

    results_df[
        "accuracy_change_vs_cnn1"
    ] = (
        results_df["accuracy"]
        - baseline_accuracy
    )

    results_df[
        "inference_time_ratio_vs_cnn1"
    ] = (
        results_df[
            "inference_time_seconds"
        ]
        / baseline_time
    )

    # --------------------------------------------------------
    # Print comparison
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FINAL PERFORMANCE COMPARISON")
    print("=" * 70)

    display_columns = [
        "model",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "parameters",
        "model_size_mb",
        "inference_time_seconds",
        "latency_ms_per_image",
        "throughput_images_per_second"
    ]

    print(
        results_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------

    print("\n")
    print("-" * 70)
    print("PERFORMANCE ANALYSIS")
    print("-" * 70)

    if cnn2_result["accuracy"] > cnn1_result["accuracy"]:

        print(
            "CNN 2 has higher accuracy than CNN 1."
        )

    else:

        print(
            "CNN 1 has higher accuracy than CNN 2."
        )

    if cnn2_result[
        "inference_time_seconds"
    ] > cnn1_result[
        "inference_time_seconds"
    ]:

        ratio = (
            cnn2_result[
                "inference_time_seconds"
            ]
            /
            cnn1_result[
                "inference_time_seconds"
            ]
        )

        print(
            f"CNN 2 is approximately "
            f"{ratio:.2f}x slower than CNN 1."
        )

    if cnn2_result[
        "parameters"
    ] > cnn1_result[
        "parameters"
    ]:

        ratio = (
            cnn2_result[
                "parameters"
            ]
            /
            cnn1_result[
                "parameters"
            ]
        )

        print(
            f"CNN 2 has approximately "
            f"{ratio:.2f}x more parameters."
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n")
    print(
        f"Saved: {OUTPUT_PATH}"
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("PRODUCTION BENCHMARK COMPLETE! ✅")
    print("=" * 70)

    print("\nBest accuracy:")

    if (
        cnn1_result["accuracy"]
        >=
        cnn2_result["accuracy"]
    ):

        print(
            f"CNN 1 → "
            f"{cnn1_result['accuracy']:.4%}"
        )

    else:

        print(
            f"CNN 2 → "
            f"{cnn2_result['accuracy']:.4%}"
        )

    print("\nBenchmark saved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()