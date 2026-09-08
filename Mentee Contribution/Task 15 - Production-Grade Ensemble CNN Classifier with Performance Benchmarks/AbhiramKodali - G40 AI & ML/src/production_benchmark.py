from pathlib import Path
import json
import time

import numpy as np
import psutil
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score

from data_loader import load_datasets


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"


MODEL_NAMES = [
    "cnn_small",
    "cnn_standard",
    "cnn_deep",
]


def dataset_to_arrays(dataset):
    """Convert a TensorFlow dataset into NumPy arrays."""
    images = []
    labels = []

    for batch_images, batch_labels in dataset:
        images.append(batch_images.numpy())
        labels.append(batch_labels.numpy().ravel())

    return (
        np.concatenate(images),
        np.concatenate(labels).astype(int),
    )


def benchmark_model(model, X_test):
    """Measure latency, throughput, and memory increase."""

    process = psutil.Process()

    # Warm-up
    model.predict(X_test[:1], verbose=0)

    memory_before = process.memory_info().rss

    start = time.perf_counter()

    model.predict(
        X_test,
        verbose=0,
    )

    elapsed = time.perf_counter() - start

    memory_after = process.memory_info().rss

    latency_ms = (
        elapsed / len(X_test)
    ) * 1000

    throughput = (
        len(X_test) / elapsed
        if elapsed > 0
        else 0
    )

    memory_increase_mb = max(
        0,
        (
            memory_after - memory_before
        ) / (1024 * 1024),
    )

    return (
        latency_ms,
        throughput,
        memory_increase_mb,
    )


def create_robust_test_data(X_test):
    """
    Create mildly perturbed test images to evaluate
    robustness against small input noise.
    """

    rng = np.random.default_rng(42)

    noise = rng.normal(
        loc=0.0,
        scale=0.03,
        size=X_test.shape,
    )

    robust_images = np.clip(
        X_test + noise,
        0.0,
        1.0,
    )

    return robust_images.astype(np.float32)


def main():

    print("\nLoading test dataset...")

    _, _, test_ds = load_datasets()

    X_test, y_test = dataset_to_arrays(test_ds)

    robust_X_test = create_robust_test_data(
        X_test
    )

    results = []

    print(
        f"Test images: {len(X_test)}"
    )

    print(
        "\n" + "=" * 60
    )
    print(
        "PRODUCTION BENCHMARK RESULTS"
    )
    print(
        "=" * 60
    )

    for model_name in MODEL_NAMES:

        model_path = (
            MODEL_DIR
            / f"{model_name}.keras"
        )

        print(
            f"\nLoading {model_name}.keras..."
        )

        model = load_model(
            model_path
        )

        # Clean test accuracy
        clean_probabilities = (
            model.predict(
                X_test,
                verbose=0,
            ).ravel()
        )

        clean_predictions = (
            clean_probabilities >= 0.5
        ).astype(int)

        clean_accuracy = accuracy_score(
            y_test,
            clean_predictions,
        )

        # Robustness accuracy
        robust_probabilities = (
            model.predict(
                robust_X_test,
                verbose=0,
            ).ravel()
        )

        robust_predictions = (
            robust_probabilities >= 0.5
        ).astype(int)

        robust_accuracy = accuracy_score(
            y_test,
            robust_predictions,
        )

        robustness_drop = (
            clean_accuracy
            - robust_accuracy
        )

        # Latency, throughput and memory
        (
            latency_ms,
            throughput,
            memory_mb,
        ) = benchmark_model(
            model,
            X_test,
        )

        parameter_count = (
            model.count_params()
        )

        model_size_mb = (
            model_path.stat().st_size
            / (1024 * 1024)
        )

        result = {
            "model": model_name,
            "clean_accuracy": round(
                float(clean_accuracy),
                4,
            ),
            "robust_accuracy": round(
                float(robust_accuracy),
                4,
            ),
            "robustness_accuracy_drop": round(
                float(robustness_drop),
                4,
            ),
            "parameter_count": int(
                parameter_count
            ),
            "model_size_mb": round(
                float(model_size_mb),
                4,
            ),
            "latency_ms_per_image": round(
                float(latency_ms),
                4,
            ),
            "throughput_images_per_second": round(
                float(throughput),
                4,
            ),
            "memory_increase_mb": round(
                float(memory_mb),
                4,
            ),
        }

        results.append(result)

        print(
            f"\n{model_name}"
        )
        print(
            f"Clean Accuracy       : "
            f"{clean_accuracy:.4f}"
        )
        print(
            f"Robust Accuracy      : "
            f"{robust_accuracy:.4f}"
        )
        print(
            f"Accuracy Drop        : "
            f"{robustness_drop:.4f}"
        )
        print(
            f"Parameters           : "
            f"{parameter_count:,}"
        )
        print(
            f"Model Size           : "
            f"{model_size_mb:.4f} MB"
        )
        print(
            f"Latency              : "
            f"{latency_ms:.4f} ms/image"
        )
        print(
            f"Throughput           : "
            f"{throughput:.4f} images/sec"
        )
        print(
            f"Memory Increase      : "
            f"{memory_mb:.4f} MB"
        )

    results_path = (
        BASE_DIR
        / "production_benchmarks.json"
    )

    with open(
        results_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=4,
        )

    print(
        "\n" + "=" * 60
    )

    print(
        "Production benchmark results saved to:"
    )

    print(
        results_path
    )


if __name__ == "__main__":
    main()