from pathlib import Path
import time
import json
import psutil

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from data_loader import load_datasets
from models import (
    build_cnn_small,
    build_cnn_standard,
    build_cnn_deep,
)


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
PLOTS_DIR = BASE_DIR / "plots"

MODEL_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)


MODEL_BUILDERS = {
    "CNN_Small": build_cnn_small,
    "CNN_Standard": build_cnn_standard,
    "CNN_Deep": build_cnn_deep,
}


def dataset_to_arrays(dataset):
    """Convert a TensorFlow dataset into NumPy arrays."""
    images = []
    labels = []

    for batch_images, batch_labels in dataset:
        images.append(batch_images.numpy())
        labels.append(batch_labels.numpy().ravel())

    return np.concatenate(images), np.concatenate(labels).astype(int)


def plot_history(history, model_name):
    """Save accuracy and loss plots for one model."""

    epochs = range(1, len(history.history["accuracy"]) + 1)

    plt.figure()
    plt.plot(
        epochs,
        history.history["accuracy"],
        label="Training Accuracy",
    )
    plt.plot(
        epochs,
        history.history["val_accuracy"],
        label="Validation Accuracy",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name} Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        PLOTS_DIR / f"{model_name.lower()}_accuracy.png",
        dpi=150,
    )
    plt.close()

    plt.figure()
    plt.plot(
        epochs,
        history.history["loss"],
        label="Training Loss",
    )
    plt.plot(
        epochs,
        history.history["val_loss"],
        label="Validation Loss",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        PLOTS_DIR / f"{model_name.lower()}_loss.png",
        dpi=150,
    )
    plt.close()


def calculate_model_size(model_path):
    """Return model file size in MB."""
    return model_path.stat().st_size / (1024 * 1024)


def benchmark_model(model, X_test):
    """Measure inference latency, throughput, and memory usage."""

    process = psutil.Process()

    # Warm-up prediction
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

    memory_mb = max(
        0,
        (
            memory_after - memory_before
        ) / (1024 * 1024),
    )

    return (
        latency_ms,
        throughput,
        memory_mb,
    )


def main():
    train_ds, val_ds, test_ds = load_datasets()

    X_test, y_test = dataset_to_arrays(test_ds)

    results = []

    for model_name, builder in MODEL_BUILDERS.items():

        print("\n" + "=" * 60)
        print(f"Training {model_name}")
        print("=" * 60)

        model = builder()
        model.summary()

        model_path = (
            MODEL_DIR
            / f"{model_name.lower()}.keras"
        )

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=4,
                restore_best_weights=True,
            ),
            ModelCheckpoint(
                model_path,
                monitor="val_accuracy",
                save_best_only=True,
            ),
        ]

        start_time = time.perf_counter()

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=15,
            callbacks=callbacks,
            verbose=1,
        )

        training_time = (
            time.perf_counter()
            - start_time
        )

        plot_history(
            history,
            model_name,
        )

        # Load best checkpoint
        best_model = builder()

        best_model.load_weights(
            model_path
        )

        probabilities = (
            best_model.predict(
                X_test,
                verbose=0,
            ).ravel()
        )

        predictions = (
            probabilities >= 0.5
        ).astype(int)

        accuracy = accuracy_score(
            y_test,
            predictions,
        )

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0,
        )

        cm = confusion_matrix(
            y_test,
            predictions,
        )

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=[
                "ok_front",
                "def_front",
            ],
        )

        disp.plot()

        plt.title(
            f"{model_name} Confusion Matrix"
        )

        plt.tight_layout()

        plt.savefig(
            PLOTS_DIR
            / f"{model_name.lower()}_confusion_matrix.png",
            dpi=150,
        )

        plt.close()

        (
            latency_ms,
            throughput,
            memory_mb,
        ) = benchmark_model(
            best_model,
            X_test,
        )

        size_mb = calculate_model_size(
            model_path
        )

        parameter_count = (
            best_model.count_params()
        )

        results.append(
            {
                "model": model_name,
                "accuracy": round(
                    float(accuracy),
                    4,
                ),
                "precision": round(
                    float(precision),
                    4,
                ),
                "recall": round(
                    float(recall),
                    4,
                ),
                "f1_score": round(
                    float(f1),
                    4,
                ),
                "parameter_count": int(
                    parameter_count
                ),
                "model_size_mb": round(
                    float(size_mb),
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
                "training_time_seconds": round(
                    float(training_time),
                    2,
                ),
            }
        )

        print(
            f"\n{model_name} Results"
        )

        print(
            f"Accuracy   : {accuracy:.4f}"
        )

        print(
            f"Precision  : {precision:.4f}"
        )

        print(
            f"Recall     : {recall:.4f}"
        )

        print(
            f"F1 Score   : {f1:.4f}"
        )

        print(
            f"Parameters : {parameter_count:,}"
        )

        print(
            f"Model Size : {size_mb:.4f} MB"
        )

        print(
            f"Latency    : {latency_ms:.4f} ms/image"
        )

        print(
            f"Throughput : {throughput:.4f} images/sec"
        )

        print(
            f"Memory     : {memory_mb:.4f} MB"
        )

    results_path = (
        BASE_DIR
        / "model_benchmarks.json"
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

    print("\n" + "=" * 60)
    print(
        "Individual CNN benchmark results"
    )
    print("=" * 60)

    for result in results:

        print(
            f"{result['model']}: "
            f"accuracy={result['accuracy']:.4f}, "
            f"F1={result['f1_score']:.4f}, "
            f"latency="
            f"{result['latency_ms_per_image']:.4f} ms, "
            f"memory="
            f"{result['memory_increase_mb']:.4f} MB"
        )

    print(
        f"\nBenchmark results saved to: "
        f"{results_path}"
    )


if __name__ == "__main__":
    main()