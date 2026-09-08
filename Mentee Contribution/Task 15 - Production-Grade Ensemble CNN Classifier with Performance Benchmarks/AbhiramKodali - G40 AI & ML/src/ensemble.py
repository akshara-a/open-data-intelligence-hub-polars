import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model

from data_loader import load_datasets


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
PLOT_DIR = BASE_DIR / "plots"
BENCHMARK_FILE = BASE_DIR / "production_benchmarks.json"

MODEL_FILES = [
    "cnn_small.keras",
    "cnn_standard.keras",
    "cnn_deep.keras",
]

CLASS_NAMES = ["ok_front", "def_front"]


def create_robust_test_data(X_test):
    """Add controlled Gaussian noise to create a robustness test set."""
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.03, X_test.shape)
    X_robust = np.clip(X_test + noise, 0.0, 1.0)
    return X_robust.astype(np.float32)


def evaluate_predictions(y_true, probabilities):
    """Calculate classification metrics from predicted probabilities."""
    predictions = (probabilities >= 0.5).astype(int).reshape(-1)
    y_true = np.asarray(y_true).astype(int).reshape(-1)

    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, predictions, zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(
            y_true, predictions
        ).tolist(),
    }


def majority_vote(probabilities):
    """Perform majority voting using binary predictions."""
    binary_predictions = np.array(
        [(prob >= 0.5).astype(int).reshape(-1) for prob in probabilities]
    )

    votes = np.sum(binary_predictions, axis=0)
    return (votes >= 2).astype(int)


def soft_vote(probabilities):
    """Perform soft voting using average probabilities."""
    average_probability = np.mean(
        [prob.reshape(-1) for prob in probabilities],
        axis=0,
    )

    return (average_probability >= 0.5).astype(int)


def evaluate_labels(y_true, predictions):
    """Calculate metrics from binary predictions."""
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    predictions = np.asarray(predictions).astype(int).reshape(-1)

    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, predictions, zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(
            y_true, predictions
        ).tolist(),
    }


def calculate_disagreement(probabilities):
    """Calculate how often the individual models disagree."""
    predictions = np.array(
        [(prob >= 0.5).astype(int).reshape(-1) for prob in probabilities]
    )

    disagreement = np.any(
        predictions != predictions[0],
        axis=0,
    )

    disagreement_cases = int(np.sum(disagreement))
    total_cases = len(disagreement)

    return disagreement_cases, disagreement_cases / total_cases

def load_current_benchmarks():
    """Read the latest individual-model benchmark results."""
    if not BENCHMARK_FILE.exists():
        raise FileNotFoundError(
            f"Benchmark file not found: {BENCHMARK_FILE}"
        )

    with open(BENCHMARK_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    return {
        item["model"]: item
        for item in data
    }

def calculate_ensemble_performance(benchmark_data):
    """
    Estimate sequential ensemble latency from the latest
    individual-model benchmark results.
    """
    latencies = []

    for model_file in MODEL_FILES:
        model_name = Path(model_file).stem

        if model_name not in benchmark_data:
            raise KeyError(
                f"{model_name} not found in production_benchmarks.json"
            )

        latencies.append(
            float(benchmark_data[model_name]["latency_ms_per_image"])
        )

    combined_latency = sum(latencies)

    throughput = 1000.0 / combined_latency

    return combined_latency, throughput


def save_confusion_matrix(matrix, filename, title):
    """Save a confusion matrix as a simple PNG image."""
    import matplotlib.pyplot as plt

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix)

    ax.set_title(title)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)

    for row in range(len(matrix)):
        for col in range(len(matrix[row])):
            ax.text(
                col,
                row,
                matrix[row][col],
                ha="center",
                va="center",
            )

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / filename, dpi=150)
    plt.close(fig)


def main():
    print("=" * 60)
    print("TASK 15 ENSEMBLE EVALUATION")
    print("=" * 60)

    print("\nLoading test dataset...")

    _, _, test_ds = load_datasets()

    X_test = []
    y_test = []

    for images, labels in test_ds:
        X_test.append(images.numpy())
        y_test.append(labels.numpy())

    X_test = np.concatenate(X_test, axis=0)
    y_test = np.concatenate(y_test, axis=0).astype(int).reshape(-1)

    print(f"Test images: {len(X_test)}")

    print("\nLoading trained CNN models...")
    print("-" * 50)

    models = []

    for model_file in MODEL_FILES:
        model_path = MODEL_DIR / model_file

        print(f"Loading: {model_file}")

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )

        models.append(load_model(model_path))

    print("\nGenerating predictions...")

    clean_probabilities = []

    for model in models:
        probabilities = model.predict(
            X_test,
            verbose=0,
        ).reshape(-1)

        clean_probabilities.append(probabilities)

    X_robust = create_robust_test_data(X_test)

    robust_probabilities = []

    for model in models:
        probabilities = model.predict(
            X_robust,
            verbose=0,
        ).reshape(-1)

        robust_probabilities.append(probabilities)

    # ---------------------------------------------------------
    # Majority Voting
    # ---------------------------------------------------------

    majority_clean_predictions = majority_vote(
        clean_probabilities
    )

    majority_robust_predictions = majority_vote(
        robust_probabilities
    )

    majority_clean = evaluate_labels(
        y_test,
        majority_clean_predictions,
    )

    majority_robust = evaluate_labels(
        y_test,
        majority_robust_predictions,
    )

    majority_drop = (
        majority_clean["accuracy"]
        - majority_robust["accuracy"]
    )

    # ---------------------------------------------------------
    # Soft Voting
    # ---------------------------------------------------------

    soft_clean_predictions = soft_vote(
        clean_probabilities
    )

    soft_robust_predictions = soft_vote(
        robust_probabilities
    )

    soft_clean = evaluate_labels(
        y_test,
        soft_clean_predictions,
    )

    soft_robust = evaluate_labels(
        y_test,
        soft_robust_predictions,
    )

    soft_drop = (
        soft_clean["accuracy"]
        - soft_robust["accuracy"]
    )

    # ---------------------------------------------------------
    # Disagreement
    # ---------------------------------------------------------

    disagreement_cases, disagreement_rate = calculate_disagreement(
        clean_probabilities
    )

    # ---------------------------------------------------------
    # Current Benchmark-Based Ensemble Performance
    # ---------------------------------------------------------

    benchmark_data = load_current_benchmarks()

    combined_latency, ensemble_throughput = (
        calculate_ensemble_performance(benchmark_data)
    )

    # ---------------------------------------------------------
    # Display Results
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("TASK 15 ENSEMBLE RESULTS")
    print("=" * 60)

    print("\nMajority Voting - Clean Test Set")
    print("-" * 40)
    print(f"Accuracy : {majority_clean['accuracy']:.4f}")
    print(f"Precision: {majority_clean['precision']:.4f}")
    print(f"Recall   : {majority_clean['recall']:.4f}")
    print(f"F1 Score : {majority_clean['f1']:.4f}")

    print("\nConfusion Matrix:")
    print(np.array(majority_clean["confusion_matrix"]))

    print("\nMajority Voting - Robust Test Set")
    print("-" * 40)
    print(f"Accuracy : {majority_robust['accuracy']:.4f}")
    print(f"Accuracy Drop: {majority_drop:.4f}")

    print("\nSoft Voting - Clean Test Set")
    print("-" * 40)
    print(f"Accuracy : {soft_clean['accuracy']:.4f}")
    print(f"Precision: {soft_clean['precision']:.4f}")
    print(f"Recall   : {soft_clean['recall']:.4f}")
    print(f"F1 Score : {soft_clean['f1']:.4f}")

    print("\nConfusion Matrix:")
    print(np.array(soft_clean["confusion_matrix"]))

    print("\nSoft Voting - Robust Test Set")
    print("-" * 40)
    print(f"Accuracy : {soft_robust['accuracy']:.4f}")
    print(f"Accuracy Drop: {soft_drop:.4f}")

    print("\nModel Disagreement")
    print("-" * 40)
    print(f"Disagreement cases: {disagreement_cases}")
    print(f"Disagreement rate : {disagreement_rate:.4f}")

    print("\nEnsemble Performance")
    print("-" * 40)
    print(
        f"Combined latency: "
        f"{combined_latency:.4f} ms/image"
    )
    print(
        f"Estimated throughput: "
        f"{ensemble_throughput:.4f} images/sec"
    )

    # ---------------------------------------------------------
    # Save Confusion Matrices
    # ---------------------------------------------------------

    save_confusion_matrix(
        majority_clean["confusion_matrix"],
        "ensemble_majority_confusion_matrix.png",
        "Majority Voting - Clean Test Set",
    )

    save_confusion_matrix(
        soft_clean["confusion_matrix"],
        "ensemble_soft_confusion_matrix.png",
        "Soft Voting - Clean Test Set",
    )

    # ---------------------------------------------------------
    # Save Results
    # ---------------------------------------------------------

    results = {
        "majority_voting": {
            "clean": majority_clean,
            "robust": majority_robust,
            "robust_accuracy_drop": majority_drop,
        },
        "soft_voting": {
            "clean": soft_clean,
            "robust": soft_robust,
            "robust_accuracy_drop": soft_drop,
        },
        "model_disagreement": {
            "cases": disagreement_cases,
            "rate": disagreement_rate,
        },
        "ensemble_performance": {
            "combined_latency_ms_per_image": combined_latency,
            "estimated_throughput_images_per_second": ensemble_throughput,
            "latency_basis": (
                "sum of current individual-model "
                "benchmark latencies"
            ),
        },
    }

    output_file = BASE_DIR / "ensemble_results.json"

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print("\nEnsemble results saved to:")
    print(output_file)

    print("\nConfusion matrices saved to:")
    print(PLOT_DIR)


if __name__ == "__main__":
    main()