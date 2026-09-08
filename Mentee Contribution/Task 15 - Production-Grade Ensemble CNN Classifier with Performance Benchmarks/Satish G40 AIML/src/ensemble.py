import json
import os
from typing import Dict, List

import numpy as np
import tensorflow as tf

from src.config import MODEL_PATHS, RESULTS_DIR


def load_ensemble_models(model_names: List[str]) -> Dict[str, tf.keras.Model]:
    models = {}
    for model_name in model_names:
        path = MODEL_PATHS[model_name]
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}. Train it before ensemble evaluation.")
        models[model_name] = tf.keras.models.load_model(str(path))
    return models


def majority_voting(probabilities: List[np.ndarray]) -> np.ndarray:
    """Count class votes across models and return the winning class per sample."""
    votes = np.stack([np.argmax(prob, axis=1) for prob in probabilities], axis=1)
    final_predictions = []
    for sample_votes in votes:
        counts = np.bincount(sample_votes, minlength=probabilities[0].shape[1])
        final_predictions.append(int(np.argmax(counts)))
    return np.asarray(final_predictions, dtype=int)


def soft_voting(probabilities: List[np.ndarray]) -> np.ndarray:
    """Average class probabilities and take the argmax."""
    stacked = np.stack(probabilities, axis=0)
    mean_probs = stacked.mean(axis=0)
    return np.argmax(mean_probs, axis=1)


def weighted_soft_voting(probabilities: List[np.ndarray], weights: List[float]) -> np.ndarray:
    """Weighted average of probabilities with weights derived from validation accuracy."""
    if len(probabilities) != len(weights):
        raise ValueError("Probability and weight lengths must match.")
    weighted = np.zeros_like(probabilities[0], dtype=np.float64)
    total_weight = float(sum(weights))
    for prob, weight in zip(probabilities, weights):
        weighted += prob * weight
    weighted /= total_weight
    return np.argmax(weighted, axis=1)


def predict_with_ensemble(model_names: List[str], x: np.ndarray, weights: List[float] | None = None) -> Dict[str, np.ndarray]:
    models = load_ensemble_models(model_names)
    probs = [model.predict(x, verbose=0) for model in models.values()]
    predictions = {
        "majority_voting": majority_voting(probs),
        "soft_voting": soft_voting(probs),
    }
    if weights is not None:
        predictions["weighted_soft_voting"] = weighted_soft_voting(probs, weights)
    return predictions


def compute_ensemble_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score(y_true, y_pred, average="macro")),
    }


def evaluate_ensemble(model_names: List[str], x_test: np.ndarray, y_test: np.ndarray, weights: List[float] | None = None) -> Dict[str, Dict[str, float]]:
    models = load_ensemble_models(model_names)
    probs = [model.predict(x_test, verbose=0) for model in models.values()]
    ensemble_predictions = {
        "majority_voting": majority_voting(probs),
        "soft_voting": soft_voting(probs),
    }
    if weights is not None:
        ensemble_predictions["weighted_soft_voting"] = weighted_soft_voting(probs, weights)

    results = {}
    for name, preds in ensemble_predictions.items():
        results[name] = compute_ensemble_metrics(y_test, preds)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_DIR / "ensemble_metrics.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    return results


if __name__ == "__main__":
    from src.data_loader import load_cifar10
    _, _, _, _, x_test, y_test = load_cifar10()
    weights = [0.8, 1.0, 1.2]
    metrics = evaluate_ensemble(["baseline_cnn", "regularized_cnn", "deep_cnn"], x_test, y_test, weights)
    print(metrics)
