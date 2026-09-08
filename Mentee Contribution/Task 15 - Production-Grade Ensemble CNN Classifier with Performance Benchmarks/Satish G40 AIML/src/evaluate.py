import json
from typing import Dict, List

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_recall_fscore_support

from src.config import CLASS_NAMES, RESULTS_DIR, MODEL_PATHS


def load_model(model_name: str) -> tf.keras.Model:
    model_path = MODEL_PATHS[model_name]
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Train the models first.")
    return tf.keras.models.load_model(str(model_path))


def evaluate_single_model(model_name: str, x_test: np.ndarray, y_test: np.ndarray) -> Dict[str, object]:
    model = load_model(model_name)
    y_pred_logits = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_logits, axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)

    result = {
        "model": model_name,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "loss": float(model.evaluate(x_test, y_test, verbose=0)[0]),
        "classification_report": classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    return result


def evaluate_all_models(model_names: List[str], x_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
    rows = []
    for name in model_names:
        metrics = evaluate_single_model(name, x_test, y_test)
        rows.append({
            "Model": name,
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1-score": metrics["f1_score"],
            "Loss": metrics["loss"],
        })

    results_df = pd.DataFrame(rows)
    results_df.to_csv(RESULTS_DIR / "model_evaluation.csv", index=False)
    return results_df


if __name__ == "__main__":
    try:
        from src.data_loader import load_cifar10
        x_train, y_train, x_val, y_val, x_test, y_test = load_cifar10()
        model_names = ["baseline_cnn", "regularized_cnn", "deep_cnn"]
        df = evaluate_all_models(model_names, x_test, y_test)
        print(df.round(4).to_string(index=False))
    except Exception as exc:
        print(f"Evaluation failed: {exc}")
