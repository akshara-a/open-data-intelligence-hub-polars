import json
from typing import Dict, List

import numpy as np


def analyze_disagreement(model_predictions: List[np.ndarray], y_true: np.ndarray, ensemble_confidences: np.ndarray | None = None) -> Dict[str, float]:
    """Compare agreement patterns across model predictions and summarize disagreement behavior."""
    if len(model_predictions) == 0:
        raise ValueError("At least one model prediction is needed.")

    pred_stack = np.stack(model_predictions, axis=0)
    total_samples = pred_stack.shape[1]

    if ensemble_confidences is None:
        ensemble_confidences = np.ones(total_samples, dtype=np.float32)

    all_agree = np.zeros(total_samples, dtype=bool)
    two_agree = np.zeros(total_samples, dtype=bool)
    all_disagree = np.zeros(total_samples, dtype=bool)

    for i in range(total_samples):
        unique_votes = np.unique(pred_stack[:, i])
        if len(unique_votes) == 1:
            all_agree[i] = True
        elif len(unique_votes) == 2:
            two_agree[i] = True
        elif len(unique_votes) == pred_stack.shape[0]:
            all_disagree[i] = True

    percent_all_agree = float(np.mean(all_agree)) * 100.0
    percent_two_agree = float(np.mean(two_agree)) * 100.0
    percent_all_disagree = float(np.mean(all_disagree)) * 100.0

    if np.any(all_disagree):
        ensemble_pred = np.argmax(ensemble_confidences[all_disagree]) if ensemble_confidences.ndim > 1 else ensemble_confidences[all_disagree]
        disagreement_accuracy = float(np.mean(ensemble_pred == y_true[all_disagree])) if ensemble_pred.size > 0 else 0.0
    else:
        disagreement_accuracy = 0.0

    agreement_confidence = float(np.mean(ensemble_confidences[all_agree])) if np.any(all_agree) else 0.0
    disagreement_confidence = float(np.mean(ensemble_confidences[all_disagree])) if np.any(all_disagree) else 0.0

    return {
        "percent_all_models_agree": percent_all_agree,
        "percent_two_models_agree": percent_two_agree,
        "percent_all_models_disagree": percent_all_disagree,
        "ensemble_accuracy_on_disagreement": disagreement_accuracy,
        "confidence_agreement_cases": agreement_confidence,
        "confidence_disagreement_cases": disagreement_confidence,
    }


def compute_disagreement_summary(model_predictions: List[np.ndarray], y_true: np.ndarray, ensemble_confidences: np.ndarray | None = None) -> Dict[str, float]:
    return analyze_disagreement(model_predictions, y_true, ensemble_confidences)
