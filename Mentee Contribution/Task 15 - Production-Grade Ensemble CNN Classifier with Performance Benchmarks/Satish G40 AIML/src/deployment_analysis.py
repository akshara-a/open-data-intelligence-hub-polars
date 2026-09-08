from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.config import RESULTS_DIR


def analyze_deployment_value(metrics_df: pd.DataFrame, baseline_accuracy: float) -> Dict[str, float | str]:
    """Compare ensemble gains against latency, memory, throughput, and size costs."""
    best = metrics_df.sort_values("Accuracy", ascending=False).iloc[0]
    accuracy_gain = float(best["Accuracy"] - baseline_accuracy)
    latency_ratio = float(best["Latency"] / metrics_df.loc[metrics_df["Model"] == "CNN 1", "Latency"].iloc[0]) if "CNN 1" in metrics_df["Model"].values else 1.0
    memory_ratio = float(best["Memory Usage"] / metrics_df.loc[metrics_df["Model"] == "CNN 1", "Memory Usage"].iloc[0]) if "CNN 1" in metrics_df["Model"].values else 1.0
    size_ratio = float(best["Model Size"] / metrics_df.loc[metrics_df["Model"] == "CNN 1", "Model Size"].iloc[0]) if "CNN 1" in metrics_df["Model"].values else 1.0
    throughput_drop = float((1.0 - (best["Throughput"] / metrics_df.loc[metrics_df["Model"] == "CNN 1", "Throughput"].iloc[0])) * 100.0) if "CNN 1" in metrics_df["Model"].values else 0.0

    if accuracy_gain > 0 and latency_ratio <= 2 and memory_ratio <= 2 and size_ratio <= 2 and throughput_drop < 50:
        recommendation = "Deploy ensemble"
    else:
        recommendation = "Keep individual CNNs"

    return {
        "accuracy_gain": accuracy_gain,
        "latency_ratio": latency_ratio,
        "memory_ratio": memory_ratio,
        "size_ratio": size_ratio,
        "throughput_drop_percent": throughput_drop,
        "recommendation": recommendation,
    }


def export_deployment_summary(metrics_df: pd.DataFrame, baseline_accuracy: float, output_path: str | Path | None = None) -> Dict[str, float | str]:
    summary = analyze_deployment_value(metrics_df, baseline_accuracy)
    if output_path is None:
        output_path = RESULTS_DIR / "deployment_analysis.json"
    else:
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    return summary
