from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.config import RESULTS_DIR
from src.deployment_analysis import export_deployment_summary


def build_final_comparison_table(
    individual_metrics: Dict[str, Dict[str, float]],
    ensemble_metrics: Dict[str, Dict[str, float]],
    benchmark_results: Dict[str, Dict[str, float]],
    robustness_results: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    rows = []
    model_labels = {
        "baseline_cnn": "CNN 1",
        "regularized_cnn": "CNN 2",
        "deep_cnn": "CNN 3",
    }

    for model_name, metrics in individual_metrics.items():
        label = model_labels.get(model_name, model_name)
        robustness = robustness_results.get(model_name, {})
        robustness_score = float(sum(robustness.values()) / len(robustness)) if robustness else 0.0
        benchmark = benchmark_results.get(model_name, {})
        rows.append({
            "Model": label,
            "Accuracy": float(metrics.get("accuracy", 0.0)),
            "Precision": float(metrics.get("precision", 0.0)),
            "Recall": float(metrics.get("recall", 0.0)),
            "F1-score": float(metrics.get("f1_score", 0.0)),
            "Parameters": int(benchmark.get("parameter_count", 0)),
            "Model Size": float(benchmark.get("model_size_mb", 0.0)),
            "Latency": float(benchmark.get("average_latency_ms", 0.0)),
            "Throughput": float(benchmark.get("throughput_images_per_second", 0.0)),
            "Memory Usage": float(benchmark.get("approx_memory_usage_mb", 0.0)),
            "Robustness": robustness_score,
        })

    for ensemble_name, metrics in ensemble_metrics.items():
        benchmark = benchmark_results.get(ensemble_name, {})
        robustness = robustness_results.get(ensemble_name, {})
        robustness_score = float(sum(robustness.values()) / len(robustness)) if robustness else 0.0
        rows.append({
            "Model": {
                "majority_voting": "Majority Voting",
                "soft_voting": "Soft Voting",
                "weighted_soft_voting": "Weighted Soft Voting",
            }.get(ensemble_name, ensemble_name),
            "Accuracy": float(metrics.get("accuracy", 0.0)),
            "Precision": float(metrics.get("precision", 0.0)),
            "Recall": float(metrics.get("recall", 0.0)),
            "F1-score": float(metrics.get("f1_score", 0.0)),
            "Parameters": int(benchmark.get("parameter_count", 0)),
            "Model Size": float(benchmark.get("model_size_mb", 0.0)),
            "Latency": float(benchmark.get("average_latency_ms", 0.0)),
            "Throughput": float(benchmark.get("throughput_images_per_second", 0.0)),
            "Memory Usage": float(benchmark.get("approx_memory_usage_mb", 0.0)),
            "Robustness": robustness_score,
        })

    comparison = pd.DataFrame(rows)
    return comparison.sort_values("Accuracy", ascending=False, ignore_index=True)


def generate_final_report(
    individual_metrics: Dict[str, Dict[str, float]],
    ensemble_metrics: Dict[str, Dict[str, float]],
    benchmark_results: Dict[str, Dict[str, float]],
    robustness_results: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    comparison_df = build_final_comparison_table(individual_metrics, ensemble_metrics, benchmark_results, robustness_results)
    output_path = RESULTS_DIR / "final_comparison.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(output_path, index=False)

    baseline_accuracy = float(comparison_df.loc[comparison_df["Model"] == "CNN 1", "Accuracy"].iloc[0]) if "CNN 1" in comparison_df["Model"].values else float(comparison_df["Accuracy"].min())
    summary = export_deployment_summary(comparison_df, baseline_accuracy, RESULTS_DIR / "deployment_analysis.json")
    with open(RESULTS_DIR / "deployment_decision.txt", "w", encoding="utf-8") as file:
        file.write(f"Recommendation: {summary['recommendation']}\n")
        file.write(f"Accuracy gain: {summary['accuracy_gain']:.4f}\n")
        file.write(f"Latency ratio: {summary['latency_ratio']:.4f}\n")
        file.write(f"Memory ratio: {summary['memory_ratio']:.4f}\n")
        file.write(f"Size ratio: {summary['size_ratio']:.4f}\n")
    return comparison_df


if __name__ == "__main__":
    with open(RESULTS_DIR / "training_summary.json", "r", encoding="utf-8") as file:
        training_summary = json.load(file)
    with open(RESULTS_DIR / "benchmark_summary.json", "r", encoding="utf-8") as file:
        benchmark = json.load(file)
    with open(RESULTS_DIR / "robustness_summary.json", "r", encoding="utf-8") as file:
        robustness = json.load(file)
    with open(RESULTS_DIR / "ensemble_metrics.json", "r", encoding="utf-8") as file:
        ensembles = json.load(file)

    comparison_df = generate_final_report(training_summary["metrics"], ensembles, benchmark, robustness)
    print(comparison_df.to_string(index=False))
