import json
import os
import time
from typing import Dict, List

import numpy as np
import tensorflow as tf

from src.config import MODEL_PATHS, RESULTS_DIR


def model_parameter_count(model: tf.keras.Model) -> int:
    return int(model.count_params())


def model_size_mb(model_path: str) -> float:
    return os.path.getsize(model_path) / (1024 * 1024)


def benchmark_model(model: tf.keras.Model, x: np.ndarray, repeats: int = 20) -> Dict[str, float]:
    latencies = []
    for _ in range(repeats):
        start = time.perf_counter()
        _ = model.predict(x, verbose=0)
        elapsed = time.perf_counter() - start
        latencies.append(elapsed * 1000.0)

    avg_latency = float(np.mean(latencies))
    min_latency = float(np.min(latencies))
    max_latency = float(np.max(latencies))
    throughput = float(len(x) / (np.mean(latencies) / 1000.0)) if np.mean(latencies) > 0 else 0.0

    model_path = getattr(model, "_path", None)
    size_mb = model_size_mb(model_path) if model_path else 0.0
    params = model_parameter_count(model)
    memory_usage_mb = float((params * 4) / (1024 * 1024))

    return {
        "average_latency_ms": avg_latency,
        "min_latency_ms": min_latency,
        "max_latency_ms": max_latency,
        "throughput_images_per_second": throughput,
        "model_size_mb": size_mb,
        "parameter_count": params,
        "approx_memory_usage_mb": memory_usage_mb,
    }


def benchmark_all_models(model_names: List[str], x_test: np.ndarray) -> Dict[str, Dict[str, float]]:
    results = {}
    for model_name in model_names:
        model = tf.keras.models.load_model(str(MODEL_PATHS[model_name]))
        model._path = str(MODEL_PATHS[model_name])
        results[model_name] = benchmark_model(model, x_test[:256], repeats=10)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    metrics_path = RESULTS_DIR / "benchmark_summary.json"
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    return results
