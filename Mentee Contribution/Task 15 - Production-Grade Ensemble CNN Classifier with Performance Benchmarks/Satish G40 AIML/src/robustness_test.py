import json
import os
from typing import Dict, List

import numpy as np
import tensorflow as tf
from PIL import Image, ImageFilter

from src.config import RESULTS_DIR


def apply_rotation(x: np.ndarray, degrees: int = 15) -> np.ndarray:
    rotated = []
    for image in x:
        pil_image = Image.fromarray((image * 255).astype("uint8")).rotate(degrees)
        rotated.append(np.asarray(pil_image).astype("float32") / 255.0)
    return np.stack(rotated)


def apply_blur(x: np.ndarray, radius: int = 1) -> np.ndarray:
    blurred = []
    for image in x:
        pil_image = Image.fromarray((image * 255).astype("uint8")).filter(ImageFilter.GaussianBlur(radius=radius))
        blurred.append(np.asarray(pil_image).astype("float32") / 255.0)
    return np.stack(blurred)


def apply_noise(x: np.ndarray, std: float = 0.05) -> np.ndarray:
    noise = np.random.normal(0.0, std, size=x.shape).astype("float32")
    return np.clip(x + noise, 0.0, 1.0)


def apply_darkening(x: np.ndarray, factor: float = 0.6) -> np.ndarray:
    return np.clip(x * factor, 0.0, 1.0)


def apply_brightening(x: np.ndarray, factor: float = 1.4) -> np.ndarray:
    return np.clip(x * factor, 0.0, 1.0)


def apply_crop(x: np.ndarray, crop_fraction: float = 0.15) -> np.ndarray:
    cropped = []
    for image in x:
        h, w = image.shape[:2]
        crop_size = int(min(h, w) * crop_fraction)
        top = crop_size // 2
        left = crop_size // 2
        patch = image[top : h - top, left : w - left]
        pad_top = max(0, top)
        pad_left = max(0, left)
        padded = np.pad(patch, ((pad_top, crop_size - pad_top), (pad_left, crop_size - pad_left), (0, 0)), mode="edge")
        if padded.shape[:2] != (h, w):
            padded = np.resize(padded, (h, w, 3))
        cropped.append(padded.astype("float32"))
    return np.stack(cropped)


def evaluate_robustness(model: tf.keras.Model, x_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    variants = {
        "original": x_test,
        "rotated": apply_rotation(x_test, degrees=15),
        "blurred": apply_blur(x_test, radius=1),
        "noisy": apply_noise(x_test, std=0.05),
        "darkened": apply_darkening(x_test, factor=0.6),
        "brightened": apply_brightening(x_test, factor=1.4),
        "cropped": apply_crop(x_test, crop_fraction=0.15),
    }

    results = {}
    for variant_name, variant in variants.items():
        preds = model.predict(variant, verbose=0)
        y_pred = np.argmax(preds, axis=1)
        acc = float(np.mean(y_pred == y_test))
        results[variant_name] = acc
    return results


def compare_robustness(model_names: List[str], x_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
    results = {}
    for name in model_names:
        model = tf.keras.models.load_model(f"models/{name}.keras")
        results[name] = evaluate_robustness(model, x_test, y_test)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_DIR / "robustness_summary.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    return results
