import argparse
import time
from typing import Dict, List

import numpy as np
from PIL import Image

from src.config import CLASS_NAMES, MODEL_PATHS


def preprocess_image(image_path: str):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((32, 32))
    arr = np.asarray(img, dtype="float32") / 255.0
    return np.expand_dims(arr, axis=0)


def predict_single_image(image_path: str, threshold: float = 0.80) -> Dict[str, object]:
    models = {name: __import__("tensorflow").keras.models.load_model(str(path)) for name, path in MODEL_PATHS.items()}
    input_batch = preprocess_image(image_path)

    probabilities = [model.predict(input_batch, verbose=0)[0] for model in models.values()]
    soft_probs = np.mean(np.stack(probabilities), axis=0)
    predicted_index = int(np.argmax(soft_probs))
    confidence = float(soft_probs[predicted_index])

    start = time.perf_counter()
    _ = [model.predict(input_batch, verbose=0) for model in models.values()]
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / len(models)

    decision = "Accepted" if confidence >= threshold else "Manual Review"
    return {
        "predictedClass": CLASS_NAMES[predicted_index],
        "confidence": round(confidence, 4),
        "decision": decision,
        "inferenceTimeMs": round(elapsed_ms, 2),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict CIFAR-10 image class with an ensemble.")
    parser.add_argument("--image", required=True, help="Path to the image file.")
    parser.add_argument("--threshold", type=float, default=0.80, help="Confidence threshold for Accepted/Manual Review.")
    args = parser.parse_args()
    print(predict_single_image(args.image, args.threshold))
