import json
import os
from typing import Dict, List

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_recall_fscore_support

from src.augmentation import apply_augmentation_if_needed, build_training_augmentation
from src.config import CLASS_NAMES, EPOCHS, MODEL_PATHS, MODELS_DIR, RESULTS_DIR, SEED
from src.data_loader import build_datasets, load_cifar10, save_dataset_preview
from src.models.baseline_cnn import build_baseline_cnn
from src.models.deep_cnn import build_deep_cnn
from src.models.regularized_cnn import build_regularized_cnn


def compile_model(model: tf.keras.Model) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def create_callbacks(model_name: str):
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=7,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(MODEL_PATHS[model_name]),
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
        ),
    ]
    return callbacks


def evaluate_model(model: tf.keras.Model, x_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    metrics = model.evaluate(x_test, y_test, verbose=0, return_dict=True)
    predictions = model.predict(x_test, verbose=0)
    y_pred = np.argmax(predictions, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")

    return {
        "loss": float(metrics["loss"]),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_macro),
        "classification_report": classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }


def train_and_evaluate_all_models() -> Dict[str, Dict[str, object]]:
    tf.keras.utils.set_random_seed(SEED)
    np.random.seed(SEED)

    x_train, y_train, x_val, y_val, x_test, y_test = load_cifar10()
    save_dataset_preview(x_train, y_train, x_val, y_val, x_test, y_test)

    train_ds, val_ds, test_ds = build_datasets(x_train, y_train, x_val, y_val, x_test, y_test)
    augmentation_model = build_training_augmentation()
    train_ds = apply_augmentation_if_needed(train_ds, augmentation_model)

    model_builders = {
        "baseline_cnn": build_baseline_cnn,
        "regularized_cnn": build_regularized_cnn,
        "deep_cnn": build_deep_cnn,
    }

    results = {}
    model_histories = {}

    for model_name, builder in model_builders.items():
        model = compile_model(builder())
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            callbacks=create_callbacks(model_name),
            verbose=1,
        )
        model_histories[model_name] = history.history

        model_path = MODEL_PATHS[model_name]
        model.save(model_path)
        results[model_name] = evaluate_model(model, x_test, y_test)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_DIR / "training_summary.json", "w", encoding="utf-8") as file:
        json.dump({
            "models": list(model_histories.keys()),
            "histories": model_histories,
            "metrics": results
        }, file, indent=2)

    return results


if __name__ == "__main__":
    train_and_evaluate_all_models()
    print("Training completed and model metrics saved.")
