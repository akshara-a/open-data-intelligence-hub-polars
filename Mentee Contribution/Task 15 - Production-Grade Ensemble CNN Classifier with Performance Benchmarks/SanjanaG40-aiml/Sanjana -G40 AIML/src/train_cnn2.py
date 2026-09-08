"""
WeatherNet-05
CNN 2 Training Pipeline
"""

import os
import sys
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras


# ============================================================
# Project Root
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# Imports
# ============================================================

from src.preprocessing import build_datasets

from src.models.cnn2 import (
    build_cnn2,
    compile_cnn2
)


# ============================================================
# Configuration
# ============================================================

SEED = 42

BATCH_SIZE = 32
EPOCHS = 30

NUM_CLASSES = 5

INPUT_SHAPE = (
    128,
    128,
    3
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "cnn2.keras"
)

HISTORY_PATH = os.path.join(
    RESULTS_DIR,
    "training_history_cnn2.csv"
)


# ============================================================
# Reproducibility
# ============================================================

os.environ[
    "PYTHONHASHSEED"
] = str(SEED)

random.seed(SEED)

np.random.seed(SEED)

tf.random.set_seed(SEED)


# ============================================================
# Directories
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# Class Weights
# ============================================================

def calculate_class_weights():

    class_counts = {
        0: 4691,
        1: 883,
        2: 1349,
        3: 1312,
        4: 4392,
    }

    total = sum(
        class_counts.values()
    )

    weights = {}

    for class_id, count in class_counts.items():

        weights[class_id] = (
            total /
            (NUM_CLASSES * count)
        )

    return weights


# ============================================================
# Training Plots
# ============================================================

def save_training_plots(history):

    import matplotlib.pyplot as plt

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        history.history["accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.title(
        "CNN 2 - Training vs Validation Accuracy"
    )

    plt.legend()

    plt.grid(True)

    path = os.path.join(
        RESULTS_DIR,
        "training_accuracy_cnn2.png"
    )

    plt.savefig(
        path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {path}"
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        history.history["loss"],
        label="Training Loss"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title(
        "CNN 2 - Training vs Validation Loss"
    )

    plt.legend()

    plt.grid(True)

    path = os.path.join(
        RESULTS_DIR,
        "training_loss_cnn2.png"
    )

    plt.savefig(
        path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {path}"
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 65)

    print(
        "WEATHERNET-05 — CNN 2 TRAINING"
    )

    print("=" * 65)

    print("\nConfiguration")

    print("-" * 65)

    print(
        f"Input shape : {INPUT_SHAPE}"
    )

    print(
        f"Batch size  : {BATCH_SIZE}"
    )

    print(
        f"Epochs      : {EPOCHS}"
    )

    print(
        f"Classes     : {NUM_CLASSES}"
    )

    print(
        f"Seed        : {SEED}"
    )

    # ========================================================
    # Load Dataset
    # ========================================================

    print(
        "\nLoading WeatherNet-05 dataset..."
    )

    (
        train_ds,
        val_ds,
        test_ds,
        dataframe,
        train_indices,
        val_indices,
        test_indices
    ) = build_datasets(
        batch_size=BATCH_SIZE
    )

    print(
        "Dataset loaded successfully! ✅"
    )

    # ========================================================
    # Build Model
    # ========================================================

    print(
        "\nBuilding CNN 2..."
    )

    model = build_cnn2(
        input_shape=INPUT_SHAPE,
        num_classes=NUM_CLASSES
    )

    model = compile_cnn2(
        model
    )

    print(
        "\nModel:"
    )

    model.summary()

    # ========================================================
    # Class Weights
    # ========================================================

    class_weights = (
        calculate_class_weights()
    )

    print(
        "\nClass weights:"
    )

    for class_id, weight in class_weights.items():

        print(
            f"  Class {class_id}: "
            f"{weight:.4f}"
        )

    # ========================================================
    # Callbacks
    # ========================================================

    callbacks = [

        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),

        keras.callbacks.ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),

        keras.callbacks.CSVLogger(
            HISTORY_PATH,
            append=False
        ),

        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # ========================================================
    # Train
    # ========================================================

    print(
        "\nStarting CNN 2 training..."
    )

    print("=" * 65)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    # ========================================================
    # Plots
    # ========================================================

    print(
        "\nCreating training plots..."
    )

    save_training_plots(
        history
    )

    # ========================================================
    # Evaluate Best Model
    # ========================================================

    print(
        "\nBest CNN 2 validation performance:"
    )

    print("-" * 65)

    best_model = keras.models.load_model(
        MODEL_PATH
    )

    val_loss, val_accuracy = (
        best_model.evaluate(
            val_ds,
            verbose=1
        )
    )

    print(
        f"\nValidation Loss     : "
        f"{val_loss:.4f}"
    )

    print(
        f"Validation Accuracy : "
        f"{val_accuracy:.4%}"
    )

    # ========================================================
    # Complete
    # ========================================================

    print(
        "\n" + "=" * 65
    )

    print(
        "CNN 2 TRAINING COMPLETE! ✅"
    )

    print(
        "=" * 65
    )

    print(
        "\nBest model saved to:"
    )

    print(
        MODEL_PATH
    )

    print(
        "\nTraining history saved to:"
    )

    print(
        HISTORY_PATH
    )

    print(
        "\nNext step:"
    )

    print(
        "Evaluate CNN 2 on the test dataset."
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    main()