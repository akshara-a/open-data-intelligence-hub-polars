"""
WeatherNet-05
Training Pipeline

Trains CNN 1 baseline model.
"""

import os
import sys
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras

# ---------------------------------------------------------
# Make project root importable
# ---------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------
# Project imports
# ---------------------------------------------------------

from src.models.baseline_cnn import (
    build_baseline_cnn,
    compile_model
)

from src.preprocessing import (
    load_weathernet_datasets
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

SEED = 42

BATCH_SIZE = 32
EPOCHS = 30

NUM_CLASSES = 5
INPUT_SHAPE = (128, 128, 3)

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
    "cnn_baseline.keras"
)

HISTORY_PATH = os.path.join(
    RESULTS_DIR,
    "training_history_cnn1.csv"
)


# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ---------------------------------------------------------
# Create directories
# ---------------------------------------------------------

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------
# Calculate class weights
# ---------------------------------------------------------

def calculate_class_weights():

    # WeatherNet-05 training distribution
    class_counts = {
        0: 4691,
        1: 883,
        2: 1349,
        3: 1312,
        4: 4392,
    }

    total = sum(class_counts.values())

    weights = {}

    for class_id, count in class_counts.items():

        weights[class_id] = (
            total /
            (NUM_CLASSES * count)
        )

    return weights


# ---------------------------------------------------------
# Plot training history
# ---------------------------------------------------------

def save_training_plots(history):

    import matplotlib.pyplot as plt

    # Accuracy
    plt.figure(figsize=(8, 5))

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
        "CNN 1 - Training vs Validation Accuracy"
    )

    plt.legend()
    plt.grid(True)

    path = os.path.join(
        RESULTS_DIR,
        "training_accuracy_cnn1.png"
    )

    plt.savefig(
        path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {path}")


    # Loss
    plt.figure(figsize=(8, 5))

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
        "CNN 1 - Training vs Validation Loss"
    )

    plt.legend()
    plt.grid(True)

    path = os.path.join(
        RESULTS_DIR,
        "training_loss_cnn1.png"
    )

    plt.savefig(
        path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {path}")


# ---------------------------------------------------------
# Main training function
# ---------------------------------------------------------

def main():

    print("=" * 65)
    print("WEATHERNET-05 — CNN 1 TRAINING")
    print("=" * 65)

    print("\nConfiguration")
    print("-" * 65)
    print(f"Input shape : {INPUT_SHAPE}")
    print(f"Batch size  : {BATCH_SIZE}")
    print(f"Epochs      : {EPOCHS}")
    print(f"Classes     : {NUM_CLASSES}")
    print(f"Seed        : {SEED}")

    # -----------------------------------------------------
    # Load datasets
    # -----------------------------------------------------

    print("\nLoading WeatherNet-05 dataset...")

    train_ds, val_ds, test_ds = load_weathernet_datasets(
        batch_size=BATCH_SIZE
    )

    print("Dataset loaded successfully! ✅")

    # -----------------------------------------------------
    # Build model
    # -----------------------------------------------------

    print("\nBuilding CNN 1...")

    model = build_baseline_cnn(
        input_shape=INPUT_SHAPE,
        num_classes=NUM_CLASSES
    )

    model = compile_model(model)

    print("\nModel:")
    model.summary()

    # -----------------------------------------------------
    # Class weights
    # -----------------------------------------------------

    class_weights = calculate_class_weights()

    print("\nClass weights:")
    for class_id, weight in class_weights.items():
        print(
            f"  Class {class_id}: {weight:.4f}"
        )

    # -----------------------------------------------------
    # Callbacks
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # Train
    # -----------------------------------------------------

    print("\nStarting training...")
    print("=" * 65)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    # -----------------------------------------------------
    # Save final training plots
    # -----------------------------------------------------

    print("\nCreating training plots...")

    save_training_plots(history)

    # -----------------------------------------------------
    # Evaluate validation set
    # -----------------------------------------------------

    print("\nBest model validation performance:")
    print("-" * 65)

    best_model = keras.models.load_model(
        MODEL_PATH
    )

    val_loss, val_accuracy = best_model.evaluate(
        val_ds,
        verbose=1
    )

    print(
        f"\nValidation Loss     : {val_loss:.4f}"
    )

    print(
        f"Validation Accuracy : {val_accuracy:.4%}"
    )

    # -----------------------------------------------------
    # Final message
    # -----------------------------------------------------

    print("\n" + "=" * 65)
    print("CNN 1 TRAINING COMPLETE! ✅")
    print("=" * 65)

    print(f"\nBest model saved to:")
    print(MODEL_PATH)

    print("\nTraining history saved to:")
    print(HISTORY_PATH)

    print("\nNext step:")
    print("Evaluate CNN 1 on the test dataset.")


if __name__ == "__main__":
    main()