from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf

from data_loader import load_datasets
from model import build_model


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
PLOT_DIR = BASE_DIR / "plots"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def plot_training_history(history):
    """Save accuracy and loss curves."""
    epochs = range(1, len(history.history["accuracy"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.history["accuracy"], label="Training accuracy")
    plt.plot(epochs, history.history["val_accuracy"], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "accuracy_plot.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.history["loss"], label="Training loss")
    plt.plot(epochs, history.history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "loss_plot.png", dpi=150)
    plt.close()


def main():
    train_ds, val_ds, test_ds = load_datasets()

    model = build_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_DIR / "casting_defect_model.keras",
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=25,
        callbacks=callbacks,
        verbose=1,
    )

    plot_training_history(history)

    print("\nEvaluating the trained model on the test set...")
    results = model.evaluate(test_ds, verbose=1)

    for name, value in zip(model.metrics_names, results):
        print(f"{name}: {value:.4f}")

    print(f"\nBest model saved to: {MODEL_DIR / 'casting_defect_model.keras'}")
    print(f"Plots saved to: {PLOT_DIR}")


if __name__ == "__main__":
    main()