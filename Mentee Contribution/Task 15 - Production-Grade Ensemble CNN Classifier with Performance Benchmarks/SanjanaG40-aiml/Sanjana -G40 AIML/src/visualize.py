import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Allow imports from src/
sys.path.append(
    str(Path(__file__).resolve().parent)
)

from preprocessing import build_datasets
from augmentation import create_augmentation_pipeline


# ============================================================
# Configuration
# ============================================================

OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent
    / "results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Class Names
# ============================================================

# We have confirmed 5 numerical classes.
# We will replace these with the official WeatherNet-05
# names after verifying the dataset metadata.

CLASS_NAMES = [
    "Class 0",
    "Class 1",
    "Class 2",
    "Class 3",
    "Class 4",
]


# ============================================================
# Plot Original Images
# ============================================================

def save_sample_images(dataset):

    images, labels = next(
        iter(dataset)
    )

    plt.figure(
        figsize=(12, 8)
    )

    for i in range(12):

        plt.subplot(
            3,
            4,
            i + 1
        )

        plt.imshow(
            images[i].numpy()
        )

        label = int(
            labels[i].numpy()
        )

        plt.title(
            CLASS_NAMES[label]
        )

        plt.axis("off")

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "sample_images.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# Plot Augmented Images
# ============================================================

def save_augmented_images(dataset):

    augmentation = (
        create_augmentation_pipeline()
    )

    images, labels = next(
        iter(dataset)
    )

    augmented_images = augmentation(
        images,
        training=True
    )

    plt.figure(
        figsize=(12, 8)
    )

    for i in range(12):

        plt.subplot(
            3,
            4,
            i + 1
        )

        plt.imshow(
            augmented_images[i].numpy()
        )

        label = int(
            labels[i].numpy()
        )

        plt.title(
            CLASS_NAMES[label]
        )

        plt.axis("off")

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "augmented_images.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print(
        "Loading WeatherNet-05..."
    )

    (
        train_dataset,
        val_dataset,
        test_dataset,
        *_,
    ) = build_datasets()

    print(
        "\nCreating sample visualization..."
    )

    save_sample_images(
        train_dataset
    )

    print(
        "Creating augmentation visualization..."
    )

    save_augmented_images(
        train_dataset
    )

    print(
        "\nVisualization complete! ✅"
    )