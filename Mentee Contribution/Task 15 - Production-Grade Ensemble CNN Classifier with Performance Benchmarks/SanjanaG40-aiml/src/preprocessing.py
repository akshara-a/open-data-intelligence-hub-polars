"""
WeatherNet-05
Production TensorFlow Data Pipeline

Handles:
- Hugging Face Parquet image dictionaries
- JPEG/PNG/GIF/BMP/WebP image bytes
- Invalid image filtering
- Stratified train/validation/test split
- tf.data pipeline
- Batch loading
- Prefetching
"""

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "weather_data"
    / "dataset"
)

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
CHANNELS = 3

NUM_CLASSES = 5

DEFAULT_BATCH_SIZE = 32

RANDOM_STATE = 42


# ============================================================
# Dataset Loading
# ============================================================

def load_metadata():
    """
    Load WeatherNet-05 Parquet files.

    Expected columns:

        image
        label
    """

    parquet_files = sorted(
        DATA_DIR.glob("*.parquet")
    )

    if not parquet_files:

        raise FileNotFoundError(
            f"No Parquet files found at:\n{DATA_DIR}"
        )

    frames = []

    for file in parquet_files:

        print(f"Reading: {file.name}")

        dataframe = pd.read_parquet(
            file,
            columns=[
                "image",
                "label"
            ]
        )

        frames.append(dataframe)

    dataframe = pd.concat(
        frames,
        ignore_index=True
    )

    print(
        f"Total images: {len(dataframe)}"
    )

    return dataframe


# ============================================================
# Image Byte Extraction
# ============================================================

def extract_image_bytes(image_object):
    """
    Extract raw image bytes from a Hugging Face image object.

    WeatherNet-05 images normally look like:

        {
            "bytes": b"...",
            "path": "image.jpg"
        }

    Also supports raw bytes.
    """

    if isinstance(
        image_object,
        dict
    ):

        image_bytes = image_object.get(
            "bytes"
        )

        if image_bytes is None:

            raise ValueError(
                "Image dictionary does not contain "
                "'bytes'."
            )

        return image_bytes

    if isinstance(
        image_object,
        bytes
    ):

        return image_object

    raise TypeError(
        "Unexpected image type: "
        f"{type(image_object)}"
    )


# ============================================================
# Image Validation
# ============================================================

def is_valid_image_bytes(image_bytes):
    """
    Validate common image file signatures.

    Supported:

    JPEG
    PNG
    GIF
    BMP
    WEBP

    This is intentionally performed before TensorFlow
    receives the image bytes.
    """

    if not image_bytes:

        return False

    # --------------------------------------------------------
    # JPEG
    # --------------------------------------------------------

    if image_bytes.startswith(
        b"\xff\xd8\xff"
    ):

        return True

    # --------------------------------------------------------
    # PNG
    # --------------------------------------------------------

    if image_bytes.startswith(
        b"\x89PNG\r\n\x1a\n"
    ):

        return True

    # --------------------------------------------------------
    # GIF
    # --------------------------------------------------------

    if image_bytes.startswith(
        (
            b"GIF87a",
            b"GIF89a"
        )
    ):

        return True

    # --------------------------------------------------------
    # BMP
    # --------------------------------------------------------

    if image_bytes.startswith(
        b"BM"
    ):

        return True

    # --------------------------------------------------------
    # WEBP
    # --------------------------------------------------------

    if (
        len(image_bytes) >= 12
        and image_bytes[:4] == b"RIFF"
        and image_bytes[8:12] == b"WEBP"
    ):

        return True

    return False


# ============================================================
# Clean Dataset
# ============================================================

def clean_metadata(dataframe):
    """
    Remove malformed/unsupported image records.

    Invalid records are removed before train/validation/test
    splitting so they can never reach TensorFlow.
    """

    print("\nValidating image records...")

    valid_indices = []
    invalid_indices = []

    for index, image_object in enumerate(
        dataframe["image"]
    ):

        try:

            image_bytes = extract_image_bytes(
                image_object
            )

            if is_valid_image_bytes(
                image_bytes
            ):

                valid_indices.append(index)

            else:

                invalid_indices.append(index)

        except Exception:

            invalid_indices.append(index)

    cleaned_dataframe = dataframe.iloc[
        valid_indices
    ].reset_index(
        drop=True
    )

    print(
        f"Valid images  : {len(valid_indices)}"
    )

    print(
        f"Invalid images: {len(invalid_indices)}"
    )

    if invalid_indices:

        print(
            "\nInvalid image records removed:"
        )

        print(
            invalid_indices[:20]
        )

    print(
        f"\nClean dataset size: "
        f"{len(cleaned_dataframe)}"
    )

    return cleaned_dataframe


# ============================================================
# Convert Images to Raw Bytes
# ============================================================

def convert_images_to_bytes(dataframe):
    """
    Convert image objects into raw byte strings.
    """

    print(
        "\nExtracting image bytes..."
    )

    image_bytes = []

    for image_object in dataframe[
        "image"
    ]:

        image_bytes.append(
            extract_image_bytes(
                image_object
            )
        )

    labels = dataframe[
        "label"
    ].astype(
        np.int32
    ).to_numpy()

    return image_bytes, labels


# ============================================================
# Decode Image
# ============================================================

def decode_image(image_bytes):
    """
    Decode image bytes using TensorFlow's generic decoder.

    Supports JPEG, PNG, GIF, BMP and WebP.
    """

    image = tf.io.decode_image(
        image_bytes,
        channels=CHANNELS,
        expand_animations=False
    )

    image = tf.image.resize(
        image,
        [
            IMAGE_HEIGHT,
            IMAGE_WIDTH
        ]
    )

    image = tf.cast(
        image,
        tf.float32
    )

    image = image / 255.0

    image.set_shape(
        [
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            CHANNELS
        ]
    )

    return image


# ============================================================
# TensorFlow Dataset
# ============================================================

def create_tf_dataset(
    image_bytes,
    labels,
    batch_size=DEFAULT_BATCH_SIZE,
    training=False
):
    """
    Create optimized tf.data.Dataset.
    """

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            image_bytes,
            labels
        )
    )

    # --------------------------------------------------------
    # Shuffle training data
    # --------------------------------------------------------

    if training:

        dataset = dataset.shuffle(
            buffer_size=len(labels),
            seed=RANDOM_STATE,
            reshuffle_each_iteration=True
        )

    # --------------------------------------------------------
    # Decode
    # --------------------------------------------------------

    def process(
        image,
        label
    ):

        image = decode_image(
            image
        )

        return image, label

    dataset = dataset.map(
        process,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # --------------------------------------------------------
    # Batch
    # --------------------------------------------------------

    dataset = dataset.batch(
        batch_size
    )

    # --------------------------------------------------------
    # Prefetch
    # --------------------------------------------------------

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


# ============================================================
# Stratified Split
# ============================================================

def create_splits(
    labels
):
    """
    Create:

        70% training
        15% validation
        15% testing

    using stratification.
    """

    indices = np.arange(
        len(labels)
    )

    train_indices, temp_indices = train_test_split(
        indices,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=labels
    )

    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=labels[temp_indices]
    )

    return (
        train_indices,
        val_indices,
        test_indices
    )


# ============================================================
# Class Distribution
# ============================================================

def print_class_distribution(
    labels,
    name
):
    """
    Print class counts and percentages.
    """

    print(
        f"\n{name} class distribution:"
    )

    unique, counts = np.unique(
        labels,
        return_counts=True
    )

    total = len(labels)

    for class_id, count in zip(
        unique,
        counts
    ):

        percentage = (
            count / total
        ) * 100

        print(
            f"  Class {class_id}: "
            f"{count} images "
            f"({percentage:.2f}%)"
        )


# ============================================================
# Complete Dataset Builder
# ============================================================

def build_datasets(
    batch_size=DEFAULT_BATCH_SIZE
):
    """
    Build complete WeatherNet-05 pipeline.

    Returns:

        train_dataset
        val_dataset
        test_dataset
        metadata
        train_indices
        val_indices
        test_indices
    """

    print("=" * 60)
    print(
        "WEATHERNET-05 TF.DATA PIPELINE"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    dataframe = load_metadata()

    # --------------------------------------------------------
    # Remove malformed images
    # --------------------------------------------------------

    dataframe = clean_metadata(
        dataframe
    )

    # --------------------------------------------------------
    # Extract bytes
    # --------------------------------------------------------

    image_bytes, labels = (
        convert_images_to_bytes(
            dataframe
        )
    )

    # --------------------------------------------------------
    # Validate classes
    # --------------------------------------------------------

    unique_classes = np.unique(
        labels
    )

    print(
        "\nClasses found:",
        unique_classes
    )

    if len(unique_classes) != NUM_CLASSES:

        raise ValueError(
            f"Expected {NUM_CLASSES} classes, "
            f"found {len(unique_classes)}"
        )

    # --------------------------------------------------------
    # Overall distribution
    # --------------------------------------------------------

    print_class_distribution(
        labels,
        "Overall"
    )

    # --------------------------------------------------------
    # Stratified split
    # --------------------------------------------------------

    (
        train_indices,
        val_indices,
        test_indices
    ) = create_splits(
        labels
    )

    print(
        "\nDataset split:"
    )

    print("-" * 30)

    print(
        f"Training:   "
        f"{len(train_indices)}"
    )

    print(
        f"Validation: "
        f"{len(val_indices)}"
    )

    print(
        f"Test:       "
        f"{len(test_indices)}"
    )

    print("-" * 30)

    # --------------------------------------------------------
    # Split images
    # --------------------------------------------------------

    train_images = [
        image_bytes[i]
        for i in train_indices
    ]

    val_images = [
        image_bytes[i]
        for i in val_indices
    ]

    test_images = [
        image_bytes[i]
        for i in test_indices
    ]

    # --------------------------------------------------------
    # Split labels
    # --------------------------------------------------------

    train_labels = labels[
        train_indices
    ]

    val_labels = labels[
        val_indices
    ]

    test_labels = labels[
        test_indices
    ]

    # --------------------------------------------------------
    # Class distributions
    # --------------------------------------------------------

    print_class_distribution(
        train_labels,
        "Training"
    )

    print_class_distribution(
        val_labels,
        "Validation"
    )

    print_class_distribution(
        test_labels,
        "Test"
    )

    # --------------------------------------------------------
    # TensorFlow datasets
    # --------------------------------------------------------

    train_dataset = create_tf_dataset(
        train_images,
        train_labels,
        batch_size=batch_size,
        training=True
    )

    val_dataset = create_tf_dataset(
        val_images,
        val_labels,
        batch_size=batch_size,
        training=False
    )

    test_dataset = create_tf_dataset(
        test_images,
        test_labels,
        batch_size=batch_size,
        training=False
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        dataframe,
        train_indices,
        val_indices,
        test_indices
    )


# ============================================================
# Training-Compatible Loader
# ============================================================

def load_weathernet_datasets(
    batch_size=DEFAULT_BATCH_SIZE
):
    """
    Convenience function used by train.py.

    Returns only:

        train_dataset
        validation_dataset
        test_dataset
    """

    (
        train_dataset,
        val_dataset,
        test_dataset,
        _,
        _,
        _,
        _
    ) = build_datasets(
        batch_size=batch_size
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset
    )


# ============================================================
# Pipeline Test
# ============================================================

if __name__ == "__main__":

    (
        train_dataset,
        val_dataset,
        test_dataset,
        dataframe,
        train_indices,
        val_indices,
        test_indices
    ) = build_datasets(
        batch_size=DEFAULT_BATCH_SIZE
    )

    print(
        "\nTesting training batch..."
    )

    images, labels = next(
        iter(train_dataset)
    )

    print(
        "Image batch shape:",
        images.shape
    )

    print(
        "Label batch shape:",
        labels.shape
    )

    print(
        "Pixel minimum:",
        float(
            tf.reduce_min(images)
        )
    )

    print(
        "Pixel maximum:",
        float(
            tf.reduce_max(images)
        )
    )

    print(
        "\nFirst 10 labels:"
    )

    print(
        labels[:10].numpy()
    )

    print(
        "\nPipeline test successful! ✅"
    )