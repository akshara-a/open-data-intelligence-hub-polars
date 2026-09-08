from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO

from sklearn.model_selection import train_test_split


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "weather_data" / "dataset"

IMAGE_SIZE = (128, 128)

NUM_CLASSES = 5

RANDOM_STATE = 42


# ============================================================
# Load Parquet Dataset
# ============================================================

def load_weather_dataframe():
    """
    Load the WeatherNet-05 Parquet files into one DataFrame.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing image bytes and labels.
    """

    parquet_files = sorted(DATA_DIR.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(
            f"No Parquet files found in: {DATA_DIR}"
        )

    print(f"Found {len(parquet_files)} Parquet files.")

    dataframes = []

    for file in parquet_files:
        print(f"Loading: {file.name}")

        df = pd.read_parquet(
            file,
            columns=["image", "label"]
        )

        dataframes.append(df)

    dataframe = pd.concat(
        dataframes,
        ignore_index=True
    )

    print(f"Total images loaded: {len(dataframe)}")

    return dataframe


# ============================================================
# Decode Images
# ============================================================

def decode_image(image_data):
    """
    Decode an image stored as JPEG bytes.

    WeatherNet stores images in dictionaries such as:

        {'bytes': b'...'}

    Returns
    -------
    numpy.ndarray
        RGB image.
    """

    if isinstance(image_data, dict):
        image_bytes = image_data["bytes"]
    else:
        image_bytes = image_data

    image = Image.open(
        BytesIO(image_bytes)
    ).convert("RGB")

    image = image.resize(
        IMAGE_SIZE,
        Image.Resampling.BILINEAR
    )

    image = np.asarray(
        image,
        dtype=np.float32
    )

    # Normalize from [0, 255] to [0, 1]
    image /= 255.0

    return image


# ============================================================
# Prepare Images and Labels
# ============================================================

def prepare_arrays(dataframe):
    """
    Decode all images and create NumPy arrays.
    """

    print("\nDecoding images...")

    images = []
    labels = []

    total = len(dataframe)

    for index, row in dataframe.iterrows():

        image = decode_image(row["image"])

        images.append(image)
        labels.append(int(row["label"]))

        if (index + 1) % 1000 == 0:
            print(
                f"Processed {index + 1}/{total} images"
            )

    X = np.asarray(
        images,
        dtype=np.float32
    )

    y = np.asarray(
        labels,
        dtype=np.int64
    )

    print("\nImage array shape:", X.shape)
    print("Label array shape:", y.shape)

    return X, y


# ============================================================
# Stratified Train / Validation / Test Split
# ============================================================

def create_splits(X, y):
    """
    Create a stratified 70/15/15 split.

    First:
        70% train
        30% temporary

    Then:
        15% validation
        15% test
    """

    # First split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Split temporary set equally
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp
    )

    print("\nDataset split:")
    print("----------------------------")
    print(f"Training:   {len(X_train)}")
    print(f"Validation: {len(X_val)}")
    print(f"Test:       {len(X_test)}")
    print("----------------------------")
    print(f"Total:      {len(X_train) + len(X_val) + len(X_test)}")

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )


# ============================================================
# Display Class Distribution
# ============================================================

def show_class_distribution(y, name):
    """
    Display class distribution.
    """

    unique, counts = np.unique(
        y,
        return_counts=True
    )

    print(f"\n{name} class distribution:")

    for label, count in zip(unique, counts):

        percentage = (
            count / len(y)
        ) * 100

        print(
            f"  Class {label}: "
            f"{count} images "
            f"({percentage:.2f}%)"
        )


# ============================================================
# Main Pipeline
# ============================================================

def load_and_prepare_dataset():

    print("=" * 60)
    print("WEATHERNET-05 DATA PIPELINE")
    print("=" * 60)

    # Load dataframe
    dataframe = load_weather_dataframe()

    # Basic validation
    if dataframe["label"].nunique() != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} classes, "
            f"found {dataframe['label'].nunique()}"
        )

    print("\nOverall class distribution:")

    print(
        dataframe["label"]
        .value_counts()
        .sort_index()
    )

    # Decode images
    X, y = prepare_arrays(dataframe)

    # Create splits
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = create_splits(X, y)

    # Show distributions
    show_class_distribution(
        y_train,
        "Training"
    )

    show_class_distribution(
        y_val,
        "Validation"
    )

    show_class_distribution(
        y_test,
        "Test"
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    load_and_prepare_dataset()