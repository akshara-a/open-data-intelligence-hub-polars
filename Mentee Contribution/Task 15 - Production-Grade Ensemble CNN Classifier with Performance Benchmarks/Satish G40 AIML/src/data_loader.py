import os
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10

from src.config import BATCH_SIZE, DATA_DIR, IMG_HEIGHT, IMG_WIDTH, INPUT_SHAPE, NUM_CLASSES, SEED


def load_cifar10() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load CIFAR-10 and return train/validation/test arrays."""
    (x_train_full, y_train_full), (x_test, y_test) = cifar10.load_data()

    x_train_full = x_train_full.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    y_train_full = y_train_full.reshape(-1)
    y_test = y_test.reshape(-1)

    val_size = int(0.1 * x_train_full.shape[0])
    indices = np.arange(x_train_full.shape[0])
    rng = np.random.default_rng(SEED)
    rng.shuffle(indices)

    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    x_train = x_train_full[train_idx]
    y_train = y_train_full[train_idx]
    x_val = x_train_full[val_idx]
    y_val = y_train_full[val_idx]

    x_train = tf.convert_to_tensor(x_train)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)
    x_val = tf.convert_to_tensor(x_val)
    y_val = tf.convert_to_tensor(y_val, dtype=tf.int32)
    x_test = tf.convert_to_tensor(x_test)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.int32)

    return x_train, y_train, x_val, y_val, x_test, y_test


def build_datasets(
    x_train: tf.Tensor,
    y_train: tf.Tensor,
    x_val: tf.Tensor,
    y_val: tf.Tensor,
    x_test: tf.Tensor,
    y_test: tf.Tensor,
):
    """Create tf.data datasets with shuffling and batching."""
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(5000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds, test_ds


def save_dataset_preview(x_train: tf.Tensor, y_train: tf.Tensor, x_val: tf.Tensor, y_val: tf.Tensor, x_test: tf.Tensor, y_test: tf.Tensor) -> None:
    """Persist a lightweight dataset overview for debugging and reproducibility."""
    dataset_info = {
        "train_shape": list(x_train.shape),
        "train_labels_shape": list(y_train.shape),
        "val_shape": list(x_val.shape),
        "val_labels_shape": list(y_val.shape),
        "test_shape": list(x_test.shape),
        "test_labels_shape": list(y_test.shape),
        "input_shape": list(INPUT_SHAPE),
        "num_classes": NUM_CLASSES,
    }

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(DATA_DIR / "dataset_summary.json", "w", encoding="utf-8") as file:
        import json
        json.dump(dataset_info, file, indent=2)
