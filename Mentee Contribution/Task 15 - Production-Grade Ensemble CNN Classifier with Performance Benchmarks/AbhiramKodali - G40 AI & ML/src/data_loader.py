from pathlib import Path

import tensorflow as tf


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42


def load_datasets():
    """Load training, validation, and test datasets."""

    train_dir = DATA_DIR / "train"
    val_dir = DATA_DIR / "val"
    test_dir = DATA_DIR / "test"

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=True,
        seed=SEED,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=False,
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=False,
    )

    normalization = tf.keras.layers.Rescaling(1.0 / 255)

    train_ds = train_ds.map(
        lambda images, labels: (normalization(images), labels),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    val_ds = val_ds.map(
        lambda images, labels: (normalization(images), labels),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    test_ds = test_ds.map(
        lambda images, labels: (normalization(images), labels),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds