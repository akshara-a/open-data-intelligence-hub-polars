from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def get_augmentation():
    """Return the documented training augmentation pipeline."""
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.10),
            layers.RandomZoom(0.10),
            layers.RandomContrast(0.10),
        ],
        name="data_augmentation",
    )


def load_datasets():
    """Load train, validation, and test datasets."""
    train_dir = DATA_DIR / "train"
    val_dir = DATA_DIR / "val"
    test_dir = DATA_DIR / "test"

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        class_names=["ok_front", "def_front"],
        shuffle=True,
        seed=SEED,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        class_names=["ok_front", "def_front"],
        shuffle=False,
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        class_names=["ok_front", "def_front"],
        shuffle=False,
    )

    normalization = layers.Rescaling(1.0 / 255)

    augmentation = get_augmentation()

    train_ds = train_ds.map(
        lambda images, labels: (
            normalization(augmentation(images, training=True)),
            labels,
        ),
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


if __name__ == "__main__":
    train, val, test = load_datasets()

    print("Training batches:", tf.data.experimental.cardinality(train).numpy())
    print("Validation batches:", tf.data.experimental.cardinality(val).numpy())
    print("Test batches:", tf.data.experimental.cardinality(test).numpy())