import tensorflow as tf


def build_training_augmentation() -> tf.keras.Sequential:
    """Apply training-only augmentation for CIFAR-10 images."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        ],
        name="training_augmentation",
    )


def apply_augmentation_if_needed(train_ds: tf.data.Dataset, augmentation_model: tf.keras.Sequential):
    """Apply augmentation to the training dataset only."""
    return train_ds.map(lambda x, y: (augmentation_model(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
