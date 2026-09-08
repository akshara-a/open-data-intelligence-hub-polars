import tensorflow as tf
from tensorflow.keras import layers


def create_augmentation_pipeline():
    """
    Data augmentation applied only during training.
    """

    return tf.keras.Sequential(
        [
            layers.RandomFlip(
                mode="horizontal"
            ),

            layers.RandomRotation(
                factor=0.08
            ),

            layers.RandomZoom(
                height_factor=0.10,
                width_factor=0.10
            ),

            layers.RandomContrast(
                factor=0.10
            ),
        ],
        name="weather_augmentation"
    )


if __name__ == "__main__":

    augmentation = create_augmentation_pipeline()

    print("WeatherNet-05 augmentation pipeline:")
    augmentation.summary()