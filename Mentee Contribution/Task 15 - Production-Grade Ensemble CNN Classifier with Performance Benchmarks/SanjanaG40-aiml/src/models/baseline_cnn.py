"""
WeatherNet-05
CNN 1 - Baseline CNN

Simple production-oriented CNN baseline.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


NUM_CLASSES = 5
INPUT_SHAPE = (128, 128, 3)


def build_baseline_cnn(
    input_shape=INPUT_SHAPE,
    num_classes=NUM_CLASSES
):
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),

            # Convolutional Block 1
            layers.Conv2D(
                32,
                (3, 3),
                padding="same",
                activation="relu"
            ),
            layers.MaxPooling2D((2, 2)),

            # Convolutional Block 2
            layers.Conv2D(
                64,
                (3, 3),
                padding="same",
                activation="relu"
            ),
            layers.MaxPooling2D((2, 2)),

            # Much smaller classification head
            layers.GlobalAveragePooling2D(),

            layers.Dense(
                128,
                activation="relu"
            ),

            layers.Dense(
                num_classes,
                activation="softmax"
            ),
        ],
        name="weather_cnn_baseline"
    )

    return model


def compile_model(model):
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":

    print("=" * 60)
    print("WEATHERNET-05 — CNN 1 BASELINE")
    print("=" * 60)

    model = build_baseline_cnn()
    model = compile_model(model)

    model.summary()

    print("\nModel compiled successfully! ✅")