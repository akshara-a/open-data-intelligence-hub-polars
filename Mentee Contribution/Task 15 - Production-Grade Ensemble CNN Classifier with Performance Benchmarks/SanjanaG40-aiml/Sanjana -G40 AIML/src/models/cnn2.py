"""
WeatherNet-05
CNN 2 — Improved Deep CNN

Architecture:
- Data augmentation
- 3 convolutional blocks
- Batch normalization
- Dropout
- Global average pooling
- Dense classifier
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ============================================================
# Configuration
# ============================================================

INPUT_SHAPE = (128, 128, 3)
NUM_CLASSES = 5


# ============================================================
# Data Augmentation
# ============================================================

def build_augmentation():

    return keras.Sequential(
        [
            layers.RandomFlip(
                "horizontal"
            ),

            layers.RandomRotation(
                0.08
            ),

            layers.RandomZoom(
                0.10
            ),

            layers.RandomContrast(
                0.10
            ),
        ],
        name="cnn2_augmentation"
    )


# ============================================================
# CNN 2
# ============================================================

def build_cnn2(
    input_shape=INPUT_SHAPE,
    num_classes=NUM_CLASSES
):

    inputs = keras.Input(
        shape=input_shape,
        name="image"
    )

    # --------------------------------------------------------
    # Augmentation
    # --------------------------------------------------------

    x = build_augmentation()(inputs)

    # --------------------------------------------------------
    # Block 1
    # --------------------------------------------------------

    x = layers.Conv2D(
        32,
        3,
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.ReLU()(x)

    x = layers.Conv2D(
        32,
        3,
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.ReLU()(x)

    x = layers.MaxPooling2D()(x)

    x = layers.Dropout(
        0.20
    )(x)

    # --------------------------------------------------------
    # Block 2
    # --------------------------------------------------------

    x = layers.Conv2D(
        64,
        3,
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.ReLU()(x)

    x = layers.Conv2D(
        64,
        3,
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.ReLU()(x)

    x = layers.MaxPooling2D()(x)

    x = layers.Dropout(
        0.25
    )(x)

    # --------------------------------------------------------
    # Block 3
    # --------------------------------------------------------

    x = layers.Conv2D(
        128,
        3,
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.ReLU()(x)

    x = layers.Conv2D(
        128,
        3,
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.ReLU()(x)

    x = layers.MaxPooling2D()(x)

    x = layers.Dropout(
        0.30
    )(x)

    # --------------------------------------------------------
    # Classification head
    # --------------------------------------------------------

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(
        128,
        activation="relu"
    )(x)

    x = layers.Dropout(
        0.40
    )(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="predictions"
    )(x)

    model = keras.Model(
        inputs,
        outputs,
        name="weathernet_cnn2"
    )

    return model


# ============================================================
# Compile
# ============================================================

def compile_cnn2(model):

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=1e-3
        ),
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy"
        ]
    )

    return model


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("WEATHERNET-05 — CNN 2")
    print("=" * 60)

    model = build_cnn2()

    model = compile_cnn2(
        model
    )

    model.summary()

    print("\nCNN 2 compiled successfully! ✅")