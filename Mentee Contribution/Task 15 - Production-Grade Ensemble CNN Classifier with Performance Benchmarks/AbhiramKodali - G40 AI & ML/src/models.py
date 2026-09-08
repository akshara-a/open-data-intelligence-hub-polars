from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    RandomFlip,
    RandomRotation,
    RandomZoom,
    RandomContrast,
    Conv2D,
    MaxPooling2D,
    GlobalAveragePooling2D,
    Dense,
    Dropout,
)
from tensorflow.keras.optimizers import Adam


INPUT_SHAPE = (224, 224, 3)


def create_augmentation():
    """Create the shared training-time image augmentation pipeline."""

    return Sequential(
        [
            RandomFlip(
                mode="horizontal",
                name="random_flip",
            ),
            RandomRotation(
                factor=0.10,
                name="random_rotation",
            ),
            RandomZoom(
                height_factor=0.10,
                width_factor=0.10,
                name="random_zoom",
            ),
            RandomContrast(
                factor=0.10,
                name="random_contrast",
            ),
        ],
        name="data_augmentation",
    )


def compile_model(model):
    model.compile(
        optimizer=Adam(
            learning_rate=0.001
        ),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            "precision",
            "recall",
        ],
    )

    return model


def build_cnn_small():
    """CNN 1: compact architecture."""

    model = Sequential(
        [
            Input(
                shape=INPUT_SHAPE
            ),

            create_augmentation(),

            Conv2D(
                16,
                (3, 3),
                activation="relu",
            ),
            MaxPooling2D(),

            Conv2D(
                32,
                (3, 3),
                activation="relu",
            ),
            MaxPooling2D(),

            GlobalAveragePooling2D(),

            Dropout(0.30),

            Dense(
                64,
                activation="relu",
            ),

            Dropout(0.30),

            Dense(
                1,
                activation="sigmoid",
            ),
        ],
        name="CNN_Small",
    )

    return compile_model(model)


def build_cnn_standard():
    """CNN 2: standard architecture."""

    model = Sequential(
        [
            Input(
                shape=INPUT_SHAPE
            ),

            create_augmentation(),

            Conv2D(
                32,
                (3, 3),
                activation="relu",
            ),
            MaxPooling2D(),

            Conv2D(
                64,
                (3, 3),
                activation="relu",
            ),
            MaxPooling2D(),

            Conv2D(
                128,
                (3, 3),
                activation="relu",
            ),
            MaxPooling2D(),

            GlobalAveragePooling2D(),

            Dropout(0.40),

            Dense(
                128,
                activation="relu",
            ),

            Dropout(0.40),

            Dense(
                1,
                activation="sigmoid",
            ),
        ],
        name="CNN_Standard",
    )

    return compile_model(model)


def build_cnn_deep():
    """CNN 3: deeper architecture."""

    model = Sequential(
        [
            Input(
                shape=INPUT_SHAPE
            ),

            create_augmentation(),

            Conv2D(
                32,
                (3, 3),
                activation="relu",
            ),
            Conv2D(
                32,
                (3, 3),
                activation="relu",
            ),
            MaxPooling2D(),

            Conv2D(
                64,
                (3, 3),
                activation="relu",
            ),
            Conv2D(
                64,
                (3, 3),
                activation="relu",
            ),
            MaxPooling2D(),

            Conv2D(
                128,
                (3, 3),
                activation="relu",
            ),
            Conv2D(
                128,
                (3, 3),
                activation="relu",
            ),
            MaxPooling2D(),

            GlobalAveragePooling2D(),

            Dropout(0.40),

            Dense(
                128,
                activation="relu",
            ),

            Dropout(0.40),

            Dense(
                1,
                activation="sigmoid",
            ),
        ],
        name="CNN_Deep",
    )

    return compile_model(model)