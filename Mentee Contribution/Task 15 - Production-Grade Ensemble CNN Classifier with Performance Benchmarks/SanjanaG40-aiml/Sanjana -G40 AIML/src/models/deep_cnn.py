import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


IMG_HEIGHT = 128
IMG_WIDTH = 128
NUM_CHANNELS = 3
NUM_CLASSES = 5


def build_deep_cnn():
    """
    Build the deeper CNN for WeatherNet-05.

    Uses:
        - Four convolutional stages
        - Batch Normalization
        - Global Average Pooling
        - Dropout
    """

    model = keras.Sequential(
        [
            layers.Input(
                shape=(IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS)
            ),

            # Stage 1
            layers.Conv2D(
                32,
                (3, 3),
                padding="same",
                use_bias=False
            ),
            layers.BatchNormalization(),
            layers.ReLU(),

            layers.Conv2D(
                32,
                (3, 3),
                padding="same",
                use_bias=False
            ),
            layers.BatchNormalization(),
            layers.ReLU(),

            layers.MaxPooling2D((2, 2)),

            # Stage 2
            layers.Conv2D(
                64,
                (3, 3),
                padding="same",
                use_bias=False
            ),
            layers.BatchNormalization(),
            layers.ReLU(),

            layers.Conv2D(
                64,
                (3, 3),
                padding="same",
                use_bias=False
            ),
            layers.BatchNormalization(),
            layers.ReLU(),

            layers.MaxPooling2D((2, 2)),

            # Stage 3
            layers.Conv2D(
                128,
                (3, 3),
                padding="same",
                use_bias=False
            ),
            layers.BatchNormalization(),
            layers.ReLU(),

            layers.Conv2D(
                128,
                (3, 3),
                padding="same",
                use_bias=False
            ),
            layers.BatchNormalization(),
            layers.ReLU(),

            layers.MaxPooling2D((2, 2)),

            # Stage 4
            layers.Conv2D(
                256,
                (3, 3),
                padding="same",
                use_bias=False
            ),
            layers.BatchNormalization(),
            layers.ReLU(),

            layers.Conv2D(
                256,
                (3, 3),
                padding="same",
                use_bias=False
            ),
            layers.BatchNormalization(),
            layers.ReLU(),

            # Instead of Flatten
            layers.GlobalAveragePooling2D(),

            layers.Dropout(0.40),

            layers.Dense(
                128,
                activation="relu"
            ),

            layers.Dropout(0.30),

            layers.Dense(
                NUM_CLASSES,
                activation="softmax"
            )
        ],
        name="weather_cnn_deep"
    )

    return model


if __name__ == "__main__":
    model = build_deep_cnn()

    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("\nCNN 3 compiled successfully! ✅")