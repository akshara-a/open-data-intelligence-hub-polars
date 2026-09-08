import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


IMG_HEIGHT = 128
IMG_WIDTH = 128
NUM_CHANNELS = 3
NUM_CLASSES = 5


def build_regularized_cnn():
    """
    Build the regularized CNN for WeatherNet-05.

    Uses:
        - Batch Normalization
        - Dropout
        - Multiple convolution blocks
    """

    model = keras.Sequential(
        [
            layers.Input(
                shape=(IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS)
            ),

            # Block 1
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
            layers.Dropout(0.20),

            # Block 2
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
            layers.Dropout(0.25),

            # Classification head
            layers.Flatten(),

            layers.Dense(
                128,
                activation="relu"
            ),

            layers.Dropout(0.40),

            layers.Dense(
                NUM_CLASSES,
                activation="softmax"
            )
        ],
        name="weather_cnn_regularized"
    )

    return model


if __name__ == "__main__":
    model = build_regularized_cnn()

    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("\nCNN 2 compiled successfully! ✅")