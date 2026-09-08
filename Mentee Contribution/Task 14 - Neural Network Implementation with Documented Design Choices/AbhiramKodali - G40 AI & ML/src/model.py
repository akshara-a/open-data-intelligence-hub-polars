import tensorflow as tf
from tensorflow.keras import layers, models


IMAGE_SIZE = (224, 224)
NUM_CHANNELS = 3
DROPOUT_RATE = 0.40


def build_model():
    """Build the documented binary casting-defect CNN."""
    model = models.Sequential(
        [
            layers.Input(shape=(*IMAGE_SIZE, NUM_CHANNELS)),

            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),

            layers.GlobalAveragePooling2D(),
            layers.Dropout(DROPOUT_RATE),

            layers.Dense(128, activation="relu"),
            layers.Dropout(DROPOUT_RATE),

            layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    return model


if __name__ == "__main__":
    model = build_model()
    model.summary()