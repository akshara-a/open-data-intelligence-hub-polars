from tensorflow.keras import layers, models

from src.config import INPUT_SHAPE, NUM_CLASSES


def build_regularized_cnn() -> models.Sequential:
    """CNN with stronger regularization and normalization."""
    model = models.Sequential(
        [
            layers.Input(shape=INPUT_SHAPE),
            layers.Conv2D(48, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(48, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            layers.Conv2D(96, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(96, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.35),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.4),
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ],
        name="regularized_cnn",
    )
    return model
