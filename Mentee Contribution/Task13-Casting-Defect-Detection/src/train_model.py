import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)
from sklearn.metrics import classification_report, confusion_matrix


# ==============================
# DATASET PATHS
# ==============================

TRAIN_DIR = "data/casting_data/train"
TEST_DIR = "data/casting_data/test"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10


# ==============================
# LOAD DATASET
# ==============================

print("Loading training dataset...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True
)

print("\nLoading testing dataset...")

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)


# ==============================
# CLASS NAMES
# ==============================

class_names = train_dataset.class_names

print("\nClass Names:", class_names)


# ==============================
# NORMALIZE IMAGES
# ==============================

normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

train_dataset = train_dataset.map(
    lambda x, y: (normalization_layer(x), y)
)

test_dataset = test_dataset.map(
    lambda x, y: (normalization_layer(x), y)
)


# ==============================
# PREFETCH DATA
# ==============================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(AUTOTUNE)
test_dataset = test_dataset.prefetch(AUTOTUNE)


# ==============================
# BUILD CNN MODEL
# ==============================

model = Sequential([

    Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=(128, 128, 3)
    ),

    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),

    Dense(
        128,
        activation="relu"
    ),

    Dropout(0.5),

    Dense(
        1,
        activation="sigmoid"
    )
])


# ==============================
# COMPILE MODEL
# ==============================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ==============================
# MODEL SUMMARY
# ==============================

print("\nCNN MODEL SUMMARY")

model.summary()


# ==============================
# TRAIN MODEL
# ==============================

print("\nTraining model...\n")

history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=EPOCHS
)


# ==============================
# EVALUATE MODEL
# ==============================

print("\nEvaluating model...")

loss, accuracy = model.evaluate(test_dataset)

print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")


# ==============================
# SAVE MODEL
# ==============================

os.makedirs("models", exist_ok=True)

MODEL_PATH = "models/casting_defect_cnn.keras"

model.save(MODEL_PATH)

print(f"\nModel saved successfully: {MODEL_PATH}")


# ==============================
# GET PREDICTIONS
# ==============================

print("\nGenerating predictions...")

y_true = []
y_pred = []

for images, labels in test_dataset:

    predictions = model.predict(images, verbose=0)

    predictions = (predictions > 0.5).astype(int)

    y_true.extend(labels.numpy().astype(int).flatten())

    y_pred.extend(predictions.flatten())


# ==============================
# CLASSIFICATION REPORT
# ==============================

print("\nCLASSIFICATION REPORT\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )
)


# ==============================
# CONFUSION MATRIX
# ==============================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\nCONFUSION MATRIX\n")

print(cm)


# ==============================
# CREATE REPORT DIRECTORY
# ==============================

os.makedirs("reports", exist_ok=True)


# ==============================
# ACCURACY GRAPH
# ==============================

plt.figure()

plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])

plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend([
    "Training Accuracy",
    "Validation Accuracy"
])

plt.savefig(
    "reports/accuracy_graph.png"
)

plt.close()


# ==============================
# LOSS GRAPH
# ==============================

plt.figure()

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])

plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend([
    "Training Loss",
    "Validation Loss"
])

plt.savefig(
    "reports/loss_graph.png"
)

plt.close()


print("\nGraphs saved in reports folder.")

print("\nTASK 13 COMPLETED SUCCESSFULLY! 🎉")