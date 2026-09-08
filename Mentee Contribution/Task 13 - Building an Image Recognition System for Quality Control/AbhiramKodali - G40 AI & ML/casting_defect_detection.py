import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, models
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


# ============================================================
# Configuration
# ============================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 25
SEED = 42
THRESHOLD = 0.50

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
SAMPLE_DIR = BASE_DIR / "sample_images"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)
SAMPLE_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "best_casting_defect_model.keras"


# ============================================================
# 1. Check Dataset
# ============================================================

print("=" * 60)
print("CASTING DEFECT DETECTION - CNN")
print("=" * 60)

print("\nChecking dataset folders...")

required_folders = [
    TRAIN_DIR / "ok_front",
    TRAIN_DIR / "def_front",
    TEST_DIR / "ok_front",
    TEST_DIR / "def_front",
]

for folder in required_folders:
    print(f"{folder}: {'FOUND' if folder.exists() else 'MISSING'}")

if not all(folder.exists() for folder in required_folders):
    raise FileNotFoundError(
        "\nDataset folders are missing.\n"
        "Expected structure:\n"
        "data/train/ok_front\n"
        "data/train/def_front\n"
        "data/test/ok_front\n"
        "data/test/def_front\n"
    )


# ============================================================
# 2. Count Images
# ============================================================

def count_images(folder):
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    return sum(
        1
        for file in folder.rglob("*")
        if file.is_file() and file.suffix.lower() in extensions
    )


train_ok = count_images(TRAIN_DIR / "ok_front")
train_def = count_images(TRAIN_DIR / "def_front")
test_ok = count_images(TEST_DIR / "ok_front")
test_def = count_images(TEST_DIR / "def_front")

print("\nDataset counts:")
print(f"Training non-defective: {train_ok}")
print(f"Training defective:     {train_def}")
print(f"Testing non-defective:  {test_ok}")
print(f"Testing defective:      {test_def}")

print("\nClass mapping:")
print("0 = Non-defective (ok_front)")
print("1 = Defective (def_front)")


# ============================================================
# 3. Load Training and Validation Data
# ============================================================

class_names = ["ok_front", "def_front"]

print("\nLoading training dataset...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    class_names=class_names,
    validation_split=0.20,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
)

print("\nLoading validation dataset...")

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    class_names=class_names,
    validation_split=0.20,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
)

print("\nLoading test dataset...")

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    class_names=class_names,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False,
)


# ============================================================
# 4. Improve Dataset Performance
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(AUTOTUNE)
validation_dataset = validation_dataset.prefetch(AUTOTUNE)
test_dataset = test_dataset.prefetch(AUTOTUNE)


# ============================================================
# 5. Display Sample Images
# ============================================================

print("\nDisplaying sample training images...")

plt.figure(figsize=(10, 8))

for images, labels in train_dataset.take(1):
    for i in range(min(8, len(images))):
        ax = plt.subplot(2, 4, i + 1)

        plt.imshow(images[i].numpy().astype("uint8"))

        label = int(labels[i].numpy()[0])

        if label == 0:
            title = "Non-defective"
        else:
            title = "Defective"

        plt.title(title)
        plt.axis("off")

plt.tight_layout()

sample_path = REPORT_DIR / "sample_images.png"
plt.savefig(sample_path, dpi=150)
plt.show()


# ============================================================
# 6. Data Augmentation
# ============================================================

data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.10),
        layers.RandomTranslation(0.05, 0.05),
        layers.RandomContrast(0.10),
    ],
    name="data_augmentation",
)


# ============================================================
# 7. Build CNN
# ============================================================

print("\nBuilding CNN model...")

model = models.Sequential(
    [
        layers.Input(shape=(224, 224, 3)),

        # Augmentation is active during training only.
        data_augmentation,

        # Normalize pixels from 0-255 to 0-1.
        layers.Rescaling(1.0 / 255),

        # Convolution block 1
        layers.Conv2D(
            32,
            kernel_size=3,
            activation="relu",
        ),
        layers.MaxPooling2D(),

        # Convolution block 2
        layers.Conv2D(
            64,
            kernel_size=3,
            activation="relu",
        ),
        layers.MaxPooling2D(),

        # Convolution block 3
        layers.Conv2D(
            128,
            kernel_size=3,
            activation="relu",
        ),
        layers.MaxPooling2D(),

        # Feature reduction
        layers.GlobalAveragePooling2D(),

        # Regularization
        layers.Dropout(0.40),

        layers.Dense(
            64,
            activation="relu",
        ),

        layers.Dropout(0.30),

        # Binary classification output
        layers.Dense(
            1,
            activation="sigmoid",
        ),
    ]
)

model.summary()


# ============================================================
# 8. Compile Model
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ],
)


# ============================================================
# 9. Training Callbacks
# ============================================================

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=0.000001,
    ),

    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_PATH),
        monitor="val_loss",
        save_best_only=True,
    ),
]


# ============================================================
# 10. Train Model
# ============================================================

print("\nStarting model training...")
print(f"Maximum epochs: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=callbacks,
)


# ============================================================
# 11. Training Graphs
# ============================================================

training_accuracy = history.history["accuracy"]
validation_accuracy = history.history["val_accuracy"]

training_loss = history.history["loss"]
validation_loss = history.history["val_loss"]

epochs_range = range(1, len(training_accuracy) + 1)


# Accuracy graph
plt.figure(figsize=(8, 5))

plt.plot(
    epochs_range,
    training_accuracy,
    label="Training Accuracy",
)

plt.plot(
    epochs_range,
    validation_accuracy,
    label="Validation Accuracy",
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid(True)

accuracy_path = REPORT_DIR / "accuracy_graph.png"
plt.savefig(accuracy_path, dpi=150)
plt.show()


# Loss graph
plt.figure(figsize=(8, 5))

plt.plot(
    epochs_range,
    training_loss,
    label="Training Loss",
)

plt.plot(
    epochs_range,
    validation_loss,
    label="Validation Loss",
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)

loss_path = REPORT_DIR / "loss_graph.png"
plt.savefig(loss_path, dpi=150)
plt.show()


# ============================================================
# 12. Evaluate Model
# ============================================================

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

test_results = model.evaluate(
    test_dataset,
    verbose=1,
)

print("\nTest results:")

for name, value in zip(
    model.metrics_names,
    test_results,
):
    print(f"{name}: {value:.4f}")


# ============================================================
# 13. Generate Test Predictions
# ============================================================

print("\nGenerating predictions...")

prediction_probabilities = model.predict(
    test_dataset,
    verbose=1,
)

predicted_labels = (
    prediction_probabilities.flatten() >= THRESHOLD
).astype(int)


actual_labels = np.concatenate(
    [
        labels.numpy().flatten()
        for images, labels in test_dataset
    ]
).astype(int)


# ============================================================
# 14. Classification Report
# ============================================================

print("\nClassification Report:")
print(
    classification_report(
        actual_labels,
        predicted_labels,
        target_names=[
            "Non-defective",
            "Defective",
        ],
    )
)


# ============================================================
# 15. Confusion Matrix
# ============================================================

matrix = confusion_matrix(
    actual_labels,
    predicted_labels,
)

print("\nConfusion Matrix:")
print(matrix)

tn, fp, fn, tp = matrix.ravel()

print("\nConfusion Matrix Details:")
print(f"True Negatives:  {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives:  {tp}")

print("\nFalse-negative analysis:")
print(
    "False negatives are defective products classified "
    "as non-defective."
)

print(
    f"Number of false negatives: {fn}"
)


# Save confusion matrix
display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=[
        "Non-defective",
        "Defective",
    ],
)

display.plot()

plt.title("Casting Defect Confusion Matrix")

confusion_path = REPORT_DIR / "confusion_matrix.png"
plt.savefig(confusion_path, dpi=150)
plt.show()


# ============================================================
# 16. Single Image Prediction Function
# ============================================================

def predict_product(
    image_path,
    model,
    threshold=0.50,
):
    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE,
    )

    image_array = tf.keras.utils.img_to_array(
        image
    )

    image_array = tf.expand_dims(
        image_array,
        axis=0,
    )

    defect_probability = float(
        model.predict(
            image_array,
            verbose=0,
        )[0][0]
    )

    if defect_probability >= threshold:
        predicted_class = "Defective"
        recommended_action = (
            "Send for manual inspection"
        )
    else:
        predicted_class = "Non-defective"
        recommended_action = (
            "Product may proceed"
        )

    print(f"\nImage: {image_path}")
    print(
        f"Prediction: {predicted_class}"
    )
    print(
        f"Defect probability: "
        f"{defect_probability:.2%}"
    )
    print(
        f"Decision threshold: "
        f"{threshold:.0%}"
    )
    print(
        f"Recommended action: "
        f"{recommended_action}"
    )

    return predicted_class, defect_probability


# ============================================================
# 17. Predict Five Unseen Images
# ============================================================

print("\n" + "=" * 60)
print("UNSEEN IMAGE PREDICTIONS")
print("=" * 60)

sample_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
}

sample_images = [
    file
    for file in SAMPLE_DIR.rglob("*")
    if file.is_file()
    and file.suffix.lower() in sample_extensions
]

if len(sample_images) == 0:
    print(
        "\nNo sample images found yet."
    )
    print(
        "Place at least five unseen product "
        "images inside:"
    )
    print(SAMPLE_DIR)

else:
    for image_path in sample_images[:5]:
        predict_product(
            str(image_path),
            model,
            threshold=THRESHOLD,
        )


# ============================================================
# 18. Save Final Model
# ============================================================

final_model_path = MODEL_DIR / "casting_defect_model.keras"

model.save(final_model_path)

print("\n" + "=" * 60)
print("TASK 13 COMPLETED")
print("=" * 60)

print(f"Best model: {MODEL_PATH}")
print(f"Final model: {final_model_path}")
print(f"Accuracy graph: {accuracy_path}")
print(f"Loss graph: {loss_path}")
print(f"Confusion matrix: {confusion_path}")
