from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "casting_defect_model.keras"
TEST_DIR = BASE_DIR / "data" / "test"
UNSEEN_DIR = BASE_DIR / "data" / "unseen"
PLOTS_DIR = BASE_DIR / "plots"

IMG_SIZE = (224, 224)

model = load_model(MODEL_PATH)

# Class mapping used by the evaluation script.
# 0 = def_front, 1 = ok_front
class_names = ["ok_front", "def_front"]

test_images = []
test_labels = []

for label, class_name in enumerate(class_names):
    class_dir = TEST_DIR / class_name

    for image_path in sorted(class_dir.glob("*.png")):
        image = load_img(image_path, target_size=IMG_SIZE)
        image_array = img_to_array(image) / 255.0

        test_images.append(image_array)
        test_labels.append(label)

X_test = np.array(test_images)
y_test = np.array(test_labels)

# Generate predictions
probabilities = model.predict(X_test, verbose=0).ravel()

# Sigmoid output >= 0.5 is class 1 (ok_front)
predictions = (probabilities >= 0.5).astype(int)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, zero_division=0)
recall = recall_score(y_test, predictions, zero_division=0)

print("\nTask 14 Test Evaluation")
print("-----------------------")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, predictions)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names,
)

disp.plot()
plt.title("Task 14 - Confusion Matrix")
plt.tight_layout()

confusion_path = PLOTS_DIR / "confusion_matrix.png"
plt.savefig(confusion_path, dpi=150)
plt.close()

print(f"\nConfusion matrix saved to: {confusion_path}")

# Five unseen-image predictions
print("\nFive Unseen Image Predictions")
print("-----------------------------")

unseen_images = sorted(UNSEEN_DIR.glob("*.png"))[:5]

for image_path in unseen_images:
    image = load_img(image_path, target_size=IMG_SIZE)
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    probability = float(model.predict(image_array, verbose=0)[0][0])

    predicted_index = int(probability >= 0.5)
    predicted_class = class_names[predicted_index]

    print(
        f"{image_path.name}: "
        f"{predicted_class} "
        f"(probability={probability:.4f})"
    )

print(f"\nUnseen images evaluated: {len(unseen_images)}")