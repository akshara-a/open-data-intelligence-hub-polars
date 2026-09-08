# Casting Quality Inspection Using Convolutional Neural Network

## 📌 Project Overview

This project implements an image-based casting quality inspection system using a **Convolutional Neural Network (CNN)**.

The system classifies casting images into two categories:

* `def_front` — Defective casting
* `ok_front` — Acceptable casting

A trained CNN model is used to automatically analyze casting images and predict the quality class. A **Streamlit web application** provides an interactive interface where users can upload a casting image and receive the predicted class along with the model's confidence.

---

## 🎯 Objectives

The main objectives of this project are:

* Build a CNN for automated casting image classification.
* Preprocess and normalize casting images.
* Train and validate the neural network.
* Evaluate the model using unseen test data.
* Analyze performance using accuracy, precision, recall, F1-score, and a confusion matrix.
* Save the trained model for later use.
* Develop a user-friendly Streamlit application for real-time image prediction.

---

## 📂 Project Structure

```text
SanjanaG40-aiml/
│
├── app.py
├── casting_cnn_model.keras
├── requirements.txt
├── README.md
├── casting_cnn.ipynb
│
└── data/
    └── casting_data/
        ├── train/
        └── test/
```

### File Description

| File                      | Description                                                  |
| ------------------------- | ------------------------------------------------------------ |
| `app.py`                  | Streamlit web application for casting image prediction       |
| `casting_cnn_model.keras` | Trained CNN model                                            |
| `casting_cnn.ipynb`       | Complete model development, training and evaluation notebook |
| `requirements.txt`        | Python dependencies required to run the project              |
| `README.md`               | Project documentation                                        |
| `data/`                   | Casting image dataset                                        |

---

## 📊 Dataset

The dataset contains images belonging to two classes:

```text
def_front
ok_front
```

The available data used during model development consisted of:

* **Training images:** 5,307
* **Validation images:** 1,326
* **Test images:** 715

The test dataset was kept separate from the training process and was used only for final model evaluation.

---

## 🧠 Model Architecture

A Convolutional Neural Network was selected because CNNs are well suited for image classification and can automatically learn visual patterns such as edges, shapes, textures, and defect characteristics.

The implemented architecture consists of:

```text
Input Image
    ↓
Rescaling (1/255)
    ↓
Conv2D (32 filters) + ReLU
    ↓
MaxPooling
    ↓
Conv2D (64 filters) + ReLU
    ↓
MaxPooling
    ↓
Conv2D (128 filters) + ReLU
    ↓
MaxPooling
    ↓
Flatten
    ↓
Dense (128 neurons) + ReLU
    ↓
Dropout (0.5)
    ↓
Dense (1 neuron) + Sigmoid
    ↓
Binary Classification
```

### Input

Images are resized to:

```text
224 × 224 × 3
```

### Output

The final sigmoid layer produces a probability for the binary classification task.

The prediction is interpreted as:

```text
Probability < 0.5  →  def_front
Probability ≥ 0.5  →  ok_front
```

---

## ⚙️ Training Configuration

The model was trained using the following configuration:

| Parameter         | Value                        |
| ----------------- | ---------------------------- |
| Model             | Convolutional Neural Network |
| Input size        | 224 × 224 × 3                |
| Optimizer         | Adam                         |
| Loss function     | Binary Cross-Entropy         |
| Activation        | ReLU + Sigmoid               |
| Dropout           | 0.5                          |
| Epochs            | 10                           |
| Evaluation metric | Accuracy                     |

Pixel values were normalized from the range `0–255` to `0–1` using a scaling factor of `1/255`.

---

## 📈 Model Performance

The trained model achieved the following results on the unseen test dataset:

```text
Test Accuracy: 98.32%
Test Loss:     0.0440
```

The model correctly classified:

```text
703 / 715 test images
```

Only 12 test images were incorrectly classified.

### Classification Report

| Class                | Precision | Recall | F1-Score | Support |
| -------------------- | --------: | -----: | -------: | ------: |
| `def_front`          |      1.00 |   0.98 |     0.99 |     453 |
| `ok_front`           |      0.96 |   1.00 |     0.98 |     262 |
| **Overall Accuracy** |           |        | **0.98** | **715** |

---

## 🔍 Confusion Matrix

The final confusion matrix was:

```text
                  Predicted
                def_front   ok_front

Actual
def_front          442         11

ok_front             1        261
```

### Interpretation

* **442** defective casting images were correctly classified as `def_front`.
* **261** acceptable casting images were correctly classified as `ok_front`.
* **11** defective images were incorrectly classified as acceptable.
* **1** acceptable image was incorrectly classified as defective.

The results demonstrate strong performance across both classes.

---

## 🧪 Individual Image Testing

The trained model was also tested using individual unseen images.

Example results:

| Actual Class | Predicted Class | Confidence |
| ------------ | --------------- | ---------: |
| `def_front`  | `def_front`     |     92.30% |
| `ok_front`   | `ok_front`      |    100.00% |

Both example images were classified correctly.

---

## 🌐 Streamlit Web Application

The project includes a Streamlit-based web application for interactive predictions.

The application provides:

* Casting image upload
* Image preview
* CNN-based prediction
* Prediction confidence
* Defective/acceptable classification
* Model performance information
* Explanation of the prediction workflow

### Application Workflow

```text
Upload Casting Image
        ↓
Image Preprocessing
        ↓
Resize to 224 × 224
        ↓
Pixel Normalization
        ↓
CNN Model
        ↓
Prediction Probability
        ↓
Classification
        ↓
Confidence Display
```

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
```

Navigate into the project directory:

```bash
cd SanjanaG40-aiml
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Make sure the trained model:

```text
casting_cnn_model.keras
```

is located in the same directory as:

```text
app.py
```

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will open in the browser.

Upload a supported image format such as:

```text
.jpg
.jpeg
.png
.bmp
.webp
```

Then click:

```text
Analyze Casting
```

The application will display the predicted class and confidence.

---

## 💾 Saved Model

The trained CNN model is saved as:

```text
casting_cnn_model.keras
```

The saved model allows predictions to be performed without retraining the neural network.

---

## 🛠️ Technologies Used

* **Python**
* **TensorFlow**
* **Keras**
* **NumPy**
* **Pillow**
* **Matplotlib**
* **Scikit-learn**
* **Streamlit**
* **Jupyter Notebook**

---

## 📌 Design Decisions

### Why CNN?

CNNs are specifically designed for image-processing tasks. They can automatically learn spatial features from casting images, making them appropriate for defect classification.

### Why normalization?

Normalizing pixel values to the range `0–1` provides more stable and efficient neural-network training.

### Why MaxPooling?

MaxPooling reduces the spatial dimensions of feature maps while retaining important features. This reduces computational requirements and helps the network learn more robust representations.

### Why Dropout?

A dropout rate of `0.5` was included before the output layer to reduce the risk of overfitting during training.

### Why Adam?

Adam provides adaptive learning rates and is an effective general-purpose optimizer for training CNN models.

### Why Binary Cross-Entropy?

The problem contains exactly two classes, making binary cross-entropy an appropriate loss function.

### Why Sigmoid?

The final sigmoid activation produces a probability between 0 and 1, which is suitable for binary classification.

---

## ⚠️ Prediction Confidence

The model's overall test accuracy and the confidence of an individual prediction represent different measurements.

For example:

```text
Test Accuracy: 98.32%
Individual Prediction Confidence: 59.22%
```

The 98.32% value represents performance across the complete test dataset, while the individual confidence represents the model's certainty about one uploaded image.

For low-confidence predictions, additional manual inspection is recommended.

---

## 🔮 Future Improvements

Possible improvements include:

* Data augmentation for improved generalization.
* Hyperparameter tuning.
* Early stopping and learning-rate scheduling.
* Transfer learning using pretrained CNN architectures.
* More extensive testing on real-world casting images.
* Explainable AI techniques such as Grad-CAM.
* Deployment using a cloud hosting platform.
* Addition of prediction history and reporting features.

---

## 📋 Conclusion

The project successfully implements a CNN-based casting quality inspection system for binary image classification.

The trained model achieved a **98.32% test accuracy** on 715 unseen test images, with strong precision, recall, and F1-scores for both `def_front` and `ok_front` classes.

The model was saved as a reusable Keras model and integrated into a Streamlit web application, allowing users to upload casting images and obtain AI-based quality predictions.

Overall, the project demonstrates the complete machine-learning workflow from dataset preparation and CNN implementation to model evaluation, model persistence, and application deployment.

---

## 👩‍💻 Project

**Casting Quality Inspection using Convolutional Neural Network**

**Author:** SanjanaG40-aiml

**Task:** Task 14 – Neural Network Implementation with Documented Design Choices
