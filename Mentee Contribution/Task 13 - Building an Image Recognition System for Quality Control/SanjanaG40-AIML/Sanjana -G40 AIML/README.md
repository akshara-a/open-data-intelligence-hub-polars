# 🔍 Casting Defect Detection System

A deep learning-based image classification system for detecting defects in industrial casting components.

The project uses a **MobileNet-based image classification model** to classify casting images into two categories:

- `ok_front` — Casting without visible defects
- `def_front` — Defective casting

A **Streamlit web interface** is provided so users can upload a casting image and receive a prediction with confidence.

---

## 📌 Project Overview

Quality inspection of industrial castings is an important part of manufacturing. Manual inspection can be time-consuming and may be affected by human error.

This project applies computer vision and deep learning to automatically identify defective casting components from images.

The final system consists of:

1. Image preprocessing
2. Data augmentation
3. Deep learning model training
4. Validation and threshold selection
5. Model evaluation
6. Streamlit prediction interface

---

## 🧠 Model

The final model uses **MobileNet** for image classification.

### Input

- Image size: `224 × 224`
- Image format: JPG, JPEG, or PNG
- Color mode: RGB

### Output Classes

| Class | Meaning |
|---|---|
| `ok_front` | Good / acceptable casting |
| `def_front` | Defective casting |

The model produces a probability representing the likelihood of the defective class.

A decision threshold of **0.52** is used for the final prediction.

---

## 📊 Final Model Performance

The final MobileNet model was evaluated on **197 test images**.

| Metric | Score |
|---|---:|
| Accuracy | **94.92%** |
| Precision | **95.00%** |
| Recall | **96.61%** |
| F1-score | **95.80%** |

### Classification Report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| `ok_front` | 0.95 | 0.92 | 0.94 | 79 |
| `def_front` | 0.95 | 0.97 | 0.96 | 118 |
| **Accuracy** | | | **0.95** | **197** |

### Confusion Matrix

```text
[[73   6]
 [ 4 114]]