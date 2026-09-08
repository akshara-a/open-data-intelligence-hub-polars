# Task 13 - Casting Defect Detection Using CNN

## Objective

Build a Convolutional Neural Network (CNN) to classify casting images into defective and non-defective categories.

## Dataset

The dataset contains two classes:

- `def_front` - Defective casting images
- `ok_front` - Non-defective casting images

Dataset structure:

```text
data/casting_data/
├── train/
│   ├── def_front/
│   └── ok_front/
└── test/
    ├── def_front/
    └── ok_front/
```

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Scikit-learn
- Pillow

## Model Architecture

The CNN model uses:

- Conv2D layers
- MaxPooling2D layers
- Flatten layer
- Dense layer
- Dropout layer
- Sigmoid output layer

## Training Configuration

- Image Size: 128 x 128
- Batch Size: 32
- Epochs: 10
- Optimizer: Adam
- Loss Function: Binary Crossentropy

## Training Results

- Training Images: 6633
- Testing Images: 715
- Test Loss: 0.0051
- Test Accuracy: 100%

## Classification Results

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| def_front | 1.00 | 1.00 | 1.00 |
| ok_front | 1.00 | 1.00 | 1.00 |

## Confusion Matrix

```text
[[453   0]
 [  0 262]]
```

## Output

### Trained Model

```text
models/casting_defect_cnn.keras
```

### Training Graphs

```text
reports/
├── accuracy_graph.png
└── loss_graph.png
```

## Project Structure

```text
Task13-Casting-Defect-Detection/
├── data/
├── models/
│   └── casting_defect_cnn.keras
├── notebooks/
├── reports/
│   ├── accuracy_graph.png
│   └── loss_graph.png
├── sample_images/
├── src/
│   └── train_model.py
├── .gitignore
├── README.md
└── requirements.txt
```

## How to Run

### 1. Create and activate virtual environment

```bash
python -m venv .venv
```

### 2. Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install requirements

```bash
python -m pip install -r requirements.txt
```

### 4. Run the model

```bash
python src/train_model.py
```

## Author

G Rajesh

SURE Trust G40 AI/ML Batch