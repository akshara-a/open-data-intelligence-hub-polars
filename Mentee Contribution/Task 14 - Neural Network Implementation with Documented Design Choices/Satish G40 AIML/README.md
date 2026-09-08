# Binary Image Classification Using a Convolutional Neural Network

## Project Objective

This beginner-friendly TensorFlow/Keras mini project classifies casting product images for quality inspection:

- `ok_front` -> Non-defective -> Class 0
- `def_front` -> Defective -> Class 1

The complete implementation is in `binary_image_classification_cnn.ipynb` and can run in Google Colab or Jupyter Notebook.

## Dataset Description

The project uses Kaggle's **Casting Product Image Data for Quality Inspection** dataset. The notebook expects image folders named `ok_front` and `def_front`. Set `DATASET_SOURCE_DIR` in Section 2 to the extracted dataset location. If split folders are not already present, the notebook creates reproducible `70%` training, `15%` validation, and `15%` test sets using seed `42`.

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn
- Pillow (PIL)
- pathlib
- Pandas for the optional comparison table

## Project Folder Structure

```text
binary-image-classification/
├── data/
│   ├── train/
│   │   ├── ok_front/
│   │   └── def_front/
│   ├── validation/
│   │   ├── ok_front/
│   │   └── def_front/
│   └── test/
│       ├── ok_front/
│       └── def_front/
├── models/
├── outputs/
│   ├── graphs/
│   └── predictions/
├── binary_image_classification_cnn.ipynb
├── requirements.txt
└── README.md
```

## CNN Architecture

```text
Input: 224 x 224 x 3
Data augmentation
Rescaling: 1 / 255
Conv2D: 32 filters, 3 x 3, ReLU
MaxPooling2D
Conv2D: 64 filters, 3 x 3, ReLU
MaxPooling2D
Conv2D: 128 filters, 3 x 3, ReLU
MaxPooling2D
GlobalAveragePooling2D
Dropout: 0.40
Dense: 64 neurons, ReLU
Dense: 1 neuron, Sigmoid
```

## Installation Instructions

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
```

For Google Colab, run `!pip install -r requirements.txt` after uploading the project files, or install the listed packages in a cell.

## How to Run the Project

1. Download and extract the Kaggle dataset.
2. Put it in a folder containing `ok_front` and `def_front`, or update `DATASET_SOURCE_DIR` in Section 2.
3. Open `binary_image_classification_cnn.ipynb`.
4. Run the cells from top to bottom.
5. Review saved graphs in `outputs/graphs/` and models in `models/`.

The notebook creates missing directories automatically. Test data is loaded with `shuffle=False` so actual labels remain aligned with predictions.

## Model Design Choices

| Decision | Selected value | Reason |
|---|---|---|
| Image size | 224 x 224 | Balances detail and computation |
| Problem type | Binary classification | There are two classes |
| Model type | CNN | Designed for image patterns |
| Conv filters | 32, 64, 128 | Learns increasingly complex features |
| Kernel size | 3 x 3 | Efficient local feature extraction |
| Hidden activation | ReLU | Fast and effective non-linearity |
| Pooling | MaxPooling | Reduces feature dimensions |
| Output activation | Sigmoid | Produces a binary probability |
| Optimizer | Adam | Adaptive and beginner-friendly |
| Learning rate | 0.001 | Good starting value |
| Loss | Binary cross-entropy | Suitable for two classes |
| Batch size | 32 | Balances speed and memory |
| Maximum epochs | 25 | Enough training with early stopping |
| Dropout | 0.40 | Helps reduce overfitting |
| Augmentation | Flip, rotation, zoom, contrast | Improves robustness |
| Metrics | Accuracy, precision, recall | Measures general and defect detection performance |

## Training Process

The model uses Adam, binary cross-entropy, data augmentation during training only, early stopping, learning-rate reduction, and a best-model checkpoint saved as `models/best_model.keras`. The final model is saved as `models/final_cnn_model.keras`.

## Evaluation Metrics

The notebook reports test loss, accuracy, precision, recall, a confusion matrix, and a classification report containing precision, recall, F1-score, and support. Recall for the `Defective` class is particularly important because a false negative could approve a defective product.

## Confusion Matrix

The matrix uses the labels `Non-defective` and `Defective`. It reports true negatives, false positives, false negatives, and true positives. The plot is saved as `outputs/graphs/confusion_matrix.png`.

## Sample Predictions

Section 22 resizes an image, creates a batch, predicts its defective probability, displays it, and prints an action. A probability of at least `0.5` means `Defective` and sends the product for manual inspection; a lower probability means `Non-defective` and approves it. Replace the demonstration paths with external unseen images when available.

## Bonus Experiment

The notebook trains a second model with dropout changed from `0.40` to `0.20`, keeping the dataset and other settings the same. It compares original and changed test accuracy, precision, and recall using actual evaluation values.

## Results

Run the notebook to fill in these values:

- Completed epochs: `[generated after training]`
- Test accuracy: `[generated after training]`
- Test precision: `[generated after training]`
- Test recall: `[generated after training]`
- Defective recall: `[generated after training]`
- Overfitting observed: `[generated from training history]`
- Dropout experiment observation: `[generated after bonus training]`

## Conclusion

After the notebook is run, the final cell generates a conclusion using actual results. It discusses whether the CNN learned the task, test accuracy, precision, recall, false-negative risk, overfitting, the effect of augmentation and dropout, and a possible future improvement such as transfer learning..
