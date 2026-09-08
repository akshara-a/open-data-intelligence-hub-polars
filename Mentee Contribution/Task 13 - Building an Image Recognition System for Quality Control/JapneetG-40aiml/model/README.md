
# Automated Casting Defect Detection Using CNN

## Project Overview

This project uses a Convolutional Neural Network (CNN) to classify
casting product images as defective or non-defective.

## Classes

0 = Non-defective
1 = Defective

## Image Size

224 x 224 pixels

## Model

The CNN contains:

- Data augmentation
- Image normalization
- Three Conv2D layers
- MaxPooling layers
- Global Average Pooling
- Dropout
- Dense layer
- Sigmoid output layer

## Training

Optimizer: Adam

Learning Rate: 0.001

Loss Function: Binary Cross-Entropy

Batch Size: 32

Maximum Epochs: 25

## Regularization

The project uses:

- Data augmentation
- Dropout
- Early stopping
- ReduceLROnPlateau
- Model checkpointing

## Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- False positives
- False negatives

Recall for defective products is particularly important because
false negatives could allow defective products to pass inspection.

## Running the Project

1. Install the required Python libraries.

   pip install -r requirements.txt

2. Place the casting dataset inside the data directory.

3. Run the Jupyter Notebook or Google Colab notebook.

4. Train the CNN.

5. Evaluate the model using the separate test dataset.

6. Use the prediction function to classify unseen casting images.

## Output

The system produces:

- Product classification
- Defect probability
- Decision threshold
- Recommended inspection action

## Saved Model

The final trained model is stored as:

casting_defect_detection_final.keras
