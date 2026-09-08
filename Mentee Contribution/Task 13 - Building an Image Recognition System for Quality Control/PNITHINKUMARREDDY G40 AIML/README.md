# Automated Casting Defect Detection Using a CNN

## Overview
Binary image classification system that detects defective vs non-defective metal castings using a Convolutional Neural Network (TensorFlow/Keras).

## Dataset
Kaggle: Casting Product Image Data for Quality Inspection
- Non-defective (ok_front) = 0
- Defective (def_front) = 1

## How to Run
1. Install dependencies: pip install -r requirements.txt
2. Download the dataset from Kaggle and place it as casting_data/train and casting_data/test
3. Open notebooks/casting_defect_detection.ipynb
4. Run all cells in order (Colab with GPU runtime recommended)

## Results
Test accuracy: 91.3% at default threshold (0.50). See reports/findings_report.md for the full write-up including threshold tuning analysis.

## Folder Structure
- notebooks/ - the full training notebook
- models/ - saved best model (.keras)
- reports/ - accuracy/loss graphs, confusion matrix, findings report
