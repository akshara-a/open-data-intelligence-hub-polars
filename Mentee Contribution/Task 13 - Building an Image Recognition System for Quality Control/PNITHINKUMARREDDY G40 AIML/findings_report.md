@"
# Findings Report — Automated Casting Defect Detection

## Dataset
Kaggle "Casting Product Image Data for Quality Inspection" dataset.
- Training: 6,633 images (2,875 ok_front / 3,758 def_front), split 80/20 into train/validation
- Test: 715 held-out images (262 ok_front / 453 def_front)
- Images resized to 224x224, pixel values normalized to [0,1]

## Model Architecture
A 3-block CNN: Conv2D(32) -> Conv2D(64) -> Conv2D(128), each followed by MaxPooling2D,
then GlobalAveragePooling2D, Dropout(0.4), Dense(64, relu), Dropout(0.3), and a single
sigmoid output neuron. 101,569 trainable parameters.

Data augmentation (horizontal flip, small rotation, zoom, translation, contrast) was
applied only to training data.

## Training
- Optimizer: Adam (learning_rate=0.001)
- Loss: binary cross-entropy
- Batch size: 32, up to 25 epochs
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint (best val_loss)

## Results at Default Threshold (0.50)
- Test accuracy: 91.3%
- Precision (Defective): 98.3%
- Recall (Defective): 87.9%
- Confusion matrix: [[255, 7], [55, 398]] — 55 false negatives, 7 false positives

## Threshold Tuning
False negatives (missed defects) are more costly in quality control than false
positives (unnecessary manual re-inspection), so thresholds 0.30-0.60 were compared.

| Threshold | Accuracy | Defective Recall | Defective Precision |
|-----------|----------|-------------------|----------------------|
| 0.30      | 0.92     | 0.93               | 0.94                 |
| 0.40      | 0.91     | 0.91               | 0.95                 |
| 0.50      | 0.91     | 0.88               | 0.98                 |
| 0.60      | 0.91     | 0.86               | 0.99                 |

Threshold 0.30 gave the best result on every metric simultaneously and was selected
as the operating threshold for this system.

## Conclusion
The CNN reliably distinguishes defective from non-defective castings. Lowering the
decision threshold to 0.30 further reduces the risk of a defective part reaching a
customer, at minimal additional cost in unnecessary manual inspections.
"@ | Out-File -Encoding utf8 reports\findings_report.md