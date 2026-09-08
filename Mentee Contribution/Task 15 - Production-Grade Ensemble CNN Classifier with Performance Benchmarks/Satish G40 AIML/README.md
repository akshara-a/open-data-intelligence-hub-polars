# Production-Grade Ensemble CNN Classifier with Performance Benchmarks

A production-oriented deep learning project for CIFAR-10 image classification using TensorFlow/Keras. The project trains three distinct CNN models, combines them through ensemble methods, benchmarks their performance, and evaluates robustness under different data corruptions.

## Project Goal

This project is designed to help you:

- train and compare multiple CNN architectures on CIFAR-10,
- evaluate both individual and ensemble models,
- benchmark inference time, throughput, parameter count, model size, and memory usage,
- assess robustness to common image perturbations,
- analyze model disagreement patterns,
- deploy a prediction pipeline with confidence thresholds.

## Project Structure

```text
ensemble-cnn-classifier/
├── data/
├── models/
├── results/
├── notebooks/
│   └── exploration.ipynb
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── augmentation.py
│   ├── train.py
│   ├── evaluate.py
│   ├── ensemble.py
│   ├── benchmark.py
│   ├── robustness_test.py
│   ├── disagreement_analysis.py
│   ├── predict.py
│   ├── final_report.py
│   ├── deployment_analysis.py
│   └── models/
│       ├── baseline_cnn.py
│       ├── regularized_cnn.py
│       └── deep_cnn.py
├── tests/
│   └── test_ensemble.py
├── requirements.txt
├── README.md
├── .gitignore
└── .gitkeep
```

## Models Included

1. Baseline CNN
   - Convolutional blocks with pooling
   - Simple feature extraction

2. Regularized CNN
   - Batch normalization
   - Dropout
   - Improved generalization

3. Deep CNN
   - Additional convolutional depth
   - GlobalAveragePooling
   - More expressive feature extractor

## Ensemble Methods

- Majority Voting
- Soft Voting
- Weighted Soft Voting

The weighted voting uses model validation accuracy as the weighting signal.

## Dataset

- CIFAR-10
- 10 classes
- 32x32 RGB images
- Automatically downloaded through TensorFlow/Keras

## Setup

This project requires Python 3.10-3.12 for TensorFlow compatibility. Python 3.14 is not supported by TensorFlow 2.15+.

### Windows

```powershell
cd ensemble-cnn-classifier
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Linux/macOS

```bash
cd ensemble-cnn-classifier
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training the Models

Run the full pipeline:

```bash
python -m src.train
```

This will:

- load CIFAR-10,
- split training and validation data,
- apply augmentation to the training set only,
- train all three CNN models,
- save them in the `models/` folder,
- evaluate on the test set,
- save benchmark and summary metrics in the `results/` folder.

## Evaluating Models

```bash
python -m src.evaluate
```

This prints a summary table of the individual model metrics.

## Running Prediction

Use an image and a confidence threshold:

```bash
python -m src.predict --image path/to/image.png --threshold 0.80
```

Example output:

```json
{
  "predictedClass": "cat",
  "confidence": 0.94,
  "decision": "Accepted",
  "inferenceTimeMs": 25.4
}
```

If the confidence is below the threshold, the result is marked as `Manual Review`.

## Running Tests

```bash
pytest -q
```

## Important Notes

- TensorFlow and Keras are required for training and inference.
- Model weights are stored in the `models/` folder.
- The project writes results to `results/`.
- The project is structured for production-oriented experimentation and extension.

## Typical Workflow

```bash
python -m src.train
python -m src.evaluate
python -m src.predict --image sample.png --threshold 0.80
pytest -q
```

## License

This project is intended for educational and research-oriented use.
