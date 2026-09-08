# WeatherNet-05 — Production-Grade Ensemble CNN Classifier

## Overview

This project implements and benchmarks CNN-based image classification models for the WeatherNet-05 dataset.

The project compares:

- CNN 1 — lightweight baseline CNN
- CNN 2 — deeper CNN with batch normalization, dropout, and data augmentation
- Validation-based weighted ensemble
- Production inference performance

The objective is not only to maximize classification accuracy, but also to evaluate model size, inference latency, throughput, and practical production suitability.

---

## Dataset

Dataset: WeatherNet-05

Total original images:

- 18,039

During preprocessing, image records were validated.

Valid images:

- 17,977

Invalid images:

- 62

The invalid records were removed before training and evaluation.

### Classes

The dataset contains 5 classes:

```text
Class 0
Class 1
Class 2
Class 3
Class 4