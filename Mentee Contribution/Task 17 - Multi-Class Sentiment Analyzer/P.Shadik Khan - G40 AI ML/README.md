# Task 17 - Multi-Class Sentiment Analyzer

## Project Overview

This project implements a basic NLP-based multi-class sentiment analyzer.

The model classifies text into three sentiment classes:

- Positive
- Neutral
- Negative

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## Methodology

Dataset  
?  
Text Cleaning  
?  
Train-Test Split  
?  
TF-IDF Vectorization  
?  
Logistic Regression  
?  
Sentiment Prediction  
?  
Model Evaluation  
?  
Confusion Matrix  
?  
Error Analysis

## Model

TF-IDF is used to convert text into numerical features.

Logistic Regression is used for multi-class sentiment classification.

## Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## Error Analysis

Incorrect predictions are examined for:

- Negation
- Mixed sentiment
- Neutral sentiment
- Sarcasm
- Rare vocabulary
- Lack of context
- Short sentences

## How to Run

Install dependencies:

pip install -r requirements.txt

Run:

python sentiment_analysis.py

## Output

The results folder contains:

- confusion_matrix.png
- error_analysis.csv
- model_results.txt
