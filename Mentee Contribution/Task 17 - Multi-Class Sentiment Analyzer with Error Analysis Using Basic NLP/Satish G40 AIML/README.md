# Multi-Class Sentiment Analyzer

## Introduction
This college mini-project classifies short text into Positive, Neutral, or Negative sentiment using basic natural language processing. It includes evaluation, a confusion matrix, error analysis, and a Streamlit interface.

## Objective
Build a transparent machine-learning pipeline that cleans text, learns TF-IDF features, predicts sentiment with Logistic Regression, and explains mistakes.

## Features
- Three classes: Positive, Neutral, and Negative
- Missing-value handling and basic text cleaning
- 80/20 stratified train/test split with `random_state=42`
- TF-IDF unigram and bigram features
- Accuracy, precision, recall, F1-score, and classification report
- Saved confusion matrix, class distribution chart, and error-analysis CSV
- Command-line prediction and Streamlit app with probabilities

## Technologies
Python, pandas, numpy, re, matplotlib, scikit-learn, joblib, and Streamlit.

## Dataset
`data/sentiment_data.csv` contains the columns `text` and `sentiment` with a balanced sample of short reviews and statements across all three classes.

## Methodology
Dataset -> text cleaning -> train/test split -> TF-IDF -> Logistic Regression -> prediction -> evaluation -> confusion matrix -> error analysis.

### Preprocessing
Text is lowercased, URLs and numbers are removed, punctuation and special characters are removed, extra spaces are collapsed, and missing text values become empty strings.

### TF-IDF explanation
TF-IDF gives higher weight to words that are important in one document but uncommon across the complete training collection. `TfidfVectorizer(max_features=5000, ngram_range=(1,2))` captures both individual words and two-word phrases. It is fitted only on training data.

### Logistic Regression explanation
Logistic Regression learns a weighted decision boundary from the TF-IDF features. Its class probabilities provide a useful confidence estimate for each prediction.

## Evaluation metrics
Accuracy is the fraction of correct predictions. Precision measures how often predicted examples are correct. Recall measures how many examples of a class were found. F1-score balances precision and recall. The script calculates all values from the held-out test set; nothing is hard-coded.

## Confusion matrix
`outputs/confusion_matrix.png` shows actual classes on the vertical axis and predicted classes on the horizontal axis for Positive, Neutral, and Negative.

## Error analysis
`outputs/error_analysis.csv` stores incorrect rows as `text | actual | predicted`. The training output also reports total test samples, incorrect predictions, error rate, and actual-to-predicted error counts. Common error sources include negation, mixed sentiment, neutral language, sarcasm, lack of context, rare words, and short sentences.

## Limitations
The dataset is a small educational sample. Bag-of-words features do not understand deep context, world knowledge, sarcasm, or word order reliably. Confidence is a model probability, not a guarantee.

## Improvements
Use a larger domain-specific dataset, tune hyperparameters with cross-validation, add stemming or lemmatization, inspect calibration, and compare additional classical models such as Linear SVM or Naive Bayes.

## Installation
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Running instructions
```powershell
python src/train.py
python src/predict.py
streamlit run app.py
```
Run training before the app so the model and output files exist.

## Example predictions
```powershell
python src/predict.py "The service was quick and excellent"
python src/predict.py "The experience was ordinary and acceptable"
python src/predict.py "I regret buying this product"
```

## Viva questions
1. Why is the vectorizer fitted only on training data?
2. What does TF-IDF measure?
3. Why use stratification in the split?
4. When can accuracy be misleading?
5. What does a confusion matrix reveal?
6. Why can negation be difficult for a bag-of-words model?
7. What is the difference between precision and recall?
8. Why is Logistic Regression suitable for sparse text features?
9. What does `max_features=5000` control?
10. How would you improve performance with a larger dataset?
