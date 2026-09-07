# Multi-Class Sentiment Analyzer

## Project Overview

This project implements a Multi-Class Sentiment Analyzer using Natural Language Processing (NLP) and Machine Learning techniques. The model classifies text into three sentiment categories:

- Positive
- Neutral
- Negative

The project follows a complete NLP pipeline including data loading, text cleaning, feature extraction, model training, evaluation, and error analysis.

## Project Objectives

1. Load and explore a multi-class sentiment dataset
2. Clean and preprocess text data
3. Convert text to numerical features using TF-IDF
4. Train a Logistic Regression classifier
5. Evaluate model performance using multiple metrics
6. Perform error analysis to identify model weaknesses
7. Save the trained model for future use

## Dataset

Source: https://huggingface.co/datasets/Sp1786/multiclass-sentiment-analysis-dataset

Dataset Statistics:

- Total Samples: 41,643
- Classes: Positive, Neutral, Negative

Class Distribution:

| Class | Samples | Percentage |
|-------|---------|------------|
| Neutral | 15,507 | 37.2% |
| Positive | 13,968 | 33.5% |
| Negative | 12,168 | 29.2% |

Features:

- text: The review or sentence text
- sentiment: Sentiment label (positive, neutral, negative)

## Technologies Used

| Library | Purpose |
|---------|---------|
| pandas | Data manipulation and analysis |
| numpy | Numerical operations |
| re | Regular expressions for text cleaning |
| matplotlib | Data visualization |
| scikit-learn | Machine learning pipeline |
| joblib | Model serialization |

Key scikit-learn Modules:

- TfidfVectorizer: Text feature extraction
- LogisticRegression: Classification model
- train_test_split: Data splitting
- accuracy_score: Accuracy calculation
- classification_report: Precision, recall, and F1-score
- confusion_matrix: Confusion matrix generation
- ConfusionMatrixDisplay: Confusion matrix visualization

## Project Structure

sentiment-analysis/
|
├── sentiment_analysis.ipynb
├── sentiment_model.pkl
├── tfidf_vectorizer.pkl
├── README.md
└── requirements.txt

## Methodology

### 1. Load Dataset

- Load train, validation, and test splits
- Combine the required datasets into a single DataFrame
- Check the dataset structure and class distribution

### 2. Text Preprocessing

The text data is cleaned using the following steps:

- Convert text to lowercase
- Remove URLs
- Remove mentions
- Remove special characters
- Remove numbers
- Remove extra spaces

### 3. Feature Extraction

Text is converted into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency).

Configuration:

- max_features: 5000
- ngram_range: (1, 2)
- TF-IDF is fitted on training data only

Both unigrams and bigrams are used to capture individual words as well as word combinations.

### 4. Model Training

A Logistic Regression classifier is trained on the TF-IDF features.

Configuration:

- Algorithm: Logistic Regression
- max_iter: 1000
- random_state: 42

### 5. Model Evaluation

The trained model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Error Analysis

### 6. Model Saving and Testing

The trained model and TF-IDF vectorizer are saved using joblib so they can be reused later for predictions on new text.

## Results

### Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 67.68% |
| Error Rate | 32.32% |
| Total Test Samples | 8,329 |
| Total Errors | 2,692 |

### Per-Class Performance

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Negative | 0.694 | 0.639 | 0.665 |
| Neutral | 0.603 | 0.664 | 0.632 |
| Positive | 0.757 | 0.724 | 0.740 |

## Error Analysis

| Error Type | Count |
|------------|-------|
| Negative to Neutral | 734 |
| Positive to Neutral | 623 |
| Neutral to Negative | 538 |
| Neutral to Positive | 504 |
| Positive to Negative | 148 |
| Negative to Positive | 145 |

## Key Insights

1. Positive sentiment is the easiest class for the model to classify, achieving 0.757 precision and 0.724 recall.
2. Neutral sentiment is the most challenging class, with a precision of 0.603.
3. The model frequently confuses neutral sentiment with slightly positive or negative sentiment.
4. Strong sentiment expressions are generally classified more accurately.
5. Most classification errors occur between Neutral and Positive and Neutral and Negative classes.

## How to Use

### 1. Install Dependencies

Run the following command:

pip install pandas numpy matplotlib scikit-learn joblib

### 2. Run the Jupyter Notebook

Open the following file:

sentiment_analysis.ipynb

Run the notebook cells in order to:

- Load the dataset
- Preprocess the text
- Generate TF-IDF features
- Train the Logistic Regression model
- Evaluate performance
- Perform error analysis
- Save the trained model and vectorizer

### 3. Load the Saved Model

The trained files are:

sentiment_model.pkl
tfidf_vectorizer.pkl

They can be loaded using joblib:

import joblib

model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

### 4. Predict Sentiment for New Text

Example:

text = ["I really enjoyed this product"]

text_features = vectorizer.transform(text)
prediction = model.predict(text_features)

print("Predicted Sentiment:", prediction[0])

## Future Improvements

The model's performance could potentially be improved by:

- Using advanced text preprocessing techniques
- Applying class balancing techniques
- Increasing the TF-IDF feature size
- Experimenting with different n-gram ranges
- Hyperparameter tuning for Logistic Regression
- Testing other classifiers such as Linear SVM
- Using word embeddings
- Using transformer-based models such as BERT

## Conclusion

This project demonstrates a complete Natural Language Processing sentiment classification pipeline using TF-IDF and Logistic Regression.

The model achieves an overall accuracy of 67.68% across three sentiment classes: Positive, Neutral, and Negative.

The error analysis shows that the primary challenge is distinguishing neutral sentiment from mildly positive or negative text. Despite this limitation, the project provides a strong baseline for multi-class sentiment analysis and can be further improved using advanced NLP techniques.

## License

This project is intended for educational and learning purposes.