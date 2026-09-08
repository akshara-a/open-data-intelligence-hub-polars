"""
sentiment.py
------------
Sentiment classification: TF-IDF + Logistic Regression.

Classes: positive / negative / neutral
"""

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from preprocessing import preprocess


def build_pipeline() -> Pipeline:
    """TF-IDF (uni+bigrams) + Logistic Regression baseline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])


def train_sentiment_model(texts, labels, test_size: float = 0.2, random_state: int = 42):
    """
    Cleans the input texts, splits into train/test, fits the pipeline,
    and prints an evaluation report.

    Returns the fitted pipeline.
    """
    cleaned_texts = [preprocess(t) for t in texts]

    x_train, x_test, y_train, y_test = train_test_split(
        cleaned_texts, labels, test_size=test_size, random_state=random_state
    )

    model = build_pipeline()
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    print("Sentiment model evaluation")
    print("-" * 40)
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    return model


def predict_sentiment(model, text: str) -> str:
    """Predict the sentiment (positive / negative / neutral) for one feedback string."""
    cleaned = preprocess(text)
    return model.predict([cleaned])[0]


def save_model(model, path: str = "sentiment_model.joblib") -> None:
    joblib.dump(model, path)


def load_model(path: str = "sentiment_model.joblib"):
    return joblib.load(path)


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("../data/feedback.csv")
    model = train_sentiment_model(df["feedback"], df["sentiment"])

    example = "The support team was excellent"
    print(f"\nExample: {example!r} -> {predict_sentiment(model, example)}")
