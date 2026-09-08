"""
category_classifier.py
-----------------------
Feedback category classification.

Two modes are provided:
    1. Single-label   -> TF-IDF + Logistic Regression
    2. Multi-label     -> TF-IDF + OneVsRestClassifier(LogisticRegression)
                          (a feedback message can belong to more than one
                          category, e.g. "app is slow and payment failed"
                          -> ["performance", "payment"])
"""

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

from preprocessing import preprocess


# ---------------------------------------------------------------------
# Single-label category classification
# ---------------------------------------------------------------------

def build_single_label_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])


def train_category_model(texts, categories, test_size: float = 0.2, random_state: int = 42):
    """Train a single-label feedback category classifier."""
    cleaned_texts = [preprocess(t) for t in texts]

    x_train, x_test, y_train, y_test = train_test_split(
        cleaned_texts, categories, test_size=test_size, random_state=random_state
    )

    model = build_single_label_pipeline()
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    print("Category model evaluation")
    print("-" * 40)
    print(classification_report(y_test, predictions, zero_division=0))

    return model


def predict_category(model, text: str) -> str:
    cleaned = preprocess(text)
    return model.predict([cleaned])[0]


# ---------------------------------------------------------------------
# Multi-label category classification
# ---------------------------------------------------------------------

def train_multilabel_category_model(texts, label_lists):
    """
    texts: list[str]
    label_lists: list[list[str]] e.g. [["payment"], ["performance", "payment"]]

    Returns (vectorizer, classifier, mlb) so predictions can be decoded
    back into category names with mlb.inverse_transform().
    """
    cleaned_texts = [preprocess(t) for t in texts]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    x = vectorizer.fit_transform(cleaned_texts)

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(label_lists)

    classifier = OneVsRestClassifier(LogisticRegression(max_iter=1000))
    classifier.fit(x, y)

    return vectorizer, classifier, mlb


def predict_multilabel_categories(vectorizer, classifier, mlb, text: str) -> list:
    cleaned = preprocess(text)
    x = vectorizer.transform([cleaned])
    prediction = classifier.predict(x)
    return list(mlb.inverse_transform(prediction)[0])


def save_artifacts(obj, path: str) -> None:
    joblib.dump(obj, path)


def load_artifacts(path: str):
    return joblib.load(path)


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("../data/feedback.csv")
    model = train_category_model(df["feedback"], df["category"])

    example = "The application freezes frequently"
    print(f"\nExample: {example!r} -> {predict_category(model, example)}")

    # Small multi-label demo
    demo_texts = [
        "Payment keeps failing",
        "The application is very slow",
        "I love the new dashboard",
        "Support solved my issue quickly",
        "Login OTP is not arriving",
        "The app is slow and payment fails",
    ]
    demo_labels = [
        ["payment"],
        ["performance"],
        ["ui"],
        ["support"],
        ["login"],
        ["performance", "payment"],
    ]
    vectorizer, classifier, mlb = train_multilabel_category_model(demo_texts, demo_labels)
    multi_example = "The app is very slow and payment keeps failing"
    print(
        f"Multi-label example: {multi_example!r} -> "
        f"{predict_multilabel_categories(vectorizer, classifier, mlb, multi_example)}"
    )
