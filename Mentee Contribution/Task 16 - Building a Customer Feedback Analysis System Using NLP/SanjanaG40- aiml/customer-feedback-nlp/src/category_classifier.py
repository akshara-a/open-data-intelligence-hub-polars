"""
Category classifier module for Customer Feedback Analysis System.

Implements single-label category classification using TF-IDF + Logistic Regression.
For multi-label classification, see multilabel_classifier.py.

Categories follow the schema:
  payment, login, performance, support, ui, bug, feature_request, general
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from .preprocessing import CUSTOM_STOP_WORDS, preprocess_text

# Feature-engine stop words: function words only, NEVER negation words.
FEATURE_STOP_WORDS = CUSTOM_STOP_WORDS

# Default category labels
DEFAULT_CATEGORIES: List[str] = [
    "payment",
    "login",
    "performance",
    "support",
    "ui",
    "bug",
    "feature_request",
    "general",
]


def build_category_pipeline(
    max_features: int = 5000,
    ngram_range: tuple = (1, 2),
    C: float = 1.0,
    random_state: int = 42,
) -> Pipeline:
    """
    Build a TF-IDF + Logistic Regression pipeline for category classification.

    Similar to the sentiment pipeline but configured for multi-class
    category prediction.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            min_df=2,
            max_df=0.95,
            strip_accents="unicode",
            stop_words=list(FEATURE_STOP_WORDS),
        )),
        ("classifier", LogisticRegression(
            C=C,
            max_iter=1000,
            random_state=random_state,
            solver="lbfgs",
        )),
    ])


def prepare_category_data(
    df: pd.DataFrame,
    text_column: str = "feedback",
    label_column: str = "category",
    test_size: float = 0.2,
    random_state: int = 42,
    clean: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare data for category classification.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    df = df.copy()
    df = df.dropna(subset=[text_column, label_column])

    if clean:
        df["clean_text"] = df[text_column].apply(
            lambda x: preprocess_text(x, clean=True, remove_stops=False, lemmatize=False)
        )
    else:
        df["clean_text"] = df[text_column]

    X = df["clean_text"].values
    y = df[label_column].str.lower().values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )
    return X_train, X_test, y_train, y_test


def train_category_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    **kwargs,
) -> Pipeline:
    """Train the category classification pipeline."""
    pipeline = build_category_pipeline(**kwargs)
    pipeline.fit(X_train, y_train)
    return pipeline


def predict_category(
    model: Pipeline,
    texts: List[str],
    clean: bool = True,
) -> List[str]:
    """
    Predict category for a list of raw text inputs.
    """
    if clean:
        cleaned = [preprocess_text(t, clean=True, remove_stops=False) for t in texts]
    else:
        cleaned = texts
    return list(model.predict(cleaned))


def predict_category_proba(
    model: Pipeline,
    texts: List[str],
    clean: bool = True,
    top_k: int = 3,
) -> List[List[Dict[str, float]]]:
    """
    Predict top-k category probabilities for each text input.

    Returns a list of lists, where each inner list contains
    {"category": str, "probability": float} dicts sorted by probability.
    """
    if clean:
        cleaned = [preprocess_text(t, clean=True, remove_stops=False) for t in texts]
    else:
        cleaned = texts

    probas = model.predict_proba(cleaned)
    classes = model.classes_

    results = []
    for prob in probas:
        top_indices = np.argsort(prob)[::-1][:top_k]
        top_cats = [
            {"category": classes[i], "probability": float(prob[i])}
            for i in top_indices
        ]
        results.append(top_cats)
    return results
