"""
Multi-label classification module for Customer Feedback Analysis System.

Handles cases where a single feedback message can belong to multiple
categories simultaneously, e.g.:

    "The app is very slow and payment keeps failing."
    -> ["performance", "payment"]

Uses sklearn's MultiLabelBinarizer + OneVsRestClassifier to decompose
the multi-label problem into multiple binary problems.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline

from .preprocessing import preprocess_text

# Feature-engine stop words: function words only, NEVER negation words.
from .preprocessing import CUSTOM_STOP_WORDS

FEATURE_STOP_WORDS = CUSTOM_STOP_WORDS


def build_multilabel_pipeline(
    max_features: int = 5000,
    ngram_range: tuple = (1, 2),
    C: float = 1.0,
    random_state: int = 42,
    min_df: int = 1,
) -> Pipeline:
    """
    Build a TF-IDF + OneVsRest(LogisticRegression) pipeline for multi-label
    classification.

    OneVsRestClassifier trains one binary classifier per category:
    - "Is this feedback about payment?" -> yes/no
    - "Is this feedback about login?"   -> yes/no
    - etc.

    This allows a single feedback to be assigned to multiple categories.

    Parameters
    ----------
    min_df : int
        Minimum document frequency to keep a term. For small datasets
        (e.g. a few hundred rows) use 1 so vocabulary is not wiped out.
        For large datasets, 2 is safer against noise.
    """
    base_estimator = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            min_df=min_df,
            max_df=0.95,
            strip_accents="unicode",
            stop_words=list(FEATURE_STOP_WORDS),
        )),
        ("classifier", LogisticRegression(
            C=C,
            max_iter=1000,
            random_state=random_state,
            solver="lbfgs",
            class_weight="balanced",  # handle imbalanced categories
        )),
    ])

    return Pipeline([
        ("ovr", OneVsRestClassifier(base_estimator, n_jobs=-1)),
    ])


def prepare_multilabel_data(
    df: pd.DataFrame,
    text_column: str = "feedback",
    label_column: str = "categories",
    test_size: float = 0.2,
    random_state: int = 42,
    clean: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MultiLabelBinarizer]:
    """
    Prepare data for multi-label classification.

    Expects the label column to contain comma-separated category strings,
    e.g. "payment,performance" or lists.

    Returns
    -------
    X_train, X_test, y_train, y_test, mlb
        The MultiLabelBinarizer is returned so the caller can inverse
        transform predictions back to label names.
    """
    df = df.copy()
    df = df.dropna(subset=[text_column, label_column])

    if clean:
        df["clean_text"] = df[text_column].apply(
            lambda x: preprocess_text(x, clean=True, remove_stops=False, lemmatize=False)
        )
    else:
        df["clean_text"] = df[text_column]

    # Parse labels: handle both list and comma-separated string formats
    def parse_labels(val):
        if isinstance(val, list):
            return [l.strip().lower() for l in val]
        if isinstance(val, str):
            return [l.strip().lower() for l in val.split(",")]
        return []

    df["label_list"] = df[label_column].apply(parse_labels)

    X = df["clean_text"].values
    y_raw = df["label_list"].tolist()

    # Fit MultiLabelBinarizer on the full label set
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
    )
    return X_train, X_test, y_train, y_test, mlb


def train_multilabel_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    **kwargs,
) -> Pipeline:
    """Train the multi-label classification pipeline."""
    pipeline = build_multilabel_pipeline(**kwargs)
    pipeline.fit(X_train, y_train)
    return pipeline


def predict_multilabel(
    model: Pipeline,
    texts: List[str],
    mlb: MultiLabelBinarizer,
    threshold: float = 0.5,
    clean: bool = True,
) -> List[List[str]]:
    """
    Predict categories for a list of texts.

    Parameters
    ----------
    model : Pipeline
        Trained multi-label pipeline.
    texts : list of str
        Raw text inputs.
    mlb : MultiLabelBinarizer
        Fitted binarizer to decode predictions.
    threshold : float
        Probability threshold for assigning a label.

    Returns
    -------
    list of list of str
        Predicted category labels for each input.
    """
    if clean:
        cleaned = [preprocess_text(t, clean=True, remove_stops=False) for t in texts]
    else:
        cleaned = texts

    try:
        probas = model.predict_proba(cleaned)
        binary_preds = (probas >= threshold).astype(int)
    except AttributeError:
        # Fallback if predict_proba is not available
        binary_preds = model.predict(cleaned)

    return mlb.inverse_transform(binary_preds)


def predict_multilabel_with_fallback(
    model: Pipeline,
    texts: List[str],
    mlb: MultiLabelBinarizer,
    threshold: float = 0.5,
    top_k: int = 2,
    clean: bool = True,
) -> List[List[str]]:
    """
    Predict categories using a threshold, with a top-k fallback.

    This is a practical pattern for small/noisy datasets: if NO category
    exceeds the probability threshold, we surface the highest-confidence
    categories instead of returning "no category".

    In the notebooks the pure threshold predictions are always reported
    so the reader can judge the raw model behavior; this function simply
    makes the interactive `predict.py` demo more useful.
    """
    if clean:
        cleaned = [preprocess_text(t, clean=True, remove_stops=False) for t in texts]
    else:
        cleaned = texts

    probas = model.predict_proba(cleaned)
    results = []
    for prob in probas:
        above = prob >= threshold
        if above.sum() > 0:
            cats = [mlb.classes_[i] for i in np.where(above)[0]]
        else:
            # Fall back to top-k by probability
            top = np.argsort(prob)[::-1][:top_k]
            cats = [mlb.classes_[i] for i in top]
        results.append(cats)
    return results


def predict_multilabel_topk(
    model: Pipeline,
    texts: List[str],
    mlb: MultiLabelBinarizer,
    top_k: int = 3,
    clean: bool = True,
) -> List[List[Dict[str, float]]]:
    """
    Predict top-k categories with probabilities for each text.
    """
    if clean:
        cleaned = [preprocess_text(t, clean=True, remove_stops=False) for t in texts]
    else:
        cleaned = texts

    probas = model.predict_proba(cleaned)
    classes = mlb.classes_

    results = []
    for prob in probas:
        top_indices = np.argsort(prob)[::-1][:top_k]
        top_cats = [
            {"category": classes[i], "probability": float(prob[i])}
            for i in top_indices
        ]
        results.append(top_cats)
    return results
