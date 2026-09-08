"""
Sentiment analysis module for Customer Feedback Analysis System.

Implements both classical (TF-IDF + Logistic Regression) and
transformer-based sentiment classification.

Uses sklearn Pipelines throughout to prevent data leakage:
the TF-IDF vectorizer is fitted ONLY on training data.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from .preprocessing import CUSTOM_STOP_WORDS, preprocess_text

# Feature-engine stop words: function words only, NEVER negation words.
FEATURE_STOP_WORDS = CUSTOM_STOP_WORDS


def build_sentiment_pipeline(
    max_features: int = 5000,
    ngram_range: tuple = (1, 2),
    C: float = 1.0,
    random_state: int = 42,
) -> Pipeline:
    """
    Build a TF-IDF + Logistic Regression pipeline for sentiment classification.

    Pipeline ensures data leakage prevention:
    - TfidfVectorizer is fitted only on training data during cross-validation
    - No separate preprocessing step that sees test data

    Parameters
    ----------
    max_features : int
        Maximum vocabulary size for TF-IDF.
    ngram_range : tuple
        Range of n-grams (1,2) = unigrams + bigrams.
    C : float
        Inverse regularization strength for Logistic Regression.
    random_state : int
        Random seed for reproducibility.
    """
    # NOTE on stop words: we only pass a minimal set of boring FUNCTION WORDS
    # (the, is, and...) to TF-IDF. Negation words (not, never, no) are
    # deliberately EXCLUDED from that set because they encode sentiment:
    #   "not good"  must not become "good"
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,       # Apply log normalization
            min_df=2,                # Ignore very rare terms
            max_df=0.95,             # Ignore terms in >95% of docs
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


def prepare_sentiment_data(
    df: pd.DataFrame,
    text_column: str = "feedback",
    label_column: str = "sentiment",
    test_size: float = 0.2,
    random_state: int = 42,
    clean: bool = True,
    sample_weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Prepare data for sentiment classification.

    1. Cleans text (basic cleaning only, no stop word removal to preserve negation)
    2. Encodes labels as integers
    3. Performs stratified train/test split

    Parameters
    ----------
    sample_weights : np.ndarray, optional
        Optional per-row weights (e.g. to upweight a small, high-quality
        supplement dataset). Must have the same length as the dataframe.
        If given, the aligned train/test split of the weights is returned.

    Returns
    -------
    X_train, X_test, y_train, y_test, train_weights
        train_weights is None when sample_weights was not provided.
    """
    df = df.copy()

    # Drop rows with missing text or labels
    df = df.dropna(subset=[text_column, label_column])

    # Clean text
    if clean:
        df["clean_text"] = df[text_column].apply(
            lambda x: preprocess_text(x, clean=True, remove_stops=False, lemmatize=False)
        )
    else:
        df["clean_text"] = df[text_column]

    X = df["clean_text"].values
    y = df[label_column].str.lower().values

    if sample_weights is not None:
        weights = np.asarray(sample_weights, dtype=float)
        weights = weights[df.index.values]

    # Stratified split to maintain class distribution
    if sample_weights is not None:
        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
            X, y, weights, test_size=test_size, random_state=random_state, stratify=y,
        )
        return X_train, X_test, y_train, y_test, w_train

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )
    return X_train, X_test, y_train, y_test, None

    return X_train, X_test, y_train, y_test


def train_sentiment_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_weights: Optional[np.ndarray] = None,
    **kwargs,
) -> Pipeline:
    """
    Train the sentiment classification pipeline.

    Parameters
    ----------
    sample_weights : np.ndarray, optional
        Per-sample weights for the classifier. Useful when combining a
        large automatically-labeled corpus with a small, high-quality
        supplement that should count more heavily.

    Returns the fitted Pipeline object.
    """
    pipeline = build_sentiment_pipeline(**kwargs)
    if sample_weights is not None:
        pipeline.fit(X_train, y_train, classifier__sample_weight=sample_weights)
    else:
        pipeline.fit(X_train, y_train)
    return pipeline


def predict_sentiment(
    model: Pipeline,
    texts: List[str],
    clean: bool = True,
) -> List[str]:
    """
    Predict sentiment for a list of raw text inputs.

    Parameters
    ----------
    model : Pipeline
        Trained sentiment pipeline.
    texts : list of str
        Raw text inputs.
    clean : bool
        Whether to apply preprocessing to inputs.

    Returns
    -------
    list of str
        Predicted sentiment labels.
    """
    if clean:
        cleaned = [preprocess_text(t, clean=True, remove_stops=False) for t in texts]
    else:
        cleaned = texts
    return list(model.predict(cleaned))


def predict_sentiment_proba(
    model: Pipeline,
    texts: List[str],
    clean: bool = True,
) -> np.ndarray:
    """
    Predict sentiment probabilities for a list of texts.

    Useful for confidence-based filtering or ranking.
    """
    if clean:
        cleaned = [preprocess_text(t, clean=True, remove_stops=False) for t in texts]
    else:
        cleaned = texts
    return model.predict_proba(cleaned)
