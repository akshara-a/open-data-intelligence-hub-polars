"""
Keyword extraction module for Customer Feedback Analysis System.

Extracts important keywords and phrases from customer feedback using
TF-IDF scoring and n-gram analysis.

Approach:
1. TF-IDF term scoring — identifies terms that are distinctive
   within the feedback corpus
2. N-gram phrase extraction — captures multi-word expressions
   like "payment gateway", "slow loading", "login issue"

This module is deliberately simple and explainable: no neural networks,
just frequency statistics and matrix factorization concepts.
"""

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .preprocessing import CUSTOM_STOP_WORDS, NEGATION_WORDS, clean_text


def extract_keywords_tfidf(
    texts: List[str],
    top_n: int = 10,
    ngram_range: tuple = (1, 2),
    max_features: int = 5000,
) -> List[str]:
    """
    Extract the top-N most important keywords from a collection of texts
    using TF-IDF scores.

    TF-IDF (Term Frequency - Inverse Document Frequency) works by:
    - TF:  How often a word appears in a document (higher = more relevant)
    - IDF: How rare a word is across all documents (rarer = more informative)

    A word that appears frequently in ONE document but rarely in others
    gets a high TF-IDF score, making it a good keyword.

    Parameters
    ----------
    texts : list of str
        Collection of texts to analyze.
    top_n : int
        Number of top keywords to return.
    ngram_range : tuple
        Range of n-grams to consider.
    max_features : int
        Maximum vocabulary size.

    Returns
    -------
    list of str
        Top keywords sorted by importance.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        min_df=1,
        max_df=0.95,
        strip_accents="unicode",
    )

    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    # Average TF-IDF score across all documents
    mean_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()

    # Get top-N indices
    top_indices = mean_scores.argsort()[::-1][:top_n]

    return [feature_names[i] for i in top_indices]


def extract_keywords_for_text(
    text: str,
    vectorizer: Optional[TfidfVectorizer] = None,
    tfidf_matrix: Optional[Any] = None,
    feature_names: Optional[np.ndarray] = None,
    top_n: int = 5,
) -> List[Tuple[str, float]]:
    """
    Extract keywords specific to a single text using a pre-fitted TF-IDF model.

    This finds terms that are important *within* this particular feedback,
    not just globally important.

    Parameters
    ----------
    text : str
        The feedback text to extract keywords from.
    vectorizer : TfidfVectorizer, optional
        Pre-fitted vectorizer. If None, fits a new one on the single text.
    tfidf_matrix : sparse matrix, optional
        Pre-computed TF-IDF matrix.
    feature_names : array, optional
        Feature names from the vectorizer.
    top_n : int
        Number of keywords to extract.

    Returns
    -------
    list of (str, float)
        Keywords with their TF-IDF scores.
    """
    if vectorizer is None or tfidf_matrix is None:
        # Fit on the single text (fallback)
        local_vec = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words="english",
        )
        matrix = local_vec.fit_transform([text])
        names = local_vec.get_feature_names_out()
        scores = matrix.toarray().flatten()
    else:
        # Use the pre-fitted model
        cleaned = clean_text(text)
        row = vectorizer.transform([cleaned])
        scores = row.toarray().flatten()
        names = feature_names

    # Get non-zero scores
    nonzero_mask = scores > 0
    nonzero_scores = scores[nonzero_mask]
    nonzero_names = names[nonzero_mask]

    # Sort by score descending
    top_indices = nonzero_scores.argsort()[::-1][:top_n]
    return [(nonzero_names[i], float(nonzero_scores[i])) for i in top_indices]


def extract_phrases_ngrams(
    text: str,
    n: int = 2,
    min_freq: int = 1,
) -> List[str]:
    """
    Extract n-gram phrases from a single text.

    Simple approach: tokenize, generate n-grams, filter out those
    composed entirely of stop words.

    Parameters
    ----------
    text : str
        Input text.
    n : int
        N-gram size (2 = bigrams, 3 = trigrams).
    min_freq : int
        Minimum frequency to keep (useful when processing multiple texts).

    Returns
    -------
    list of str
        Extracted n-gram phrases.
    """
    cleaned = clean_text(text).lower()
    tokens = cleaned.split()

    # Strip trailing punctuation from tokens so phrases read naturally:
    # "failing." -> "failing", "checkout." -> "checkout"
    tokens = [t.strip(".,!?'\";:()") for t in tokens]
    tokens = [t for t in tokens if t]

    # Filter stop words but keep negation
    meaningful_tokens = [
        t for t in tokens
        if t in NEGATION_WORDS or t not in CUSTOM_STOP_WORDS
    ]

    if len(meaningful_tokens) < n:
        return []

    phrases = []
    for i in range(len(meaningful_tokens) - n + 1):
        gram = " ".join(meaningful_tokens[i:i + n])
        phrases.append(gram)

    return phrases


def extract_feedback_keywords(
    text: str,
    top_n: int = 5,
) -> List[str]:
    """
    High-level keyword extraction for a single feedback string.

    Combines:
    1. TF-IDF single-word keywords
    2. Bigram phrases (meaningful 2-word combinations)

    Returns a deduplicated list of the most important terms.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []

    # Single-word keywords via TF-IDF on this single document
    vec = TfidfVectorizer(
        ngram_range=(1, 1),
        sublinear_tf=True,
        stop_words="english",
    )
    try:
        matrix = vec.fit_transform([cleaned])
    except ValueError:
        return []

    names = vec.get_feature_names_out()
    scores = matrix.toarray().flatten()
    top_indices = scores.argsort()[::-1][:top_n]
    single_words = [names[i] for i in top_indices if scores[i] > 0]

    # Bigram phrases
    bigrams = extract_phrases_ngrams(cleaned, n=2)
    # Deduplicate while preserving order
    seen = set()
    unique_bigrams = []
    for bg in bigrams:
        if bg not in seen:
            seen.add(bg)
            unique_bigrams.append(bg)

    # Combine: take top bigrams first, then fill with single words
    result = []
    for bg in unique_bigrams[:top_n]:
        result.append(bg)
    for w in single_words:
        if w not in " ".join(result):
            result.append(w)
        if len(result) >= top_n:
            break

    return result[:top_n]


def build_keyword_index(
    df: pd.DataFrame,
    text_column: str = "feedback",
    ngram_range: tuple = (1, 2),
    max_features: int = 5000,
) -> Tuple[TfidfVectorizer, np.ndarray, np.ndarray]:
    """
    Build a TF-IDF index over the entire feedback dataset.

    Returns the fitted vectorizer, TF-IDF matrix, and feature names.
    This index is used for per-document keyword extraction.
    """
    texts = df[text_column].fillna("").tolist()
    cleaned = [clean_text(t) for t in texts]

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        min_df=1,
        max_df=0.95,
        strip_accents="unicode",
    )
    tfidf_matrix = vectorizer.fit_transform(cleaned)
    feature_names = vectorizer.get_feature_names_out()

    return vectorizer, tfidf_matrix, feature_names


def extract_keywords_with_index(
    text: str,
    vectorizer: Optional[TfidfVectorizer] = None,
    feature_names: Optional[np.ndarray] = None,
    top_n: int = 5,
) -> List[str]:
    """
    Extract keywords for a single feedback using a corpus-fitted TF-IDF index.

    This is more informative than the standalone version because TF-IDF
    weights are computed against the WHOLE corpus of feedback:
      - a word that is rare across all feedback (e.g. "gateway", "checkout")
        has a high IDF and is therefore ranked as an important keyword
      - a word that appears everywhere (e.g. "app") contributes little

    The top scoring corpus terms are merged with meaningful 2-word phrases
    (which are extracted locally from the text).

    Returns
    -------
    list of str
        Important keywords and phrases for the given feedback.
    """
    if vectorizer is None or feature_names is None:
        return extract_feedback_keywords(text, top_n=top_n)

    cleaned = clean_text(text)
    if not cleaned:
        return []

    # 1. Rank terms by their corpus-aware TF-IDF score in this document
    row = vectorizer.transform([cleaned])
    feature_names = np.asarray(feature_names)
    scores = row.toarray().flatten()
    active = np.where(scores > 0)[0]

    if len(active) == 0:
        return extract_feedback_keywords(text, top_n=top_n)

    # Only consider single-word features (skip n-gram features here so the
    # result stays readable; phrases come from step 2).
    single_mask = np.array([len(feature_names[i].split()) == 1
                            for i in active])
    single_idx = active[single_mask]
    if len(single_idx) > 0:
        ranked = single_idx[np.argsort(scores[single_idx])[::-1]]
        top_terms = [feature_names[i] for i in ranked[:top_n]]
    else:
        top_terms = []

    # Distinguish content words from filler: keep only terms that survive
    # the stop-word filter (but keep negation).
    top_terms = [
        t for t in top_terms
        if (t in NEGATION_WORDS or t not in CUSTOM_STOP_WORDS)
    ]

    # 2. Extract meaningful 2-word phrases from the text
    bigrams = extract_phrases_ngrams(cleaned, n=2)
    seen = set()
    unique_bigrams = []
    for bg in bigrams:
        if bg not in seen:
            seen.add(bg)
            unique_bigrams.append(bg)

    # 3. Merge: corpus-ranked terms + phrases, de-duplicated
    result = []
    for t in top_terms[:max(1, top_n // 2)]:
        if t not in result:
            result.append(t)
    for bg in unique_bigrams[:top_n]:
        if len(result) >= top_n:
            break
        if bg not in result:
            result.append(bg)

    return result[:top_n]
