"""
Text preprocessing module for Customer Feedback Analysis System.

Handles cleaning, tokenization, and normalization of raw text data.
Uses spaCy for NLP operations and provides utilities for
preparing text for machine learning models.
"""

import re
from typing import List, Optional

import pandas as pd

# Load spaCy English model.
# Use try/except so the project degrades gracefully if spaCy or its model
# is not installed (user can run: pip install spacy && python -m spacy download en_core_web_sm).
try:
    import spacy

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        nlp = None  # Model not downloaded; fall back to regex-only cleaning
except ImportError:
    nlp = None  # spaCy not installed; fall back to regex-only cleaning


# ---------------------------------------------------------------------------
# Custom stop words: we keep negation words because they are critical for
# sentiment analysis.  Removing "not", "no", "never" etc. would flip the
# meaning of sentences like "not good" -> "good".
# ---------------------------------------------------------------------------
CUSTOM_STOP_WORDS: set = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "and", "but", "or",
    "so", "yet", "both", "neither", "each", "every", "all", "any", "few",
    "more", "most", "other", "some", "such", "than", "too", "very",
    "just", "about", "also", "that", "this", "these", "those", "it",
    "its", "i", "me", "my", "we", "our", "you", "your", "he", "him",
    "his", "she", "her", "they", "them", "their", "what", "which",
    "who", "whom", "how", "where", "when", "while", "if", "then",
}
# Negation words we explicitly NEVER remove
NEGATION_WORDS: set = {"not", "no", "never", "neither", "nobody", "nothing",
                        "nowhere", "nor", "cannot", "can't", "don't",
                        "doesn't", "didn't", "won't", "wouldn't", "couldn't",
                        "shouldn't", "isn't", "aren't", "wasn't", "weren't"}


def clean_text(text: str) -> str:
    """
    Apply basic text cleaning: lowercase, remove URLs, emails,
    special characters, and extra whitespace.

    This does NOT remove stop words (see `remove_stop_words` for that).
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove Twitter handles (@username)
    text = re.sub(r"@\w+", "", text)

    # Remove hashtag symbols but keep the text
    text = re.sub(r"#", "", text)

    # Keep only letters, numbers, and basic punctuation.
    # We keep apostrophes so contractions like "don't" survive.
    text = re.sub(r"[^a-zA-Z0-9\s'.,!?]", "", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def remove_stop_words(text: str, aggressive: bool = False) -> str:
    """
    Remove stop words from text while preserving negation words.

    Parameters
    ----------
    text : str
        Input text.
    aggressive : bool
        If True, removes more common words.  Default False keeps the
        text closer to its original form.
    """
    if not isinstance(text, str):
        return ""

    words = text.split()
    # Always keep negation words
    filtered = [
        w for w in words
        if w in NEGATION_WORDS or w not in CUSTOM_STOP_WORDS
    ]
    return " ".join(filtered)


def lemmatize_text(text: str) -> str:
    """
    Lemmatize text using spaCy.

    Converts words to their base/dictionary form:
        "running" -> "run"
        "better"  -> "good"
        "feet"    -> "foot"

    Falls back to the original text if spaCy is not available.
    """
    if nlp is None:
        return text

    doc = nlp(text)
    return " ".join(token.lemma_ for token in doc)


def preprocess_text(
    text: str,
    clean: bool = True,
    remove_stops: bool = False,
    lemmatize: bool = False,
) -> str:
    """
    Full preprocessing pipeline for a single text string.

    Parameters
    ----------
    text : str
        Raw input text.
    clean : bool
        Apply basic cleaning (lowercase, remove URLs, etc.).
    remove_stops : bool
        Remove stop words.  Default False because we want to preserve
        negation for sentiment analysis.
    lemmatize : bool
        Apply lemmatization via spaCy.
    """
    if clean:
        text = clean_text(text)
    if remove_stops:
        text = remove_stop_words(text)
    if lemmatize:
        text = lemmatize_text(text)
    return text


def preprocess_dataframe(
    df: pd.DataFrame,
    text_column: str = "text",
    clean: bool = True,
    remove_stops: bool = False,
    lemmatize: bool = False,
    new_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Apply preprocessing to an entire DataFrame column.

    Returns a copy of the DataFrame with the new cleaned column.
    If new_column is None, overwrites the original column.
    """
    df = df.copy()
    if new_column is None:
        new_column = text_column

    df[new_column] = df[text_column].apply(
        lambda x: preprocess_text(
            x, clean=clean, remove_stops=remove_stops, lemmatize=lemmatize
        )
    )
    return df


def tokenize_text(text: str) -> List[str]:
    """
    Tokenize text into individual tokens.

    Uses spaCy if available, otherwise falls back to whitespace splitting.
    """
    if nlp is not None:
        doc = nlp(text)
        return [token.text for token in doc]
    return text.split()
