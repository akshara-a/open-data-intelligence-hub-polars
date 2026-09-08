"""
preprocessing.py
-----------------
Basic NLP text preprocessing utilities for the Customer Feedback
Analysis System.

Covers:
    - Text cleaning
    - Tokenization
    - Careful stop-word removal (keeps negations like "not", "never")
    - Optional lemmatization (spaCy, with a safe fallback)
"""

import re

# Negation words are deliberately kept even when removing stop words,
# because dropping them can flip the meaning of a sentence
# (e.g. "not good" -> "good").
NEGATION_WORDS = {
    "not", "no", "never", "n't", "cannot", "cant", "without",
}

DEFAULT_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "and", "or", "but", "with",
    "this", "that", "these", "those", "it", "its", "as", "so", "very",
    "i", "my", "me", "we", "our", "you", "your",
} - NEGATION_WORDS


def clean_text(text: str) -> str:
    """
    Lowercase, strip, and normalize whitespace / punctuation noise.

    Example:
        "   APP is VERY slow!!!   " -> "app is very slow"
    """
    text = text.lower().strip()
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # URLs
    text = re.sub(r"\S+@\S+", " ", text)                    # emails
    text = re.sub(r"<[^>]+>", " ", text)                     # HTML tags
    text = re.sub(r"[^a-z0-9\s']", " ", text)                # special chars
    text = re.sub(r"\s+", " ", text).strip()                 # extra spaces
    return text


def tokenize(text: str) -> list:
    """Split cleaned text into simple whitespace tokens."""
    return text.split()


def remove_stop_words(tokens: list, stop_words: set = None) -> list:
    """
    Remove common stop words while preserving negation words so that
    sentiment-bearing phrases like "not good" are not damaged.
    """
    stop_words = stop_words if stop_words is not None else DEFAULT_STOP_WORDS
    return [t for t in tokens if t not in stop_words]


def lemmatize_tokens(tokens: list) -> list:
    """
    Lemmatize tokens using spaCy if it is installed and a model is
    available; otherwise fall back to returning the tokens unchanged.
    """
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        except OSError:
            return tokens
        doc = nlp(" ".join(tokens))
        return [token.lemma_ for token in doc]
    except ImportError:
        return tokens


def preprocess(text: str, remove_stops: bool = True, lemmatize: bool = False) -> str:
    """
    Full preprocessing pipeline used before vectorization:
        clean -> tokenize -> (stop words) -> (lemmatize) -> join

    Returns a single cleaned string ready for TfidfVectorizer.
    """
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)

    if remove_stops:
        tokens = remove_stop_words(tokens)
    if lemmatize:
        tokens = lemmatize_tokens(tokens)

    return " ".join(tokens)


if __name__ == "__main__":
    samples = [
        "   APP is VERY slow!!!   ",
        "The application is not good.",
        "Payment failed again",
    ]
    for s in samples:
        print(f"{s!r:45} -> {preprocess(s)!r}")
