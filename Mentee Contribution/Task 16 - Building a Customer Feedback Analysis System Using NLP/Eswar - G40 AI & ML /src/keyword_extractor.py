"""
keyword_extractor.py
---------------------
Simple, beginner-friendly keyword / key-phrase extraction using TF-IDF
scores over unigrams and bigrams.
"""

from sklearn.feature_extraction.text import TfidfVectorizer

from preprocessing import preprocess


def extract_keywords_from_corpus(texts, top_n: int = 5, ngram_range=(1, 2)):
    """
    Fits a TF-IDF vectorizer on the whole corpus and returns the top_n
    highest-scoring keywords/phrases for EACH document.

    Returns: list[list[str]] (one keyword list per input document)
    """
    cleaned_texts = [preprocess(t, remove_stops=True) for t in texts]

    vectorizer = TfidfVectorizer(ngram_range=ngram_range)
    tfidf_matrix = vectorizer.fit_transform(cleaned_texts)
    feature_names = vectorizer.get_feature_names_out()

    all_keywords = []
    for row in tfidf_matrix:
        row_data = row.toarray().flatten()
        top_indices = row_data.argsort()[::-1][:top_n]
        keywords = [feature_names[i] for i in top_indices if row_data[i] > 0]
        all_keywords.append(keywords)

    return all_keywords


def extract_keywords_for_text(text: str, corpus, top_n: int = 5, ngram_range=(1, 2)):
    """
    Extract keywords for a single piece of feedback, using `corpus`
    (e.g. the full dataset) to fit the TF-IDF vocabulary so that
    document-frequency weighting is meaningful.
    """
    cleaned_corpus = [preprocess(t, remove_stops=True) for t in corpus]
    cleaned_text = preprocess(text, remove_stops=True)

    vectorizer = TfidfVectorizer(ngram_range=ngram_range)
    vectorizer.fit(cleaned_corpus)

    vector = vectorizer.transform([cleaned_text]).toarray().flatten()
    feature_names = vectorizer.get_feature_names_out()

    top_indices = vector.argsort()[::-1][:top_n]
    return [feature_names[i] for i in top_indices if vector[i] > 0]


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("../data/feedback.csv")
    example = "The payment gateway failed after entering the OTP"

    keywords = extract_keywords_for_text(example, df["feedback"], top_n=5)
    print(f"Feedback: {example!r}")
    print(f"Keywords: {keywords}")
