"""
embeddings.py
-------------
Sentence embeddings (via sentence-transformers) and cosine similarity,
used for detecting semantically similar customer feedback.

If sentence-transformers is not installed, a lightweight TF-IDF based
fallback is used so the module still runs end to end.
"""

import numpy as np


def _cosine_similarity(a, b) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class SentenceEmbedder:
    """
    Wraps sentence-transformers' all-MiniLM-L6-v2 model when available.
    Falls back to a TF-IDF vector representation otherwise, so the rest
    of the pipeline (similarity search) still works without the extra
    dependency.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._backend = None
        self._fallback_vectorizer = None
        try:
            from sentence_transformers import SentenceTransformer
            self._backend = SentenceTransformer(model_name)
        except Exception:
            self._backend = None

    def encode(self, sentences):
        if self._backend is not None:
            return self._backend.encode(sentences)

        # Fallback: TF-IDF vectors used purely for similarity comparison.
        from sklearn.feature_extraction.text import TfidfVectorizer
        if self._fallback_vectorizer is None:
            self._fallback_vectorizer = TfidfVectorizer().fit(sentences)
        return self._fallback_vectorizer.transform(sentences).toarray()


def find_most_similar(query: str, candidates, embedder: SentenceEmbedder = None):
    """
    Returns (best_match, similarity_score) for the candidate feedback
    message that is most semantically similar to the query.
    """
    embedder = embedder or SentenceEmbedder()
    all_sentences = [query] + list(candidates)
    vectors = embedder.encode(all_sentences)

    query_vector, candidate_vectors = vectors[0], vectors[1:]
    scores = [_cosine_similarity(query_vector, v) for v in candidate_vectors]

    best_idx = int(np.argmax(scores))
    return candidates[best_idx], scores[best_idx]


if __name__ == "__main__":
    feedback_a = "Payment failed during checkout."
    feedback_b = "Unable to complete my card transaction."
    feedback_c = "The application interface looks beautiful."

    embedder = SentenceEmbedder()
    vectors = embedder.encode([feedback_a, feedback_b, feedback_c])

    print(f"A vs B similarity: {_cosine_similarity(vectors[0], vectors[1]):.3f}")
    print(f"A vs C similarity: {_cosine_similarity(vectors[0], vectors[2]):.3f}")

    match, score = find_most_similar(feedback_a, [feedback_b, feedback_c], embedder)
    print(f"\nMost similar to {feedback_a!r}: {match!r} (score={score:.3f})")
