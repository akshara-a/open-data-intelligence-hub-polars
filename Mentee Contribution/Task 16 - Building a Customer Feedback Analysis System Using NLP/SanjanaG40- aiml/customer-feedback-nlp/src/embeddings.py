"""
Sentence embeddings module for Customer Feedback Analysis System.

Uses the sentence-transformers library to compute dense vector
representations of text, enabling semantic similarity search.

Model: all-MiniLM-L6-v2
- 384-dimensional embeddings
- Optimized for semantic similarity tasks
- Fast inference on CPU
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


# Default model name
DEFAULT_MODEL = "all-MiniLM-L6-v2"


class EmbeddingEngine:
    """
    Manages sentence embedding computation and similarity search.

    Wraps sentence-transformers to provide a simple interface for:
    1. Encoding texts into dense vectors
    2. Computing cosine similarity between texts
    3. Finding similar feedback from a corpus

    Usage
    -----
    engine = EmbeddingEngine()
    engine.build_index(texts)
    results = engine.find_similar("Payment failed", top_k=3)
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            )
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embeddings: Optional[np.ndarray] = None
        self.texts: Optional[List[str]] = None

    def encode(
        self,
        texts: List[str],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode a list of texts into dense vector representations.

        Parameters
        ----------
        texts : list of str
            Input texts to encode.
        batch_size : int
            Batch size for encoding (affects memory usage).
        show_progress : bool
            Show progress bar during encoding.

        Returns
        -------
        np.ndarray
            Array of shape (n_texts, 384) containing the embeddings.
        """
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

    def build_index(
        self,
        texts: List[str],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> None:
        """
        Build an embedding index for similarity search.

        Encodes all texts and stores them for later retrieval.
        """
        self.texts = list(texts)
        self.embeddings = self.encode(
            texts, batch_size=batch_size, show_progress=show_progress
        )

    def compute_similarity(
        self,
        text_a: str,
        text_b: str,
    ) -> float:
        """
        Compute cosine similarity between two texts.

        Returns a float between -1 and 1, where:
        - 1.0 = identical meaning
        - 0.0 = unrelated
        - -1.0 = opposite meaning (rare in practice)

        Example
        -------
        >>> engine.compute_similarity(
        ...     "Payment failed during checkout.",
        ...     "Unable to complete my card transaction."
        ... )
        0.72  # High similarity — both about payment problems
        """
        emb_a = self.encode([text_a])[0]
        emb_b = self.encode([text_b])[0]

        sim = cosine_similarity(
            emb_a.reshape(1, -1),
            emb_b.reshape(1, -1),
        )
        return float(sim[0, 0])

    def find_similar(
        self,
        query: str,
        top_k: int = 3,
    ) -> List[Dict]:
        """
        Find the most semantically similar feedback messages from the index.

        Parameters
        ----------
        query : str
            The query text to search for.
        top_k : int
            Number of similar items to return.

        Returns
        -------
        list of dict
            Each dict contains:
            - "text": the matching feedback text
            - "index": position in the original corpus
            - "similarity": cosine similarity score (0-1)

        Example
        -------
        >>> results = engine.find_similar("Login problem", top_k=3)
        >>> for r in results:
        ...     print(f"{r['similarity']:.3f} | {r['text'][:80]}")
        0.856 | I cannot log in to my account since yesterday.
        0.782 | The login page keeps showing an error.
        0.651 | I forgot my password and cannot reset it.
        """
        if self.embeddings is None or self.texts is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        query_embedding = self.encode([query])

        similarities = cosine_similarity(query_embedding, self.embeddings)[0]

        # Get top-k indices (excluding exact matches if desired)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "text": self.texts[idx],
                "index": int(idx),
                "similarity": float(similarities[idx]),
            })

        return results

    def compute_similarity_matrix(
        self,
        texts: List[str],
    ) -> np.ndarray:
        """
        Compute a full pairwise similarity matrix for a list of texts.

        Returns an (n x n) matrix where entry [i][j] is the cosine
        similarity between texts[i] and texts[j].

        Useful for visualizing clusters of similar feedback.
        """
        embeddings = self.encode(texts)
        return cosine_similarity(embeddings)
