import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .preprocessing import clean_text

def extract_keywords(text, max_keywords=8):
    cleaned = clean_text(text)
    if not cleaned:
        return []
    v = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
    matrix = v.fit_transform([cleaned])
    terms = v.get_feature_names_out()
    scores = matrix.toarray()[0]
    ranked = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)
    return [term for term, score in ranked if score > 0][:max_keywords]

def similar_feedback(query, training_texts, top_k=3):
    query = clean_text(query)
    texts = [clean_text(x) for x in training_texts]
    if not query or not texts:
        return []
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    matrix = v.fit_transform(texts + [query])
    scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    indices = np.argsort(scores)[::-1][:top_k]
    return [(training_texts[i], float(scores[i])) for i in indices]
