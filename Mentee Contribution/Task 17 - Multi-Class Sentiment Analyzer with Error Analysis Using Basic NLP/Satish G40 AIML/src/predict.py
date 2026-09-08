"""Command-line and reusable prediction interface."""

from pathlib import Path
import sys

import joblib

from preprocessing import clean_text


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "sentiment_model.pkl"
VECTORIZER_PATH = ROOT_DIR / "models" / "tfidf_vectorizer.pkl"


def predict_sentiment(sentence: str) -> dict[str, object]:
    """Clean a sentence and return its label, confidence, and all probabilities."""
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        raise FileNotFoundError("Model files are missing. Run 'python src/train.py' first.")

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    cleaned_sentence = clean_text(sentence)
    features = vectorizer.transform([cleaned_sentence])
    probabilities = model.predict_proba(features)[0]
    predicted = model.classes_[probabilities.argmax()]

    return {
        "sentiment": predicted,
        "confidence": float(probabilities.max()),
        "probabilities": {
            label: float(probability)
            for label, probability in zip(model.classes_, probabilities)
        },
    }


if __name__ == "__main__":
    sentence = " ".join(sys.argv[1:]).strip() or "I really enjoyed this experience"
    try:
        result = predict_sentiment(sentence)
        print(f"Text: {sentence}")
        print(f"Prediction: {result['sentiment']}")
        print(f"Confidence: {result['confidence']:.2%}")
        for label, probability in result["probabilities"].items():
            print(f"{label}: {probability:.2%}")
    except FileNotFoundError as error:
        print(error)
