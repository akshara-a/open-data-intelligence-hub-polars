"""
Training script for Customer Feedback Analysis System.

Trains and saves all models:
1. Sentiment classifier (TF-IDF + Logistic Regression)
2. Multi-label category classifier (TF-IDF + OneVsRest)
3. Builds keyword index (TF-IDF)
4. Trains sentence embedding index

Optionally trains the Transformt model (slow, use flag).

Usage:
    python train.py                     # Train classical models
    python train.py --transformer       # Also train DistilBERT
"""

import argparse
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Ensure project root is on path so `src` is importable
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.sentiment import prepare_sentiment_data, train_sentiment_model
from src.multilabel_classifier import (
    prepare_multilabel_data,
    train_multilabel_model,
)
from src.keyword_extractor import build_keyword_index
from src.evaluation import (
    compare_models,
    evaluate_single_label,
    evaluate_multilabel,
)


MODEL_DIR = PROJECT_ROOT / "models"


def setup_models_dir() -> Path:
    """Ensure the models directory exists."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return MODEL_DIR


def load_processed_data() -> dict:
    """
    Load all processed datasets needed for training.

    - sentiment:      feedback_sentiment.csv     (14,640 real labels)
    - single-label:   feedback_sentiment_cat.csv (airline complaint categories)
    - multi-label:    manual_categories.csv      (manually labeled supplement)
    """
    processed_dir = PROJECT_ROOT / "data" / "processed"
    required = {
        "sentiment": "feedback_sentiment.csv",
        "single_category": "feedback_sentiment_cat.csv",
        "manual_multilabel": "manual_categories.csv",
    }
    for name, fname in required.items():
        if not (processed_dir / fname).exists():
            raise FileNotFoundError(
                f"Processed data file missing: {fname}\n"
                "Run 'python make_dataset.py' to build the datasets."
            )

    return {
        name: pd.read_csv(processed_dir / fname)
        for name, fname in required.items()
    }


def train_sentiment(
    airline_df: pd.DataFrame,
    manual_df: pd.DataFrame,
) -> dict:
    """
    Train the sentiment classifier and return evaluation results.

    Two models are evaluated:
    1. BASELINE:  TF-IDF + LogisticRegression trained on the airline data only.
       This is the honest benchmark for the raw dataset.
    2. DEPLOYED:  same pipeline but trained on airline data PLUS the small
       manually-labeled app-domain supplement. The supplement rows are
       upweighted (positive x40, negative x20, neutral x8) so the model
       handles app-domain feedback better (e.g. "support solved my issue").
       Matching this deployment choice is documented in the README.
    """
    print("\n" + "=" * 60)
    print("  Training Sentiment Classifier (TF-IDF + Logistic Regression)")
    print("=" * 60)

    # --- 1. Honest baseline on the airline dataset only ---
    X_train, X_test, y_train, y_test, _ = prepare_sentiment_data(
        airline_df, text_column="feedback", label_column="sentiment"
    )
    baseline = train_sentiment_model(X_train, y_train)
    baseline_results = evaluate_single_label(
        y_test, baseline.predict(X_test),
        labels=["negative", "neutral", "positive"],
        task_name="Sentiment (airline baseline)",
    )

    # --- 2. Deployed model with the app-domain supplement ---
    supplement = manual_df[["feedback", "sentiment"]].copy()
    supplement["source"] = "manual"
    airline_df = airline_df.copy()
    airline_df["source"] = "airline"

    combined = pd.concat([airline_df, supplement], ignore_index=True).reset_index(drop=True)

    weights = np.ones(len(combined))
    manual_mask = combined["source"] == "manual"
    sent = combined["sentiment"].str.lower().values
    weights[manual_mask & (sent == "positive")] = 40.0
    weights[manual_mask & (sent == "negative")] = 20.0
    weights[manual_mask & (sent == "neutral")] = 8.0

    X_train, X_test, y_train, y_test, w_train = prepare_sentiment_data(
        combined, text_column="feedback", label_column="sentiment",
        sample_weights=weights,
    )
    model = train_sentiment_model(X_train, y_train, sample_weights=w_train)

    y_pred = model.predict(X_test)
    results = evaluate_single_label(
        y_test, y_pred,
        labels=["negative", "neutral", "positive"],
        task_name="Sentiment (deployed, + augmented)",
        plot=True,
        save_path=str(MODEL_DIR / "sentiment_confusion_matrix.png"),
    )

    joblib.dump(model, MODEL_DIR / "sentiment_model.joblib")
    print(f"\n  Model saved to {MODEL_DIR / 'sentiment_model.joblib'}")

    return {
        "baseline": baseline_results,
        "deployed": results,
    }


def train_multilabel(df: pd.DataFrame) -> dict:
    """Train the multi-label category classifier and return metrics."""
    print("\n" + "=" * 60)
    print("  Training Multi-Label Category Classifier")
    print("=" * 60)

    X_train, X_test, y_train, y_test, mlb = prepare_multilabel_data(
        df, text_column="feedback", label_column="categories"
    )

    model = train_multilabel_model(X_train, y_train)

    # Binary prediction for evaluation
    y_pred_binary = model.predict(X_test)
    results = evaluate_multilabel(
        y_test, y_pred_binary,
        labels=list(mlb.classes_),
        task_name="Multi-Label Category",
    )

    joblib.dump(model, MODEL_DIR / "multilabel_model.joblib")
    joblib.dump(mlb, MODEL_DIR / "multilabel_binarizer.joblib")
    print(f"\n  Model saved to {MODEL_DIR / 'multilabel_model.joblib'}")

    return {
        "micro_f1": results["micro_f1"],
        "macro_f1": results["macro_f1"],
    }


def save_keyword_index(df: pd.DataFrame) -> None:
    """Build and persist TF-IDF keyword index over the corpus."""
    print("\n" + "=" * 60)
    print("  Building Keyword/TF-IDF Index")
    print("=" * 60)

    vectorizer, tfidf_matrix, feature_names = build_keyword_index(
        df, text_column="feedback"
    )
    joblib.dump(vectorizer, MODEL_DIR / "keyword_vectorizer.joblib")
    joblib.dump(feature_names, MODEL_DIR / "keyword_feature_names.joblib")
    joblib.dump(tfidf_matrix, MODEL_DIR / "keyword_tfidf_matrix.joblib")
    joblib.dump(df["feedback"].tolist(), MODEL_DIR / "corpus_texts.joblib")
    print(f"  Index built with {tfidf_matrix.shape[1]} features")
    print(f"  Saved to {MODEL_DIR}")


def train_embeddings(df: pd.DataFrame) -> None:
    """Build sentence embedding index for semantic similarity."""
    print("\n" + "=" * 60)
    print("  Building Sentence Embedding Index")
    print("=" * 60)

    try:
        from src.embeddings import EmbeddingEngine

        engine = EmbeddingEngine()
        engine.build_index(
            df["feedback"].tolist(),
            show_progress=False,
        )
        np.save(MODEL_DIR / "embedding_index.npy", engine.embeddings)
        with open(MODEL_DIR / "embedding_texts.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(engine.texts))
        print(f"  Embeddings saved to {MODEL_DIR / 'embedding_index.npy'}")
    except ImportError as e:
        print(f"  [SKIP] {e}")


def train_transformer(df: pd.DataFrame) -> None:
    """Fine-tune DistilBERT for sentiment classification."""
    print("\n" + "=" * 60)
    print("  Training Transformer (DistilBERT) - Sentiment")
    print("=" * 60)

    try:
        from src.transformer_model import (
            TransformerSentimentClassifier,
            SENTIMENT_MAP,
        )

        df = df.dropna(subset=["feedback", "sentiment"])

        # Limit to sentiment column (binary: positive/negative for simplicity,
        # but keep neutral if present)
        classifier = TransformerSentimentClassifier(num_labels=3)

        train_ds, test_ds = classifier.prepare_data(
            df["feedback"].tolist(),
            df["sentiment"].tolist(),
            SENTIMENT_MAP,
        )

        classifier.train(
            train_ds,
            output_dir=str(MODEL_DIR / "transformer_sentiment"),
            epochs=3,
        )

        # Save the model
        classifier.model.save_pretrained(MODEL_DIR / "transformer_sentiment")
        classifier.tokenizer.save_pretrained(MODEL_DIR / "transformer_sentiment")

        # Evaluate
        result = classifier.evaluate(test_ds)
        print("\n  Transformer Evaluation Results:")
        for k, v in result.items():
            print(f"    {k}: {v:.4f}")

    except ImportError as e:
        print(f"  [SKIP] {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train models for Customer Feedback Analysis System."
    )
    parser.add_argument(
        "--transformer",
        action="store_true",
        help="Also train the DistilBERT transformer model (slow).",
    )
    args = parser.parse_args()

    if not os.path.exists(MODEL_DIR):
        setup_models_dir()

    try:
        datasets = load_processed_data()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    sentiment_df = datasets["sentiment"]
    single_df = datasets["single_category"]
    multilabel_df = datasets["manual_multilabel"]

    sentiment_results = train_sentiment(sentiment_df, multilabel_df)

    compare_models(
        {
            "airline-only": {k: v for k, v in sentiment_results["baseline"].items()
                             if k in ("accuracy", "precision", "recall", "f1")},
            "deployed": {k: v for k, v in sentiment_results["deployed"].items()
                         if k in ("accuracy", "precision", "recall", "f1")},
        }
    )

    save_keyword_index(sentiment_df)
    train_embeddings(sentiment_df)

    print("\n" + "=" * 60)
    print("  NOTE: single-label category training is demonstrated in")
    print("  notebooks/04_multilabel_classification.ipynb using the airline")
    print("  complaint categories (feedback_sentiment_cat.csv).")
    print("=" * 60)

    multilabel_results = train_multilabel(multilabel_df)

    if args.transformer:
        train_transformer(sentiment_df)

    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print(f"  All models saved in: {MODEL_DIR}")


if __name__ == "__main__":
    main()
