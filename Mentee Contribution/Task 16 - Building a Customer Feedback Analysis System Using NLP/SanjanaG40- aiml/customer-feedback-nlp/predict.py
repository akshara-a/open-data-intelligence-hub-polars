"""
Prediction interface for Customer Feedback Analysis System.

Allows the user to enter one or multiple feedback messages and get
a full analysis: sentiment, categories, and important keywords/phrases.

Usage:
    python predict.py                   # Single input (prompt)
    python predict.py --multi           # Multiple inputs
    python predict.py --file input.txt  # Read from file

Examples:
    $ python predict.py
    Enter your feedback: The application is very slow and payment keeps failing.

    ========================================
    CUSTOMER FEEDBACK ANALYSIS
    ========================================

    Feedback:
    The application is very slow and payment keeps failing.

    Sentiment:
    Negative

    Categories:
    - Performance
    - Payment

    Important Keywords/Phrases:
    - application slow
    - payment failing
"""

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import preprocess_text
from src.keyword_extractor import (
    extract_feedback_keywords,
    extract_keywords_with_index,
)
from src.multilabel_classifier import predict_multilabel_with_fallback


def load_models():
    """Load all trained models. Returns a dict of models or raises if missing."""
    model_dir = PROJECT_ROOT / "models"

    required = [
        "sentiment_model.joblib",
        "multilabel_model.joblib",
        "multilabel_binarizer.joblib",
    ]

    missing = [m for m in required if not (model_dir / m).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing model files: {', '.join(missing)}\n"
            "Please run 'python train.py' to train the models first."
        )

    models = {
        "sentiment": joblib.load(model_dir / "sentiment_model.joblib"),
        "multilabel": joblib.load(model_dir / "multilabel_model.joblib"),
        "mlb": joblib.load(model_dir / "multilabel_binarizer.joblib"),
    }

    # Optional: corpus TF-IDF index for better keyword extraction
    kw_vec_path = model_dir / "keyword_vectorizer.joblib"
    kw_names_path = model_dir / "keyword_feature_names.joblib"
    models["kw_vectorizer"] = (
        joblib.load(kw_vec_path) if kw_vec_path.exists() else None
    )
    models["kw_feature_names"] = (
        joblib.load(kw_names_path) if kw_names_path.exists() else None
    )

    return models


CATEGORY_DISPLAY_NAMES = {
    "payment": "Payment",
    "login": "Login",
    "performance": "Performance",
    "support": "Support",
    "ui": "UI",
    "bug": "Bug",
    "feature_request": "Feature Request",
    "general": "General",
}


def display_category(category: str) -> str:
    """Map a category slug to a human-friendly display name."""
    return CATEGORY_DISPLAY_NAMES.get(category.lower(), category.title())


def analyze_feedback(
    feedback: str,
    models: dict,
    extract_keywords: bool = True,
    num_keywords: int = 5,
) -> dict:
    """
    Analyze a single feedback message.

    Returns a dict with sentiment, categories, and keywords.
    """
    # 1. Sentiment prediction
    sentiment = models["sentiment"].predict(
        [preprocess_text(feedback, clean=True, remove_stops=False)]
    )[0]

    # 2. Multi-label category prediction
    #    We use the threshold + top-k fallback so that low-confidence texts
    #    still get the most likely categories instead of an empty result.
    try:
        cleaned = preprocess_text(feedback, clean=True, remove_stops=False)
        categories = predict_multilabel_with_fallback(
            models["multilabel"], [cleaned], models["mlb"]
        )[0]
        categories = [c for c in categories if c]
    except Exception:
        categories = []

    # 3. Keyword extraction
    #    Preferred method uses the corpus-fitted TF-IDF index (words that are
    #    rare across all feedback are more informative). Falls back to the
    #    standalone method if the index is not available.
    keywords = []
    if extract_keywords:
        try:
            keywords = extract_keywords_with_index(
                feedback,
                vectorizer=models.get("kw_vectorizer"),
                feature_names=models.get("kw_feature_names"),
                top_n=num_keywords,
            )
        except Exception:
            keywords = extract_feedback_keywords(feedback, top_n=num_keywords)

    return {
        "feedback": feedback,
        "sentiment": sentiment.capitalize() if sentiment else "Unknown",
        "categories": (
            [display_category(c) for c in categories]
            if categories
            else ["General"]
        ),
        "keywords": keywords,
    }


def format_output(result: dict) -> str:
    """Format the analysis result into a nice display string."""
    lines = []
    lines.append("=" * 42)
    lines.append("CUSTOMER FEEDBACK ANALYSIS")
    lines.append("=" * 42)
    lines.append("")
    lines.append("Feedback:")
    lines.append(f"  {result['feedback']}")
    lines.append("")
    lines.append("Sentiment:")
    lines.append(f"  {result['sentiment']}")
    lines.append("")
    lines.append("Categories:")
    for i, cat in enumerate(result["categories"], 1):
        lines.append(f"  {cat}")
    lines.append("")
    lines.append("Important Keywords/Phrases:")
    for i, kw in enumerate(result["keywords"], 1):
        lines.append(f"  {kw}")
    lines.append("")
    lines.append("=" * 42)
    return "\n".join(lines)


def interactive_single(models: dict) -> None:
    """Interactive single-input mode."""
    print("\nEnter your feedback (or 'q' to quit):")
    while True:
        try:
            feedback = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if feedback.lower() in ("q", "quit", "exit"):
            break

        if not feedback:
            print("  Please enter some text.")
            continue

        try:
            result = analyze_feedback(feedback, models)
            print("\n" + format_output(result))
        except Exception as e:
            print(f"  Error analyzing feedback: {e}")


def interactive_multi(models: dict) -> None:
    """Multi-input mode: read multiple feedback strings."""
    print("Enter feedback (one per line). Enter a blank line to finish:")
    texts = []
    while True:
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            break

        if not line.strip():
            break
        texts.append(line.strip())

    if not texts:
        print("No input provided.")
        return

    print("\nProcessing {} feedback messages...\n".format(len(texts)))
    for fb in texts:
        try:
            result = analyze_feedback(fb, models)
            print(format_output(result))
            print()
        except Exception as e:
            print(f"  Error analyzing: {fb[:50]}... -> {e}")
            print()


def read_from_file(path: str, models: dict) -> None:
    """Read feedback from a file (one per line or JSON list)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: File not found: {path}")
        return

    # Try JSON first
    try:
        data = json.loads(content)
        if isinstance(data, list):
            texts = [str(x) for x in data]
        else:
            texts = [str(data)]
    except json.JSONDecodeError:
        texts = [line.strip() for line in content.splitlines() if line.strip()]

    print(f"Processing {len(texts)} feedback messages...\n")
    for fb in texts:
        try:
            result = analyze_feedback(fb, models)
            print(format_output(result))
            print()
        except Exception as e:
            print(f"  Error analyzing: {fb[:50]}... -> {e}")
            print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Customer Feedback Analysis System - Prediction Interface"
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--multi",
        action="store_true",
        help="Read multiple feedback messages (one per line).",
    )
    mode_group.add_argument(
        "--file",
        type=str,
        metavar="PATH",
        help="Read feedback from a file (one per line or JSON list).",
    )
    parser.add_argument(
        "--no-keywords",
        action="store_true",
        help="Skip keyword extraction.",
    )
    args = parser.parse_args()

    try:
        models = load_models()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    if args.file:
        read_from_file(args.file, models)
    elif args.multi:
        interactive_multi(models)
    else:
        print("=" * 42)
        print("CUSTOMER FEEDBACK ANALYSIS SYSTEM")
        print("Using trained NLP models to analyze feedback")
        print("=" * 42)
        interactive_single(models)


if __name__ == "__main__":
    main()
