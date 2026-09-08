"""
Task 17 - Multi-Class Sentiment Analyzer with Error Analysis.

Workflow:
Dataset -> Text Cleaning -> Train/Test Split -> TF-IDF ->
Logistic Regression -> Sentiment Prediction -> Evaluation ->
Confusion Matrix -> Error Analysis
"""

from pathlib import Path
import json
import re

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split


SEED = 42

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "sentiment_data.csv"
PLOTS_DIR = BASE_DIR / "plots"
REPORTS_DIR = BASE_DIR / "reports"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(text: str) -> str:
    """
    Perform basic text cleaning while preserving negation words.
    """
    text = str(text).lower()

    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)

    # Keep letters, numbers, apostrophes and spaces.
    # Important words such as "not", "never", and "cannot" remain.
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def classify_error_type(text: str) -> str:
    """
    Assign a simple explanatory category to an incorrect prediction.

    This is not another ML model. It is a rule-based aid for inspecting
    why a prediction may have been difficult for the basic NLP model.
    """
    text_lower = text.lower()

    negation_patterns = [
        "not ",
        "never ",
        "cannot",
        "can't",
        "nothing wrong",
        "neither ",
    ]

    mixed_patterns = [
        "but",
        "although",
        "however",
        "yet",
        "though",
    ]

    sarcasm_patterns = [
        "wonderful, another",
        "great job making",
    ]

    neutral_patterns = [
        "okay",
        "average",
        "nothing particularly",
        "neither good nor bad",
    ]

    if any(pattern in text_lower for pattern in sarcasm_patterns):
        return "sarcasm_or_irony"

    if any(pattern in text_lower for pattern in negation_patterns):
        return "negation"

    if any(pattern in text_lower for pattern in mixed_patterns):
        return "mixed_opinion"

    if any(pattern in text_lower for pattern in neutral_patterns):
        return "neutral_or_ambiguous_wording"

    return "context_or_vocabulary"


def evaluate_model(model, x_test, y_test) -> dict:
    """Return accuracy and classification metrics."""
    predictions = model.predict(x_test)

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
    }


def save_classification_report(model, x_test, y_test) -> None:
    """Save detailed per-class evaluation metrics."""
    predictions = model.predict(x_test)

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(report).transpose().to_csv(
        REPORTS_DIR / "classification_report.csv"
    )


def save_confusion_matrix(model, x_test, y_test) -> None:
    """Save the sentiment confusion matrix."""
    predictions = model.predict(x_test)
    labels = ["negative", "neutral", "positive"]

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=labels,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels,
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    display.plot(ax=ax, xticks_rotation=45)
    ax.set_title("Multi-Class Sentiment Confusion Matrix")
    fig.tight_layout()

    fig.savefig(
        PLOTS_DIR / "sentiment_confusion_matrix.png",
        dpi=150,
    )

    plt.close(fig)

    pd.DataFrame(
        cm,
        index=labels,
        columns=labels,
    ).to_csv(REPORTS_DIR / "confusion_matrix.csv")


def save_errors(
    model,
    x_test,
    y_test,
    original_text,
) -> pd.DataFrame:
    """Save incorrect predictions for detailed error analysis."""
    predictions = model.predict(x_test)

    errors = pd.DataFrame(
        {
            "text": original_text,
            "actual": y_test,
            "predicted": predictions,
        }
    )

    errors = errors[errors["actual"] != errors["predicted"]].copy()

    if not errors.empty:
        errors["error_type"] = errors["text"].apply(
            classify_error_type
        )
    else:
        errors["error_type"] = pd.Series(dtype=str)

    errors.to_csv(
        REPORTS_DIR / "incorrect_predictions.csv",
        index=False,
    )

    return errors


def save_error_analysis(errors: pd.DataFrame) -> None:
    """Create a summary of the observed model errors."""
    if errors.empty:
        summary = {
            "total_errors": 0,
            "message": (
                "No incorrect predictions were observed on the test set. "
                "A larger or more challenging dataset would be needed "
                "for deeper empirical error analysis."
            ),
        }
    else:
        type_counts = errors["error_type"].value_counts().to_dict()

        summary = {
            "total_errors": int(len(errors)),
            "error_types": {
                key: int(value)
                for key, value in type_counts.items()
            },
        }

    with open(
        REPORTS_DIR / "error_analysis_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2)


def save_class_distribution(df: pd.DataFrame) -> None:
    """Save the sentiment class distribution plot."""
    counts = df["sentiment"].value_counts().reindex(
        ["negative", "neutral", "positive"]
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    counts.plot(kind="bar", ax=ax)

    ax.set_title("Sentiment Class Distribution")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of Records")

    fig.tight_layout()
    fig.savefig(
        PLOTS_DIR / "sentiment_distribution.png",
        dpi=150,
    )

    plt.close(fig)


def predict_sentiment(
    model,
    vectorizer,
    sentences: list[str],
) -> list[str]:
    """
    Predict sentiment for new sentences.

    This function demonstrates how the trained model can be reused
    for new customer feedback.
    """
    cleaned = [clean_text(sentence) for sentence in sentences]
    features = vectorizer.transform(cleaned)

    return model.predict(features).tolist()


def main() -> None:
    """Run the complete sentiment-analysis workflow."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Run generate_dataset.py first."
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = {"text", "sentiment"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing columns: {sorted(missing_columns)}"
        )

    df["clean_text"] = df["text"].apply(clean_text)

    save_class_distribution(df)

    train_indices, test_indices = train_test_split(
        df.index,
        test_size=0.25,
        random_state=SEED,
        stratify=df["sentiment"],
    )

    train_df = df.loc[train_indices]
    test_df = df.loc[test_indices]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
        max_features=5000,
    )

    x_train = vectorizer.fit_transform(
        train_df["clean_text"]
    )
    x_test = vectorizer.transform(
        test_df["clean_text"]
    )

    model = LogisticRegression(
        max_iter=1000,
        random_state=SEED,
    )

    model.fit(
        x_train,
        train_df["sentiment"],
    )

    metrics = evaluate_model(
        model,
        x_test,
        test_df["sentiment"],
    )

    save_classification_report(
        model,
        x_test,
        test_df["sentiment"],
    )

    save_confusion_matrix(
        model,
        x_test,
        test_df["sentiment"],
    )

    errors = save_errors(
        model,
        x_test,
        test_df["sentiment"],
        test_df["text"].values,
    )

    save_error_analysis(errors)

    # Demonstrate prediction on new, unseen sentences.
    new_sentences = [
        "The application is fast and very easy to use.",
        "The service is available today.",
        "The application keeps crashing and is very frustrating.",
    ]

    predictions = predict_sentiment(
        model,
        vectorizer,
        new_sentences,
    )

    prediction_examples = [
        {
            "text": text,
            "predicted_sentiment": prediction,
        }
        for text, prediction in zip(
            new_sentences,
            predictions,
        )
    ]

    with open(
        REPORTS_DIR / "sample_predictions.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            prediction_examples,
            file,
            indent=2,
        )

    summary = {
        "total_records": len(df),
        "training_records": len(train_df),
        "test_records": len(test_df),
        "tfidf_features": x_train.shape[1],
        "model": "TF-IDF + Logistic Regression",
        "ngram_range": [1, 2],
        "max_iterations": 1000,
        "random_seed": SEED,
        "metrics": metrics,
        "incorrect_predictions": len(errors),
        "class_distribution": (
            df["sentiment"].value_counts().to_dict()
        ),
    }

    with open(
        REPORTS_DIR / "metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2)

    print(f"Training records: {len(train_df)}")
    print(f"Test records: {len(test_df)}")
    print(f"TF-IDF features: {x_train.shape[1]}")

    print("\n=== SENTIMENT RESULTS ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    print("\n=== ERROR ANALYSIS ===")
    print(f"Incorrect predictions: {len(errors)}")

    if not errors.empty:
        print("\nIncorrect predictions:")
        print(
            errors[
                ["text", "actual", "predicted", "error_type"]
            ].to_string(index=False)
        )

    print("\n=== SAMPLE PREDICTIONS ===")
    for example in prediction_examples:
        print(
            f"{example['predicted_sentiment']:>8} | "
            f"{example['text']}"
        )

    print("\nReports and plots have been saved.")


if __name__ == "__main__":
    main()