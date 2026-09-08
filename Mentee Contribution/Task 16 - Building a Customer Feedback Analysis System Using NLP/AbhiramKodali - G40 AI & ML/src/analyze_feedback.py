"""
Task 16 - Customer Feedback Analysis System Using NLP.

This script:
- Cleans customer feedback while preserving negation.
- Uses TF-IDF with unigrams and bigrams.
- Trains separate sentiment and category classifiers.
- Evaluates both models.
- Creates confusion matrices.
- Extracts important words and phrases.
- Saves incorrect predictions for error analysis.
- Produces an overall feedback summary.
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
DATA_PATH = BASE_DIR / "data" / "feedback.csv"
PLOTS_DIR = BASE_DIR / "plots"
REPORTS_DIR = BASE_DIR / "reports"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(text: str) -> str:
    """
    Clean feedback text without removing important negation words.

    We keep words such as 'not', 'no', 'never', and 'cannot' because
    they can strongly affect sentiment.
    """
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)

    # Keep letters, numbers, apostrophes, and spaces.
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def evaluate_model(model, x_test, y_test, label_name: str) -> dict:
    """Evaluate a classifier and return its main metrics."""
    predictions = model.predict(x_test)

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
    }

    pd.DataFrame(report).transpose().to_csv(
        REPORTS_DIR / f"{label_name}_classification_report.csv"
    )

    return metrics


def save_confusion_matrix(model, x_test, y_test, label_name: str) -> None:
    """Create and save a confusion-matrix plot."""
    predictions = model.predict(x_test)
    labels = sorted(pd.Series(y_test).unique())

    cm = confusion_matrix(y_test, predictions, labels=labels)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels,
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    display.plot(ax=ax, xticks_rotation=45)
    ax.set_title(f"{label_name.title()} Confusion Matrix")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"{label_name}_confusion_matrix.png", dpi=150)
    plt.close(fig)


def save_errors(model, x_test, y_test, original_text, label_name: str) -> pd.DataFrame:
    """Save incorrect predictions for error analysis."""
    predictions = model.predict(x_test)

    errors = pd.DataFrame(
        {
            "feedback": original_text,
            "actual": y_test,
            "predicted": predictions,
        }
    )

    errors = errors[errors["actual"] != errors["predicted"]].copy()

    errors.to_csv(
        REPORTS_DIR / f"{label_name}_errors.csv",
        index=False,
    )

    return errors


def extract_important_terms(vectorizer, model, label_name: str) -> pd.DataFrame:
    """
    Extract highly weighted TF-IDF terms for each class.

    For binary logistic regression, positive coefficients indicate
    stronger association with the second class.
    """
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_

    rows = []

    if len(model.classes_) == 2:
        for class_index, class_name in enumerate(model.classes_):
            if class_index == 0:
                weights = -coefficients[0]
            else:
                weights = coefficients[0]

            top_indices = weights.argsort()[-15:][::-1]

            for rank, index in enumerate(top_indices, start=1):
                rows.append(
                    {
                        "model": label_name,
                        "class": class_name,
                        "rank": rank,
                        "term": feature_names[index],
                        "weight": float(weights[index]),
                    }
                )
    else:
        for class_index, class_name in enumerate(model.classes_):
            weights = coefficients[class_index]
            top_indices = weights.argsort()[-15:][::-1]

            for rank, index in enumerate(top_indices, start=1):
                rows.append(
                    {
                        "model": label_name,
                        "class": class_name,
                        "rank": rank,
                        "term": feature_names[index],
                        "weight": float(weights[index]),
                    }
                )

    return pd.DataFrame(rows)


def create_distribution_plot(df: pd.DataFrame) -> None:
    """Create a simple feedback distribution plot."""
    sentiment_counts = df["sentiment"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(7, 5))
    sentiment_counts.plot(kind="bar", ax=ax)
    ax.set_title("Customer Feedback by Sentiment")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of Feedback Records")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "sentiment_distribution.png", dpi=150)
    plt.close(fig)


def build_summary(df: pd.DataFrame) -> dict:
    """Build a high-level summary of the feedback dataset."""
    sentiment_counts = df["sentiment"].value_counts().to_dict()
    category_counts = df["category"].value_counts().to_dict()

    return {
        "total_feedback_records": len(df),
        "sentiment_distribution": sentiment_counts,
        "category_distribution": category_counts,
        "most_common_sentiment": df["sentiment"].mode().iloc[0],
        "most_common_category": df["category"].mode().iloc[0],
    }


def main() -> None:
    """Run the complete customer-feedback NLP pipeline."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Run generate_dataset.py first."
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = {"feedback", "sentiment", "category"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing columns: {sorted(missing_columns)}"
        )

    df["clean_feedback"] = df["feedback"].apply(clean_text)

    create_distribution_plot(df)

    # Keep the same train/test split for both tasks.
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

    x_train = vectorizer.fit_transform(train_df["clean_feedback"])
    x_test = vectorizer.transform(test_df["clean_feedback"])

    print(f"Training records: {len(train_df)}")
    print(f"Test records: {len(test_df)}")
    print(f"TF-IDF features: {x_train.shape[1]}")

    # ---------------------------------------------------------
    # Sentiment classification
    # ---------------------------------------------------------
    sentiment_model = LogisticRegression(
        max_iter=1000,
        random_state=SEED,
    )

    sentiment_model.fit(
        x_train,
        train_df["sentiment"],
    )

    sentiment_metrics = evaluate_model(
        sentiment_model,
        x_test,
        test_df["sentiment"],
        "sentiment",
    )

    save_confusion_matrix(
        sentiment_model,
        x_test,
        test_df["sentiment"],
        "sentiment",
    )

    sentiment_errors = save_errors(
        sentiment_model,
        x_test,
        test_df["sentiment"],
        test_df["feedback"].values,
        "sentiment",
    )

    # ---------------------------------------------------------
    # Category classification
    # ---------------------------------------------------------
    category_model = LogisticRegression(
        max_iter=1000,
        random_state=SEED,
    )

    category_model.fit(
        x_train,
        train_df["category"],
    )

    category_metrics = evaluate_model(
        category_model,
        x_test,
        test_df["category"],
        "category",
    )

    save_confusion_matrix(
        category_model,
        x_test,
        test_df["category"],
        "category",
    )

    category_errors = save_errors(
        category_model,
        x_test,
        test_df["category"],
        test_df["feedback"].values,
        "category",
    )

    # ---------------------------------------------------------
    # Important terms / phrases
    # ---------------------------------------------------------
    sentiment_terms = extract_important_terms(
        vectorizer,
        sentiment_model,
        "sentiment",
    )

    category_terms = extract_important_terms(
        vectorizer,
        category_model,
        "category",
    )

    important_terms = pd.concat(
        [sentiment_terms, category_terms],
        ignore_index=True,
    )

    important_terms.to_csv(
        REPORTS_DIR / "important_terms.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Error-analysis summary
    # ---------------------------------------------------------
    error_summary = {
        "sentiment_error_count": len(sentiment_errors),
        "category_error_count": len(category_errors),
        "sentiment_error_rate": (
            len(sentiment_errors) / len(test_df)
            if len(test_df)
            else 0
        ),
        "category_error_rate": (
            len(category_errors) / len(test_df)
            if len(test_df)
            else 0
        ),
    }

    with open(
        REPORTS_DIR / "error_analysis_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(error_summary, file, indent=2)

    # ---------------------------------------------------------
    # Overall feedback summary
    # ---------------------------------------------------------
    summary = build_summary(df)

    summary["models"] = {
        "sentiment": "TF-IDF + Logistic Regression",
        "category": "TF-IDF + Logistic Regression",
    }

    summary["test_size"] = len(test_df)
    summary["sentiment_metrics"] = sentiment_metrics
    summary["category_metrics"] = category_metrics
    summary["tfidf_features"] = x_train.shape[1]

    with open(
        REPORTS_DIR / "feedback_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2)

    print("\n=== SENTIMENT RESULTS ===")
    for key, value in sentiment_metrics.items():
        print(f"{key}: {value:.4f}")

    print("\n=== CATEGORY RESULTS ===")
    for key, value in category_metrics.items():
        print(f"{key}: {value:.4f}")

    print("\n=== ERROR ANALYSIS ===")
    print(f"Sentiment errors: {len(sentiment_errors)}")
    print(f"Category errors: {len(category_errors)}")

    print("\n=== OVERALL FEEDBACK SUMMARY ===")
    print(f"Most common sentiment: {summary['most_common_sentiment']}")
    print(f"Most common category: {summary['most_common_category']}")

    print("\nReports and plots have been saved.")


if __name__ == "__main__":
    main()