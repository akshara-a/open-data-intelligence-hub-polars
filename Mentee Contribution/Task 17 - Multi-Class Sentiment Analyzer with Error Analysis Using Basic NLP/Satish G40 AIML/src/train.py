"""Train, evaluate, and save the basic NLP sentiment analyzer."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from error_analysis import build_error_analysis, save_error_analysis
from preprocessing import prepare_text_column


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "sentiment_data.csv"
MODEL_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"
LABELS = ["Negative", "Neutral", "Positive"]


def save_distribution_plot(labels: pd.Series) -> None:
    counts = labels.value_counts().reindex(["Positive", "Neutral", "Negative"], fill_value=0)
    plt.figure(figsize=(8, 5))
    bars = plt.bar(counts.index, counts.values, color=["#238636", "#6e7781", "#cf222e"])
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of samples")
    plt.bar_label(bars)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sentiment_distribution.png", dpi=160)
    plt.close()


def save_confusion_matrix(actual: pd.Series, predicted: np.ndarray) -> None:
    matrix = confusion_matrix(actual, predicted, labels=LABELS)
    fig, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(LABELS)),
        yticks=np.arange(len(LABELS)),
        xticklabels=LABELS,
        yticklabels=LABELS,
        xlabel="Predicted",
        ylabel="Actual",
        title="Confusion Matrix",
    )
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                matrix[row, column],
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def train() -> dict[str, object]:
    """Run the full training pipeline and return metrics for the app."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    dataframe = pd.read_csv(DATA_PATH)
    required_columns = {"text", "sentiment"}
    if not required_columns.issubset(dataframe.columns):
        raise ValueError("Dataset must contain 'text' and 'sentiment' columns.")

    dataframe["text"] = dataframe["text"].fillna("")
    dataframe["sentiment"] = dataframe["sentiment"].fillna("Neutral").str.title()
    dataframe = dataframe[dataframe["sentiment"].isin(LABELS)].copy()
    dataframe["clean_text"] = prepare_text_column(dataframe)
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    print("First 5 rows:")
    print(dataframe[["text", "sentiment"]].head().to_string(index=False))
    print(f"\nShape: {dataframe.shape}")
    print(f"Columns: {list(dataframe.columns)}")
    print(f"Missing values:\n{dataframe[['text', 'sentiment']].isna().sum()}")
    print(f"Class distribution:\n{dataframe['sentiment'].value_counts().to_string()}")
    save_distribution_plot(dataframe["sentiment"])

    x_train, x_test, y_train, y_test = train_test_split(
        dataframe["clean_text"],
        dataframe["sentiment"],
        test_size=0.2,
        random_state=42,
        stratify=dataframe["sentiment"],
    )
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    x_train_features = vectorizer.fit_transform(x_train)
    x_test_features = vectorizer.transform(x_test)
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train_features, y_train)
    predictions = model.predict(x_test_features)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, average="weighted", zero_division=0),
        "recall": recall_score(y_test, predictions, average="weighted", zero_division=0),
        "f1": f1_score(y_test, predictions, average="weighted", zero_division=0),
        "classification_report": classification_report(
            y_test, predictions, labels=LABELS, zero_division=0
        ),
    }
    print("\nEvaluation:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-score: {metrics['f1']:.4f}")
    print(metrics["classification_report"])

    save_confusion_matrix(y_test, predictions)
    results, errors = build_error_analysis(dataframe.loc[x_test.index, "text"], y_test, predictions)
    save_error_analysis(errors, OUTPUT_DIR / "error_analysis.csv")
    error_counts = errors.groupby(["actual", "predicted"]).size().sort_values(ascending=False)
    total_samples = len(y_test)
    error_rate = len(errors) / total_samples if total_samples else 0
    print(f"Total test samples: {total_samples}")
    print(f"Incorrect predictions: {len(errors)}")
    print(f"Error rate: {error_rate:.2%}")
    print(f"Actual -> predicted errors:\n{error_counts.to_string() if not error_counts.empty else 'None'}")
    print("Common error sources: negation, mixed sentiment, neutral language, sarcasm, lack of context, rare words, and short sentences.")

    joblib.dump(model, MODEL_DIR / "sentiment_model.pkl")
    joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.pkl")
    pd.Series(metrics).to_json(OUTPUT_DIR / "metrics.json")
    return metrics


if __name__ == "__main__":
    train()
