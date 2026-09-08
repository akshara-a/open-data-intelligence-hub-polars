"""
Data loading and dataset construction for Customer Feedback Analysis System.

This module loads the raw Twitter US Airline Sentiment dataset and builds
processed CSVs in the format required by the project:

    processed/feedback_sentiment.csv      -> feedback, sentiment
    processed/feedback_sentiment_cat.csv  -> feedback, sentiment, category
    processed/feedback_multilabel.csv     -> feedback, sentiment, categories
    processed/manual_categories.csv       -> feedback, sentiment, categories

DATASET LABELING STRATEGY (documents the preprocessing decisions)
----------------------------------------------------------------

1. SENTIMENT
   The original dataset provides a real crowd-labeled `airline_sentiment`
   column with values: positive / neutral / negative.
   We keep these labels exactly as they are (no fabrication).

2. SINGLE-LABEL CATEGORY
   The dataset only labels NEGATIVE tweets with a `negativereason`
   (e.g. "Customer Service Issue", "Late Flight"). These reasons are
   airline-specific and do not exactly match the project's app-feedback
   category schema (payment, login, performance, support, ui, bug,
   feature_request, general).

   We therefore map each negativereason to the closest project category
   and clearly document this as an APPROXIMATE mapping:

       Customer Service Issue        -> support
       Flight Attendant Complaints   -> support
       Flight Booking Problems       -> payment   (booking/checkout flow)
       Lost Luggage / Damaged Luggage-> bug       (system/service failure)
       Late Flight / Cancelled Flight/ Bad Flight -> performance
       longlines                     -> performance
       Can't Tell                    -> general

   Tweets without a negativereason DO NOT receive a category label.
   This keeps the single-label category demo honest: categories shown
   here are derived from real, crowd-sourced complaint labels.

3. MULTI-LABEL CATEGORY
   The original dataset does NOT contain reliable multi-label category
   annotations. To demonstrate multi-label classification without
   fabricating labels, we provide a SMALL MANUALLY LABELED supplementary
   dataset (`data/processed/manual_categories.csv`). These rows are
   hand-labeled (single annotator) and clearly marked as such.

MANUAL LABEL SET SCHEMA
    payment, login, performance, support, ui, bug, feature_request, general
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Category schema & negativereason mapping
# ---------------------------------------------------------------------------
CATEGORIES = [
    "payment",
    "login",
    "performance",
    "support",
    "ui",
    "bug",
    "feature_request",
    "general",
]

NEGATIVE_REASON_TO_CATEGORY: Dict[str, str] = {
    "Customer Service Issue": "support",
    "Flight Attendant Complaints": "support",
    "Flight Booking Problems": "payment",
    "Lost Luggage": "bug",
    "Damaged Luggage": "bug",
    "Late Flight": "performance",
    "Cancelled Flight": "performance",
    "Bad Flight": "performance",
    "longlines": "performance",
    "Can't Tell": "general",
}

RAW_TWEETS_PATH = Path(__file__).parent.parent / "data" / "raw" / "Tweets.csv"
MANUAL_CATEGORIES_PATH = (
    Path(__file__).parent.parent / "data" / "processed" / "manual_categories.csv"
)
PROCESSED_NAMES = {
    "sentiment": "feedback_sentiment.csv",
    "single": "feedback_sentiment_cat.csv",
    "multilabel_airline": "feedback_airline_multilabel.csv",
}


def load_raw_tweets(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the raw Twitter US Airline Sentiment dataset.

    Parameters
    ----------
    path : Path, optional
        Path to Tweets.csv. Defaults to data/raw/Tweets.csv.

    Returns
    -------
    pd.DataFrame
        Raw dataset as loaded from disk.
    """
    if path is None:
        path = RAW_TWEETS_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. "
            "Download the Twitter US Airline Sentiment dataset and "
            "place Tweets.csv in data/raw/."
        )
    return pd.read_csv(path)


def build_sentiment_dataset(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Build the base sentiment dataset: feedback + sentiment.

    Uses the real `airline_sentiment` labels from the dataset.
    """
    if df is None:
        df = load_raw_tweets()

    result = pd.DataFrame({
        "feedback": df["text"].astype(str).str.strip(),
        "sentiment": df["airline_sentiment"].str.strip().str.lower(),
    })
    result = result.dropna(subset=["feedback", "sentiment"])
    result = result[result["feedback"].str.len() > 0]
    return result.reset_index(drop=True)


def build_single_label_category_dataset(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Build a single-label category dataset by mapping `negativereason`.

    Only rows with a non-null negativereason get a category.
    The mapping is documented in the module docstring.
    """
    if df is None:
        df = load_raw_tweets()

    rows = []
    for _, row in df.iterrows():
        text = str(row.get("text", "")).strip()
        reason = row.get("negativereason")
        sentiment = str(row.get("airline_sentiment", "")).strip().lower()
        if not text or pd.isna(reason) or str(reason).strip() == "":
            continue

        reason = str(reason).strip()
        category = NEGATIVE_REASON_TO_CATEGORY.get(reason, "general")

        rows.append({
            "feedback": text,
            "sentiment": sentiment,
            "category": category,
        })

    return pd.DataFrame(rows).reset_index(drop=True)


def build_airline_multilabel_dataset(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Build a multi-label version of the airline data.

    Because the original dataset has only ONE negativereason per tweet,
    this is technically single-label with a `categories` column.

    The real multi-label demonstration uses the manually labeled set.
    """
    single = build_single_label_category_dataset(df)
    single["categories"] = single["category"]
    return single[["feedback", "sentiment", "categories"]].reset_index(drop=True)


def load_manual_multilabel_dataset(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the manually labeled multi-label dataset.

    This is the supplementary dataset used to demonstrate multi-label
    classification. It is clearly documented as manually labeled
    (see the dataset header and README section on labeling strategy).
    """
    if path is None:
        path = MANUAL_CATEGORIES_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Manual multi-label dataset not found at {path}."
        )
    return pd.read_csv(path)


def save_processed_datasets(
    output_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """
    Build and save all processed datasets.

    Returns a mapping of dataset name -> file path.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw_tweets()

    sentiment = build_sentiment_dataset(df)
    sentiment_path = output_dir / PROCESSED_NAMES["sentiment"]
    sentiment.to_csv(sentiment_path, index=False)

    single = build_single_label_category_dataset(df)
    single_path = output_dir / PROCESSED_NAMES["single"]
    single.to_csv(single_path, index=False)

    multi = build_airline_multilabel_dataset(df)
    multi_path = output_dir / PROCESSED_NAMES["multilabel_airline"]
    multi.to_csv(multi_path, index=False)

    return {
        "sentiment": sentiment_path,
        "single_category": single_path,
        "airline_multilabel": multi_path,
    }


if __name__ == "__main__":
    paths = save_processed_datasets()
    for name, path in paths.items():
        print(f"{name}: {path}")