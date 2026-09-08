"""
Build the processed datasets from the raw Twitter US Airline Sentiment data.

Usage:
    python make_dataset.py
    python make_dataset.py --max-rows 3000

Outputs:
    data/processed/feedback_sentiment.csv      (feedback, sentiment)
    data/processed/feedback_sentiment_cat.csv  (feedback, sentiment, category)
    data/processed/feedback_airline_multilabel.csv
    data/processed/manual_categories.csv       (manually labeled)

See src/data_loader.py (module docstring) for the full labeling strategy.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import (
    load_raw_tweets,
    build_sentiment_dataset,
    build_single_label_category_dataset,
    build_airline_multilabel_dataset,
)

MANUAL_PATH = PROJECT_ROOT / "data" / "processed" / "manual_categories.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build processed datasets.")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optionally subsample the raw data (useful for quick tests).",
    )
    args = parser.parse_args()

    df = load_raw_tweets()
    if args.max_rows:
        df = df.sample(n=args.max_rows, random_state=42)

    processed_dir = PROJECT_ROOT / "data" / "processed"

    # 1. Sentiment only
    sentiment = build_sentiment_dataset(df)
    path1 = processed_dir / "feedback_sentiment.csv"
    sentiment.to_csv(path1, index=False)
    print(f"  feedback_sentiment.csv: {sentiment.shape[0]} rows -> {path1}")

    # 2. Single-label category (derived from negativereason)
    single = build_single_label_category_dataset(df)
    path2 = processed_dir / "feedback_sentiment_cat.csv"
    single.to_csv(path2, index=False)
    print(f"  feedback_sentiment_cat.csv: {single.shape[0]} rows -> {path2}")

    # 3. Multi-label format version of the airline data
    multi = build_airline_multilabel_dataset(df)
    path3 = processed_dir / "feedback_airline_multilabel.csv"
    multi.to_csv(path3, index=False)
    print(f"  feedback_airline_multilabel.csv: {multi.shape[0]} rows -> {path3}")

    # 4. Validate the manual dataset exists
    if MANUAL_PATH.exists():
        manual = pd.read_csv(MANUAL_PATH)
        print(f"  manual_categories.csv: {manual.shape[0]} rows -> {MANUAL_PATH}")
        print("    NOTE: this file is MANUALLY LABELED (see README).")
    else:
        print("  [WARN] manual_categories.csv not found.")
        print("         Create it to enable the multi-label demonstration.")


if __name__ == "__main__":
    main()