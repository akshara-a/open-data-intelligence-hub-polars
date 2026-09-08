"""Text cleaning utilities for the sentiment analyzer."""

import re

import pandas as pd


def clean_text(text: object) -> str:
    """Normalize text using basic NLP preprocessing rules."""
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def prepare_text_column(dataframe: pd.DataFrame, column: str = "text") -> pd.Series:
    """Return a cleaned text series, safely handling missing values."""
    if column not in dataframe.columns:
        raise KeyError(f"Required column '{column}' was not found.")
    return dataframe[column].fillna("").map(clean_text)
