"""Error analysis helpers for model evaluation."""

from pathlib import Path

import pandas as pd


def build_error_analysis(
    texts: pd.Series, actual: pd.Series, predicted: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build all predictions and the subset of incorrect predictions."""
    results = pd.DataFrame(
        {
            "text": texts.to_numpy(),
            "actual": actual.to_numpy(),
            "predicted": getattr(predicted, "to_numpy", lambda: predicted)(),
        }
    )
    return results, results[results["actual"] != results["predicted"]].copy()


def save_error_analysis(errors: pd.DataFrame, output_path: Path) -> None:
    """Save incorrect predictions as a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    errors.to_csv(output_path, index=False)
