"""
Generate a reproducible synthetic customer-feedback dataset for Task 16.

The dataset contains:
- feedback
- sentiment: positive / neutral / negative
- category: payment / login / performance / support / ui / bug / feature_request

The generated CSV is intentionally kept out of Git via .gitignore.
"""

from pathlib import Path
import random

import pandas as pd


SEED = 42
SAMPLES_PER_COMBINATION = 18

OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "feedback.csv"
)


TEMPLATES = {
    "payment": {
        "positive": [
            "Payment completed successfully on the first attempt.",
            "The payment process was quick and reliable.",
            "Checkout worked smoothly and my payment went through.",
        ],
        "neutral": [
            "The payment was processed today.",
            "I used the payment option during checkout.",
            "The payment transaction was completed.",
        ],
        "negative": [
            "Payment keeps failing during checkout.",
            "My card payment was declined again.",
            "The payment transaction failed several times.",
        ],
    },
    "login": {
        "positive": [
            "Login worked quickly and I accessed my account.",
            "The sign in process was smooth.",
            "I logged in without any trouble.",
        ],
        "neutral": [
            "I logged into the application this morning.",
            "The account login page is available.",
            "I used my credentials to sign in.",
        ],
        "negative": [
            "I cannot log in to my account.",
            "The login OTP is not arriving.",
            "The sign in page keeps rejecting my credentials.",
        ],
    },
    "performance": {
        "positive": [
            "The application loads very quickly now.",
            "The latest version feels fast and responsive.",
            "Pages open smoothly without delays.",
        ],
        "neutral": [
            "The application performance is acceptable.",
            "Pages load at a normal speed.",
            "The application response time is average.",
        ],
        "negative": [
            "The application is extremely slow.",
            "Pages take too long to load.",
            "The app freezes whenever I open a page.",
        ],
    },
    "support": {
        "positive": [
            "The support team solved my issue very quickly.",
            "Customer support was helpful and professional.",
            "The support agent gave me a clear solution.",
        ],
        "neutral": [
            "I contacted support about my account.",
            "A support ticket was created yesterday.",
            "The support team responded to my request.",
        ],
        "negative": [
            "Support did not resolve my issue.",
            "The support response was slow and unhelpful.",
            "I am still waiting for the support team to fix this.",
        ],
    },
    "ui": {
        "positive": [
            "I love the new dashboard interface.",
            "The updated design looks clean and intuitive.",
            "The new interface is easy to navigate.",
        ],
        "neutral": [
            "The dashboard has the updated interface.",
            "The application uses the new navigation layout.",
            "The interface looks different after the update.",
        ],
        "negative": [
            "The new interface is confusing to use.",
            "The dashboard layout is difficult to navigate.",
            "The updated design makes the app harder to use.",
        ],
    },
    "bug": {
        "positive": [
            "The issue appears to be fixed in the latest version.",
            "The application now works correctly after the update.",
            "The previous error no longer occurs.",
        ],
        "neutral": [
            "I noticed an issue while using the application.",
            "There is a problem on one of the application screens.",
            "I found an unexpected behavior in the app.",
        ],
        "negative": [
            "The application crashes when I submit the form.",
            "There is a serious bug in the checkout screen.",
            "The app shows an error every time I save changes.",
        ],
    },
    "feature_request": {
        "positive": [
            "I would love to have dark mode in the application.",
            "A download option would be a useful addition.",
            "Please add a way to export reports.",
        ],
        "neutral": [
            "I would like to suggest a new application feature.",
            "A new export option could be added to the dashboard.",
            "It would be useful to have another notification setting.",
        ],
        "negative": [
            "It is frustrating that the app does not have dark mode.",
            "The lack of an export option is disappointing.",
            "I am unhappy that this useful feature is still missing.",
        ],
    },
}


PREFIXES = [
    "",
    "Overall, ",
    "For my account, ",
    "Recently, ",
    "After the update, ",
    "In my experience, ",
]

SUFFIXES = [
    "",
    " Please look into it.",
    " This happened more than once.",
    " I noticed this today.",
    " This is affecting my work.",
]


def build_dataset() -> pd.DataFrame:
    """Create balanced synthetic customer-feedback records."""
    random.seed(SEED)

    rows = []

    for category, sentiment_templates in TEMPLATES.items():
        for sentiment, templates in sentiment_templates.items():
            for _ in range(SAMPLES_PER_COMBINATION):
                template = random.choice(templates)
                prefix = random.choice(PREFIXES)
                suffix = random.choice(SUFFIXES)

                feedback = f"{prefix}{template}{suffix}".strip()

                rows.append(
                    {
                        "feedback": feedback,
                        "sentiment": sentiment,
                        "category": category,
                    }
                )

    random.shuffle(rows)

    return pd.DataFrame(rows)


def main() -> None:
    """Generate and save the dataset."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = build_dataset()
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Dataset written to: {OUTPUT_PATH}")
    print(f"Rows: {len(df)}")
    print("\nSentiment distribution:")
    print(df["sentiment"].value_counts().sort_index())

    print("\nCategory distribution:")
    print(df["category"].value_counts().sort_index())

    print("\nFirst five records:")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()