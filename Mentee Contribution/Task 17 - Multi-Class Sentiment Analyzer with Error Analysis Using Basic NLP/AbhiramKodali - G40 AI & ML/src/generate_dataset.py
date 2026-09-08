"""
Task 17 - Generate a reproducible synthetic multi-class sentiment dataset.

Classes:
- positive
- neutral
- negative

The dataset includes ordinary examples and a smaller set of challenging
examples involving negation, mixed opinions, neutral wording, and sarcasm.
"""

from pathlib import Path
import random

import pandas as pd


SEED = 42

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_PATH = BASE_DIR / "data" / "sentiment_data.csv"


STANDARD_TEMPLATES = {
    "positive": [
        "I really enjoyed this product.",
        "The service was excellent and helpful.",
        "I am very happy with the experience.",
        "The application works smoothly.",
        "Everything worked perfectly for me.",
        "The quality is impressive.",
        "I would definitely recommend this.",
        "The new update is useful and easy to understand.",
        "Customer support solved my problem quickly.",
        "The overall experience was pleasant.",
        "I am satisfied with the result.",
        "This is a great improvement.",
    ],
    "neutral": [
        "I used the product today.",
        "The application is available on my device.",
        "I received the package this morning.",
        "The update was released yesterday.",
        "I contacted customer support about my account.",
        "The product has several available options.",
        "The application contains a dashboard.",
        "I checked the information on the website.",
        "The service is currently active.",
        "I completed the requested process.",
        "The order status has been updated.",
        "The account contains my current details.",
    ],
    "negative": [
        "I am very disappointed with this product.",
        "The service was slow and frustrating.",
        "The application keeps crashing.",
        "I had a terrible experience.",
        "The product quality is poor.",
        "The problem has not been fixed.",
        "I would not recommend this service.",
        "The latest update made the application worse.",
        "Customer support failed to solve my problem.",
        "The experience was frustrating and unpleasant.",
        "The application is unreliable.",
        "I am unhappy with the result.",
    ],
}


PREFIXES = [
    "",
    "Honestly, ",
    "For me, ",
    "Recently, ",
    "In my experience, ",
]

SUFFIXES = [
    "",
    " I noticed this today.",
    " This was my experience.",
    " That is what I observed.",
]


CHALLENGE_EXAMPLES = [
    ("The product is not bad at all.", "positive"),
    ("I am not unhappy with the service.", "positive"),
    ("The application is not perfect, but it works well enough.", "positive"),
    ("I expected more, although the basic service is acceptable.", "neutral"),
    ("The product arrived on time, but the packaging was damaged.", "neutral"),
    ("The interface looks good, but the application keeps freezing.", "negative"),
    ("I love the design, but I cannot use the main feature.", "negative"),
    ("Not exactly a disaster, but I would not use it again.", "negative"),
    ("The update is fine, nothing particularly impressive.", "neutral"),
    ("It works, but I am still waiting for the promised feature.", "neutral"),
    ("Wonderful, another update that broke the login.", "negative"),
    ("Great job making the process slower than before.", "negative"),
    ("The service is okay and completed the request.", "neutral"),
    ("I thought it would be awful, but it turned out better than expected.", "positive"),
    ("There is nothing wrong with the product, and I am quite pleased.", "positive"),
    ("The result is neither good nor bad; it simply works.", "neutral"),
    ("I cannot say I dislike the new interface.", "positive"),
    ("The feature sounds useful, but I have not tried it yet.", "neutral"),
]


def build_dataset() -> pd.DataFrame:
    """Build a balanced, reproducible sentiment dataset."""
    random.seed(SEED)

    rows = []

    # 120 examples per class = 360 standard examples.
    for sentiment, templates in STANDARD_TEMPLATES.items():
        for _ in range(30):
            for _ in range(4):
                template = random.choice(templates)
                prefix = random.choice(PREFIXES)
                suffix = random.choice(SUFFIXES)

                feedback = f"{prefix}{template}{suffix}".strip()

                rows.append(
                    {
                        "text": feedback,
                        "sentiment": sentiment,
                        "source_type": "standard",
                    }
                )

    # Add challenging examples to make error analysis meaningful.
    for text, sentiment in CHALLENGE_EXAMPLES:
        rows.append(
            {
                "text": text,
                "sentiment": sentiment,
                "source_type": "challenge",
            }
        )

    random.shuffle(rows)

    return pd.DataFrame(rows)


def main() -> None:
    """Generate and save the sentiment dataset."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = build_dataset()
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Dataset written to: {OUTPUT_PATH}")
    print(f"Rows: {len(df)}")

    print("\nSentiment distribution:")
    print(df["sentiment"].value_counts().sort_index())

    print("\nSource-type distribution:")
    print(df["source_type"].value_counts().sort_index())

    print("\nFirst ten records:")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()