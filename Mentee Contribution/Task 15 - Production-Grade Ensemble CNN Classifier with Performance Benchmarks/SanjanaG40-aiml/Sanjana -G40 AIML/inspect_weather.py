import pandas as pd
from pathlib import Path
from collections import Counter

DATA_DIR = Path("data/weather_data/dataset")

files = sorted(DATA_DIR.glob("*.parquet"))

print("=" * 60)
print("WEATHERNET-05 DATASET INSPECTION")
print("=" * 60)

total_images = 0
label_counter = Counter()

for file in files:
    print(f"\nReading: {file.name}")

    df = pd.read_parquet(file, columns=["label"])

    total_images += len(df)

    label_counts = df["label"].value_counts().sort_index()

    print(f"Images: {len(df)}")
    print("Labels:")

    for label, count in label_counts.items():
        print(f"  Label {label}: {count}")

    label_counter.update(df["label"].tolist())

print("\n" + "=" * 60)
print("TOTAL DATASET")
print("=" * 60)

print(f"\nTotal images: {total_images}")

print("\nOverall class distribution:")

for label, count in sorted(label_counter.items()):
    percentage = (count / total_images) * 100

    print(
        f"  Label {label}: "
        f"{count:5d} images "
        f"({percentage:.2f}%)"
    )

print("\nNumber of classes:", len(label_counter))