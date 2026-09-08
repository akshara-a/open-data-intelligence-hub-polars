import os
import numpy as np
import pandas as pd

# Reproducible dataset
np.random.seed(42)

# Number of customers
n = 1500

# Generate customer information
data = {
    "CustomerID": range(1001, 1001 + n),
    "Age": np.random.randint(18, 66, n),
    "Gender": np.random.choice(["Male", "Female"], n),
    "DeviceType": np.random.choice(
        ["Mobile", "Desktop", "Tablet"], n,
        p=[0.55, 0.30, 0.15]
    ),
    "TrafficSource": np.random.choice(
        ["Search", "Social Media", "Email", "Direct", "Advertisement"],
        n
    ),
    "PagesViewed": np.random.randint(1, 25, n),
    "TimeOnSite": np.round(np.random.uniform(1, 30, n), 2),
    "ProductsViewed": np.random.randint(1, 15, n),
    "CartItems": np.random.randint(0, 8, n),
    "PreviousPurchases": np.random.randint(0, 12, n),
    "AverageOrderValue": np.round(np.random.uniform(200, 10000, n), 2),
    "DiscountUsed": np.random.choice([0, 1], n),
    "EmailClicked": np.random.choice([0, 1], n),
    "AdClicked": np.random.choice([0, 1], n),
    "DaysSinceLastVisit": np.random.randint(0, 61, n),
    "SessionCount": np.random.randint(1, 20, n)
}

df = pd.DataFrame(data)

# Create realistic purchase probability
score = (
    -3.5
    + 0.30 * df["CartItems"]
    + 0.10 * df["PreviousPurchases"]
    + 0.05 * df["ProductsViewed"]
    + 0.03 * df["TimeOnSite"]
    + 0.35 * df["DiscountUsed"]
    + 0.30 * df["EmailClicked"]
    + 0.25 * df["AdClicked"]
    - 0.015 * df["DaysSinceLastVisit"]
)

probability = 1 / (1 + np.exp(-score))

df["Purchase"] = np.random.binomial(1, probability)

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Save dataset
file_path = "data/ecommerce_customer_data.csv"
df.to_csv(file_path, index=False)

print("=" * 60)
print("TASK 9 - E-COMMERCE DATASET GENERATED SUCCESSFULLY")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nPurchase Distribution:")
print(df["Purchase"].value_counts())

print("\nPurchase Percentage:")
print((df["Purchase"].value_counts(normalize=True) * 100).round(2))

print(f"\nDataset saved to: {file_path}")