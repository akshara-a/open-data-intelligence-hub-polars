import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

# Number of records
n = 2000

# Generate basic data
user_ids = np.arange(10001, 10001 + n)

product_ids = np.random.randint(5001, 5201, n)

categories = np.random.choice(
    ["Electronics", "Fashion", "Home", "Beauty", "Sports"],
    size=n
)

price = np.round(
    np.random.uniform(100, 50000, n),
    2
)

browsing_time = np.round(
    np.random.gamma(shape=3, scale=4, size=n),
    2
)

previous_purchases = np.random.poisson(
    lam=5,
    size=n
)

cart_addition = np.random.choice(
    [0, 1],
    size=n,
    p=[0.4, 0.6]
)

age = np.random.randint(
    18,
    66,
    n
)

gender = np.random.choice(
    ["Male", "Female", "Other"],
    size=n,
    p=[0.48, 0.48, 0.04]
)

location = np.random.choice(
    ["Chennai", "Bangalore", "Hyderabad", "Mumbai", "Delhi"],
    size=n
)

discount_applied = np.random.choice(
    [0, 1],
    size=n,
    p=[0.55, 0.45]
)

# Generate total spending
total_spending = np.round(
    previous_purchases * np.random.uniform(500, 5000, n)
    + np.random.uniform(0, 10000, n),
    2
)

# Generate Rating
rating = (
    2.5
    + 0.03 * browsing_time
    + 0.08 * previous_purchases
    + 0.25 * cart_addition
    + 0.15 * discount_applied
    + np.random.normal(0, 0.45, n)
)

rating = np.clip(rating, 1, 5)
rating = np.round(rating, 1)

# Generate Purchase Probability
purchase_score = (
    -2
    + 0.12 * browsing_time
    + 1.2 * cart_addition
    + 0.10 * previous_purchases
    + 0.20 * discount_applied
    + 0.25 * rating
)

purchase_probability = 1 / (
    1 + np.exp(-purchase_score)
)

purchase_status = np.random.binomial(
    1,
    purchase_probability
)

# Create DataFrame
df = pd.DataFrame({
    "User_ID": user_ids,
    "Product_ID": product_ids,
    "Category": categories,
    "Price": price,
    "Rating": rating,
    "Browsing_Time": browsing_time,
    "Previous_Purchases": previous_purchases,
    "Cart_Addition": cart_addition,
    "Purchase_Status": purchase_status,
    "Age": age,
    "Gender": gender,
    "Location": location,
    "Discount_Applied": discount_applied,
    "Total_Spending": total_spending
})

# Add missing values for preprocessing
for column in ["Rating", "Browsing_Time", "Total_Spending"]:
    indices = np.random.choice(
        df.index,
        size=20,
        replace=False
    )

    df.loc[indices, column] = np.nan

# Add duplicate rows
duplicates = df.sample(
    25,
    random_state=42
)

df = pd.concat(
    [df, duplicates],
    ignore_index=True
)

# Output path
output_path = Path(
    "Mentee Contribution/Task 6 - Building a Recommendation System for E-Commerce/"
    "SNITHINKUMAR/data/ecommerce_recommendation_dataset.csv"
)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

# Save dataset
df.to_csv(
    output_path,
    index=False
)

print("=" * 60)
print("E-COMMERCE RECOMMENDATION DATASET CREATED")
print("=" * 60)

print(f"\nDataset shape: {df.shape}")

print("\nColumns:")
for column in df.columns:
    print("-", column)

print(f"\nSaved successfully to:\n{output_path}")