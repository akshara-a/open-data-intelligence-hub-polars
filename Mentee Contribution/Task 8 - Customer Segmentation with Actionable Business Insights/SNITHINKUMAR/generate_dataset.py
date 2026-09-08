import pandas as pd
import numpy as np
import os

# Make results reproducible
np.random.seed(42)

# Number of customers
n_customers = 1500

# Generate customer data
customer_id = np.arange(1001, 1001 + n_customers)

age = np.random.randint(18, 70, n_customers)

gender = np.random.choice(
    ["Male", "Female"],
    n_customers
)

annual_income = np.random.randint(
    200000,
    2000001,
    n_customers
)

purchase_frequency = np.random.randint(
    1,
    31,
    n_customers
)

average_order_value = np.random.randint(
    300,
    10001,
    n_customers
)

# Calculate realistic total spending
total_spending = (
    purchase_frequency
    * average_order_value
    * np.random.uniform(0.7, 1.3, n_customers)
).round(2)

days_since_last_purchase = np.random.randint(
    1,
    366,
    n_customers
)

website_visits = np.random.randint(
    1,
    101,
    n_customers
)

discount_usage = np.random.randint(
    0,
    101,
    n_customers
)

customer_rating = np.clip(
    np.random.normal(3.8, 0.8, n_customers),
    1,
    5
).round(1)

product_category = np.random.choice(
    [
        "Electronics",
        "Fashion",
        "Home",
        "Beauty",
        "Sports"
    ],
    n_customers
)

# Create purchase likelihood
purchase_score = (
    0.04 * purchase_frequency
    + 0.01 * website_visits
    - 0.004 * days_since_last_purchase
    + 0.15 * customer_rating
)

purchase_likelihood = (
    purchase_score > np.median(purchase_score)
).astype(int)

# Create DataFrame
df = pd.DataFrame({
    "CustomerID": customer_id,
    "Age": age,
    "Gender": gender,
    "AnnualIncome": annual_income,
    "TotalSpending": total_spending,
    "PurchaseFrequency": purchase_frequency,
    "AverageOrderValue": average_order_value,
    "DaysSinceLastPurchase": days_since_last_purchase,
    "WebsiteVisits": website_visits,
    "DiscountUsage": discount_usage,
    "CustomerRating": customer_rating,
    "ProductCategory": product_category,
    "PurchaseLikelihood": purchase_likelihood
})

# Create data directory
os.makedirs("data", exist_ok=True)

# Save dataset
file_path = "data/customer_data.csv"

df.to_csv(file_path, index=False)

print("Customer dataset generated successfully!")
print("Dataset shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nPurchase likelihood distribution:")
print(df["PurchaseLikelihood"].value_counts())

print("\nDataset saved to:", file_path)