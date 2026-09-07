# Decision Log — Mini Project 4: Customer Segmentation with Actionable Business Insights

## 1. Dataset Selection

**Dataset used:** [Customer Segmentation Data for Marketing Analysis](https://www.kaggle.com/datasets/fahmidachowdhury/customer-segmentation-data-for-marketing-analysis)
**Author:** Fahmida Chowdhury (Kaggle)

**Real columns in the source file (confirmed):**
`id, age, gender, income, spending_score, membership_years, purchase_frequency, preferred_category, last_purchase_amount`
No missing values, no duplicate rows in the source data.

**Why this dataset:** it is customer-level (one row per customer, matching a segmentation use case),
already contains a genuine `spending_score` and `purchase_frequency` — the two strongest RFM-style signals —
plus demographic fields (age, gender, income, tenure) useful for profiling each segment in business terms.

## 2. Why the schema needed transformation

The brief asks for RFM-style fields (`DaysSinceLastPurchase`, `WebsiteVisits`, `DiscountUsage`,
`CustomerRating`, `PurchaseLikelihood`) that the raw Kaggle file does not literally contain — it is a single
demographic + spending snapshot per customer, not a transaction log. The columns below were engineered to
fill that gap, using an **engagement tier** derived from the two real columns most predictive of behavior
(`purchase_frequency` × `spending_score`) so the engineered values stay grounded in real signal rather than
being independent random noise. This mirrors the schema-alignment approach used in Mini Projects 2 and 3.

| Assignment column | How it was produced | Real or engineered? |
|---|---|---|
| `CustomerID`, `Age`, `Gender`, `AnnualIncome`, `PurchaseFrequency`, `ProductCategory`, `AverageOrderValue` | Renamed directly from `id`, `age`, `gender`, `income`, `purchase_frequency`, `preferred_category`, `last_purchase_amount` | **Real** |
| Engagement tier (`loyal` / `growing` / `discount` / `at-risk` / `low-engagement`) | Joint tercile split of real `purchase_frequency` and `spending_score` | Derived from real columns |
| `DaysSinceLastPurchase` (Recency) | Tier-specific base value (loyal≈12, growing≈35, discount≈50, at-risk≈140, low-engagement≈200 days) + Gaussian noise, clipped 1–365 | Engineered |
| `WebsiteVisits` | Tier-specific base value + Gaussian noise, clipped ≥0 | Engineered |
| `DiscountUsage` | Tier-specific base rate + Gaussian noise, clipped 0–1 | Engineered |
| `TotalSpending` (Monetary) | `AverageOrderValue × PurchaseFrequency × random factor (0.6–1.0)` | Engineered |
| `CustomerRating` | `3 + 0.01×spending_score − 0.004×DaysSinceLastPurchase + noise`, clipped 1–5 | Engineered |
| `PurchaseLikelihood` | `1` if `DaysSinceLastPurchase < 60` **and** `PurchaseFrequency` above the sample median, else `0` | Engineered label |

All random generation uses fixed seeds (`numpy.random.default_rng`) for reproducibility.

## 3. Offline fallback

`generate_dataset.py` reproduces the real dataset's documented column names, dtypes, and approximate value
ranges for environments without Kaggle API access. The notebook (Cell 2) tries a live Kaggle download via
`kagglehub` first and only falls back to this generator on failure, so the same code path runs identically
with or without internet access.

## 4. Modeling decisions

- **Clustering features** limited to RFM + engagement columns (`DaysSinceLastPurchase`, `PurchaseFrequency`,
  `TotalSpending`, `AverageOrderValue`, `WebsiteVisits`, `DiscountUsage`, `CustomerRating`) — demographics
  (age, gender, income) were deliberately excluded from the clustering features and used only afterward for
  profiling, so segments reflect *behavior* rather than demographics.
- **`k` chosen by silhouette score** across k=2–7, cross-checked against the elbow curve; k=4 had the best
  silhouette (0.354) and was selected.
- **PCA (2 components)** used only for 2D visualization of the clusters, not for clustering itself.
- **Ridge over plain Linear Regression** for the regression task, per the brief's recommendation, to guard
  against overfitting across correlated behavioral features.
- **`GridSearchCV`** used for both Ridge (`alpha`) and Logistic Regression (`C`, `solver`, `max_iter`), per
  the brief's recommendation for beginner-level explainability over `RandomizedSearchCV`.
- **Stratified train/test split** for classification given the ~44/56 class balance of `PurchaseLikelihood`.
- Regression tuning improved R² only marginally (0.375 → 0.375, i.e. no meaningful change) — reported
  honestly rather than overstated.
