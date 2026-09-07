# Customer Segmentation — Model Comparison & Evaluation Report

## 1. Business Problem
An e-commerce company wants to understand different types of customers to improve marketing campaigns,
retention, product recommendations, and promotional strategy. This project builds a customer segmentation
solution (primary) plus two supporting models: a regression model estimating customer rating, and a
classification model estimating purchase likelihood — all on the same customer base.

## 2. Dataset
**Source:** [Customer Segmentation Data for Marketing Analysis](https://www.kaggle.com/datasets/fahmidachowdhury/customer-segmentation-data-for-marketing-analysis) (Kaggle, Fahmida Chowdhury)
2,000 customers (in this run), 9 real columns (id, age, gender, income, spending_score, membership_years,
purchase_frequency, preferred_category, last_purchase_amount), no missing values, no duplicates. RFM and
engagement columns (`DaysSinceLastPurchase`, `WebsiteVisits`, `DiscountUsage`, `CustomerRating`,
`PurchaseLikelihood`, `TotalSpending`) were engineered from real behavioral signal — full detail in
`Decision_Log.md`.

## 3. Data Preparation
- Verified 0 missing values, 0 duplicate rows (before and after schema transformation).
- Removed the customer identifier from all model feature sets (used only for profiling/joins).
- Scaled numeric features separately per task with `StandardScaler`.
- 80/20 train-test split for regression and classification (stratified for classification).
- Outliers checked visually via boxplots on income and order value; none required removal.

## 4. Clustering — K-Means (Primary Model)
**Features:** Recency, Frequency, Monetary (`DaysSinceLastPurchase`, `PurchaseFrequency`, `TotalSpending`,
`AverageOrderValue`) plus `WebsiteVisits`, `DiscountUsage`, `CustomerRating`.

| k | Silhouette Score |
|---|---|
| 2 | 0.326 |
| 3 | (see notebook Step 4 chart) |
| **4** | **0.354 (selected)** |
| 5–7 | lower |

**Final model:** k=4, Inertia=4,951.6, Silhouette=0.354. Full segment names, sizes, revenue share, and
recommended actions are in `Customer_Segment_Report.md`.

## 5. Regression — Ridge Regression
**Target:** `CustomerRating` · **Features:** income, purchase frequency, recency, website visits, discount
usage, membership tenure, average order value.

| | MAE | RMSE | R² |
|---|---|---|---|
| Baseline (alpha=1.0) | 0.319 | 0.404 | 0.375 |
| Tuned (GridSearchCV, best alpha=10) | 0.319 | 0.405 | 0.375 |

Tuning `alpha` over {0.01, 0.1, 1, 10, 100} produced no meaningful change — the model was already near its
ceiling for this feature set. R²≈0.38 means behavioral features (recency, discounts, visits) explain a
moderate share of rating variance; MAE of ~0.32 stars is tight enough to flag customers whose predicted
satisfaction is trending low.

## 6. Classification — Logistic Regression
**Target:** `PurchaseLikelihood` · **Features:** purchase frequency, recency, total spending, rating, website
visits, discount usage.

| | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Baseline | 0.953 | 0.964 | 0.925 | 0.944 | 0.989 |
| Tuned (GridSearchCV, best C=1, solver=liblinear) | 0.955 | 0.964 | 0.931 | 0.947 | 0.989 |

This is the strongest model in the comparison. High precision (0.964) means marketing spend aimed at
"likely" customers is rarely wasted; recall of 0.93 means the model misses relatively few genuine buyers.

## 7. Hyperparameter Tuning Summary

| Model | Method | Grid | Best Params |
|---|---|---|---|
| K-Means | Manual sweep + silhouette selection | n_clusters ∈ {2..7} | n_clusters=4 |
| Ridge | GridSearchCV (5-fold, scoring=R²) | alpha ∈ {0.01, 0.1, 1, 10, 100} | alpha=10 |
| Logistic Regression | GridSearchCV (5-fold, scoring=F1) | C ∈ {0.01,0.1,1,10}, solver ∈ {liblinear, lbfgs}, max_iter ∈ {100,200,500} | C=1, solver=liblinear, max_iter=100 |

## 8. Model Comparison Table

| Model | Objective | Baseline Performance | Tuned Performance | Selected Model |
|---|---|---|---|---|
| K-Means | Customer segmentation | k=2, silhouette=0.326 | k=4, silhouette=0.354 | **Yes** |
| Ridge Regression | Predict customer rating | RMSE=0.404, R²=0.375 | RMSE=0.405, R²=0.375 | **Yes** |
| Logistic Regression | Predict purchase likelihood | F1=0.944, AUC=0.989 | F1=0.947, AUC=0.989 | **Yes** |

## 9. Business-Focused Evaluation
- **Can the segments be clearly distinguished?** Yes — recency, frequency, and spending each vary by an
  order of magnitude across clusters (e.g. recency ranges from ~13 to ~174 days).
- **Are segment sizes meaningful?** Yes — each segment is 20–33% of the base, large enough to justify
  distinct campaigns.
- **Can marketing take different actions per segment?** Yes — see `Customer_Segment_Report.md` for four
  distinct, non-overlapping action plans.
- **Does the classification model identify high-potential customers?** Yes — F1 0.947, AUC 0.989.
- **Does the regression model provide useful estimates?** Moderately — R²=0.375 is usable for relative
  ranking/flagging, not precise point estimates.
- **Can the recommendations improve revenue or retention?** The revenue concentration finding (60% of revenue
  from 22% of customers) directly supports a retention-first budget reallocation.

## 10. Final Conclusion
K-Means clustering (the primary model, k=4) produced four clearly separable, business-actionable customer
segments with a stark revenue concentration story. The supporting classification model is strong enough for
direct marketing targeting today (F1 0.947); the regression model is moderately useful and would improve with
richer product-level features. Recommended workflow: use clustering to set segment-level strategy, use
classification to prioritize individual customers within a campaign window, and use the regression model as a
secondary signal for service-quality follow-up where predicted rating is low.
