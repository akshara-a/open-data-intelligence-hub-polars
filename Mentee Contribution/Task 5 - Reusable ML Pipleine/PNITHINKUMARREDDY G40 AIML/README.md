# Mini Project 2 - Reusable ML Pipeline for Customer Churn Prediction

## Student
Surendra Reddy - G40 AIML

## Phase
Phase 1: ML Engineering Fundamentals

---

## Project Overview

Built a reusable end-to-end machine learning pipeline using scikit-learn
to predict customer churn for a subscription-based business.

The pipeline handles all preprocessing and model training in one connected
workflow that can be saved and reused for future customer data.

---

## Dataset

| Detail | Value |
|---|---|
| Name | Telco Customer Churn Dataset |
| Source | Kaggle |
| Rows | 7043 |
| Columns | 21 |
| Target | Churn (Yes / No) |
| Churn Rate | 26.54% |

---

## Pipeline Architecture

```
Raw Data
   ↓
Train-Test Split (80:20 with stratify=y)
   ↓
ColumnTransformer
   ├── Numerical Pipeline
   │     ├── SimpleImputer (strategy=median)
   │     └── StandardScaler
   └── Categorical Pipeline
         ├── SimpleImputer (strategy=most_frequent)
         └── OneHotEncoder (handle_unknown=ignore)
   ↓
RandomForestClassifier (random_state=42)
   ↓
Evaluation
   ↓
Saved Pipeline (.pkl)
   ↓
Reusable Prediction for New Customers
```

---

## Model Results

| Metric | Value |
|---|---|
| Accuracy | 77.5% |
| Precision (Churn) | 0.59 |
| Recall (Churn) | 0.50 |
| F1-Score (Churn) | 0.54 |
| True Negatives | 904 |
| False Positives | 131 |
| False Negatives | 186 |
| True Positives | 188 |

---

## New Customer Prediction

```python
new_customer = pd.DataFrame({
    "Gender"         : ["Female"],
    "Age"            : [32],
    "Tenure"         : [5],
    "MonthlyCharges" : [850],
    "TotalCharges"   : [4250],
    "ContractType"   : ["Monthly"],
    "PaymentMethod"  : ["Credit Card"],
    "InternetService": ["Fiber"],
    "SupportTickets" : [3]
})

prediction = loaded_pipeline.predict(new_customer)
# Output: No Churn — 32.0% churn probability
```

---

## Files

| File | Description |
|---|---|
| customer_churn_pipeline.ipynb | Complete pipeline code — 17 cells |
| decision_log.md | All technical decisions with reasons |
| model_evaluation_report.md | Full model performance and business interpretation |
| README.md | This file |

---

## How to Reuse the Pipeline

```python
import joblib
import pandas as pd

# Load saved pipeline
pipeline = joblib.load("customer_churn_pipeline.pkl")

# Predict for new customer data
prediction = pipeline.predict(new_customer_dataframe)
print("Churn Prediction:", prediction[0])
```

No manual preprocessing needed — the pipeline handles everything automatically.

---

## Key Learnings

- A reusable sklearn Pipeline connects preprocessing and model training in one object
- ColumnTransformer applies different preprocessing to numeric and categorical columns
- OneHotEncoder converts text categories into 0/1 numeric columns
- fit() must only be called on training data to prevent data leakage
- joblib saves the complete pipeline including all preprocessing steps
- Recall is the most important metric for churn — missing a churning customer costs the business
