# Mini Project 2 - Model Evaluation Report

## Project Title
Reusable Customer Churn Prediction Pipeline using scikit-learn

## Student Name
Surendra Reddy - G40 AIML

---

## 1. Project Overview

| Detail | Value |
|---|---|
| Project | Reusable ML Pipeline for Customer Churn Prediction |
| Dataset | Telco Customer Churn Dataset |
| Model | RandomForestClassifier |
| Framework | scikit-learn Pipeline with ColumnTransformer |
| Language | Python |

---

## 2. Dataset Summary

| Detail | Value |
|---|---|
| Total rows | 7043 |
| Total columns | 21 (after preparation) |
| Target column | Churn (Yes / No) |
| Churn — Yes | 1869 customers (26.54%) |
| Churn — No | 5174 customers (73.46%) |
| Missing values | TotalCharges — 11 missing values (handled by SimpleImputer) |
| Training rows | 5634 (80%) |
| Testing rows | 1409 (20%) |

---

## 3. Model Performance

### Accuracy
```
77.5%
```

### Confusion Matrix

```
[[904  131]
 [186  188]]
```

| | Predicted No Churn | Predicted Churn |
|---|---|---|
| Actual No Churn | 904 (True Negative) | 131 (False Positive) |
| Actual Churn | 186 (False Negative) | 188 (True Positive) |

### Confusion Matrix Interpretation

| Cell | Count | Meaning |
|---|---|---|
| True Negative (top-left) | 904 | Correctly predicted No Churn |
| False Positive (top-right) | 131 | Predicted Churn but customer did not actually churn |
| False Negative (bottom-left) | 186 | Predicted No Churn but customer actually churned |
| True Positive (bottom-right) | 188 | Correctly predicted Churn |

---

## 4. Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| No (No Churn) | 0.83 | 0.87 | 0.85 | 1035 |
| Yes (Churn) | 0.59 | 0.50 | 0.54 | 374 |
| Accuracy | | | 0.78 | 1409 |
| Macro Avg | 0.71 | 0.69 | 0.70 | 1409 |
| Weighted Avg | 0.77 | 0.78 | 0.77 | 1409 |

---

## 5. Metric Explanations

| Metric | Meaning | Our Result |
|---|---|---|
| Accuracy | Overall correct predictions out of total | 77.5% — model is correct 77.5% of the time |
| Precision (Churn) | Out of predicted churn customers, how many actually churned | 0.59 — 59% of predicted churn were real churn |
| Recall (Churn) | Out of actual churn customers, how many were correctly found | 0.50 — found 50% of real churn customers |
| F1-Score (Churn) | Balance between precision and recall | 0.54 — moderate performance on churn class |
| Macro Avg | Average across both classes equally | 0.70 |
| Weighted Avg | Average weighted by class size | 0.77 |

---

## 6. New Customer Prediction Result

A new customer with the following profile was tested:

| Feature | Value |
|---|---|
| Gender | Female |
| Age | 32 |
| Tenure | 5 months |
| MonthlyCharges | 850 |
| TotalCharges | 4250 |
| ContractType | Monthly |
| PaymentMethod | Credit Card |
| InternetService | Fiber |
| SupportTickets | 3 |

**Churn Prediction : No**
**Churn Probability : 32.0%**

The pipeline automatically handled missing value imputation, scaling,
and OneHotEncoding for this new customer without any manual preprocessing.

---

## 7. Business Interpretation

| Finding | Business Action |
|---|---|
| 26.54% of customers churned | 1 in 4 customers is leaving — retention is critical |
| Monthly contract customers churn most | Offer incentives to switch to yearly or two-year contracts |
| Recall for churn is 50% | 186 real churn customers were missed — improve model to catch more |
| New customer with 5 months tenure predicted No Churn | Short tenure customers on monthly contracts are borderline risk |
| Pipeline is fully reusable | Every month new customer data can be scored without rewriting code |

---

## 8. Recommendations to Improve Model

| Action | Expected Impact |
|---|---|
| Use class_weight="balanced" in RandomForestClassifier | Improves recall for minority churn class |
| Add more features from Telco dataset (OnlineSecurity, TechSupport) | Better pattern learning |
| Try GridSearchCV for hyperparameter tuning | Higher accuracy and F1-score |
| Increase training data | More churn examples improve model learning |

---

## 9. Conclusion

A reusable scikit-learn pipeline was successfully built for customer churn prediction.
The pipeline handles missing values using SimpleImputer, scales numeric columns
using StandardScaler, encodes categorical columns using OneHotEncoder, and trains
a RandomForestClassifier — all in one connected workflow.

The model achieved 77.5% accuracy on 1409 unseen test rows.
The complete pipeline was saved using joblib and successfully reused
to predict churn for a new customer without repeating any preprocessing steps.

This pipeline can be deployed by any subscription business to identify
at-risk customers early and take proactive retention actions.
