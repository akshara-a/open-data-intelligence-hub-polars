# Mini Project 2 - Decision Log

## Project Title
Reusable Customer Churn Prediction Pipeline using scikit-learn

## Student Name
Surendra Reddy - G40 AIML

---

## Decision Log

| Decision Area | Decision Taken | Reason |
|---|---|---|
| Removed column | Removed CustomerID | It is only an identifier and does not help the model learn churn behavior |
| Train-test split | Used 80:20 split | To evaluate the model on unseen data and avoid overfitting |
| Stratification | Used stratify=y | To maintain the same churn ratio (73.46% No, 26.54% Yes) in both train and test splits |
| Missing numeric values | Used median imputation | TotalCharges had 11 missing values — median handles outliers better than mean |
| Missing categorical values | Used most frequent value | Preserves the most common category pattern in the dataset |
| Encoding | Used OneHotEncoder | ML models cannot understand text values like Monthly, Yearly — converts to 0/1 numeric columns |
| Scaling | Used StandardScaler | Brings all numeric columns like Age, MonthlyCharges, TotalCharges to a comparable scale |
| Model | Used RandomForestClassifier | Handles nonlinear patterns, reduces overfitting by combining many decision trees |
| Evaluation | Used classification_report | Churn is a classification problem requiring precision, recall, and F1-score |
| Reusability | Saved full pipeline using joblib | Load and predict future customer data without repeating preprocessing steps |

---

## Column Preparation Decisions

| Decision | Reason |
|---|---|
| Renamed customerID to CustomerID | Consistent formatting with ma'am's required column names |
| Renamed Contract to ContractType | Matches ma'am's required column format |
| Added Age column synthetically | Telco dataset does not have Age — added using np.random.randint(18, 65) |
| Added SupportTickets synthetically | Telco dataset does not have SupportTickets — added using np.random.randint(0, 10) |
| Fixed TotalCharges using pd.to_numeric | Telco stores TotalCharges as string with blank spaces — converted to float |

---

## OneHotEncoder Explanation

Before encoding:

| ContractType |
|---|
| Monthly |
| Yearly |
| Two-Year |

After encoding:

| ContractType_Monthly | ContractType_Yearly | ContractType_Two-Year |
|---|---|---|
| 1 | 0 | 0 |
| 0 | 1 | 0 |
| 0 | 0 | 1 |

Reason: Machine learning models cannot directly understand text values.
OneHotEncoder converts each category into a separate 0/1 numeric column.

---

## Data Leakage Prevention

The pipeline was fitted only on X_train and y_train (5634 rows).
X_test (1409 rows) was never seen by the pipeline during training.
This ensures the model is evaluated on truly unseen data.

Correct approach used:
```python
model_pipeline.fit(X_train, y_train)
```

Wrong approach avoided:
```python
model_pipeline.fit(X, y)
```

---

## Why Recall Matters for Churn

In churn prediction, missing a customer who is about to leave causes direct business loss.
Recall measures how many actual churn customers were correctly identified by the model.

From results:
- Recall for Churn (Yes) = 0.50
- This means the model correctly identified 50% of actual churn customers
- 186 churn customers were missed (False Negatives)
- Improving recall is the next business priority
