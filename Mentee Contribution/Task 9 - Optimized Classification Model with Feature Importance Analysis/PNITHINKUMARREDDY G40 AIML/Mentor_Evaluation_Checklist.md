# Mentor Evaluation Checklist — Mini Project 5

| Requirement | Status | Notes |
|---|---|---|
| Real Kaggle dataset used, with link | ✅ | `henrysue/online-shoppers-intention` (UCI mirror) |
| Target variable clearly defined as binary classification | ✅ | `Revenue` (Purchase = 1 / 0) |
| Target-class distribution checked | ✅ | ~83.5% / 16.5% — imbalance confirmed |
| Meaningful EDA with business-meaning explanations | ✅ | Step 2, each chart annotated in-notebook |
| Data leakage prevented | ✅ | `ColumnTransformer` inside `Pipeline`, fit only on train fold |
| Appropriate encoding and scaling | ✅ | Median/most-frequent imputation, `StandardScaler`, `OneHotEncoder` |
| At least 3 classification models trained | ✅ | Logistic Regression, Decision Tree, Random Forest |
| Baseline performance recorded | ✅ | Model_Comparison_Report.md Section 7 |
| Cross-validation used during optimization | ✅ | 5-fold `GridSearchCV` for all 3 models |
| Optimization metric selected and justified | ✅ | F1-score, justified in Section 8 |
| Best hyperparameters reported | ✅ | Notebook Step 6 output |
| Final model evaluated on unseen test data | ✅ | Section 10 |
| More than accuracy used for evaluation | ✅ | Precision, Recall, F1, ROC-AUC, confusion matrix |
| Confusion matrix presented | ✅ | `outputs/confusion_matrix_best_model.png` |
| False positives / false negatives analyzed | ✅ | Discussed alongside confusion matrix, Section 11 |
| Baseline vs. optimized comparison | ✅ | Section 10, reported honestly (not all models improved) |
| Feature importance presented | ✅ | `Feature_Importance_Report.md`, top 10 with direction |
| Important features connected to business actions | ✅ | `Business_Recommendations.md` |
| Classification threshold adjustment considered | ✅ | Section 13, threshold sweep + recommendation |
| Class imbalance addressed | ✅ | `class_weight="balanced"` tested in every grid |
| Realistic limitations provided | ✅ | Section 16 + Feature Importance caveat on causation |
| Saved model pipeline | ✅ | `models/purchase_prediction_model.pkl` (preprocessing + classifier) |
| README with run instructions | ✅ | `README.md` |
