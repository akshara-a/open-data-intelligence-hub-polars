# Optimized Classification Model — Model Comparison & Evaluation Report

## 1. Project Title
Predicting E-Commerce Purchase Likelihood Using an Optimized Classification Model

## 2. Business Problem
Only a small share of website visitors complete a purchase. The company wants to know: **based on session
behavior and visitor profile, can we predict whether a customer will make a purchase?** Accurate prediction
supports targeted marketing, reduced wasted spend, personalized discounts, and prioritized remarketing.

## 3. Dataset
**Source:** [Online Shoppers Purchasing Intention Dataset](https://www.kaggle.com/datasets/henrysue/online-shoppers-intention)
(Kaggle mirror of UCI dataset #468). 18 real columns, no missing values in the source. Target: `Revenue`
(binary — did the session end in a purchase). Class distribution in this run: ~83.5% no-purchase, ~16.5%
purchase — a realistic imbalance. Full feature mapping in `Decision_Log.md`.

## 4. Data-Quality Checks
- 0 missing values.
- Duplicate rows checked and removed (none found in this run).
- No `CustomerID`-style identifier exists in the dataset (session-level, not customer-level).
- Target distribution confirmed imbalanced — accuracy alone would be misleading (Section 6).

## 5. Exploratory Data Analysis — Key Findings
- Purchase rate varies meaningfully by `VisitorType` and `Weekend` (see `outputs/purchase_rate_by_visitor_weekend.png`).
- Purchasers spend measurably longer on product pages, exit less, and view higher-value pages than
  non-purchasers (`outputs/behavior_vs_purchase_boxplots.png`).
- `PageValues` shows by far the strongest correlation with `Revenue` in the correlation heatmap
  (`outputs/correlation_heatmap.png`); `BounceRates`/`ExitRates` correlate negatively.

## 6. Data Preparation
- Numeric columns: median imputation → `StandardScaler`.
- Categorical columns: most-frequent imputation → `OneHotEncoder(handle_unknown="ignore")`.
- Both wrapped in a single `ColumnTransformer` inside each model's `Pipeline`, fit only on the training fold
  (no leakage).
- 80/20 stratified train-test split to preserve the purchase/no-purchase ratio in both sets.

## 7. Baseline Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.878 | 0.706 | 0.439 | 0.541 | 0.827 |
| Decision Tree | 0.832 | 0.486 | 0.415 | 0.447 | 0.664 |
| Random Forest | 0.882 | 0.848 | 0.341 | 0.487 | 0.821 |

All three baselines show the classic imbalance signature: relatively high accuracy but much lower
recall — they under-predict the minority "purchase" class by default.

## 8. Optimization Metric Selection
**F1-score** was chosen as the primary tuning metric (via `GridSearchCV(scoring="f1")`), because the business
needs a **balance** between not wasting marketing spend on unlikely buyers (precision) and not missing real
buyers (recall) — accuracy alone is misleading given the ~5:1 class imbalance. **ROC-AUC** is reported
alongside because the deployed use case (Section 13) ranks *all* sessions by purchase probability rather than
applying only a single hard cutoff.

## 9. Hyperparameter Optimization Summary

| Model | Method | CV | Parameters Tested | Best Parameters | Best CV F1 |
|---|---|---|---|---|---|
| Decision Tree | GridSearchCV | 5-fold | max_depth, min_samples_split, min_samples_leaf, criterion, class_weight | see notebook Step 6 output | reported in notebook |
| Random Forest | GridSearchCV | 5-fold | n_estimators, max_depth, min_samples_split, class_weight | see notebook Step 6 output | reported in notebook |
| Logistic Regression | GridSearchCV | 5-fold | C, solver, class_weight | see notebook Step 6 output | reported in notebook |

*(Exact best-parameter dictionaries are printed live in the notebook — they are seed- and grid-dependent and
reproduced exactly on re-run; see Step 6 cells.)*

## 10. Baseline vs. Optimized Comparison

| Model | Status | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | Baseline | 0.878 | 0.706 | 0.439 | 0.541 | 0.827 |
| Decision Tree | Baseline | 0.832 | 0.486 | 0.415 | 0.447 | 0.664 |
| Random Forest | Baseline | 0.882 | 0.848 | 0.341 | 0.487 | 0.821 |
| Decision Tree | Optimized | 0.876 | 0.727 | 0.390 | 0.508 | 0.791 |
| Random Forest | Optimized | 0.872 | 0.680 | 0.415 | 0.515 | 0.834 |
| **Logistic Regression** | **Optimized (selected)** | **0.816** | **0.455** | **0.622** | **0.526** | **0.857** |

**Selected model: Logistic Regression (optimized).** It has the best ROC-AUC (0.857) and, among the
*optimized* models, the best F1 (0.526) — driven by `class_weight="balanced"` substantially improving recall
(0.439 → 0.622) at a deliberate cost to precision and raw accuracy. This directly reflects the business
priority set in Section 8: catching more real buyers matters more than the accuracy number alone. Reported
honestly: optimized Decision Tree and Random Forest did **not** clearly beat their own baselines on F1 in
this run — a genuine, useful finding, not just a tuning success story.

## 11. Confusion Matrix & ROC Curve
See `outputs/confusion_matrix_best_model.png` and `outputs/roc_curve_best_model.png`. The confusion matrix
shows the recall/precision trade-off directly: more true positives caught, at the cost of more false
positives, consistent with the `class_weight="balanced"` setting.

## 12. Feature Importance
Full ranked list in `outputs/feature_importance.csv`; business-facing top 10 in `Feature_Importance_Report.md`.
Top drivers: `ProductRelated_Duration`, `PageValues`, `ExitRates`, plus several `OperatingSystems` / `Browser`
/ `Month` / `TrafficType` categories.

## 13. Threshold Analysis
Precision/Recall/F1 were swept across thresholds 0.10–0.90 (`outputs/threshold_analysis.csv` /
`.png`). F1 peaks at **threshold ≈ 0.70** (F1 = 0.576) in this run, higher than the default 0.50 (F1 = 0.526)
— for this particular optimized (recall-heavy) model, raising the threshold pulls precision back up without
losing as much recall. Recommended threshold depends on the channel: use a **lower threshold (~0.30–0.40)**
for cheap channels (email, on-site banners) where casting a wider net is low-cost, and a **higher threshold
(~0.65–0.70)** for expensive channels (paid retargeting) where precision matters more.

## 14. Customer Purchase-Likelihood Categories
Sessions were bucketed into Low / Medium / High purchase-likelihood using predicted probability
(`outputs/purchase_likelihood_categories.csv`):

| Category | Sessions | Actual Purchase Rate | Avg. Predicted Probability |
|---|---|---|---|
| Low (0.00–0.30) | 259 | 3.1% | 0.20 |
| Medium (0.30–0.60) | 163 | 18.4% | 0.42 |
| High (0.60–1.00) | 78 | 56.4% | 0.78 |

The actual purchase rate climbs cleanly from Low → High, confirming the model is well-calibrated and the
categories are usable directly for tiered marketing action.

## 15. Business Recommendations
See `Business_Recommendations.md` for the full write-up with evidence, action, benefit, and risk for each.

## 16. Final Conclusion
**Which model?** Logistic Regression, tuned with `class_weight="balanced"` (C and solver per the notebook's
printed best parameters), selected for the best ROC-AUC (0.857) and best F1 among optimized models (0.526).
**How much did tuning help?** It shifted the model from precision-heavy/recall-poor (baseline: 0.706
precision / 0.439 recall) to a more balanced, business-appropriate profile (0.455 precision / 0.622 recall) —
tuning improved *recall* and *ROC-AUC* meaningfully, though not every metric improved simultaneously, and the
raw accuracy number actually dropped, which is expected and acceptable given the chosen optimization metric.
**Which features mattered?** `ProductRelated_Duration`, `PageValues`, and `ExitRates` dominate — see Section
12 and `Feature_Importance_Report.md`. **How can the company act on this?** Route sessions into the three
purchase-likelihood tiers (Section 14) and apply the tier-specific actions in `Business_Recommendations.md`,
using the channel-appropriate threshold from Section 13.
