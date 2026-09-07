# Decision Log — Mini Project 5: Optimized Classification Model with Feature Importance Analysis

## 1. Dataset Selection

**Dataset used:** [Online Shoppers Purchasing Intention Dataset](https://www.kaggle.com/datasets/henrysue/online-shoppers-intention)
(Kaggle mirror; original source: UCI Machine Learning Repository, dataset #468, Sakar, Polat, Katircioglu &
Kastro, 2019 — DOI 10.24432/C5F88Q)

**Real columns (confirmed):** `Administrative, Administrative_Duration, Informational,
Informational_Duration, ProductRelated, ProductRelated_Duration, BounceRates, ExitRates, PageValues,
SpecialDay, Month, OperatingSystems, Browser, Region, TrafficType, VisitorType, Weekend, Revenue`.
12,330 sessions in the original dataset, no missing values. `Revenue` (renamed conceptually to "Purchase" in
the write-up) is real, binary, and imbalanced — roughly 84.5% did not purchase, 15.5% did — a realistic
class-imbalance scenario the brief specifically asks students to handle (Task 11).

**Why this dataset:** unlike the customer-level datasets used in earlier mini-projects, this one is
**session-level real behavioral data** (Google-Analytics-style page counts, durations, bounce/exit rates,
page value) that maps directly onto the brief's suggested features (`PagesViewed`, `TimeOnSite`,
`ProductsViewed`, browsing engagement) without needing heavy synthetic engineering. It is also one of the
most widely used real datasets for exactly this kind of purchase-intent classification exercise.

## 2. Feature mapping to the assignment brief

| Assignment feature | Real dataset column | Note |
|---|---|---|
| `PagesViewed` | `Administrative + Informational + ProductRelated` | Sum of real page-count columns |
| `TimeOnSite` | `Administrative_Duration + Informational_Duration + ProductRelated_Duration` | Sum of real duration columns |
| `ProductsViewed` | `ProductRelated` | Real column, used directly |
| `DeviceType` (proxy) | `OperatingSystems`, `Browser` | Real anonymized categorical codes |
| `TrafficSource` | `TrafficType` | Real anonymized categorical code |
| `ReviewScoreViewed` (proxy) | `PageValues` | Real Google Analytics "value of pages visited" metric — closest real analog |
| `Purchase` (target) | `Revenue` | Real, binary, used as-is |

No columns from the brief's suggested list (`CartItems`, `DiscountUsed`, `EmailClicked`, `AdClicked`,
`PreviousPurchases`, `AverageOrderValue`) were fabricated, since the real dataset already provides strong,
genuinely predictive session-behavior signal (`BounceRates`, `ExitRates`, `PageValues`, `SpecialDay`,
`VisitorType`, `Weekend`) without needing invented values. This keeps every model input traceable to real
data rather than synthetic noise. `CustomerID` does not exist in this dataset (sessions, not customers, are
the unit of analysis), so the "remove identifier columns" step from the brief is naturally satisfied.

## 3. Offline fallback

`generate_dataset.py` reproduces the real dataset's documented column names, dtypes, and approximate
distributions/correlations (in particular, `PageValues` as the dominant predictor of `Revenue`, matching the
pattern reported in the original paper) for environments without Kaggle API access. The notebook (Cell 2)
tries a live Kaggle download via `kagglehub` first and only falls back to this generator on failure.

## 4. Modeling decisions

- **Three algorithms compared** per the brief's minimum requirement: Logistic Regression, Decision Tree,
  Random Forest — all wrapped in a single `Pipeline` with a shared `ColumnTransformer` preprocessor to
  prevent train/test leakage.
- **Categorical columns cast to a uniform string dtype** before imputation — mixing native `bool` (`Weekend`)
  with `int` (`OperatingSystems`, etc.) and `str` (`Month`) dtypes in one `ColumnTransformer` block breaks
  `SimpleImputer`'s internal array conversion; casting to `str` first resolved this cleanly.
- **F1-score chosen as the primary optimization/tuning metric** (Task 7), with ROC-AUC reported alongside —
  justified in `Model_Comparison_Report.md` Section 5.
- **`class_weight="balanced"` included in every hyperparameter grid** (Logistic Regression, Decision Tree,
  Random Forest) to directly test whether imbalance-aware weighting improves recall (Task 11) — it did, for
  Logistic Regression in particular.
- **`GridSearchCV`, 5-fold, `scoring="f1"`** used for all three models (Task 8). Grid sizes were kept modest
  (16–320 combinations per model) to keep runtime reasonable without sacrificing meaningful coverage, per the
  brief's caution against oversized grids.
- **Final model selected by test-set F1** among the three *optimized* pipelines, not by training score, per
  the brief's explicit constraint.
- **Model comparison is fully honest:** optimized Decision Tree and Random Forest actually scored slightly
  *lower* F1 than their own baselines in this run (tuning toward F1 via `class_weight="balanced"` traded some
  precision for recall in ways that didn't always net out positively on this particular test split) — this is
  reported plainly in the comparison table rather than smoothed over.
