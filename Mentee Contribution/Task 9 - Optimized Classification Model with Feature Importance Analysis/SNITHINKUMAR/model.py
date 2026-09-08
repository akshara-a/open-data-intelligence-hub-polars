import os
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)


# ============================================================
# TASK 9
# Optimized Classification Model with Feature Importance
# Author: SNITHINKUMAR
# ============================================================

print("=" * 70)
print("TASK 9 - E-COMMERCE PURCHASE PREDICTION")
print("=" * 70)


# ============================================================
# 1. CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

DATA_PATH = "data/ecommerce_customer_data.csv"

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully.")
print("Dataset Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())


# ============================================================
# 3. DATA QUALITY CHECKS
# ============================================================

print("\n" + "=" * 70)
print("DATA QUALITY CHECKS")
print("=" * 70)

print("\nColumn Data Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

print("\nTarget Distribution:")
print(df["Purchase"].value_counts())

print("\nTarget Percentage:")
print(
    (df["Purchase"].value_counts(normalize=True) * 100)
    .round(2)
)


# ============================================================
# 4. BASIC EDA
# ============================================================

# Target distribution
plt.figure(figsize=(6, 4))
df["Purchase"].value_counts().sort_index().plot(kind="bar")

plt.title("Purchase Class Distribution")
plt.xlabel("Purchase")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig("outputs/purchase_distribution.png")
plt.close()


# Purchase rate by device
device_purchase = (
    df.groupby("DeviceType")["Purchase"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(7, 4))
device_purchase.plot(kind="bar")

plt.title("Purchase Rate by Device Type")
plt.xlabel("Device Type")
plt.ylabel("Purchase Rate")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig("outputs/purchase_rate_by_device.png")
plt.close()


# Purchase rate by traffic source
traffic_purchase = (
    df.groupby("TrafficSource")["Purchase"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(8, 4))
traffic_purchase.plot(kind="bar")

plt.title("Purchase Rate by Traffic Source")
plt.xlabel("Traffic Source")
plt.ylabel("Purchase Rate")
plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig("outputs/purchase_rate_by_traffic.png")
plt.close()


# ============================================================
# 5. PREPARE FEATURES AND TARGET
# ============================================================

# CustomerID is an identifier and should not be used for prediction.
X = df.drop(columns=["Purchase", "CustomerID"])
y = df["Purchase"]

numerical_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nNumerical Features:")
print(numerical_columns)

print("\nCategorical Features:")
print(categorical_columns)


# ============================================================
# 6. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])


# ============================================================
# 7. PREPROCESSING PIPELINE
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numerical_columns
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_columns
        )
    ]
)


# ============================================================
# 8. BASELINE MODELS
# ============================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}


baseline_results = []

trained_pipelines = {}


print("\n" + "=" * 70)
print("BASELINE MODEL RESULTS")
print("=" * 70)


for model_name, classifier in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    probabilities = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    baseline_results.append(
        {
            "Model": model_name,
            "Optimization Status": "Baseline",
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "ROC-AUC": roc_auc
        }
    )

    trained_pipelines[model_name] = pipeline

    print(f"\n{model_name}")
    print("-" * 40)

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1-Score : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {roc_auc:.4f}"
    )


# Save baseline results
baseline_df = pd.DataFrame(
    baseline_results
)

baseline_df.to_csv(
    "outputs/baseline_model_results.csv",
    index=False
)


# ============================================================
# 9. HYPERPARAMETER OPTIMIZATION
# ============================================================

print("\n" + "=" * 70)
print("RANDOM FOREST HYPERPARAMETER OPTIMIZATION")
print("=" * 70)


rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),

        (
            "classifier",
            RandomForestClassifier(
                random_state=42
            )
        )
    ]
)


parameter_grid = {

    "classifier__n_estimators":
        [100, 200],

    "classifier__max_depth":
        [None, 10, 15],

    "classifier__min_samples_split":
        [2, 5],

    "classifier__min_samples_leaf":
        [1, 2],

    "classifier__class_weight":
        [None, "balanced"]
}


grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=parameter_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1,
    verbose=1
)


print("\nRunning GridSearchCV...")
print("This may take a few minutes.\n")

grid_search.fit(
    X_train,
    y_train
)


print("\nBest Parameters:")
print(grid_search.best_params_)

print(
    "\nBest Cross-Validation F1:",
    round(grid_search.best_score_, 4)
)


best_model = grid_search.best_estimator_


# ============================================================
# 10. OPTIMIZED MODEL EVALUATION
# ============================================================

optimized_predictions = best_model.predict(
    X_test
)

optimized_probabilities = best_model.predict_proba(
    X_test
)[:, 1]


optimized_accuracy = accuracy_score(
    y_test,
    optimized_predictions
)

optimized_precision = precision_score(
    y_test,
    optimized_predictions,
    zero_division=0
)

optimized_recall = recall_score(
    y_test,
    optimized_predictions,
    zero_division=0
)

optimized_f1 = f1_score(
    y_test,
    optimized_predictions,
    zero_division=0
)

optimized_auc = roc_auc_score(
    y_test,
    optimized_probabilities
)


print("\n" + "=" * 70)
print("OPTIMIZED RANDOM FOREST RESULTS")
print("=" * 70)

print(
    f"Accuracy : {optimized_accuracy:.4f}"
)

print(
    f"Precision: {optimized_precision:.4f}"
)

print(
    f"Recall   : {optimized_recall:.4f}"
)

print(
    f"F1-Score : {optimized_f1:.4f}"
)

print(
    f"ROC-AUC  : {optimized_auc:.4f}"
)


# ============================================================
# 11. MODEL COMPARISON
# ============================================================

optimized_row = pd.DataFrame(
    [
        {
            "Model": "Random Forest",
            "Optimization Status": "Optimized",
            "Accuracy": optimized_accuracy,
            "Precision": optimized_precision,
            "Recall": optimized_recall,
            "F1-Score": optimized_f1,
            "ROC-AUC": optimized_auc
        }
    ]
)


comparison_df = pd.concat(
    [
        baseline_df,
        optimized_row
    ],
    ignore_index=True
)


comparison_df.to_csv(
    "outputs/model_comparison.csv",
    index=False
)


print("\nModel Comparison:")
print(
    comparison_df.round(4)
)


# ============================================================
# 12. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    optimized_predictions
)

print("\nConfusion Matrix:")
print(cm)


ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "No Purchase",
        "Purchase"
    ]
).plot()

plt.title(
    "Optimized Random Forest Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    "outputs/confusion_matrix.png"
)

plt.close()


# ============================================================
# 13. ROC CURVE
# ============================================================

RocCurveDisplay.from_predictions(
    y_test,
    optimized_probabilities
)

plt.title(
    "ROC Curve - Optimized Random Forest"
)

plt.tight_layout()

plt.savefig(
    "outputs/roc_curve.png"
)

plt.close()


# ============================================================
# 14. FEATURE IMPORTANCE
# ============================================================

feature_names = (
    best_model
    .named_steps["preprocessor"]
    .get_feature_names_out()
)


importance_values = (
    best_model
    .named_steps["classifier"]
    .feature_importances_
)


feature_importance = pd.DataFrame(
    {
        "Feature": feature_names,
        "Importance": importance_values
    }
).sort_values(
    by="Importance",
    ascending=False
)


feature_importance.to_csv(
    "outputs/feature_importance.csv",
    index=False
)


print("\nTop 10 Important Features:")
print(
    feature_importance.head(10)
)


top_features = (
    feature_importance
    .head(10)
    .sort_values(
        by="Importance"
    )
)


plt.figure(
    figsize=(9, 6)
)

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel(
    "Feature Importance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Top 10 Features Influencing Purchase Prediction"
)

plt.tight_layout()

plt.savefig(
    "outputs/feature_importance.png"
)

plt.close()


# ============================================================
# 15. THRESHOLD ANALYSIS
# ============================================================

threshold_results = []


for threshold in [
    0.30,
    0.40,
    0.50,
    0.60,
    0.70
]:

    threshold_predictions = (
        optimized_probabilities
        >= threshold
    ).astype(int)

    threshold_results.append(
        {
            "Threshold": threshold,

            "Precision":
                precision_score(
                    y_test,
                    threshold_predictions,
                    zero_division=0
                ),

            "Recall":
                recall_score(
                    y_test,
                    threshold_predictions,
                    zero_division=0
                ),

            "F1-Score":
                f1_score(
                    y_test,
                    threshold_predictions,
                    zero_division=0
                )
        }
    )


threshold_df = pd.DataFrame(
    threshold_results
)


threshold_df.to_csv(
    "outputs/threshold_analysis.csv",
    index=False
)


print("\nThreshold Analysis:")
print(
    threshold_df.round(4)
)


# ============================================================
# 16. CUSTOMER PURCHASE-LIKELIHOOD CATEGORIES
# ============================================================

customer_results = X_test.copy()

customer_results[
    "ActualPurchase"
] = y_test.values

customer_results[
    "PurchaseProbability"
] = optimized_probabilities


customer_results[
    "PurchaseLikelihood"
] = pd.cut(
    customer_results[
        "PurchaseProbability"
    ],
    bins=[
        0.0,
        0.30,
        0.60,
        1.0
    ],
    labels=[
        "Low",
        "Medium",
        "High"
    ],
    include_lowest=True
)


customer_results.to_csv(
    "outputs/customer_purchase_predictions.csv",
    index=False
)


print("\nPurchase Likelihood Categories:")

print(
    customer_results[
        "PurchaseLikelihood"
    ].value_counts()
)


# ============================================================
# 17. SAVE FINAL MODEL
# ============================================================

MODEL_PATH = (
    "models/purchase_prediction_model.pkl"
)

joblib.dump(
    best_model,
    MODEL_PATH
)


print(
    f"\nFinal model saved to: {MODEL_PATH}"
)


# ============================================================
# 18. BUSINESS RECOMMENDATIONS
# ============================================================

recommendations = """
BUSINESS RECOMMENDATIONS

1. High purchase-likelihood customers should receive
   personalized product recommendations and cart reminders.

2. Customers with medium purchase likelihood can be targeted
   using small discounts, reviews, and promotional emails.

3. Avoid expensive marketing campaigns for low-likelihood
   customers until stronger engagement signals appear.

4. Use the most influential behavioral features to identify
   customers showing strong buying intent.

5. Use purchase probabilities instead of only class predictions
   to prioritize remarketing campaigns.

6. Thresholds can be adjusted according to business goals.
   Lower thresholds increase recall, while higher thresholds
   generally increase precision.

LIMITATIONS

1. This project uses a synthetic educational dataset.
2. Feature importance indicates predictive influence,
   not causation.
3. Real-world customer behavior may change over time.
4. The model should be retrained when new customer data
   becomes available.
"""


with open(
    "outputs/business_recommendations.txt",
    "w"
) as file:

    file.write(
        recommendations
    )


print(recommendations)


print("=" * 70)
print("TASK 9 COMPLETED SUCCESSFULLY")
print("=" * 70)