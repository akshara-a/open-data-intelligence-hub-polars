import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.cluster import KMeans

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    silhouette_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import joblib

warnings.filterwarnings("ignore")

# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "ecommerce_recommendation_dataset.csv"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CHART_DIR = os.path.join(BASE_DIR, "charts")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 60)
print("E-COMMERCE RECOMMENDATION SYSTEM")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print("\nDataset Loaded Successfully!")
print("Dataset Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())


# ============================================================
# DATA PREPROCESSING
# ============================================================

print("\n" + "=" * 60)
print("DATA PREPROCESSING")
print("=" * 60)

print("\nMissing Values Before:")
print(df.isnull().sum())

# Remove duplicates
duplicates_before = df.duplicated().sum()
df = df.drop_duplicates()

# Fill numerical missing values
numeric_columns = df.select_dtypes(include=np.number).columns

for col in numeric_columns:
    df[col] = df[col].fillna(df[col].median())

# Fill categorical missing values
categorical_columns = df.select_dtypes(exclude=np.number).columns

for col in categorical_columns:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nDuplicate Rows Before:", duplicates_before)
print("Duplicate Rows After:", df.duplicated().sum())

print("\nMissing Values After:")
print(df.isnull().sum().sum())


# ============================================================
# EXPLORATORY DATA ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print("\nNumerical Summary:")
print(df.describe())

# ------------------------------------------------------------
# Chart 1: Purchase Status by Category
# ------------------------------------------------------------

category_purchase = (
    df.groupby("Category")["Purchase_Status"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 6))
category_purchase.plot(kind="bar")
plt.title("Purchase Rate by Product Category")
plt.xlabel("Category")
plt.ylabel("Purchase Rate")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    os.path.join(CHART_DIR, "chart_1_purchase_rate_by_category.png")
)

plt.close()


# ------------------------------------------------------------
# Chart 2: Rating by Category
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

df.groupby("Category")["Rating"].mean().sort_values(
    ascending=False
).plot(kind="bar")

plt.title("Average Rating by Category")
plt.xlabel("Category")
plt.ylabel("Average Rating")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    os.path.join(CHART_DIR, "chart_2_average_rating_by_category.png")
)

plt.close()


# ------------------------------------------------------------
# Chart 3: Total Spending Distribution
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="Total_Spending",
    bins=30,
    kde=True
)

plt.title("Distribution of Total Spending")
plt.tight_layout()

plt.savefig(
    os.path.join(CHART_DIR, "chart_3_total_spending_distribution.png")
)

plt.close()


# ------------------------------------------------------------
# Chart 4: Browsing Time vs Purchase Status
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

sns.boxplot(
    data=df,
    x="Purchase_Status",
    y="Browsing_Time"
)

plt.title("Browsing Time vs Purchase Status")
plt.xlabel("Purchase Status")
plt.ylabel("Browsing Time")

plt.tight_layout()

plt.savefig(
    os.path.join(CHART_DIR, "chart_4_browsing_time_vs_purchase.png")
)

plt.close()


print("\nEDA Charts Created Successfully!")


# ============================================================
# PART A - REGRESSION
# RATING PREDICTION USING RIDGE REGRESSION
# ============================================================

print("\n" + "=" * 60)
print("PART A: REGRESSION - RATING PREDICTION")
print("=" * 60)

regression_features = [
    "Price",
    "Browsing_Time",
    "Previous_Purchases",
    "Discount_Applied",
    "Age",
    "Category",
    "Total_Spending"
]

X_reg = df[regression_features]
y_reg = df["Rating"]

numeric_features_reg = [
    "Price",
    "Browsing_Time",
    "Previous_Purchases",
    "Discount_Applied",
    "Age",
    "Total_Spending"
]

categorical_features_reg = [
    "Category"
]

preprocessor_reg = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_features_reg
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features_reg
        )
    ]
)

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg,
    y_reg,
    test_size=0.2,
    random_state=42
)

ridge_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor_reg),
        ("model", Ridge())
    ]
)

# Hyperparameter Tuning
param_grid_reg = {
    "model__alpha": [0.01, 0.1, 1, 10, 100]
}

grid_reg = GridSearchCV(
    ridge_pipeline,
    param_grid_reg,
    cv=5,
    scoring="neg_mean_squared_error"
)

grid_reg.fit(
    X_train_reg,
    y_train_reg
)

best_ridge = grid_reg.best_estimator_

y_pred_reg = best_ridge.predict(X_test_reg)

mae = mean_absolute_error(
    y_test_reg,
    y_pred_reg
)

mse = mean_squared_error(
    y_test_reg,
    y_pred_reg
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test_reg,
    y_pred_reg
)

print("\nBest Ridge Parameters:")
print(grid_reg.best_params_)

print("\nRegression Results")
print("MAE:", round(mae, 4))
print("MSE:", round(mse, 4))
print("RMSE:", round(rmse, 4))
print("R2 Score:", round(r2, 4))

joblib.dump(
    best_ridge,
    os.path.join(MODEL_DIR, "ridge_rating_model.pkl")
)


# Regression Results DataFrame

regression_results = pd.DataFrame({
    "Metric": [
        "MAE",
        "MSE",
        "RMSE",
        "R2 Score"
    ],
    "Value": [
        mae,
        mse,
        rmse,
        r2
    ]
})

regression_results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "regression_results.csv"
    ),
    index=False
)


# ============================================================
# PART B - CLASSIFICATION
# PURCHASE LIKELIHOOD PREDICTION
# ============================================================

print("\n" + "=" * 60)
print("PART B: CLASSIFICATION - PURCHASE PREDICTION")
print("=" * 60)

classification_features = [
    "Browsing_Time",
    "Cart_Addition",
    "Previous_Purchases",
    "Rating",
    "Price",
    "Discount_Applied",
    "Total_Spending"
]

X_cls = df[classification_features]
y_cls = df["Purchase_Status"]

X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X_cls,
    y_cls,
    test_size=0.2,
    random_state=42,
    stratify=y_cls
)

preprocessor_cls = StandardScaler()

logistic_pipeline = Pipeline(
    steps=[
        ("scaler", preprocessor_cls),
        (
            "model",
            LogisticRegression(
                max_iter=500
            )
        )
    ]
)

param_grid_cls = {
    "model__C": [
        0.01,
        0.1,
        1,
        10,
        100
    ]
}

grid_cls = GridSearchCV(
    logistic_pipeline,
    param_grid_cls,
    cv=5,
    scoring="f1"
)

grid_cls.fit(
    X_train_cls,
    y_train_cls
)

best_logistic = grid_cls.best_estimator_

y_pred_cls = best_logistic.predict(
    X_test_cls
)

y_prob_cls = best_logistic.predict_proba(
    X_test_cls
)[:, 1]

accuracy = accuracy_score(
    y_test_cls,
    y_pred_cls
)

precision = precision_score(
    y_test_cls,
    y_pred_cls,
    zero_division=0
)

recall = recall_score(
    y_test_cls,
    y_pred_cls,
    zero_division=0
)

f1 = f1_score(
    y_test_cls,
    y_pred_cls,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test_cls,
    y_prob_cls
)

print("\nBest Logistic Regression Parameters:")
print(grid_cls.best_params_)

print("\nClassification Results")
print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))
print("ROC-AUC:", round(roc_auc, 4))

joblib.dump(
    best_logistic,
    os.path.join(
        MODEL_DIR,
        "logistic_purchase_model.pkl"
    )
)


classification_results = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ],
    "Value": [
        accuracy,
        precision,
        recall,
        f1,
        roc_auc
    ]
})

classification_results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "classification_results.csv"
    ),
    index=False
)


# ------------------------------------------------------------
# Confusion Matrix
# ------------------------------------------------------------

cm = confusion_matrix(
    y_test_cls,
    y_pred_cls
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()

plt.title(
    "Logistic Regression Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        CHART_DIR,
        "chart_5_confusion_matrix.png"
    )
)

plt.close()


# ============================================================
# PART C - CLUSTERING
# CUSTOMER SEGMENTATION USING K-MEANS
# ============================================================

print("\n" + "=" * 60)
print("PART C: CLUSTERING - CUSTOMER SEGMENTATION")
print("=" * 60)

cluster_features = [
    "Browsing_Time",
    "Previous_Purchases",
    "Rating",
    "Total_Spending",
    "Cart_Addition",
    "Discount_Applied"
]

X_cluster = df[
    cluster_features
].copy()

scaler_cluster = StandardScaler()

X_cluster_scaled = scaler_cluster.fit_transform(
    X_cluster
)


# ------------------------------------------------------------
# Elbow Method
# ------------------------------------------------------------

inertia_values = []

k_values = range(
    2,
    9
)

for k in k_values:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(
        X_cluster_scaled
    )

    inertia_values.append(
        kmeans.inertia_
    )


plt.figure(
    figsize=(8, 6)
)

plt.plot(
    list(k_values),
    inertia_values,
    marker="o"
)

plt.title(
    "Elbow Method for Optimal K"
)

plt.xlabel(
    "Number of Clusters (K)"
)

plt.ylabel(
    "Inertia"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        CHART_DIR,
        "chart_6_elbow_method.png"
    )
)

plt.close()


# ------------------------------------------------------------
# Silhouette Score Analysis
# ------------------------------------------------------------

silhouette_results = []

for k in k_values:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(
        X_cluster_scaled
    )

    score = silhouette_score(
        X_cluster_scaled,
        labels
    )

    silhouette_results.append(
        {
            "K": k,
            "Silhouette_Score": score,
            "Inertia": kmeans.inertia_
        }
    )


silhouette_df = pd.DataFrame(
    silhouette_results
)

best_k = silhouette_df.loc[
    silhouette_df[
        "Silhouette_Score"
    ].idxmax(),
    "K"
]

best_k = int(best_k)

print("\nBest Number of Clusters:", best_k)


# ------------------------------------------------------------
# Final K-Means Model
# ------------------------------------------------------------

final_kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

df["Customer_Segment"] = final_kmeans.fit_predict(
    X_cluster_scaled
)

final_silhouette = silhouette_score(
    X_cluster_scaled,
    df["Customer_Segment"]
)

final_inertia = final_kmeans.inertia_

print("Final Inertia:", round(final_inertia, 4))
print(
    "Final Silhouette Score:",
    round(final_silhouette, 4)
)

joblib.dump(
    final_kmeans,
    os.path.join(
        MODEL_DIR,
        "kmeans_customer_segmentation.pkl"
    )
)

joblib.dump(
    scaler_cluster,
    os.path.join(
        MODEL_DIR,
        "cluster_scaler.pkl"
    )
)


# ------------------------------------------------------------
# Cluster Summary
# ------------------------------------------------------------

cluster_summary = df.groupby(
    "Customer_Segment"
)[
    [
        "Browsing_Time",
        "Previous_Purchases",
        "Rating",
        "Total_Spending",
        "Cart_Addition",
        "Discount_Applied"
    ]
].mean()

print("\nCustomer Segment Summary:")
print(
    cluster_summary.round(2)
)

cluster_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "customer_segment_summary.csv"
    )
)

silhouette_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "kmeans_evaluation.csv"
    ),
    index=False
)


# ------------------------------------------------------------
# Cluster Visualization
# ------------------------------------------------------------

plt.figure(
    figsize=(9, 7)
)

sns.scatterplot(
    data=df,
    x="Total_Spending",
    y="Previous_Purchases",
    hue="Customer_Segment",
    palette="viridis"
)

plt.title(
    "Customer Segments"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        CHART_DIR,
        "chart_7_customer_segments.png"
    )
)

plt.close()


# ============================================================
# PART D - MODEL COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

comparison = pd.DataFrame({
    "Model": [
        "Ridge Regression",
        "Logistic Regression",
        "K-Means Clustering"
    ],

    "Business Objective": [
        "Predict Customer Rating",
        "Predict Purchase Likelihood",
        "Customer Segmentation"
    ],

    "Primary Metric": [
        "R2 Score",
        "F1 Score",
        "Silhouette Score"
    ],

    "Score": [
        r2,
        f1,
        final_silhouette
    ]
})

print(
    comparison.round(4)
)

comparison.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "model_comparison.csv"
    ),
    index=False
)


# ============================================================
# SAVE FINAL DATASET
# ============================================================

df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "final_ecommerce_dataset_with_segments.csv"
    ),
    index=False
)


# ============================================================
# FINAL BUSINESS INTERPRETATION
# ============================================================

print("\n" + "=" * 60)
print("FINAL BUSINESS INTERPRETATION")
print("=" * 60)

print("""
1. Ridge Regression predicts customer product ratings.

2. Logistic Regression identifies customers
   who are likely to purchase products.

3. K-Means groups customers based on their
   browsing and purchasing behaviour.

4. High-value customer segments can receive
   loyalty rewards and premium recommendations.

5. Discount-sensitive customers can receive
   personalized coupons.

6. Customers with high purchase probability
   can be targeted using email campaigns
   and personalized recommendations.
""")


print("=" * 60)
print("TASK 6 COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nGenerated Outputs:")
print("- Regression Results")
print("- Classification Results")
print("- K-Means Evaluation")
print("- Customer Segment Summary")
print("- Model Comparison")
print("- Trained ML Models")
print("- 7 Visualization Charts")