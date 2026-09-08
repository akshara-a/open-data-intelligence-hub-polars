import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression


# ============================================================
# TASK 8 - CUSTOMER SEGMENTATION WITH ACTIONABLE BUSINESS INSIGHTS
# ============================================================

print("=" * 70)
print("TASK 8 - CUSTOMER SEGMENTATION WITH ACTIONABLE BUSINESS INSIGHTS")
print("=" * 70)


# ============================================================
# 1. CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs("outputs", exist_ok=True)
os.makedirs("outputs/visualizations", exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

DATA_PATH = "data/customer_data.csv"

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully!")

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())


# ============================================================
# 3. EXPLORATORY DATA ANALYSIS
# ============================================================

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nStatistical Summary:")
print(df.describe())


# ============================================================
# 4. DATA CLEANING
# ============================================================

df = df.drop_duplicates()

numeric_columns = df.select_dtypes(include=np.number).columns

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())

categorical_columns = df.select_dtypes(exclude=np.number).columns

for column in categorical_columns:
    if df[column].isnull().any():
        df[column] = df[column].fillna(df[column].mode()[0])

print("\nData cleaning completed successfully!")


# ============================================================
# 5. CUSTOMER SPENDING DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

plt.hist(
    df["TotalSpending"],
    bins=30,
    edgecolor="black"
)

plt.title("Customer Spending Distribution")
plt.xlabel("Total Spending")
plt.ylabel("Number of Customers")
plt.tight_layout()

plt.savefig(
    "outputs/visualizations/spending_distribution.png"
)

plt.close()


# ============================================================
# 6. RECENCY VS FREQUENCY
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    df["DaysSinceLastPurchase"],
    df["PurchaseFrequency"],
    alpha=0.5
)

plt.title("Recency vs Purchase Frequency")
plt.xlabel("Days Since Last Purchase")
plt.ylabel("Purchase Frequency")
plt.tight_layout()

plt.savefig(
    "outputs/visualizations/recency_vs_frequency.png"
)

plt.close()


# ============================================================
# 7. CORRELATION MATRIX
# ============================================================

numeric_df = df.select_dtypes(include=np.number)

correlation = numeric_df.corr()

plt.figure(figsize=(12, 8))

plt.imshow(
    correlation,
    cmap="coolwarm",
    aspect="auto"
)

plt.colorbar()

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=90
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)

plt.title("Customer Feature Correlation Matrix")
plt.tight_layout()

plt.savefig(
    "outputs/visualizations/correlation_matrix.png"
)

plt.close()

print("\nEDA visualizations saved successfully!")


# ============================================================
# 8. SELECT CLUSTERING FEATURES
# ============================================================

clustering_features = [
    "TotalSpending",
    "PurchaseFrequency",
    "AverageOrderValue",
    "DaysSinceLastPurchase",
    "WebsiteVisits",
    "DiscountUsage",
    "CustomerRating"
]

X_cluster = df[clustering_features].copy()

print("\nClustering Features:")
print(clustering_features)


# ============================================================
# 9. SCALE CLUSTERING FEATURES
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_cluster)

print("\nFeature scaling completed!")
print("Scaled Data Shape:", X_scaled.shape)


# ============================================================
# 10. ELBOW METHOD AND SILHOUETTE SCORE
# ============================================================

print("\n" + "=" * 70)
print("K-MEANS CLUSTER ANALYSIS")
print("=" * 70)

k_values = list(range(2, 9))

inertia_values = []
silhouette_scores = []

for k in k_values:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    inertia_values.append(model.inertia_)

    score = silhouette_score(
        X_scaled,
        labels
    )

    silhouette_scores.append(score)

    print(
        f"K = {k} | "
        f"Inertia = {model.inertia_:.2f} | "
        f"Silhouette Score = {score:.4f}"
    )


# ============================================================
# 11. ELBOW METHOD GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    k_values,
    inertia_values,
    marker="o"
)

plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "outputs/visualizations/elbow_method.png"
)

plt.close()


# ============================================================
# 12. SILHOUETTE SCORE GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    k_values,
    silhouette_scores,
    marker="o"
)

plt.title("Silhouette Score Comparison")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "outputs/visualizations/silhouette_scores.png"
)

plt.close()


# ============================================================
# 13. SELECT BEST K
# ============================================================

best_index = np.argmax(silhouette_scores)

best_k = k_values[best_index]

best_silhouette = silhouette_scores[best_index]

print("\nBest Number of Clusters:", best_k)

print(
    "Best Silhouette Score:",
    round(best_silhouette, 4)
)


# ============================================================
# 14. FINAL K-MEANS MODEL
# ============================================================

final_kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

df["Cluster"] = final_kmeans.fit_predict(
    X_scaled
)

print("\nFinal K-Means model created successfully!")

print("\nCustomers in each cluster:")

print(
    df["Cluster"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 15. CLUSTER PROFILING
# ============================================================

cluster_profile = (
    df.groupby("Cluster")[clustering_features]
    .mean()
    .round(2)
)

cluster_profile["CustomerCount"] = (
    df.groupby("Cluster").size()
)

revenue_by_cluster = (
    df.groupby("Cluster")["TotalSpending"].sum()
)

total_revenue = df["TotalSpending"].sum()

cluster_profile["RevenueContributionPercent"] = (
    revenue_by_cluster /
    total_revenue *
    100
).round(2)

print("\nCluster Characteristics:")

print(cluster_profile)


# ============================================================
# 16. ASSIGN BUSINESS SEGMENT NAMES
# ============================================================

# Determine high-value cluster automatically
high_value_cluster = (
    cluster_profile["TotalSpending"].idxmax()
)

segment_names = {}

for cluster in cluster_profile.index:

    if cluster == high_value_cluster:

        segment_names[cluster] = (
            "High-Value Loyal Customers"
        )

    else:

        segment_names[cluster] = (
            "Regular Customers"
        )

df["SegmentName"] = (
    df["Cluster"].map(segment_names)
)

print("\nCustomer Segment Distribution:")

print(
    df["SegmentName"]
    .value_counts()
)


# ============================================================
# 17. PCA CLUSTER VISUALIZATION
# ============================================================

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=df["Cluster"],
    alpha=0.6
)

plt.title("Customer Segments - PCA Visualization")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.colorbar(
    scatter,
    label="Cluster"
)

plt.tight_layout()

plt.savefig(
    "outputs/visualizations/customer_clusters.png"
)

plt.close()


# ============================================================
# 18. CLUSTER-WISE CUSTOMER COUNT
# ============================================================

cluster_counts = (
    df["SegmentName"]
    .value_counts()
)

plt.figure(figsize=(8, 5))

cluster_counts.plot(
    kind="bar"
)

plt.title("Customer Count by Segment")
plt.xlabel("Customer Segment")
plt.ylabel("Number of Customers")
plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig(
    "outputs/visualizations/cluster_customer_count.png"
)

plt.close()


# ============================================================
# 19. CLUSTER-WISE AVERAGE SPENDING
# ============================================================

average_spending = (
    df.groupby("SegmentName")["TotalSpending"]
    .mean()
)

plt.figure(figsize=(8, 5))

average_spending.plot(
    kind="bar"
)

plt.title("Average Spending by Customer Segment")
plt.xlabel("Customer Segment")
plt.ylabel("Average Total Spending")
plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig(
    "outputs/visualizations/cluster_average_spending.png"
)

plt.close()


# ============================================================
# 20. REGRESSION MODEL
# ============================================================

print("\n" + "=" * 70)
print("REGRESSION - PREDICT TOTAL CUSTOMER SPENDING")
print("=" * 70)

regression_features = [
    "PurchaseFrequency",
    "AverageOrderValue",
    "DaysSinceLastPurchase",
    "WebsiteVisits",
    "DiscountUsage",
    "CustomerRating"
]

X_reg = df[regression_features]

y_reg = df["TotalSpending"]

X_reg_train, X_reg_test, y_reg_train, y_reg_test = (
    train_test_split(
        X_reg,
        y_reg,
        test_size=0.20,
        random_state=42
    )
)

reg_scaler = StandardScaler()

X_reg_train_scaled = reg_scaler.fit_transform(
    X_reg_train
)

X_reg_test_scaled = reg_scaler.transform(
    X_reg_test
)


# ============================================================
# 21. LINEAR REGRESSION
# ============================================================

linear_model = LinearRegression()

linear_model.fit(
    X_reg_train_scaled,
    y_reg_train
)

linear_predictions = linear_model.predict(
    X_reg_test_scaled
)

linear_mae = mean_absolute_error(
    y_reg_test,
    linear_predictions
)

linear_rmse = np.sqrt(
    mean_squared_error(
        y_reg_test,
        linear_predictions
    )
)

linear_r2 = r2_score(
    y_reg_test,
    linear_predictions
)

print("\nLinear Regression Results:")

print("MAE:", round(linear_mae, 2))
print("RMSE:", round(linear_rmse, 2))
print("R2 Score:", round(linear_r2, 4))


# ============================================================
# 22. RIDGE REGRESSION
# ============================================================

ridge_model = Ridge()

ridge_model.fit(
    X_reg_train_scaled,
    y_reg_train
)

ridge_predictions = ridge_model.predict(
    X_reg_test_scaled
)

ridge_rmse = np.sqrt(
    mean_squared_error(
        y_reg_test,
        ridge_predictions
    )
)

ridge_r2 = r2_score(
    y_reg_test,
    ridge_predictions
)

print("\nBaseline Ridge Regression:")

print("RMSE:", round(ridge_rmse, 2))
print("R2 Score:", round(ridge_r2, 4))


# ============================================================
# 23. RIDGE HYPERPARAMETER TUNING
# ============================================================

ridge_grid = {
    "alpha": [
        0.01,
        0.1,
        1,
        10,
        100
    ]
}

ridge_search = GridSearchCV(
    Ridge(),
    ridge_grid,
    cv=5,
    scoring="neg_mean_squared_error"
)

ridge_search.fit(
    X_reg_train_scaled,
    y_reg_train
)

best_ridge = ridge_search.best_estimator_

tuned_ridge_predictions = best_ridge.predict(
    X_reg_test_scaled
)

tuned_ridge_rmse = np.sqrt(
    mean_squared_error(
        y_reg_test,
        tuned_ridge_predictions
    )
)

tuned_ridge_r2 = r2_score(
    y_reg_test,
    tuned_ridge_predictions
)

print("\nOptimized Ridge Regression:")

print(
    "Best Parameters:",
    ridge_search.best_params_
)

print(
    "RMSE:",
    round(tuned_ridge_rmse, 2)
)

print(
    "R2 Score:",
    round(tuned_ridge_r2, 4)
)


# ============================================================
# 24. ACTUAL VS PREDICTED REGRESSION GRAPH
# ============================================================

plt.figure(figsize=(7, 6))

plt.scatter(
    y_reg_test,
    tuned_ridge_predictions,
    alpha=0.6
)

minimum = min(
    y_reg_test.min(),
    tuned_ridge_predictions.min()
)

maximum = max(
    y_reg_test.max(),
    tuned_ridge_predictions.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.title("Actual vs Predicted Total Spending")
plt.xlabel("Actual Spending")
plt.ylabel("Predicted Spending")
plt.tight_layout()

plt.savefig(
    "outputs/visualizations/regression_actual_vs_predicted.png"
)

plt.close()


# ============================================================
# 25. CLASSIFICATION MODEL
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION - PURCHASE LIKELIHOOD")
print("=" * 70)

classification_features = [
    "DaysSinceLastPurchase",
    "PurchaseFrequency",
    "TotalSpending",
    "CustomerRating",
    "WebsiteVisits",
    "DiscountUsage",
    "Cluster"
]

X_class = df[classification_features]

y_class = df["PurchaseLikelihood"]

X_class_train, X_class_test, y_class_train, y_class_test = (
    train_test_split(
        X_class,
        y_class,
        test_size=0.20,
        random_state=42,
        stratify=y_class
    )
)

class_scaler = StandardScaler()

X_class_train_scaled = class_scaler.fit_transform(
    X_class_train
)

X_class_test_scaled = class_scaler.transform(
    X_class_test
)


# ============================================================
# 26. BASELINE LOGISTIC REGRESSION
# ============================================================

logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

logistic_model.fit(
    X_class_train_scaled,
    y_class_train
)

class_predictions = logistic_model.predict(
    X_class_test_scaled
)

class_probabilities = logistic_model.predict_proba(
    X_class_test_scaled
)[:, 1]

accuracy = accuracy_score(
    y_class_test,
    class_predictions
)

precision = precision_score(
    y_class_test,
    class_predictions
)

recall = recall_score(
    y_class_test,
    class_predictions
)

f1 = f1_score(
    y_class_test,
    class_predictions
)

roc_auc = roc_auc_score(
    y_class_test,
    class_probabilities
)

print("\nBaseline Logistic Regression:")

print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))
print("ROC-AUC:", round(roc_auc, 4))


# ============================================================
# 27. LOGISTIC REGRESSION HYPERPARAMETER TUNING
# ============================================================

logistic_grid = {
    "C": [
        0.01,
        0.1,
        1,
        10,
        100
    ]
}

logistic_search = GridSearchCV(
    LogisticRegression(
        max_iter=1000,
        random_state=42
    ),
    logistic_grid,
    cv=5,
    scoring="f1"
)

logistic_search.fit(
    X_class_train_scaled,
    y_class_train
)

best_logistic = logistic_search.best_estimator_

tuned_predictions = best_logistic.predict(
    X_class_test_scaled
)

tuned_probabilities = best_logistic.predict_proba(
    X_class_test_scaled
)[:, 1]

tuned_accuracy = accuracy_score(
    y_class_test,
    tuned_predictions
)

tuned_precision = precision_score(
    y_class_test,
    tuned_predictions
)

tuned_recall = recall_score(
    y_class_test,
    tuned_predictions
)

tuned_f1 = f1_score(
    y_class_test,
    tuned_predictions
)

tuned_roc_auc = roc_auc_score(
    y_class_test,
    tuned_probabilities
)

print("\nOptimized Logistic Regression:")

print(
    "Best Parameters:",
    logistic_search.best_params_
)

print(
    "Accuracy:",
    round(tuned_accuracy, 4)
)

print(
    "Precision:",
    round(tuned_precision, 4)
)

print(
    "Recall:",
    round(tuned_recall, 4)
)

print(
    "F1 Score:",
    round(tuned_f1, 4)
)

print(
    "ROC-AUC:",
    round(tuned_roc_auc, 4)
)


# ============================================================
# 28. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_class_test,
    tuned_predictions
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

display.plot()

plt.title(
    "Purchase Likelihood Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    "outputs/visualizations/confusion_matrix.png"
)

plt.close()


# ============================================================
# 29. MODEL COMPARISON
# ============================================================

model_comparison = pd.DataFrame({
    "Model": [
        "K-Means",
        "Linear Regression",
        "Ridge Regression",
        "Logistic Regression"
    ],

    "Objective": [
        "Customer Segmentation",
        "Predict Total Spending",
        "Predict Total Spending",
        "Predict Purchase Likelihood"
    ],

    "BaselinePerformance": [
        f"Silhouette={best_silhouette:.4f}",
        f"RMSE={linear_rmse:.2f}, R2={linear_r2:.4f}",
        f"RMSE={ridge_rmse:.2f}, R2={ridge_r2:.4f}",
        f"F1={f1:.4f}, ROC-AUC={roc_auc:.4f}"
    ],

    "TunedPerformance": [
        f"Best K={best_k}",
        "Not Tuned",
        f"RMSE={tuned_ridge_rmse:.2f}, R2={tuned_ridge_r2:.4f}",
        f"F1={tuned_f1:.4f}, ROC-AUC={tuned_roc_auc:.4f}"
    ]
})

model_comparison.to_csv(
    "outputs/model_comparison.csv",
    index=False
)


# ============================================================
# 30. SAVE CUSTOMER SEGMENTS
# ============================================================

df.to_csv(
    "outputs/customer_segments.csv",
    index=False
)

cluster_profile.to_csv(
    "outputs/cluster_profile.csv"
)


# ============================================================
# 31. BUSINESS RECOMMENDATIONS
# ============================================================

business_insights = []

for cluster in cluster_profile.index:

    segment = segment_names[cluster]

    if cluster == high_value_cluster:

        recommendation = (
            "Provide loyalty rewards, premium product "
            "recommendations, early access to new products, "
            "and avoid unnecessary discounts."
        )

    else:

        recommendation = (
            "Use personalized campaigns, product recommendations, "
            "limited promotional offers, and encourage customers "
            "to increase purchase frequency."
        )

    business_insights.append(
        {
            "Cluster": cluster,
            "SegmentName": segment,
            "CustomerCount": int(
                cluster_profile.loc[
                    cluster,
                    "CustomerCount"
                ]
            ),
            "AverageSpending": round(
                cluster_profile.loc[
                    cluster,
                    "TotalSpending"
                ],
                2
            ),
            "RevenueContributionPercent": round(
                cluster_profile.loc[
                    cluster,
                    "RevenueContributionPercent"
                ],
                2
            ),
            "RecommendedBusinessAction": recommendation
        }
    )

business_df = pd.DataFrame(
    business_insights
)

business_df.to_csv(
    "outputs/business_recommendations.csv",
    index=False
)


# ============================================================
# 32. FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("BUSINESS INSIGHTS")
print("=" * 70)

for item in business_insights:

    print(
        f"\n{item['SegmentName']}"
    )

    print(
        "Customers:",
        item["CustomerCount"]
    )

    print(
        "Average Spending:",
        item["AverageSpending"]
    )

    print(
        "Revenue Contribution:",
        str(
            item["RevenueContributionPercent"]
        ) + "%"
    )

    print(
        "Recommended Action:",
        item["RecommendedBusinessAction"]
    )


print("\n" + "=" * 70)
print("TASK 8 COMPLETED SUCCESSFULLY!")
print("=" * 70)

print("\nGenerated files:")

print(
    "1. outputs/customer_segments.csv"
)

print(
    "2. outputs/cluster_profile.csv"
)

print(
    "3. outputs/model_comparison.csv"
)

print(
    "4. outputs/business_recommendations.csv"
)

print(
    "5. outputs/visualizations/"
)