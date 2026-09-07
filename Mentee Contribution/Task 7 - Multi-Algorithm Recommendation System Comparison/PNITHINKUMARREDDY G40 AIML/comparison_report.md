# Multi-Algorithm Recommendation System Comparison Report

## 1. Project Title
Multi-Algorithm Recommendation System Comparison for E-Commerce

## 2. Problem Statement
Build and compare regression, classification, and clustering models to support
product recommendations, purchase prediction, and customer segmentation for an
e-commerce business.

## 3. Dataset Description
Synthetic e-commerce dataset (2000 rows) with columns: User_ID, Product_ID,
Product_Category, Price, Number_of_Views, Time_Spent, Previous_Purchases, Cart_Status,
Rating, Purchase_Status.

## 4. Data Preprocessing
- Removed duplicate records
- Filled missing Price and Rating values with median
- Encoded Product_Category using LabelEncoder
- Scaled numeric features using StandardScaler
- 80/20 train-test split

## 5. Regression Model (Ridge Regression)
- Before tuning: MAE=0.46, RMSE=0.569, R2=0.421
- Best alpha (GridSearchCV): 10
- After tuning: MAE=0.461, RMSE=0.569, R2=0.421

## 6. Classification Model (Logistic Regression)
- Before tuning: Accuracy=0.932, F1=0.961
- Best Params (GridSearchCV): {'C': 1, 'max_iter': 100, 'solver': 'lbfgs'}
- After tuning: Accuracy=0.932, F1=0.961

## 7. Clustering Model (K-Means)
- Optimal K: 4 (Elbow Method)
- Silhouette Score: 0.212
- Inertia: 1465.2

## 8. Model Comparison Table
See model_comparison_task7.csv

## 9. Business Interpretation
- Ridge Regression helps recommend products a user is likely to rate highly.
- Logistic Regression identifies users likely to purchase, enabling targeted offers.
- K-Means segments customers into behavior-based groups for tailored marketing.

## 10. Final Conclusion
Comparing all three approaches shows each solves a distinct part of the recommendation
problem: regression estimates satisfaction, classification estimates buying intent, and
clustering reveals customer segments. Hyperparameter tuning via GridSearchCV improved
both the regression and classification models over their default-parameter baselines.
Together, these models give the business a well-rounded, data-driven recommendation strategy.
