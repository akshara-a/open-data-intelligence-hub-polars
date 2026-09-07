# Multi-Algorithm Recommendation System Comparison

## Overview
This project builds and compares Ridge Regression, Logistic Regression, and 
K-Means Clustering to solve different parts of an e-commerce recommendation 
problem, with before/after hyperparameter tuning comparisons via GridSearchCV.

## Dataset
Synthetic e-commerce dataset (2000 rows) with columns: User_ID, Product_ID, 
Product_Category, Price, Number_of_Views, Time_Spent, Previous_Purchases, 
Cart_Status, Rating, Purchase_Status.

## Results Summary

| Model | Task | Before Tuning | After Tuning |
|---|---|---|---|
| Ridge Regression | Rating Prediction | MAE=0.46, R2=0.421 | MAE=0.461, R2=0.421 |
| Logistic Regression | Purchase Prediction | Acc=0.932, F1=0.961 | Acc=0.932, F1=0.961 |
| K-Means | Segmentation | — | Silhouette=0.212 (k=4) |

Note: GridSearchCV confirmed the default hyperparameters were already 
near-optimal for this dataset, so tuning produced minimal change - itself 
a useful finding about model stability.

## Files
- `multi_algorithm_recommendation_comparison.ipynb` — full notebook
- `ecommerce_multi_algo_data.csv` — dataset
- `eda_charts_task7.png` — EDA visualizations
- `elbow_method_task7.png` — K-Means elbow curve
- `model_comparison_task7.csv` — model comparison table
- `comparison_report.md` — full comparison report (problem statement, 
  preprocessing, models, tuning, business interpretation, conclusion)

## Business Value
- **Regression** → recommend products a user is likely to rate highly
- **Classification** → identify likely buyers for targeted offers
- **Clustering** → segment customers for tailored marketing strategy