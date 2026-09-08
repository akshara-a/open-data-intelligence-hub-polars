# Task 8 - Customer Segmentation with Actionable Business Insights

## Submitted By

**Name:** Nithin Kumar  
**Folder:** SNITHINKUMAR

---

## Project Overview

This project performs customer segmentation for an e-commerce business using machine learning.

The main objective is to group customers with similar purchasing behaviour and convert the identified customer segments into actionable business recommendations.

The project includes:

- Exploratory Data Analysis (EDA)
- Data preprocessing
- K-Means clustering
- Elbow Method
- Silhouette Score analysis
- Customer segment profiling
- Linear Regression
- Ridge Regression
- Logistic Regression
- Hyperparameter tuning using GridSearchCV
- Model evaluation
- Business recommendations
- Data visualizations

---

## Business Problem

An e-commerce company wants to better understand its customers based on purchasing behaviour, spending, engagement, recency, and discount usage.

Customer segmentation can help the company:

- Identify high-value customers
- Improve customer retention
- Personalize marketing campaigns
- Recommend suitable products
- Optimize promotional strategies
- Improve revenue contribution

---

## Dataset

A realistic synthetic customer dataset containing **1,500 customers and 13 columns** was created for this project.

### Dataset Features

- CustomerID
- Age
- Gender
- AnnualIncome
- TotalSpending
- PurchaseFrequency
- AverageOrderValue
- DaysSinceLastPurchase
- WebsiteVisits
- DiscountUsage
- CustomerRating
- ProductCategory
- PurchaseLikelihood

The dataset contains no missing values or duplicate records.

---

## Exploratory Data Analysis

EDA was performed to understand customer behaviour and data quality.

The analysis included:

- Dataset shape
- Data types
- Missing-value analysis
- Duplicate analysis
- Statistical summary
- Customer spending distribution
- Recency vs purchase frequency
- Correlation analysis

---

## Customer Segmentation

The following features were used for K-Means clustering:

- TotalSpending
- PurchaseFrequency
- AverageOrderValue
- DaysSinceLastPurchase
- WebsiteVisits
- DiscountUsage
- CustomerRating

The features were standardized using `StandardScaler`.

---

## Selecting the Number of Clusters

K-Means models were tested for cluster values from **K = 2 to K = 8**.

Both inertia and Silhouette Score were evaluated.

| K | Silhouette Score |
|---|---:|
| 2 | 0.1783 |
| 3 | 0.1480 |
| 4 | 0.1415 |
| 5 | 0.1314 |
| 6 | 0.1312 |
| 7 | 0.1335 |
| 8 | 0.1368 |

The highest Silhouette Score was obtained for:

**K = 2**

Therefore, two customer segments were selected.

---

## Customer Segments

### Segment 1 - High-Value Loyal Customers

- Number of customers: **534**
- Average spending: **157,664.40**
- Average purchase frequency: **21.68**
- Average order value: **7,381.72**
- Revenue contribution: **69.97%**

### Recommended Business Actions

- Provide loyalty rewards
- Recommend premium products
- Provide early access to new products
- Introduce exclusive membership benefits
- Avoid unnecessary discounts

---

### Segment 2 - Regular Customers

- Number of customers: **966**
- Average spending: **37,411.94**
- Average purchase frequency: **12.62**
- Average order value: **3,782.09**
- Revenue contribution: **30.03%**

### Recommended Business Actions

- Use personalized marketing campaigns
- Recommend relevant products
- Provide limited promotional offers
- Encourage repeat purchases
- Increase purchase frequency through targeted engagement

---

## Regression

Regression models were developed to predict customer total spending.

### Linear Regression

- MAE: **22,071.41**
- RMSE: **30,000.70**
- R² Score: **0.8196**

### Ridge Regression

Baseline:

- RMSE: **30,003.51**
- R² Score: **0.8196**

After GridSearchCV:

- Best Alpha: **1**
- RMSE: **30,003.51**
- R² Score: **0.8196**

The regression models explain approximately **81.96%** of the variation in total customer spending.

---

## Classification

Logistic Regression was used to predict customer purchase likelihood.

### Baseline Logistic Regression

- Accuracy: **0.9867**
- Precision: **0.9932**
- Recall: **0.9800**
- F1 Score: **0.9866**
- ROC-AUC: **0.9997**

### Optimized Logistic Regression

GridSearchCV was used for hyperparameter optimization.

Best parameter:

`C = 100`

Performance:

- Accuracy: **0.9967**
- Precision: **0.9934**
- Recall: **1.0000**
- F1 Score: **0.9967**
- ROC-AUC: **1.0000**

---

## Business Insights

The analysis identified two major customer groups.

High-Value Loyal Customers represent only 534 customers but contribute approximately **69.97% of total revenue**.

This indicates that retaining and rewarding these customers should be a major business priority.

Regular Customers represent the larger customer group. Personalized marketing, product recommendations, and targeted promotions can be used to increase their purchase frequency and customer value.

---

## Generated Outputs

The project generates:

- `customer_segments.csv`
- `cluster_profile.csv`
- `model_comparison.csv`
- `business_recommendations.csv`

### Visualizations

- Customer spending distribution
- Recency vs purchase frequency
- Correlation matrix
- Elbow Method
- Silhouette Score comparison
- PCA customer cluster visualization
- Customer count by segment
- Average spending by segment
- Regression actual vs predicted values
- Classification confusion matrix

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

---

## How to Run

Install the required libraries:

pip install -r requirements.txt

Generate the dataset:

python generate_dataset.py

Run the complete customer segmentation analysis:

python customer_segmentation.py

---

## Project Structure

SNITHINKUMAR/

- data/
  - customer_data.csv
- outputs/
  - visualizations/
  - customer_segments.csv
  - cluster_profile.csv
  - model_comparison.csv
  - business_recommendations.csv
- generate_dataset.py
- customer_segmentation.py
- requirements.txt
- README.md

---

## Conclusion

This project demonstrates an end-to-end machine learning workflow for customer segmentation and business analytics.

K-Means clustering successfully identified two major customer segments. Regression was used to predict customer spending, while Logistic Regression was used to predict purchase likelihood.

The final results were converted into actionable recommendations that can support customer retention, targeted marketing, product recommendations, and revenue growth.