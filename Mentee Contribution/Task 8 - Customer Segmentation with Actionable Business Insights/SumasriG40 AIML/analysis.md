# Analysis Report – Customer Segmentation with Actionable Business Insights

## Project Overview

This project analyzes customer behavior using machine learning techniques to identify distinct customer segments. The goal is to understand customer purchasing patterns and generate actionable business insights that can help improve marketing strategies, customer retention, and business decision-making.

---

## Dataset Information

**Dataset Name:** E-commerce Customer Behavior

**Number of Records:** Based on the provided dataset

**Features Used:**

- Customer ID
- Gender
- Age
- City
- Membership Type
- Total Spend
- Items Purchased
- Average Rating
- Discount Applied
- Days Since Last Purchase
- Satisfaction Level

---

## Data Preprocessing

The following preprocessing steps were performed:

- Loaded the dataset using Pandas.
- Checked for missing values.
- Removed missing values in the Satisfaction Level column.
- Converted categorical variables into numerical values using encoding.
- Scaled numerical features before clustering.
- Created a target variable for classification.

---

## Exploratory Data Analysis (EDA)

EDA was performed to understand customer behavior.

The following visualizations were created:

- Customer Age Distribution
- Total Spend Distribution
- Membership Type Distribution
- Average Rating Distribution
- Correlation Heatmap
- Cluster Visualization

These visualizations helped identify spending patterns and customer characteristics.

---

## Customer Segmentation

Customer segmentation was performed using the K-Means Clustering algorithm.

Steps performed:

1. Selected important numerical features.
2. Standardized the data.
3. Applied the Elbow Method to determine the optimal number of clusters.
4. Trained the K-Means model.
5. Assigned cluster labels to each customer.
6. Generated cluster summaries.

The resulting customer groups represent customers with different purchasing behaviors.

---

## Regression Analysis

A Linear Regression model was developed to predict **Total Spend** using customer-related features.

A Ridge Regression model was also implemented to reduce overfitting and improve prediction stability.

Evaluation metrics included:

- R² Score
- Root Mean Squared Error (RMSE)

---

## Classification Analysis

A Logistic Regression model was used to classify customer satisfaction levels.

Hyperparameter tuning was performed using GridSearchCV to identify the best-performing model parameters.

Evaluation metrics included:

- Accuracy
- Classification Report
- Confusion Matrix

---

## Business Insights

The analysis provides the following insights:

- High-spending customers should receive premium membership benefits.
- Customers with frequent purchases are ideal targets for loyalty programs.
- Customers with lower spending can be encouraged through personalized discounts.
- Satisfaction levels can be improved using targeted promotional campaigns.
- Customer segmentation enables more effective and personalized marketing strategies.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

---

## Conclusion

This project successfully demonstrates the application of machine learning techniques for customer segmentation and predictive analysis. By combining clustering, regression, and classification models, the analysis provides meaningful business insights that can help organizations improve customer engagement, optimize marketing efforts, and support data-driven decision-making.
