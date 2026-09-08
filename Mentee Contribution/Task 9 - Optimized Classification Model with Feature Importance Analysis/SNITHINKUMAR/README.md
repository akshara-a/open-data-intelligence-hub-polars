# Task 9 - Optimized Classification Model with Feature Importance Analysis

## Project Title

**Predicting E-Commerce Purchase Likelihood Using an Optimized Classification Model**

## Author

**SNITHINKUMAR**

---

## Project Overview

This project develops a machine learning classification system to predict whether an e-commerce customer is likely to complete a purchase.

The prediction is based on customer demographics, browsing behaviour, engagement information, and previous purchasing activity.

The project includes:

- Exploratory Data Analysis (EDA)
- Data preprocessing
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Hyperparameter optimization using GridSearchCV
- Model evaluation and comparison
- Confusion matrix analysis
- ROC-AUC analysis
- Feature importance analysis
- Classification threshold analysis
- Customer purchase-likelihood segmentation
- Business recommendations
- Model persistence using Joblib

---

## Business Problem

E-commerce websites receive many visitors, but not every visitor completes a purchase.

A machine learning model can help identify customers with a higher probability of purchasing so that marketing resources can be used more effectively.

The target variable is:

- `0` - Customer did not purchase
- `1` - Customer completed a purchase

Therefore, this is a **binary classification problem**.

---

## Dataset

A synthetic e-commerce customer dataset containing **1,500 records and 17 columns** was created for this project.

### Target Distribution

| Purchase | Customers | Percentage |
|---|---:|---:|
| No Purchase (0) | 1096 | 73.07% |
| Purchase (1) | 404 | 26.93% |

The dataset contains moderate class imbalance.

### Features

Important features include:

- Age
- Gender
- Device Type
- Traffic Source
- Pages Viewed
- Time on Site
- Products Viewed
- Cart Items
- Previous Purchases
- Average Order Value
- Discount Used
- Email Clicked
- Ad Clicked
- Days Since Last Visit
- Session Count

`CustomerID` was removed before model training because it is an identifier rather than a meaningful predictive feature.

---

## Data Preprocessing

The preprocessing pipeline performs:

- Median imputation for numerical variables
- Most-frequent imputation for categorical variables
- StandardScaler for numerical variables
- One-hot encoding for categorical variables
- 80/20 train-test split
- Stratified sampling

Using a pipeline ensures that preprocessing is applied consistently and helps prevent data leakage.

---

## Models Used

Three baseline classification algorithms were trained:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

Random Forest was then optimized using **GridSearchCV with 5-fold cross-validation**.

The optimization metric used was **F1-Score**, which provides a balance between precision and recall.

---

## Model Performance

| Model | Status | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| Logistic Regression | Baseline | 0.7433 | 0.5588 | 0.2346 | 0.3304 | 0.7509 |
| Decision Tree | Baseline | 0.6533 | 0.3678 | 0.3951 | 0.3810 | 0.5720 |
| Random Forest | Baseline | 0.7300 | 0.5000 | 0.0864 | 0.1474 | 0.7181 |
| Random Forest | Optimized | **0.7100** | **0.4625** | **0.4568** | **0.4596** | **0.7259** |

---

## Effect of Hyperparameter Optimization

Hyperparameter optimization significantly improved the Random Forest model's ability to identify purchasers.

### Baseline Random Forest

F1-Score:

`0.1474`

Recall:

`0.0864`

ROC-AUC:

`0.7181`

### Optimized Random Forest

F1-Score:

`0.4596`

Recall:

`0.4568`

ROC-AUC:

`0.7259`

Although accuracy decreased slightly from `0.7300` to `0.7100`, the optimized model achieved much better recall and F1-score.

This is useful because identifying potential purchasers is more important than relying only on overall accuracy.

---

## Confusion Matrix

The optimized Random Forest produced:

| | Predicted No Purchase | Predicted Purchase |
|---|---:|---:|
| Actual No Purchase | 176 | 43 |
| Actual Purchase | 44 | 37 |

This means:

- True Negatives = 176
- False Positives = 43
- False Negatives = 44
- True Positives = 37

---

## Feature Importance Analysis

The most influential features identified by the optimized Random Forest were:

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | Cart Items | 0.148872 |
| 2 | Average Order Value | 0.104336 |
| 3 | Time on Site | 0.103550 |
| 4 | Days Since Last Visit | 0.093749 |
| 5 | Products Viewed | 0.085172 |
| 6 | Previous Purchases | 0.083899 |
| 7 | Age | 0.075393 |
| 8 | Pages Viewed | 0.071578 |
| 9 | Session Count | 0.067985 |
| 10 | Discount Used | 0.019518 |

`CartItems` was the most important predictive feature in the final model.

Feature importance represents predictive influence and should not be interpreted as proof of causation.

---

## Threshold Analysis

Different classification thresholds were evaluated.

| Threshold | Precision | Recall | F1-Score |
|---:|---:|---:|---:|
| 0.30 | 0.3244 | 0.9012 | 0.4771 |
| 0.40 | 0.4225 | 0.7407 | **0.5381** |
| 0.50 | 0.4625 | 0.4568 | 0.4596 |
| 0.60 | 0.5769 | 0.1852 | 0.2804 |
| 0.70 | 0.6000 | 0.0370 | 0.0698 |

Among the tested thresholds, **0.40 produced the highest F1-score of 0.5381**.

A lower threshold can identify more potential purchasers but may increase false positives.

A higher threshold improves precision but may miss more potential purchasers.

---

## Customer Purchase-Likelihood Categories

Customers were divided into three categories based on predicted purchase probability:

- **Low:** 0.00 - 0.29
- **Medium:** 0.30 - 0.59
- **High:** 0.60 - 1.00

Results:

| Category | Customers |
|---|---:|
| Low | 75 |
| Medium | 199 |
| High | 26 |

These categories can help prioritize customers for different marketing strategies.

---

## Business Recommendations

1. High-likelihood customers should receive personalized product recommendations and cart reminders.

2. Medium-likelihood customers can be targeted with discounts, reviews, promotional emails, and product comparisons.

3. Expensive marketing campaigns should be limited for low-likelihood customers until stronger engagement signals appear.

4. Cart activity and other high-importance behavioural features can be used to identify customers showing strong buying intent.

5. Purchase probabilities should be used to prioritize customers for remarketing instead of relying only on binary predictions.

6. A threshold around `0.40` may be considered when the business wants a better balance between identifying purchasers and controlling false positives.

---

## Limitations

- The project uses a synthetic educational dataset.
- Real-world customer behaviour may be more complex.
- Feature importance does not prove causation.
- Model performance may change as customer behaviour changes.
- The model should be retrained and evaluated using new real-world data before production deployment.

---

## Project Structure

```text
SNITHINKUMAR/
│
├── data/
│   └── ecommerce_customer_data.csv
│
├── models/
│   └── purchase_prediction_model.pkl
│
├── outputs/
│   ├── baseline_model_results.csv
│   ├── business_recommendations.txt
│   ├── confusion_matrix.png
│   ├── customer_purchase_predictions.csv
│   ├── feature_importance.csv
│   ├── feature_importance.png
│   ├── model_comparison.csv
│   ├── purchase_distribution.png
│   ├── purchase_rate_by_device.png
│   ├── purchase_rate_by_traffic.png
│   ├── roc_curve.png
│   └── threshold_analysis.csv
│
├── generate_dataset.py
├── model.py
└── README.md