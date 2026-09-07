# ✈️ Airline Passenger Satisfaction Analysis using Machine Learning

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-red)
![Plotly](https://img.shields.io/badge/Plotly-Visualization-green)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-yellow)

---

# 📌 Project Overview

This project analyzes airline passenger satisfaction using Data Analysis and Machine Learning techniques.

The objective is to understand the factors that influence passenger satisfaction, identify different passenger segments, and build predictive models that help airlines improve customer experience and business performance.

The project follows a complete Data Science workflow:

- Data Collection
- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Passenger Segmentation
- Regression
- Classification
- Business Insights

---

# 🎯 Project Objectives

- Analyze airline passenger satisfaction.
- Identify key factors affecting customer satisfaction.
- Perform Exploratory Data Analysis (EDA).
- Engineer useful business features.
- Segment passengers using K-Means Clustering.
- Predict customer value using Regression.
- Predict passenger satisfaction using Classification.
- Generate actionable business recommendations.

---

# 📂 Project Structure

```
Airline-Passenger-Satisfaction/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── notebooks/
│   ├── 01_Data_Loading.ipynb
│   ├── 02_Data_Cleaning.ipynb
│   ├── 03_EDA.ipynb
│   ├── 04_Feature_Engineering.ipynb
│   ├── 05_Passenger_Segmentation.ipynb
│   ├── 06_Regression.ipynb
│   ├── 07_Classification.ipynb
│   ├── 08_Business_Insights.ipynb
│   └── 09_Final_Report.ipynb
│
├── models/
│
├── reports/
│
├── images/
│
├── requirements.txt
│
└── README.md
```

---

# 📊 Dataset Information

The project uses the Airline Passenger Satisfaction dataset.

### Dataset contains

- Passenger Demographics
- Flight Information
- Travel Type
- Customer Type
- Flight Distance
- Delay Information
- Service Ratings
- Passenger Satisfaction

Target Variable

```
satisfaction
```

---

# 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Plotly
- Scikit-Learn
- Jupyter Notebook

---

# 📈 Project Workflow

```
Dataset
     │
     ▼
Data Cleaning
     │
     ▼
Exploratory Data Analysis
     │
     ▼
Feature Engineering
     │
     ▼
Passenger Segmentation
     │
     ▼
Regression
     │
     ▼
Classification
     │
     ▼
Business Insights
```

---

# 🔍 Exploratory Data Analysis

The EDA phase includes:

- Missing Value Analysis
- Duplicate Value Analysis
- Passenger Age Distribution
- Flight Distance Analysis
- Customer Type Analysis
- Travel Type Analysis
- Passenger Class Analysis
- Satisfaction Analysis
- Correlation Analysis
- Service Rating Analysis

---

# ⚙️ Feature Engineering

The following business features were created:

### Total Delay

```
Departure Delay + Arrival Delay
```

### Overall Service Score

Average of all service ratings.

### Customer Value

```
Flight Distance × Travel Class Weight
```

---

# 👥 Passenger Segmentation

K-Means Clustering was applied to segment passengers.

Features used:

- Age
- Flight Distance
- Overall Service Score
- Total Delay

The resulting passenger clusters help airlines understand different customer groups.

---

# 📉 Regression

Regression models were developed to predict Customer Value.

Models:

- Linear Regression
- Ridge Regression

Evaluation Metrics

- MAE
- MSE
- RMSE
- R² Score

---

# 🤖 Classification

Classification models were developed to predict passenger satisfaction.

Model

- Logistic Regression

Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix

---

# 📊 Business Insights

The analysis provides several business recommendations.

Examples include:

- Improve in-flight service quality.
- Reduce flight delays.
- Increase loyalty benefits for premium passengers.
- Personalize marketing based on passenger segments.
- Enhance customer experience for business travelers.

---

# 📷 Sample Visualizations

The project includes visualizations such as:

- Histograms
- Pie Charts
- Bar Charts
- Correlation Heatmap
- PCA Visualization
- Cluster Distribution
- Customer Value Analysis
- Satisfaction Analysis

---

# 🚀 How to Run

Clone the repository

```bash
git clone https://github.com/your-username/airline-passenger-satisfaction.git
```

Go to the project directory

```bash
cd airline-passenger-satisfaction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook

```bash
jupyter notebook
```

Run the notebooks in order from:

```
01_Data_Loading.ipynb
```

to

```
09_Final_Report.ipynb
```

---

# 📌 Future Improvements

- Add advanced ensemble models (Random Forest, XGBoost).
- Hyperparameter tuning using GridSearchCV.
- Model explainability using SHAP.
- Automated ML pipeline.
- Deployment as a web application.

---

# 📚 Learning Outcomes

This project demonstrates:

- Data Cleaning
- Exploratory Data Analysis
- Feature Engineering
- Machine Learning
- Clustering
- Regression
- Classification
- Data Visualization
- Business Analytics
- Problem Solving

---

# 👩‍💻 Author

**Sasi Rekha**

B.Tech – Artificial Intelligence and Data Science

SKP Engineering College

---

# ⭐ If you found this project helpful, please consider giving it a star on GitHub.