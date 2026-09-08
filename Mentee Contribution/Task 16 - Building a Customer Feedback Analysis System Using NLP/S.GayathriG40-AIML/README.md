# 📝 Customer Feedback Analysis System Using NLP

## 📌 Project Overview

This project is a **Customer Feedback Analysis System** built using **Natural Language Processing (NLP)** techniques.

The system analyzes customer reviews and predicts whether the feedback is **Positive** or **Negative**. It also extracts important keywords from reviews and provides insights into the overall feedback.

The project focuses mainly on **classical NLP techniques** such as text preprocessing, Bag of Words, TF-IDF, n-grams, and Logistic Regression.

---

## 🎯 Objectives

The main objectives of this project are:

- Clean and preprocess customer feedback text.
- Perform basic exploratory data analysis on the reviews.
- Convert text into numerical features using NLP techniques.
- Perform sentiment classification.
- Identify important words and phrases from customer feedback.
- Evaluate the performance of different NLP approaches.
- Analyze incorrectly classified reviews.
- Build a reusable feedback prediction function.

---

## 📂 Dataset

The project uses the `finalReviews.csv` dataset.

### Dataset Information

| Feature | Description |
|--------|-------------|
| `review` | Customer/movie feedback text |
| `label` | Sentiment label |

The dataset contains:

- **302 reviews**
- **2 columns**
- **132 Negative samples**
- **170 Positive samples**

### Dataset Structure

```text
review,label
"Review text...",0
"Review text...",1