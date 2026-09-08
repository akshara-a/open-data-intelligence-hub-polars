# Multi-Class Sentiment Analyzer with Error Analysis

## Task 17 – NLP Assignment

A Natural Language Processing (NLP) project that classifies text into three sentiment categories — **Positive, Neutral, and Negative** — using TF-IDF feature extraction and a machine learning classifier.

The project also performs detailed **error analysis** to understand why the model makes incorrect predictions.

---

## 📌 Project Overview

Sentiment analysis is an NLP technique used to identify the opinion or sentiment expressed in a piece of text.

This project implements a complete sentiment classification pipeline:

```text
Dataset
   ↓
Data Understanding
   ↓
Data Cleaning
   ↓
Train-Test Split
   ↓
TF-IDF Feature Extraction
   ↓
Linear SVM Classification
   ↓
Prediction
   ↓
Model Evaluation
   ↓
Confusion Matrix
   ↓
Error Analysis
```

The model predicts three classes:

* 🟢 **Positive**
* ⚪ **Neutral**
* 🔴 **Negative**

---

## 🎯 Objectives

The main objectives of this project are:

1. Load and understand the sentiment dataset.
2. Clean and preprocess textual data.
3. Convert text into numerical features using **TF-IDF**.
4. Train a multi-class machine learning classifier.
5. Predict sentiment for unseen text.
6. Evaluate the model using multiple metrics.
7. Visualize the confusion matrix.
8. Analyze incorrect predictions.
9. Identify common causes of model errors.
10. Test the trained model on new sentences.

The assignment specifically requires a three-class sentiment analyzer and error analysis.

---

## 📂 Dataset

The dataset contains text comments and corresponding sentiment labels.

### Original Columns

```text
Comment
Sentiment
```

The sentiment values are converted into readable labels:

| Original Label | Sentiment |
| -------------: | --------- |
|              0 | Negative  |
|              1 | Neutral   |
|              2 | Positive  |

For consistency with the assignment, the columns are converted to:

```text
text
sentiment
```

The assignment requires a dataset containing at least `text` and `sentiment` columns.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Jupyter Notebook / VS Code

### NLP Techniques

* Text preprocessing
* TF-IDF Vectorization
* Unigram features
* Bigram features

### Machine Learning

**Linear Support Vector Machine (`LinearSVC`)**

### Evaluation

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* Error Analysis

---

## 📁 Project Structure

```text
Task-17-Sentiment-Analyzer/
│
├── sentiment_data.csv
├── sentiment_analysis.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🔄 Project Workflow

### 1. Load Dataset

The CSV dataset is loaded using Pandas.

```python
df = pd.read_csv("sentiment_data.csv")
```

The notebook also validates that the required dataset columns are available.

---

### 2. Data Understanding

The following checks are performed:

* Dataset shape
* Column names
* Data types
* Missing values
* Duplicate records
* Sentiment distribution

Example:

```python
df.shape
df.isnull().sum()
df["sentiment"].value_counts()
```

---

### 3. Data Cleaning

Text preprocessing includes:

* Converting text to lowercase
* Removing URLs
* Removing unnecessary symbols
* Normalizing whitespace
* Removing empty records
* Removing duplicate text-label pairs

A key design choice is to **preserve negation words** such as:

```text
not
no
never
```

This is useful because phrases such as:

```text
not good
not bad
never liked
```

can have meanings that are different from the individual words.

The assignment's basic cleaning approach also includes lowercase conversion, URL removal, special-character removal and whitespace normalization.

---

## 4. Train-Test Split

The dataset is divided into:

* **80% Training Data**
* **20% Testing Data**

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

`stratify=y` helps maintain similar proportions of Positive, Neutral and Negative samples in both datasets.

The assignment specifies an 80/20 split with `random_state=42` and stratification.

---

## 5. TF-IDF Feature Extraction

Machine learning algorithms require numerical input.

Therefore, textual data is converted into numerical vectors using:

**TF-IDF — Term Frequency-Inverse Document Frequency**

The notebook uses:

```python
TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    strip_accents="unicode"
)
```

### Why use unigrams and bigrams?

Unigrams capture individual words:

```text
good
bad
excellent
terrible
```

Bigrams capture two-word patterns:

```text
not good
not bad
very good
really enjoyed
very disappointing
```

This can provide more useful information for sentiment classification.

The assignment also recommends using `ngram_range=(1,2)`.

---

## 6. Machine Learning Algorithm

### Linear Support Vector Machine

The main classifier used in this improved implementation is:

```python
LinearSVC(
    C=2.0,
    class_weight="balanced",
    random_state=42
)
```

### Why Linear SVM?

Linear SVM is well suited for text classification because TF-IDF produces a high-dimensional sparse feature matrix.

Advantages include:

* Efficient for large text datasets
* Works well with sparse TF-IDF features
* Strong performance for traditional NLP classification
* Handles many features efficiently
* Supports multi-class classification

The assignment demonstrates Logistic Regression as the baseline classifier and mentions SVM as a possible alternative. This implementation uses SVM as the main model to strengthen the basic solution.

---

## 7. Model Evaluation

The model is evaluated using:

### Accuracy

Measures the percentage of correctly classified samples.

```text
Accuracy =
Correct Predictions / Total Predictions
```

### Precision

Measures how many predicted samples of a class were actually correct.

### Recall

Measures how many actual samples of a class were correctly identified.

### F1-Score

Combines precision and recall.

### Macro F1

Macro F1 gives equal importance to:

```text
Positive
Neutral
Negative
```

This is particularly useful for evaluating multi-class sentiment classification.

The assignment requires accuracy, precision, recall, F1-score and confusion-matrix based evaluation.

---

## 8. Confusion Matrix

The confusion matrix shows how predictions are distributed across the three classes.

Example structure:

| Actual / Predicted | Negative | Neutral | Positive |
| ------------------ | -------: | ------: | -------: |
| Negative           |  Correct |   Error |    Error |
| Neutral            |    Error | Correct |    Error |
| Positive           |    Error |   Error |  Correct |

The diagonal represents correct predictions, while off-diagonal values represent errors.

The notebook generates the confusion matrix using:

```python
ConfusionMatrixDisplay
```

---

# 🔎 9. Error Analysis

Error analysis is one of the main components of this project.

Instead of only asking:

> "How accurate is the model?"

we also ask:

> "Why did the model make this prediction incorrectly?"

The assignment identifies several common NLP error types. This project expands the analysis to **8 categories**.

---

## Error Type 1 – Negation

Example:

```text
The movie was not bad.
```

The word `bad` is negative, but the complete phrase can express a positive opinion.

A basic TF-IDF model may have difficulty understanding the interaction between the words.

### Common indicators

```text
not
never
no
without
hardly
```

---

## Error Type 2 – Mixed Sentiment

Example:

```text
The acting was amazing but the story was boring.
```

The sentence contains both positive and negative sentiment.

```text
amazing → positive
boring → negative
```

The model may have difficulty deciding the overall sentiment.

---

## Error Type 3 – Neutral Sentiment

Example:

```text
The movie was okay.
```

Neutral sentences often contain weak sentiment words such as:

```text
okay
average
fine
normal
ordinary
decent
```

These can be confused with positive or negative sentiment.

---

## Error Type 4 – Sarcasm

Example:

```text
Great, another boring movie.
```

The word:

```text
Great
```

normally indicates positive sentiment.

However, the complete sentence may express a negative opinion.

Traditional TF-IDF models have limited ability to understand sarcasm and implied meaning.

---

## Error Type 5 – Rare Vocabulary

Example:

```text
The film was phenomenal.
```

If `phenomenal` occurs very rarely in the training dataset, the model may not have enough information to associate it with positive sentiment.

Rare vocabulary can therefore contribute to incorrect predictions.

---

## Error Type 6 – Context-Dependent Sentences

Example:

```text
That was something.
```

or:

```text
It was fine.
```

The meaning may depend on context that is not available in the sentence itself.

A basic TF-IDF classifier has limited contextual understanding.

---

## Error Type 7 – Short Sentences

Example:

```text
Not good.
```

or:

```text
Amazing!
```

Very short sentences contain fewer features.

This can make classification more difficult.

---

## Error Type 8 – Slang / Noisy Spelling

Examples:

```text
lol this was amazing
omg terrible
sooooo good
idk about this
```

Informal language, abbreviations and repeated characters can create vocabulary variations that are difficult for a traditional NLP model.

---

## 10. Error Analysis Implementation

The notebook creates a DataFrame containing:

```text
text
actual
predicted
```

Incorrect predictions are extracted using:

```python
errors = results[
    results["actual"] != results["predicted"]
]
```

The notebook then:

* Counts incorrect predictions
* Calculates error rate
* Finds common actual → predicted confusion pairs
* Categorizes errors into 8 diagnostic types
* Displays examples for each category
* Identifies the sentiment class with the lowest F1-score

This extends the assignment's required incorrect-prediction analysis.

---

# 📊 11. Results

After running the notebook, the following values are generated automatically:

```text
Dataset samples
Training samples
Testing samples
TF-IDF features
Accuracy
Macro F1
Weighted F1
Incorrect predictions
Error rate
```

### Model

```text
TF-IDF + Linear SVM
```

### Important

The accuracy and F1-score depend on the actual dataset and train/test split.

Therefore, the final report should use the **actual values produced by the notebook** rather than a predefined example accuracy.

---

# 💡 12. Model Improvements

Compared with the basic implementation, this project includes several improvements:

### Data improvements

* Missing-value handling
* Empty-text removal
* Duplicate removal
* Sentiment-label validation

### Text improvements

* Lowercase normalization
* URL removal
* Special-character cleanup
* Negation preservation
* Unigram + bigram features

### TF-IDF improvements

```python
min_df=2
max_df=0.95
sublinear_tf=True
max_features=30000
```

### Model improvement

Instead of only using the assignment's baseline Logistic Regression, the final model uses:

```text
Linear SVM
```

with:

```python
class_weight="balanced"
```

### Evaluation improvement

In addition to accuracy, the notebook evaluates:

```text
Precision
Recall
F1-score
Macro F1
Weighted F1
Confusion Matrix
```

### Error-analysis improvement

The original five major error types are expanded to **eight diagnostic categories**.

---

# ⚠️ 13. Limitations

Although Linear SVM with TF-IDF is a strong traditional NLP approach, it has limitations.

### 1. Limited Context Understanding

TF-IDF mainly represents word and phrase importance.

It does not deeply understand sentence meaning.

### 2. Sarcasm

The model may struggle with sarcastic statements.

### 3. Negation

Bigrams help with phrases such as `not good`, but complex negation can still be difficult.

### 4. Rare Words

Words that appear very rarely may not provide enough training information.

### 5. Context

The model does not maintain a human-like understanding of broader context.

### 6. Slang

Informal language and spelling variations can reduce prediction quality.

---

# 🚀 14. Future Improvements

The project can be extended by:

1. Comparing Logistic Regression, Naive Bayes and Linear SVM.
2. Hyperparameter tuning using cross-validation.
3. Adding domain-specific preprocessing.
4. Using larger balanced datasets.
5. Using word embeddings.
6. Using advanced contextual models such as BERT or DistilBERT.

However, advanced deep-learning and Transformer models are outside the scope of this **basic NLP assignment**.

---

# 🧪 15. Example Predictions

The notebook includes a prediction function:

```python
def predict_sentiment(sentence):
    ...
```

Example:

```text
"This product is absolutely amazing"
→ positive

"The product arrived yesterday"
→ neutral

"This is a terrible experience"
→ negative
```

The actual output depends on the trained model.

---

# 📦 16. Installation

Install the required Python libraries:

```bash
pip install pandas numpy matplotlib scikit-learn jupyter
```

Or use the included `requirements.txt` file:

```bash
pip install -r requirements.txt
```

---

# ▶️ 17. How to Run

### Step 1 – Clone the repository

```bash
git clone <your-repository-url>
```

### Step 2 – Open the project

```bash
cd Task-17-Sentiment-Analyzer
```

### Step 3 – Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 – Open the notebook

```bash
jupyter notebook sentiment_analysis.ipynb
```

Or open the notebook directly in **VS Code**.

### Step 5 – Run All Cells

In VS Code:

```text
Run All
```

Check the generated:

* Dataset statistics
* Sentiment distribution
* TF-IDF information
* Accuracy
* Classification report
* Confusion matrix
* Error analysis
* New predictions

---

# 📋 18. Requirements

Create a `requirements.txt` file containing:

```text
pandas
numpy
matplotlib
scikit-learn
jupyter
```

---

# 🧠 19. Key Concepts

| Concept          | Purpose                                         |
| ---------------- | ----------------------------------------------- |
| NLP              | Processing human language                       |
| Text Cleaning    | Removes unnecessary noise                       |
| TF-IDF           | Converts text into numerical features           |
| Unigrams         | Individual words                                |
| Bigrams          | Two-word patterns                               |
| Linear SVM       | Text classification                             |
| Accuracy         | Overall correctness                             |
| Precision        | Correctness of positive predictions for a class |
| Recall           | Coverage of actual class samples                |
| F1-score         | Balance of precision and recall                 |
| Confusion Matrix | Shows class-wise predictions                    |
| Error Analysis   | Explains model mistakes                         |

---

# 🎓 20. Viva Questions

### What is NLP?

NLP stands for Natural Language Processing. It enables computers to process and analyze human language.

### What is sentiment analysis?

Sentiment analysis identifies the sentiment or opinion expressed in text.

### Why is this a multi-class problem?

Because the model predicts three classes:

```text
Positive
Neutral
Negative
```

### What is TF-IDF?

TF-IDF stands for Term Frequency-Inverse Document Frequency. It converts text into numerical features based on word importance.

### Why use bigrams?

Bigrams capture two-word patterns such as:

```text
not good
very bad
really enjoyed
```

### Why use Linear SVM?

Linear SVM works efficiently with high-dimensional sparse TF-IDF features and is effective for traditional text classification.

### Why use stratification?

It helps preserve class proportions between training and testing data.

### What is a confusion matrix?

It is a table showing correct and incorrect predictions for each class.

### What is error analysis?

Error analysis examines incorrect predictions to identify the reasons behind model mistakes.

### Why is error analysis important?

It helps identify model weaknesses and suggests directions for improvement.

---

# ✅ 21. Conclusion

This project successfully implements a **Multi-Class Sentiment Analyzer** using basic NLP and machine learning techniques.

The system classifies text into:

```text
Positive
Neutral
Negative
```

Text is cleaned and converted into numerical representations using **TF-IDF with unigram and bigram features**. A **Linear SVM classifier** is then trained to perform sentiment classification.

The model is evaluated using:

```text
Accuracy
Precision
Recall
F1-score
Confusion Matrix
```

Detailed error analysis is performed using eight categories:

```text
1. Negation
2. Mixed Sentiment
3. Neutral Sentiment
4. Sarcasm
5. Rare Vocabulary
6. Context-Dependent Sentences
7. Short Sentences
8. Slang / Noisy Spelling
```

The project demonstrates how a traditional NLP pipeline can be used to build an effective multi-class sentiment classification system while also investigating **why the model makes incorrect predictions**.
