import pandas as pd
import re
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ============================================================
# 1. Load Dataset
# ============================================================

df = pd.read_csv("sentiment_data.csv")

print("\nDataset Preview:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nSentiment Distribution:")
print(df["sentiment"].value_counts())

# ============================================================
# 2. Remove Missing Values
# ============================================================

df = df.dropna(subset=["text", "sentiment"]).copy()

# ============================================================
# 3. Text Cleaning
# ============================================================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["clean_text"] = df["text"].apply(clean_text)

# ============================================================
# 4. Input and Target
# ============================================================

X = df["clean_text"]
y = df["sentiment"]

# ============================================================
# 5. Train-Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# ============================================================
# 6. TF-IDF Vectorization
# ============================================================

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=1
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("\nTF-IDF Training Matrix:", X_train_tfidf.shape)
print("TF-IDF Testing Matrix:", X_test_tfidf.shape)

# ============================================================
# 7. Logistic Regression
# ============================================================

model = LogisticRegression(
    max_iter=2000,
    C=2.0,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_tfidf, y_train)

# ============================================================
# 8. Prediction
# ============================================================

y_pred = model.predict(X_test_tfidf)

# ============================================================
# 9. Model Evaluation
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

report = classification_report(
    y_test,
    y_pred,
    zero_division=0
)

print("\n==============================")
print("FINAL MODEL PERFORMANCE")
print("==============================")

print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy Percentage: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(report)

# ============================================================
# 10. Confusion Matrix
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=model.classes_
)

print("\nConfusion Matrix:")
print(cm)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

display.plot()
plt.title("Multi-Class Sentiment Confusion Matrix")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=300)
plt.close()

# ============================================================
# 11. Error Analysis
# ============================================================

results = pd.DataFrame({
    "text": X_test.values,
    "actual": y_test.values,
    "predicted": y_pred
})

errors = results[
    results["actual"] != results["predicted"]
].copy()

error_rate = len(errors) / len(results)

print("\n==============================")
print("ERROR ANALYSIS")
print("==============================")

print("\nIncorrect Predictions:")
print(errors.head(20))

print("\nTotal Test Samples:", len(results))
print("Incorrect Predictions:", len(errors))
print(f"Error Rate: {error_rate:.4f}")
print(f"Error Percentage: {error_rate * 100:.2f}%")

print("\nError Counts:")
print(
    errors.groupby(
        ["actual", "predicted"]
    ).size()
)

# ============================================================
# 12. Neutral Sentiment Errors
# ============================================================

neutral_errors = errors[
    errors["actual"] == "neutral"
]

print("\nNeutral Sentiment Errors:")
print(neutral_errors.head(10))

# ============================================================
# 13. Save Error Analysis
# ============================================================

errors.to_csv(
    "results/error_analysis.csv",
    index=False
)

# ============================================================
# 14. Prediction Function
# ============================================================

def predict_sentiment(sentence):
    cleaned_sentence = clean_text(sentence)

    sentence_tfidf = tfidf.transform(
        [cleaned_sentence]
    )

    prediction = model.predict(
        sentence_tfidf
    )

    return prediction[0]

# ============================================================
# 15. Test New Sentences
# ============================================================

print("\n==============================")
print("NEW SENTENCE PREDICTIONS")
print("==============================")

test_sentences = [
    "I really loved this movie",
    "The movie was okay",
    "This was a terrible movie",
    "The acting was excellent",
    "Nothing special about this film",
    "I would definitely recommend this movie",
    "The story was boring and disappointing",
    "The film was average"
]

for sentence in test_sentences:
    sentiment = predict_sentiment(sentence)
    print(f"{sentence} -> {sentiment}")

# ============================================================
# 16. Save Final Results
# ============================================================

with open(
    "results/model_results.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write("MULTI-CLASS SENTIMENT ANALYZER\n")
    file.write("==============================\n\n")

    file.write(f"Dataset Size: {len(df)}\n")
    file.write(f"Training Samples: {len(X_train)}\n")
    file.write(f"Testing Samples: {len(X_test)}\n")

    file.write(
        f"Accuracy: {accuracy:.4f}\n"
    )

    file.write(
        f"Accuracy Percentage: {accuracy * 100:.2f}%\n"
    )

    file.write(
        f"Error Rate: {error_rate:.4f}\n"
    )

    file.write(
        f"Error Percentage: {error_rate * 100:.2f}%\n\n"
    )

    file.write("Classification Report:\n")
    file.write(report)

    file.write("\n\nConfusion Matrix:\n")
    file.write(str(cm))

print("\nTask 17 completed successfully!")
print("Final results saved in the results folder.")
