import pandas as pd
from src.config import DATA_FILE, MODEL_DIR, SENTIMENT_MODEL_FILE, PRODUCT_MODEL_FILE, PRIORITY_MODEL_FILE, TRAINING_TEXTS_FILE, EVALUATION_FILE
from src.modeling import train_classifier, split_data, save_object, evaluate
from src.preprocessing import clean_text

def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)
    required = {"Consumer_complaint", "Product", "Sentiment", "Priority"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    df = df.dropna(subset=list(required)).copy()
    df["feedback"] = df["Consumer_complaint"].astype(str).map(clean_text)
    # In this dataset: 0 = negative, 1 = positive.
    df["sentiment"] = df["Sentiment"].astype(int).map({0: "negative", 1: "positive"})
    # Product is the complaint category supplied by the dataset.
    df["product"] = df["Product"].astype(str).str.strip()
    df["priority"] = df["Priority"].astype(int).map({0: "low", 1: "high"})

    train_df, test_df = split_data(df)
    sentiment_model = train_classifier(train_df["feedback"], train_df["sentiment"])
    product_model = train_classifier(train_df["feedback"], train_df["product"])
    priority_model = train_classifier(train_df["feedback"], train_df["priority"])

    s_acc, s_f1, s_report = evaluate(sentiment_model, test_df["feedback"], test_df["sentiment"])
    p_acc, p_f1, p_report = evaluate(product_model, test_df["feedback"], test_df["product"])
    r_acc, r_f1, r_report = evaluate(priority_model, test_df["feedback"], test_df["priority"])

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_object(sentiment_model, SENTIMENT_MODEL_FILE)
    save_object(product_model, PRODUCT_MODEL_FILE)
    save_object(priority_model, PRIORITY_MODEL_FILE)
    save_object(df["feedback"].tolist(), TRAINING_TEXTS_FILE)

    report = f"""CUSTOMER FEEDBACK NLP - EVALUATION

Dataset rows: {len(df)}
Training rows: {len(train_df)}
Test rows: {len(test_df)}

SENTIMENT
Accuracy: {s_acc:.4f}
Weighted F1: {s_f1:.4f}
{s_report}
PRODUCT / COMPLAINT CATEGORY
Accuracy: {p_acc:.4f}
Weighted F1: {p_f1:.4f}
{p_report}
PRIORITY
Accuracy: {r_acc:.4f}
Weighted F1: {r_f1:.4f}
{r_report}
"""
    EVALUATION_FILE.write_text(report, encoding="utf-8")
    print(report)
    print("Training completed successfully.")
    print(f"Models saved in: {MODEL_DIR}")

if __name__ == "__main__":
    main()
