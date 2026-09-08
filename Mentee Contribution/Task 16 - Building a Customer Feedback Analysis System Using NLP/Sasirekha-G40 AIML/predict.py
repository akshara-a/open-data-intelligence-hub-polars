import argparse
from src.config import SENTIMENT_MODEL_FILE, PRODUCT_MODEL_FILE, PRIORITY_MODEL_FILE, TRAINING_TEXTS_FILE
from src.modeling import load_object
from src.analysis import extract_keywords, similar_feedback
from src.preprocessing import clean_text

def predict(feedback):
    sentiment_model = load_object(SENTIMENT_MODEL_FILE)
    product_model = load_object(PRODUCT_MODEL_FILE)
    priority_model = load_object(PRIORITY_MODEL_FILE)
    training_texts = load_object(TRAINING_TEXTS_FILE)
    cleaned = clean_text(feedback)
    return {
        "sentiment": sentiment_model.predict([cleaned])[0],
        "product": product_model.predict([cleaned])[0],
        "priority": priority_model.predict([cleaned])[0],
        "keywords": extract_keywords(feedback),
        "similar": similar_feedback(feedback, training_texts),
    }

def main():
    parser = argparse.ArgumentParser(description="Analyze customer complaints.")
    parser.add_argument("feedback", nargs="+")
    args = parser.parse_args()
    result = predict(" ".join(args.feedback))
    print("\nCustomer Feedback Analysis\n--------------------------")
    print("Sentiment:", result["sentiment"].title())
    print("Complaint category:", result["product"])
    print("Priority:", result["priority"].title())
    print("Keywords:")
    for x in result["keywords"]: print(" -", x)
    print("Similar feedback:")
    for text, score in result["similar"]: print(f" - {text} (similarity={score:.3f})")

if __name__ == "__main__": main()
