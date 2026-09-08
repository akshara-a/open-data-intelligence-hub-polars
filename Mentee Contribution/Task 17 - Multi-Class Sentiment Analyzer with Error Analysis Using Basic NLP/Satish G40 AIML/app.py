"""Streamlit interface for the sentiment analyzer."""

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "src"))
from predict import predict_sentiment  # noqa: E402

MODEL_PATH = ROOT_DIR / "models" / "sentiment_model.pkl"
VECTORIZER_PATH = ROOT_DIR / "models" / "tfidf_vectorizer.pkl"
OUTPUT_DIR = ROOT_DIR / "outputs"

st.set_page_config(page_title="Sentiment Analyzer", page_icon="📊", layout="wide")
st.title("Multi-Class Sentiment Analyzer")
st.caption("Basic NLP with TF-IDF and Logistic Regression")

if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
    st.warning("Model files are missing. Run `python src/train.py` before using the app.")
    st.stop()

sentence = st.text_area("Enter a sentence", placeholder="I really enjoyed this experience.", height=120)
if st.button("Analyze", type="primary", width="stretch"):
    if not sentence.strip():
        st.error("Please enter a sentence to analyze.")
    else:
        result = predict_sentiment(sentence)
        left, right = st.columns(2)
        with left:
            st.metric("Predicted sentiment", str(result["sentiment"]))
            st.metric("Confidence", f"{result['confidence']:.2%}")
        with right:
            probability_frame = pd.DataFrame(
                {"Probability": result["probabilities"]}
            ).reindex(["Positive", "Neutral", "Negative"])
            st.bar_chart(probability_frame)

st.divider()
metrics_path = OUTPUT_DIR / "metrics.json"
if metrics_path.exists():
    metrics = pd.read_json(metrics_path, typ="series")
    st.subheader("Model evaluation")
    metric_columns = st.columns(4)
    for column, label in zip(metric_columns, ["Accuracy", "Precision", "Recall", "F1-score"]):
        column.metric(label, f"{float(metrics[label.lower().replace('-', '') if label != 'F1-score' else 'f1']):.2%}")
    st.text("Classification report")
    st.code(str(metrics["classification_report"]))

confusion_path = OUTPUT_DIR / "confusion_matrix.png"
if confusion_path.exists():
    st.subheader("Confusion matrix")
    st.image(str(confusion_path), width="stretch")

error_path = OUTPUT_DIR / "error_analysis.csv"
if error_path.exists():
    st.subheader("Error analysis")
    errors = pd.read_csv(error_path)
    st.write(f"Incorrect predictions: {len(errors)}")
    st.dataframe(errors, use_container_width=True)
    st.info("Common errors arise from negation, mixed sentiment, neutral sentences, sarcasm, lack of context, rare words, and short sentences.")
