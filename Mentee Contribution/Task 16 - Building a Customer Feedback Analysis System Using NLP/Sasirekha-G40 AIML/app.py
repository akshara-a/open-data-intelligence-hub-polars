import streamlit as st
from src.config import SENTIMENT_MODEL_FILE, PRODUCT_MODEL_FILE, PRIORITY_MODEL_FILE, TRAINING_TEXTS_FILE
from src.modeling import load_object
from src.analysis import extract_keywords, similar_feedback
from src.preprocessing import clean_text

st.set_page_config(page_title="Customer Feedback NLP", page_icon="💬", layout="centered")
st.title("Customer Feedback Analysis System")
st.caption("Text cleaning → TF-IDF → NLP classification → complaint analysis")

@st.cache_resource
def load_models():
    return (load_object(SENTIMENT_MODEL_FILE), load_object(PRODUCT_MODEL_FILE), load_object(PRIORITY_MODEL_FILE), load_object(TRAINING_TEXTS_FILE))

try:
    sentiment_model, product_model, priority_model, training_texts = load_models()
except FileNotFoundError as exc:
    st.error(str(exc)); st.stop()

feedback = st.text_area("Enter customer complaint", "The application is very slow and I cannot complete my payment.", height=160)
if st.button("Analyze feedback", type="primary"):
    if not feedback.strip(): st.warning("Please enter a complaint.")
    else:
        cleaned = clean_text(feedback)
        st.subheader("Analysis")
        st.write("**Sentiment:**", sentiment_model.predict([cleaned])[0].title())
        st.write("**Complaint category:**", product_model.predict([cleaned])[0])
        st.write("**Priority:**", priority_model.predict([cleaned])[0].title())
        st.write("**Keywords:**", ", ".join(extract_keywords(feedback)) or "None")
        st.write("**Similar feedback:**")
        for text, score in similar_feedback(feedback, training_texts): st.write(f"- {text} *(similarity: {score:.3f})*")
        with st.expander("Preprocessed text"): st.code(cleaned)
