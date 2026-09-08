import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = r"C:\Users\22053\Programming-Language-Popularity-Tracker\open-data-intelligence-hub\Mentee Contribution\Task 13 - Building an Image Recognition System for Quality Control\SanjanaG40-AIML\models\best_casting_defect_model.keras"

IMAGE_SIZE = (224, 224)

# Final MobileNet validation/test threshold
THRESHOLD = 0.52

CLASS_NAMES = ["ok_front", "def_front"]


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Casting Defect Detector",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 Casting Defect Detection")
st.write(
    "Upload a casting image to check whether it is OK or defective."
)


# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()


# -----------------------------
# Image upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a casting image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    # -----------------------------
    # Load image
    # -----------------------------
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Image")

    # Compatible with older Streamlit versions
    st.image(
        image,
        caption="Input Image",
        use_column_width=True
    )


    # -----------------------------
    # Preprocess
    # -----------------------------
    img = image.resize(IMAGE_SIZE)

    img_array = np.array(img, dtype=np.float32)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)


    # -----------------------------
    # Prediction
    # -----------------------------
    probability = float(
        model.predict(img_array, verbose=0)[0][0]
    )

    # The model output represents probability of DEFECTIVE (def_front)
    defective_probability = probability
    ok_probability = 1.0 - probability


    # -----------------------------
    # Classification
    # -----------------------------
    if defective_probability >= THRESHOLD:
        prediction = "DEFECTIVE"
        confidence = defective_probability
    else:
        prediction = "OK"
        confidence = ok_probability


    # -----------------------------
    # Display result
    # -----------------------------
    st.subheader("Prediction Result")

    if prediction == "DEFECTIVE":
        st.error("⚠️ DEFECTIVE CASTING")
    else:
        st.success("✅ OK CASTING")


    st.metric(
        label="Prediction",
        value=prediction
    )

    st.metric(
        label="Confidence",
        value=f"{confidence * 100:.2f}%"
    )


    # -----------------------------
    # Probability details
    # -----------------------------
    st.subheader("Prediction Details")

    st.write(
        f"OK probability: **{ok_probability * 100:.2f}%**"
    )

    st.write(
        f"Defective probability: **{defective_probability * 100:.2f}%**"
    )

    st.write(
        f"Decision threshold: **{THRESHOLD:.2f}**"
    )


    # -----------------------------
    # Probability bar
    # -----------------------------
    st.progress(
        min(max(defective_probability, 0.0), 1.0)
    )


    # -----------------------------
    # Explanation
    # -----------------------------
    if prediction == "DEFECTIVE":

        st.warning(
            "The model predicts that this casting contains a defect."
        )

    else:

        st.info(
            "The model predicts that this casting is acceptable."
        )