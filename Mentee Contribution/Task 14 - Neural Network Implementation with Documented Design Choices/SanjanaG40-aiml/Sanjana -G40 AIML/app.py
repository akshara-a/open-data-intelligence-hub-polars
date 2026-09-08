import os
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Casting Quality Inspection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-header {
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        background: #1f2937;
        color: white;
    }

    .main-header h1 {
        margin-bottom: 0.5rem;
        font-size: 2.2rem;
    }

    .main-header p {
        margin-bottom: 0;
        font-size: 1rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# CONSTANTS
# ============================================================

MODEL_PATH = r"C:\Users\22053\Programming-Language-Popularity-Tracker\open-data-intelligence-hub\Mentee Contribution\Task 14 - Neural Network Implementation with Documented Design Choices\SanjanaG40-aiml\models\casting_cnn_model.keras"

CLASS_NAMES = {
    0: "def_front",
    1: "ok_front"
}

IMAGE_SIZE = (224, 224)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_cnn_model():
    """
    Load the trained CNN model once and cache it.
    """
    if not os.path.exists(MODEL_PATH):
        return None

    return load_model(MODEL_PATH)


model = load_cnn_model()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-header">
        <h1>🔍 Casting Quality Inspection</h1>
        <p>
            AI-powered visual inspection using a Convolutional Neural Network
            to classify casting components as defective or acceptable.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Model Information")

    st.markdown(
        """
        **Model:** Convolutional Neural Network

        **Task:** Binary Image Classification

        **Classes:**
        - 🔴 `def_front`
        - 🟢 `ok_front`

        **Input Size:** 224 × 224 pixels

        **Optimizer:** Adam

        **Loss:** Binary Cross-Entropy

        **Training Epochs:** 10
        """
    )

    st.divider()

    st.metric(
        label="Test Accuracy",
        value="98.32%"
    )

    st.caption(
        "Performance measured on 715 unseen test images."
    )


# ============================================================
# MODEL CHECK
# ============================================================

if model is None:

    st.error(
        "⚠️ Trained model not found."
    )

    st.info(
        f"Please place '{MODEL_PATH}' in the same directory as app.py."
    )

    st.stop()


# ============================================================
# INTRODUCTION
# ============================================================

st.markdown(
    """
    <div class="card">
        <h3>📋 Inspection System</h3>
        <p>
            Upload a front-view casting image below. The trained CNN will
            analyze the image and classify it as either a defective casting
            or an acceptable casting.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📤 Upload a casting image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    help="Upload a casting image for AI-based quality inspection."
)


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_image(image):
    """
    Preprocess an uploaded image and generate a prediction.

    The model was trained using:
    - 224 x 224 image size
    - pixel normalization using 1/255
    - sigmoid output for binary classification
    """

    image = image.convert("RGB")

    resized_image = image.resize(IMAGE_SIZE)

    image_array = np.array(resized_image, dtype=np.float32)

    # Normalize pixels from [0, 255] to [0, 1]
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Generate prediction
    probability = float(
        model.predict(image_array, verbose=0)[0][0]
    )

    # Sigmoid output:
    # probability >= 0.5 -> ok_front
    # probability < 0.5 -> def_front

    if probability >= 0.5:
        predicted_class = "ok_front"
        confidence = probability
    else:
        predicted_class = "def_front"
        confidence = 1 - probability

    return predicted_class, confidence


# ============================================================
# DISPLAY AND PREDICT
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.divider()

    col1, col2 = st.columns([1, 1], gap="large")

    # --------------------------------------------------------
    # IMAGE PREVIEW
    # --------------------------------------------------------

    with col1:

        st.subheader("🖼️ Image Preview")

        st.image(
            image,
            caption="Uploaded casting image",
            width=400
        )

        st.caption(
            f"File: {uploaded_file.name}"
        )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    with col2:

        st.subheader("🤖 AI Inspection Result")

        if st.button(
            "🔍 Analyze Casting",
            type="primary",
            # use_container_width=True
        ):

            with st.spinner("Analyzing image..."):

                predicted_class, confidence = predict_image(image)

            confidence_percentage = confidence * 100

            # ------------------------------------------------
            # DEFECTIVE RESULT
            # ------------------------------------------------

            if predicted_class == 1:

                st.markdown(
                    f"""
                    <div class="prediction-defect">

                        <div class="prediction-title">
                            🔴 DEFECTIVE CASTING
                        </div>

                        <div class="confidence">
                            Confidence: {confidence_percentage:.2f}%
                        </div>

                        <p>
                            The CNN classified this image as a defective casting.
                        </p>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if confidence_percentage >= 80:
                    st.success("High-confidence prediction.")
                elif confidence_percentage >= 60:
                    st.warning("Moderate-confidence prediction. Consider additional inspection.")
                else:
                    st.warning("Low-confidence prediction. Manual inspection is recommended.")

                st.progress(confidence)

            # ------------------------------------------------
            # ACCEPTABLE RESULT
            # ------------------------------------------------

            else:

                st.markdown(
                    f"""
                    <div class="prediction-ok">

                        <div class="prediction-title">
                            🟢 ACCEPTABLE CASTING
                        </div>

                        <div class="confidence">
                            Confidence: {confidence_percentage:.2f}%
                        </div>

                        <p>
                            The CNN classified this image as an
                            acceptable casting.
                        </p>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.success(
                    "✅ Casting passed the AI visual inspection."
                )

            # ------------------------------------------------
            # CONFIDENCE BAR
            # ------------------------------------------------

            st.write("### Prediction Confidence")

            st.progress(
                confidence
            )

            st.caption(
                f"Model confidence: {confidence_percentage:.2f}%"
            )


# ============================================================
# HOW IT WORKS
# ============================================================

st.divider()

st.subheader("🧠 How the AI System Works")

step1, step2, step3, step4 = st.columns(4)

with step1:
    st.markdown(
        """
        **1️⃣ Upload**

        User uploads a casting image through the web interface.
        """
    )

with step2:
    st.markdown(
        """
        **2️⃣ Preprocess**

        Image is resized to 224 × 224 and normalized.
        """
    )

with step3:
    st.markdown(
        """
        **3️⃣ CNN Analysis**

        The trained CNN extracts visual features and generates
        a classification probability.
        """
    )

with step4:
    st.markdown(
        """
        **4️⃣ Result**

        The system displays the predicted class and confidence.
        """
    )
# ============================================================
# DISPLAY PREDICTION RESULT
# ============================================================

confidence_percentage = confidence * 100

st.subheader("🎯 AI Inspection Result")

# ------------------------------------------------------------
# DEFECTIVE CASTING
# ------------------------------------------------------------

if predicted_class == "def_front":

    st.error(
        "🔴 DEFECTIVE CASTING"
    )

    st.metric(
        label="Prediction Confidence",
        value=f"{confidence_percentage:.2f}%"
    )

    if confidence_percentage >= 80:
        st.success(
            "High-confidence prediction."
        )
    elif confidence_percentage >= 60:
        st.warning(
            "Moderate-confidence prediction. "
            "Additional inspection is recommended."
        )
    else:
        st.warning(
            "Low-confidence prediction. "
            "Manual inspection is recommended."
        )

    st.info(
        "The CNN detected visual characteristics associated "
        "with a defective casting."
    )


# ------------------------------------------------------------
# ACCEPTABLE CASTING
# ------------------------------------------------------------

else:

    st.success(
        "🟢 ACCEPTABLE CASTING"
    )

    st.metric(
        label="Prediction Confidence",
        value=f"{confidence_percentage:.2f}%"
    )

    if confidence_percentage >= 80:
        st.success(
            "High-confidence prediction."
        )
    elif confidence_percentage >= 60:
        st.warning(
            "Moderate-confidence prediction. "
            "Additional inspection is recommended."
        )
    else:
        st.warning(
            "Low-confidence prediction. "
            "Manual inspection is recommended."
        )

    st.info(
        "The CNN classified this image as an acceptable casting."
    )


# ------------------------------------------------------------
# CONFIDENCE BAR
# ------------------------------------------------------------

st.write("### 📊 Prediction Confidence")

st.progress(
    confidence
)

st.caption(
    f"Model confidence: {confidence_percentage:.2f}%"
)

# ============================================================
# PERFORMANCE SECTION
# ============================================================

st.divider()

st.subheader("📊 Model Performance")

metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric(
        "Test Accuracy",
        "98.32%"
    )

with metric2:
    st.metric(
        "Test Images",
        "715"
    )

with metric3:
    st.metric(
        "Correct Predictions",
        "703"
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        <p>
            Casting Quality Inspection System |
            CNN-based Image Classification
        </p>
        <p>
            Model trained using TensorFlow / Keras
        </p>
    </div>
    """,
    unsafe_allow_html=True
)