import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models


def build_model():
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3)
    )

    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(4, activation='softmax')
    ])

    return model

# ═══════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════
st.set_page_config(
    page_title="🌽 Maize Disease Detection System",
    page_icon="🌽",
    layout="wide"
)


# ═══════════════════════════════════════════════
# LOAD MODEL
# ═══════════════════════════════════════════════
@st.cache_resource
def load_model():
    model = build_model()
    model.load_weights("model_weights.weights.h5")
    return model

model = load_model()

# ═══════════════════════════════════════════════
# PREDICTION FUNCTION (FIXED)
# ═══════════════════════════════════════════════
def predict(image):
    image = image.resize((224, 224))

    img_array = np.array(image).astype(np.float32)

    # EfficientNet preprocessing (safe TF version)
    img_array = preprocess_input(img_array)

    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)

    return predictions[0]

# ═══════════════════════════════════════════════

# ═══════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════
st.title("🌽 Maize Disease Detection System")

st.caption(
    "AI-powered maize leaf disease identification using EfficientNetB0 and Deep Learning."
)

st.info(
    "📷 Upload a clear image of a maize leaf to receive an instant disease diagnosis and confidence score."
)

st.divider()

# ═══════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════
with st.sidebar:

    st.title("🌽 Project Overview")

    st.success("AI-Powered Crop Disease Detection")

    st.markdown("""
    ### About This Project

    Maize is one of the world's most important food crops, but its productivity can be severely affected by leaf diseases.

    This project uses Artificial Intelligence, Computer Vision, and Deep Learning to automatically detect maize leaf diseases from uploaded images.

    The model is based on EfficientNetB0, a state-of-the-art convolutional neural network trained through transfer learning.

    ### Objectives

    - Early disease identification
    - Faster crop monitoring
    - Reduced crop losses
    - Support for farmers and agricultural researchers

    ### Detectable Classes

    ✅ Healthy

    ⚠️ Blight

    ⚠️ Common Rust

    ⚠️ Gray Leaf Spot
    """)

    st.divider()

    st.subheader("🎯 Model Information")

    st.info("""
Architecture: EfficientNetB0

Framework: TensorFlow / Keras

Input Size: 224 × 224

Classes: 4

Technique: Transfer Learning
    """)

    st.divider()

    with st.expander("🦠 Supported Diseases"):

        st.markdown("""
**Blight**
- Brown lesions
- Yellow halo
- Rapid spread

**Common Rust**
- Orange powdery pustules
- Yellow spotting

**Gray Leaf Spot**
- Gray rectangular lesions
- Leaf drying

**Healthy**
- No visible disease symptoms
        """)

    st.divider()

    st.caption(
        "Developed as a Machine Learning project for maize disease classification."
    )

# ═══════════════════════════════════════════════
# FILE UPLOAD
# ═══════════════════════════════════════════════

uploaded_file = st.file_uploader(
    "📷 Select Leaf Image",
    type=["jpg", "jpeg", "png"]
    help="Supported formats: JPG, JPEG, PNG"
)

# ═══════════════════════════════════════════════
# MAIN UI
# ═══════════════════════════════════════════════
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:

        st.subheader("📷 Uploaded Image")

        try:
            st.image(image, use_container_width=True)
        except TypeError:
            st.image(image, use_column_width=True)

    with col2:

        st.subheader("🔍 Disease Prediction")

        st.write(
            "Click the button below to analyze the uploaded maize leaf."
        )

        if st.button(
            "🚀 Analyze Image",
            use_container_width=True
        ):

            predictions = predict(image)

            idx = np.argmax(predictions)
            disease = class_names[idx]
            confidence = float(predictions[idx])

            info = DISEASE_INFO[disease]

            result_col1, result_col2 = st.columns(2)

            with result_col1:

                if disease == "Healthy":
                    st.success("🌱 Healthy Leaf")
                else:
                    st.error(f"⚠️ {disease}")

            with result_col2:

                st.metric(
                    "Confidence",
                    f"{confidence * 100:.2f}%"
                )

            with st.expander(
                "📋 Disease Details",
                expanded=True
            ):

                st.write(
                    f"**Severity Level:** {info['severity']}"
                )

                st.write(
                    f"**Description:** {info['description']}"
                )

                st.write("### Symptoms")

                for symptom in info["symptoms"]:
                    st.write(f"• {symptom}")

                st.write("### Recommended Treatment")

                for treatment in info["treatment"]:
                    st.write(f"• {treatment}")

            st.session_state["predictions"] = predictions

    st.divider()

    st.subheader("📊 Prediction Analysis")

    if "predictions" in st.session_state:

        predictions = st.session_state["predictions"]

        tab1, tab2 = st.tabs(
            [
                "📈 Confidence Scores",
                "📊 Probability Chart"
            ]
        )

        with tab1:

            metric_col1, metric_col2 = st.columns(2)

            with metric_col1:

                st.metric(
                    class_names[0],
                    f"{predictions[0] * 100:.2f}%"
                )

                st.metric(
                    class_names[1],
                    f"{predictions[1] * 100:.2f}%"
                )

            with metric_col2:

                st.metric(
                    class_names[2],
                    f"{predictions[2] * 100:.2f}%"
                )

                st.metric(
                    class_names[3],
                    f"{predictions[3] * 100:.2f}%"
                )

        with tab2:

            chart_data = {
                name: float(prob)
                for name, prob in zip(
                    class_names,
                    predictions
                )
            }

            st.bar_chart(chart_data)

else:

    st.info(
        "👆 Upload a maize leaf image to begin disease detection."
    )

# ═══════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════
st.divider()

st.caption(
    "🌽 Maize Disease Detection System | Built with Streamlit, TensorFlow and EfficientNetB0"
)
