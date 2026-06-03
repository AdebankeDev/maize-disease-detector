import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models


# ═══════════════════════════════════════════════
# BUILD MODEL
# ═══════════════════════════════════════════════
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
# CLASS LABELS + DISEASE INFO
# ═══════════════════════════════════════════════
class_names = ['Blight', 'Common_rust', 'Gray_leaf_spot', 'Healthy']

DISEASE_INFO = {
    'Blight': {
        'severity': 'HIGH',
        'description': 'Fungal disease causing large brown lesions on leaves.',
        'symptoms': ['Brown spots', 'Yellow halo', 'Rapid spread'],
        'treatment': ['Apply fungicide', 'Remove infected leaves', 'Improve airflow']
    },
    'Common_rust': {
        'severity': 'MEDIUM',
        'description': 'Rust fungus causing orange powdery pustules.',
        'symptoms': ['Orange pustules', 'Yellow spots', 'Leaf damage'],
        'treatment': ['Use fungicide', 'Remove infected leaves']
    },
    'Gray_leaf_spot': {
        'severity': 'MEDIUM',
        'description': 'Gray rectangular lesions on leaves.',
        'symptoms': ['Gray patches', 'Dark borders', 'Leaf drying'],
        'treatment': ['Apply fungicide', 'Reduce humidity']
    },
    'Healthy': {
        'severity': 'NONE',
        'description': 'No disease detected.',
        'symptoms': ['Green leaves', 'Normal growth'],
        'treatment': ['Maintain care routine']
    }
}


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
# PREDICTION FUNCTION
# ═══════════════════════════════════════════════
def predict(image):
    image = image.resize((224, 224))

    img_array = np.array(image).astype(np.float32)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)

    return predictions[0]


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

This system uses **Deep Learning (EfficientNetB0)** to classify maize leaf diseases.

It helps in early detection of crop diseases to improve agricultural productivity and reduce losses.

### Objectives
- Early disease detection
- Improve crop yield
- Support farmers

### Classes
- Healthy
- Blight
- Common Rust
- Gray Leaf Spot
""")

    st.divider()

    st.subheader("🎯 Model Info")
    st.info("""
EfficientNetB0

Transfer Learning

Input: 224×224

4 Classes
""")

    st.divider()

    st.caption("Built for academic ML research project.")


# ═══════════════════════════════════════════════
# UPLOAD SECTION
# ═══════════════════════════════════════════════
uploaded_file = st.file_uploader(
    "",
    type=["jpg", "jpeg", "png"]
)


# ═══════════════════════════════════════════════
# MAIN UI
# ═══════════════════════════════════════════════
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Image")
        try:
            st.image(image, use_container_width=True)
        except TypeError:
            st.image(image, use_column_width=True)

    with col2:
        st.subheader("🔍 Prediction")

        if st.button("🚀 Analyze Image", use_container_width=True):

            predictions = predict(image)

            idx = np.argmax(predictions)
            disease = class_names[idx]
            confidence = float(predictions[idx])

            info = DISEASE_INFO[disease]

            col_a, col_b = st.columns(2)

            with col_a:
                if disease == "Healthy":
                    st.success("🌱 Healthy Leaf")
                else:
                    st.error(f"⚠️ {disease}")

            with col_b:
                st.metric("Confidence", f"{confidence * 100:.2f}%")

            with st.expander("📋 Disease Details", expanded=True):

                st.write(f"**Severity:** {info['severity']}")
                st.write(info["description"])

                st.write("### Symptoms")
                for s in info["symptoms"]:
                    st.write(f"• {s}")

                st.write("### Treatment")
                for t in info["treatment"]:
                    st.write(f"• {t}")

            st.session_state["predictions"] = predictions

    st.divider()

    st.subheader("📊 Prediction Analysis")

    if "predictions" in st.session_state:

        predictions = st.session_state["predictions"]

        tab1, tab2 = st.tabs(["📈 Scores", "📊 Chart"])

        with tab1:
            for i, name in enumerate(class_names):
                st.metric(name, f"{predictions[i] * 100:.2f}%")

        with tab2:
            st.bar_chart({
                name: float(prob)
                for name, prob in zip(class_names, predictions)
            })

else:
    st.info("👆 Upload a maize leaf image to begin detection.")


# ═══════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════
st.divider()

st.caption(
    "🌽 Maize Disease Detection System | Built with Streamlit + TensorFlow + EfficientNetB0"
)
