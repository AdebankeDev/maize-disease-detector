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
    page_title="🌽 Maize Disease Detector",
    page_icon="🌽",
    layout="wide"
)

# ═══════════════════════════════════════════════
# DISEASE INFO
# ═══════════════════════════════════════════════
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

class_names = ['Blight', 'Common_rust', 'Gray_leaf_spot', 'Healthy']

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
# HEADER
# ═══════════════════════════════════════════════
st.title("🌽 Maize Disease Detector")
st.write("AI-powered detection using EfficientNetB0")

st.divider()

# ═══════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════
with st.sidebar:
    st.header("📋 About")
    st.write("""
    Detect maize leaf diseases using deep learning.
    
    - Model: EfficientNetB0
    - Format: TFLite
    - Input: 224×224 images
    """)

    st.divider()

    st.header("Diseases")
    st.write("""
    - Blight  
    - Common Rust  
    - Gray Leaf Spot  
    - Healthy  
    """)

    st.divider()
    st.warning("For educational use only.")

# ═══════════════════════════════════════════════
# UPLOAD
# ═══════════════════════════════════════════════
uploaded_file = st.file_uploader(
    "Upload maize leaf image",
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
        import streamlit as st
        try:
            st.image(image, use_container_width=True)
        except TypeError:
            st.image(image, use_column_width=True)

    with col2:
        st.subheader("📊 Prediction")

        predictions = predict(image)

        idx = np.argmax(predictions)
        disease = class_names[idx]
        confidence = float(predictions[idx])

        info = DISEASE_INFO[disease]

        if st.button("🔍 Predict"):
            if disease == "Healthy":
                st.success(f"🌱 {disease}")
            else:
                st.error(f"⚠️ {disease}")

            st.metric("Confidence", f"{confidence*100:.2f}%")

            st.write(f"**Severity:** {info['severity']}")
            st.write(info['description'])

    st.divider()

    # ═══════════════════════════════════════════════
    # FULL BREAKDOWN
    # ═══════════════════════════════════════════════
    st.subheader("📊 Class Probabilities")

    col3, col4 = st.columns(2)

    with col3:
        for i, name in enumerate(class_names):
            st.write(f"{name}: {predictions[i]*100:.2f}%")

    with col4:
        st.bar_chart({
            name: float(prob)
            for name, prob in zip(class_names, predictions)
        })

else:
    st.info("Upload an image to start prediction.")