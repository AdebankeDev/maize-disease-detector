import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

from tensorflow.keras.applications.efficientnet import preprocess_input

# -----------------------------
# Page config (simple polish)
# -----------------------------
st.set_page_config(
    page_title="Maize Disease Detector",
    page_icon="🌽",
    layout="centered"
)

# -----------------------------
# Header
# -----------------------------
st.title("🌽 Maize Disease Detector")
st.caption("AI-powered plant disease detection using EfficientNetB0")

st.divider()

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_model.keras")

model = load_model()

# -----------------------------
# Classes
# -----------------------------
class_names = ['Blight', 'Common_rust', 'Gray_leaf_spot', 'Healthy']

# -----------------------------
# Upload section
# -----------------------------
st.subheader("📤 Upload Leaf Image")

uploaded_file = st.file_uploader(
    "Choose an image (JPG, JPEG, PNG)",
    type=["jpg", "jpeg", "png"]
)

submit = st.button("🔍 Predict Disease")

# -----------------------------
# Preprocessing
# -----------------------------
def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = np.array(image).astype(np.float32)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -----------------------------
# Prediction
# -----------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if submit:
        with st.spinner("Analyzing leaf... 🌿"):
            processed = preprocess_image(image)
            predictions = model.predict(processed, verbose=0)

            index = np.argmax(predictions[0])
            disease = class_names[index]
            confidence = float(np.max(predictions[0]))

        st.divider()

        # -----------------------------
        # Result Section (clean UI)
        # -----------------------------
        if disease == "Healthy":
            st.success(f"🌱 Result: {disease}")
        else:
            st.error(f"⚠️ Disease Detected: {disease}")

        st.metric(label="Confidence", value=f"{confidence*100:.2f}%")

        # -----------------------------
        # Probability breakdown
        # -----------------------------
        st.subheader("📊 Prediction Breakdown")

        for i, name in enumerate(class_names):
            st.progress(float(predictions[0][i]), text=f"{name}: {predictions[0][i]*100:.2f}%")

        # -----------------------------
        # Low confidence warning
        # -----------------------------
        if confidence < 0.6:
            st.warning("⚠️ Low confidence — result may be unreliable")