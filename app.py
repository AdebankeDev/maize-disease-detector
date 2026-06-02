import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# ═══════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════
st.set_page_config(
    page_title="🌽 Maize Disease Detector",
    page_icon="🌽",
    layout="wide"
)

st.title("🌽 Maize Disease Detector")
st.write("AI-powered maize disease detection using EfficientNetB0")

st.divider()

# ═══════════════════════════════════════
# CLASS LABELS
# ═══════════════════════════════════════
class_names = ['Blight', 'Common_rust', 'Gray_leaf_spot', 'Healthy']

# ═══════════════════════════════════════
# LOAD MODEL
# ═══════════════════════════════════════
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_model.keras")

model = load_model()

# ═══════════════════════════════════════
# PREPROCESSING (CORRECT + SIMPLE)
# ═══════════════════════════════════════
def preprocess(image):
    image = image.resize((224, 224))
    img = np.array(image).astype(np.float32)

    # EfficientNetB0 correct scaling
    img = (img / 127.5) - 1.0

    img = np.expand_dims(img, axis=0)
    return img

# ═══════════════════════════════════════
# PREDICTION
# ═══════════════════════════════════════
def predict(image):
    img = preprocess(image)
    preds = model.predict(img, verbose=0)
    return preds[0]

# ═══════════════════════════════════════
# UI
# ═══════════════════════════════════════
uploaded_file = st.file_uploader(
    "Upload maize leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, use_container_width=True)

    with col2:

        if st.button("🔍 Predict"):

            predictions = predict(image)

            idx = np.argmax(predictions)
            disease = class_names[idx]
            confidence = float(predictions[idx])

            if disease == "Healthy":
                st.success(f"🌱 {disease}")
            else:
                st.error(f"⚠️ {disease}")

            st.metric("Confidence", f"{confidence*100:.2f}%")

            st.divider()

            st.subheader("Class Probabilities")

            for i, name in enumerate(class_names):
                st.write(f"{name}: {predictions[i]*100:.2f}%")

            st.bar_chart({
                name: float(prob)
                for name, prob in zip(class_names, predictions)
            })

else:
    st.info("Upload an image to start prediction")