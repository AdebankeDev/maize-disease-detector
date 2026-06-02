import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# -----------------------------
# App Title
# -----------------------------
st.title("🌽 Maize Disease Detector")
st.write("Upload a maize leaf image to detect the disease.")

# -----------------------------
# Load TFLite model
# -----------------------------
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# -----------------------------
# Class labels
# -----------------------------
class_names = ['Blight', 'Common_rust', 'Gray_leaf_spot', 'Healthy']

# -----------------------------
# Prediction function
# -----------------------------
def predict(image):
    image = image.resize((224, 224))
    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])[0]
    return output

# -----------------------------
# Upload section
# -----------------------------
uploaded_file = st.file_uploader("Choose an image (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Prediction
    predictions = predict(image)

    idx = np.argmax(predictions)
    confidence = predictions[idx]
    disease = class_names[idx]

    # -----------------------------
    # Results
    # -----------------------------
    st.success(f"🌿 Predicted Disease: **{disease}**")
    st.info(f"Confidence: **{confidence * 100:.2f}%**")

    st.write("### 📊 All Predictions")
    for i, name in enumerate(class_names):
        st.write(f"- {name}: {predictions[i] * 100:.2f}%")