# 🌽 Maize Disease Detection System

An AI-powered web application that detects maize leaf diseases using a deep learning model built with **EfficientNetB0** and deployed using Streamlit.

---

## 🚀 Live Demo
https://maize-disease-detector-v1.streamlit.app/

---

## 📌 Features
- Upload maize leaf images
- Predict disease type instantly
- Displays confidence score
- Simple and interactive UI

---

## 🧠 Model Details
- Architecture: EfficientNetB0 (Transfer Learning)
- Input Size: 224 × 224
- Classes:
  - Blight
  - Common Rust
  - Gray Leaf Spot
  - Healthy
- Test Accuracy: ~91%

---

## ⚙️ Preprocessing

Images are processed using:

```python
from tensorflow.keras.applications.efficientnet import preprocess_input
