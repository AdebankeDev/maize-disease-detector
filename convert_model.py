import tensorflow as tf

# Load trained Keras model
model = tf.keras.models.load_model("best_model.keras")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optional optimization (safe for deployment)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

# Save the model
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Conversion successful: model.tflite created")