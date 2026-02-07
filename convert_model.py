import tensorflow as tf
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model_traffic_signs.h5')
TFLITE_PATH = os.path.join(BASE_DIR, 'model_traffic_signs.tflite')

# Load your existing model
model = tf.keras.models.load_model(MODEL_PATH)

# Convert to TFLite with optimization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Optional: Use float16 quantization for even smaller size
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

# Save the optimized model
with open(TFLITE_PATH, 'wb') as f:
    f.write(tflite_model)

print(f"✓ Model converted successfully!")
print(f"Original .h5 size: {os.path.getsize(MODEL_PATH) / 1024 / 1024:.2f} MB")
print(f"TFLite size: {os.path.getsize(TFLITE_PATH) / 1024 / 1024:.2f} MB")