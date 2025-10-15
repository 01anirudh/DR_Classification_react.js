import tensorflow as tf
import os

# Load your existing Keras model
model = tf.keras.models.load_model('../my_model.keras')

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optional: Optimize further with quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

# Create the directory if it doesn't exist
output_path = 'server/my_model.tflite'
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Save the TFLite model
with open(output_path, 'wb') as f:
    f.write(tflite_model)

print(f"TFLite model saved to {output_path}")
