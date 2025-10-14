from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
import numpy as np

import gdown
import os

MODEL_PATH = "server/my_model.keras"
MODEL_ID = "1a6Q9dg3KKuw0QCe7cbRVon7-oyxnIpx9"  # Your Google Drive file ID
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_ID}"

# Download only if it doesn't exist
if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)





app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

@app.route('/')
def home():
    return jsonify({"message": "Retinopathy API running"})

# Load model
model = tf.keras.models.load_model(MODEL_PATH)
class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    image = Image.open(file.stream)
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class_index = int(np.argmax(predictions[0]))
    predicted_class_name = class_names[predicted_class_index]
    confidence = float(predictions[0][predicted_class_index])
    return jsonify({
        'predicted_stage': predicted_class_name,
        'confidence': confidence
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
