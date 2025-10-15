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

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables
model = None
class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

def load_model():
    """Load the model - called at module import time"""
    global model
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        
        # Download only if file doesn't exist
        if not os.path.exists(MODEL_PATH):
            app.logger.info("Downloading model from Google Drive...")
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        
        # Load the model
        app.logger.info("Loading model...")
        model = tf.keras.models.load_model(MODEL_PATH)
        app.logger.info("Model loaded successfully!")
        
    except Exception as e:
        app.logger.error(f"Error loading model: {e}")
        raise

# Load model when module is imported (works with gunicorn)
load_model()

@app.route('/')
def home():
    return jsonify({"message": "Retinopathy API running"})

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/predict', methods=['POST'])
def predict():
    # Check if model is loaded
    if model is None:
        app.logger.error("Model not loaded")
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        app.logger.info(f"Processing prediction for file: {file.filename}")
        image = Image.open(file.stream)
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        predicted_class_index = int(np.argmax(predictions[0]))
        predicted_class_name = class_names[predicted_class_index]
        confidence = float(predictions[0][predicted_class_index])
        
        app.logger.info(f"Prediction: {predicted_class_name}, Confidence: {confidence}")
        
        return jsonify({
            'predicted_stage': predicted_class_name,
            'confidence': confidence
        })
    except Exception as e:
        app.logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Only for local development
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
