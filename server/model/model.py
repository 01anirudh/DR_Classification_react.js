from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Path to local model file
MODEL_PATH = "server/my_model.keras"

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables
model = None
class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

def load_model():
    """Load the model lazily on first request (fork-safe)"""
    global model
    
    if model is None:
        try:
            app.logger.info("Loading model in worker process...")
            
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
            
            model = tf.keras.models.load_model(MODEL_PATH)
            app.logger.info("Model loaded successfully!")
        except Exception as e:
            app.logger.error(f"Error loading model: {e}")
            raise
    
    return model

@app.route('/')
def home():
    return jsonify({"message": "Retinopathy API running"})

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy"}), 200

def preprocess_image(image):
    """Preprocess image for model prediction"""
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image.astype(np.float32)

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Load model lazily (only in the worker that handles request)
        current_model = load_model()
        
        app.logger.info(f"Processing prediction for file: {file.filename}")
        
        # Read the image from stream
        image = Image.open(file.stream)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        processed_image = preprocess_image(image)
        
        # Run prediction
        predictions = current_model.predict(processed_image, verbose=0)
        predicted_class_index = int(np.argmax(predictions[0]))
        predicted_class_name = class_names[predicted_class_index]
        confidence = float(predictions[0][predicted_class_index])
        
        app.logger.info(f"Prediction: {predicted_class_name}, Confidence: {confidence:.2f}")
        
        return jsonify({
            'predicted_stage': predicted_class_name,
            'confidence': confidence,
            'all_probabilities': {
                class_names[i]: float(predictions[0][i]) 
                for i in range(len(class_names))
            }
        })
        
    except Exception as e:
        app.logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

# Only for local development
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
