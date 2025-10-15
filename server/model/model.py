from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import traceback

# Path to TFLite model file
MODEL_PATH = "./my_model.tflite"

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables
interpreter = None
input_details = None
output_details = None
class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

def load_tflite_model():
    """Load TFLite model lazily on first request (fork-safe)"""
    global interpreter, input_details, output_details
    
    if interpreter is None:
        try:
            app.logger.info("Loading TFLite model in worker process...")
            
            # Debug: Check if model file exists
            if not os.path.exists(MODEL_PATH):
                app.logger.error(f"Model file not found at {MODEL_PATH}")
                app.logger.error(f"Current directory: {os.getcwd()}")
                app.logger.error(f"Files in current dir: {os.listdir('.')}")
                if os.path.exists('server'):
                    app.logger.error(f"Files in server/: {os.listdir('server')}")
                raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
            
            # Load TFLite model
            interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
            interpreter.allocate_tensors()
            
            # Get input and output details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            app.logger.info("TFLite model loaded successfully!")
            app.logger.info(f"Input details: {input_details}")
            app.logger.info(f"Output details: {output_details}")
            
        except Exception as e:
            app.logger.error(f"Error loading TFLite model: {e}")
            app.logger.error(traceback.format_exc())
            raise
    
    return interpreter, input_details, output_details

@app.route('/')
def home():
    return jsonify({"message": "Retinopathy API running with TFLite"})

@app.route('/health')
def health():
    """Health check endpoint"""
    try:
        model_status = "loaded" if interpreter is not None else "not loaded"
        model_exists = os.path.exists(MODEL_PATH)
        return jsonify({
            "status": "healthy",
            "model_status": model_status,
            "model_file_exists": model_exists,
            "model_type": "TFLite"
        }), 200
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

def preprocess_image(image, input_shape):
    """Preprocess image for TFLite model prediction"""
    # Get expected input size from model
    height = input_shape[1]
    width = input_shape[2]
    
    # Resize image
    image = image.resize((width, height))
    
    # Convert to numpy array and normalize
    image = np.array(image, dtype=np.float32) / 255.0
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    return image

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint using TFLite"""
    try:
        app.logger.info("=== New prediction request ===")
        
        if 'file' not in request.files:
            app.logger.error("No file part in request")
            return jsonify({'error': 'No file part in request'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            app.logger.error("Empty filename")
            return jsonify({'error': 'No selected file'}), 400
        
        # Load TFLite model lazily
        app.logger.info("Loading TFLite model...")
        current_interpreter, current_input_details, current_output_details = load_tflite_model()
        app.logger.info("TFLite model loaded")
        
        app.logger.info(f"Processing prediction for file: {file.filename}")
        
        # Read the image from stream
        image = Image.open(file.stream)
        app.logger.info(f"Image opened: size={image.size}, mode={image.mode}")
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            app.logger.info(f"Converting from {image.mode} to RGB")
            image = image.convert('RGB')
        
        # Preprocess image based on model's expected input shape
        input_shape = current_input_details[0]['shape']
        processed_image = preprocess_image(image, input_shape)
        app.logger.info(f"Image preprocessed: shape={processed_image.shape}, dtype={processed_image.dtype}")
        
        # Set input tensor
        current_interpreter.set_tensor(current_input_details[0]['index'], processed_image)
        
        # Run inference
        app.logger.info("Running TFLite inference...")
        current_interpreter.invoke()
        
        # Get output tensor
        output_data = current_interpreter.get_tensor(current_output_details[0]['index'])
        app.logger.info(f"Inference complete: output shape={output_data.shape}")
        
        # Process predictions
        predictions = output_data[0]
        predicted_class_index = int(np.argmax(predictions))
        predicted_class_name = class_names[predicted_class_index]
        confidence = float(predictions[predicted_class_index])
        
        app.logger.info(f"Prediction: {predicted_class_name}, Confidence: {confidence:.4f}")
        
        return jsonify({
            'predicted_stage': predicted_class_name,
            'confidence': confidence,
            'all_probabilities': {
                class_names[i]: float(predictions[i]) 
                for i in range(len(class_names))
            }
        })
        
    except Exception as e:
        app.logger.error(f"=== ERROR in /predict ===")
        app.logger.error(f"Error type: {type(e).__name__}")
        app.logger.error(f"Error message: {str(e)}")
        app.logger.error(f"Full traceback:\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

# Only for local development
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
