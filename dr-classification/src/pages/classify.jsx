import React, { useState } from 'react';
import axios from 'axios';
import './classify.css';

function RetinopathyClassifier() {
    const [selectedFile, setSelectedFile] = useState(null);
    const [preview, setPreview] = useState(null);
    const [prediction, setPrediction] = useState(null);
    const [error, setError] = useState(null);
    const [loading, setLoading] = useState(false);

    // File validation constants
    const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
    const ALLOWED_TYPES = ['image/jpeg', 'image/jpg', 'image/png'];

    const handleFileChange = (event) => {
        const file = event.target.files[0];
        
        // Reset states
        setPrediction(null);
        setError(null);
        setPreview(null);
        setSelectedFile(null);

        if (!file) {
            return;
        }

        // Validate file type
        if (!ALLOWED_TYPES.includes(file.type)) {
            setError('Please select a valid image file (JPEG or PNG only).');
            return;
        }

        // Validate file size
        if (file.size > MAX_FILE_SIZE) {
            setError(`File size must be less than ${MAX_FILE_SIZE / (1024 * 1024)}MB. Your file is ${(file.size / (1024 * 1024)).toFixed(2)}MB.`);
            return;
        }

        setSelectedFile(file);

        // Create preview URL
        const reader = new FileReader();
        reader.onloadend = () => {
            setPreview(reader.result);
        };
        reader.onerror = () => {
            setError('Failed to read file.');
        };
        reader.readAsDataURL(file);
    };

    const handleUpload = async () => {
        if (!selectedFile) {
            setError('Please select an image first.');
            return;
        }

        const formData = new FormData();
        formData.append('file', selectedFile);

        setLoading(true);
        setError(null);
        setPrediction(null);

        try {
            const response = await axios.post(
                'https://dr-classification-react-js-7.onrender.com/predict',
                formData,
                {
                    headers: {
                        'Content-Type': 'multipart/form-data'
                    },
                    timeout: 120000, // 120 seconds timeout
                }
            );

            setPrediction(response.data);
            console.log('Prediction successful:', response.data);
            
        } catch (err) {
            console.error('Error details:', err);
            
            // Handle different types of errors
            if (err.response) {
                // Server responded with error status
                const backendError = err.response.data?.error || 'Server error occurred';
                setError(`Error: ${backendError} (Status: ${err.response.status})`);
                console.error('Backend error:', err.response.data);
            } else if (err.request) {
                // Request was made but no response received
                setError('No response from server. Please check if the backend is running and try again.');
                console.error('No response:', err.request);
            } else {
                // Something else happened
                setError(`Request error: ${err.message}`);
                console.error('Error:', err.message);
            }
        } finally {
            setLoading(false);
        }
    };

    const handleReset = () => {
        setSelectedFile(null);
        setPreview(null);
        setPrediction(null);
        setError(null);
        // Reset file input
        const fileInput = document.querySelector('.file-input');
        if (fileInput) fileInput.value = '';
    };

    const getSeverityColor = (stage) => {
        const colors = {
            'No DR': '#4CAF50',
            'Mild': '#8BC34A',
            'Moderate': '#FFC107',
            'Severe': '#FF9800',
            'Proliferative DR': '#F44336'
        };
        return colors[stage] || '#2196F3';
    };

    return (
        <div className="classifier-container">
            <h2>Diabetic Retinopathy Classifier</h2>
            <p className="subtitle">Upload a retinal image to detect diabetic retinopathy</p>

            <div className="upload-section">
                <input 
                    type="file" 
                    onChange={handleFileChange} 
                    className="file-input"
                    accept="image/jpeg,image/jpg,image/png"
                    disabled={loading}
                />
                <p className="file-info">Accepted formats: JPEG, PNG (Max size: 10MB)</p>
            </div>

            {preview && (
                <div className="image-preview">
                    <h4>Uploaded Image:</h4>
                    <img src={preview} alt="Uploaded Preview" />
                    <p className="image-name">{selectedFile?.name}</p>
                </div>
            )}

            <div className="button-group">
                <button 
                    onClick={handleUpload} 
                    className="upload-button" 
                    disabled={loading || !selectedFile}
                >
                    {loading ? 'Processing...' : 'Classify Image'}
                </button>

                {(selectedFile || prediction) && (
                    <button 
                        onClick={handleReset} 
                        className="reset-button"
                        disabled={loading}
                    >
                        Reset
                    </button>
                )}
            </div>

            {loading && (
                <div className="spinner-container">
                    <div className="spinner"></div>
                    <p>Analyzing image... This may take up to 60 seconds.</p>
                </div>
            )}

            {error && (
                <div className="error-container">
                    <p className="error-text">⚠️ {error}</p>
                </div>
            )}

            {prediction && (
                <div className="prediction-result">
                    <h3>Prediction Result:</h3>
                    <div 
                        className="result-stage"
                        style={{ 
                            borderLeft: `5px solid ${getSeverityColor(prediction.predicted_stage)}`
                        }}
                    >
                        <p><strong>Stage:</strong> {prediction.predicted_stage}</p>
                        <p><strong>Confidence:</strong> {(prediction.confidence * 100).toFixed(2)}%</p>
                    </div>

                    {prediction.all_probabilities && (
                        <div className="all-probabilities">
                            <h4>All Probabilities:</h4>
                            <div className="probability-bars">
                                {Object.entries(prediction.all_probabilities).map(([stage, prob]) => (
                                    <div key={stage} className="probability-item">
                                        <span className="stage-name">{stage}</span>
                                        <div className="progress-bar">
                                            <div 
                                                className="progress-fill"
                                                style={{ 
                                                    width: `${prob * 100}%`,
                                                    backgroundColor: getSeverityColor(stage)
                                                }}
                                            />
                                        </div>
                                        <span className="probability-value">
                                            {(prob * 100).toFixed(1)}%
                                        </span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    <div className="disclaimer">
                        <p><em>⚕️ This is for educational purposes only. Please consult a medical professional for diagnosis.</em></p>
                    </div>
                </div>
            )}
        </div>
    );
}

export default RetinopathyClassifier;
