import React, { useState } from 'react';
import axios from 'axios';
import './classify.css'; // We'll use this for styling

function RetinopathyClassifier() {
    const [selectedFile, setSelectedFile] = useState(null);
    const [preview, setPreview] = useState(null);
    const [prediction, setPrediction] = useState(null);
    const [error, setError] = useState(null);
    const [loading, setLoading] = useState(false);

    const handleFileChange = (event) => {
        const file = event.target.files[0];
        setSelectedFile(file);
        setPrediction(null);
        setError(null);

        // Create preview URL
        if (file) {
            const reader = new FileReader();
            reader.onloadend = () => {
                setPreview(reader.result);
            };
            reader.readAsDataURL(file);
        } else {
            setPreview(null);
        }
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
                'https://retinascan-backend.onrender.com/predict',
                formData,
                { headers: { 'Content-Type': 'multipart/form-data' } }
            );

            setPrediction(response.data);
        } catch (err) {
            setError('An error occurred during prediction.');
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="classifier-container">
            <h2>Diabetic Retinopathy Classifier</h2>

            <input type="file" onChange={handleFileChange} className="file-input" />

            {preview && (
                <div className="image-preview">
                    <h4>Uploaded Image:</h4>
                    <img src={preview} alt="Uploaded Preview" />
                </div>
            )}

            <button onClick={handleUpload} className="upload-button" disabled={loading}>
                {loading ? 'Processing...' : 'Classify'}
            </button>

            {loading && (
                <div className="spinner-container">
                    <div className="spinner"></div>
                </div>
            )}

            {error && <p className="error-text">{error}</p>}

            {prediction && (
                <div className="prediction-result">
                    <h3>Prediction Result:</h3>
                    <p><strong>Stage:</strong> {prediction.predicted_stage}</p>
                    <p><strong>Confidence:</strong> {(prediction.confidence * 100).toFixed(2)}%</p>
                </div>
            )}
        </div>
    );
}

export default RetinopathyClassifier;
