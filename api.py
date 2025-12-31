#!/usr/bin/env python3
"""
Resume Classifier - Flask API

Simple REST API for resume classification.

Endpoints:
- POST /predict - Classify a single resume
- POST /predict_batch - Classify multiple resumes
- GET /info - Get model information
- GET /health - Health check
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, request, jsonify

from src.config import APIConfig, DEFAULT_API_CONFIG
from src.model import ResumePredictor

# Initialize Flask app
app = Flask(__name__)

# Configuration
api_config = DEFAULT_API_CONFIG

# Load model predictor
print("Loading model...")
predictor = ResumePredictor(model_dir=api_config.model_dir)
print("Model loaded successfully!")


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'message': 'API is running'})


@app.route('/info', methods=['GET'])
def info():
    """Get model information."""
    model_info = predictor.get_model_info()
    return jsonify(model_info)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict category for a single resume with hierarchical output.

    Request body:
    {
        "resume_text": "Your resume text here...",
        "return_details": true/false (optional, default: false)
    }

    Response (return_details=false):
    {
        "predicted_category": "Python Developer",  // or parent if low confidence
        "confidence": 0.85,
        "tier": "high",  // "high", "medium", or "low"
        "requires_review": false,
        "hierarchy_path": ["Technology", "Software Development", "Python Developer"]
    }
    
    Response (return_details=true):
    Includes additional fields: specific_category, top_5_predictions, etc.
    """
    try:
        data = request.get_json()

        if not data or 'resume_text' not in data:
            return jsonify({'error': 'Missing resume_text in request body'}), 400

        resume_text = data['resume_text']
        return_details = data.get('return_details', False)

        result = predictor.predict(resume_text, return_details=return_details)

        if return_details:
            # Already a dict
            return jsonify(result)
        else:
            # HierarchicalPrediction object - convert to dict
            return jsonify({
                'predicted_category': result.category,
                'confidence': result.confidence,
                'tier': result.tier,
                'requires_review': result.requires_review,
                'hierarchy_path': result.path,
                'specific_category': result.specific_category
            })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Predict categories for multiple resumes with hierarchical output.

    Request body:
    {
        "resumes": ["resume 1 text...", "resume 2 text...", ...]
    }

    Response:
    {
        "predictions": [
            {
                "predicted_category": "Python Developer",
                "confidence": 0.85,
                "tier": "high",
                "requires_review": false,
                "hierarchy_path": ["Technology", "Software Development", "Python Developer"]
            },
            ...
        ]
    }
    """
    try:
        data = request.get_json()

        if not data or 'resumes' not in data:
            return jsonify({'error': 'Missing resumes in request body'}), 400

        resumes = data['resumes']

        if not isinstance(resumes, list):
            return jsonify({'error': 'resumes must be a list'}), 400

        results = predictor.predict_batch(resumes)
        
        # Convert HierarchicalPrediction objects to dicts
        predictions = [
            {
                'predicted_category': r.category,
                'confidence': r.confidence,
                'tier': r.tier,
                'requires_review': r.requires_review,
                'hierarchy_path': r.path,
                'specific_category': r.specific_category
            }
            for r in results
        ]

        return jsonify({'predictions': predictions})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def run_server(config: APIConfig = None):
    """Run the Flask server with the given configuration."""
    config = config or api_config
    
    print("\n" + "="*70)
    print("RESUME CLASSIFIER API")
    print("="*70)
    print("\nAPI Endpoints:")
    print("  GET  /health          - Health check")
    print("  GET  /info            - Model information")
    print("  POST /predict         - Classify single resume")
    print("  POST /predict_batch   - Classify multiple resumes")
    print("\n" + "="*70)
    print(f"Starting server on http://{config.host}:{config.port}")
    print("="*70 + "\n")

    app.run(host=config.host, port=config.port, debug=config.debug)


if __name__ == '__main__':
    run_server()
