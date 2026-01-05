"""
Resume Classifier API Server

Core Flask application separated from CLI interface.
Provides REST API for resume classification predictions.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, request, jsonify

from src.config import DEFAULT_API_CONFIG
from src.model import ResumePredictor


def create_app(model_dir: str = "models"):
    """
    Create and configure Flask application.

    Args:
        model_dir: Directory containing trained model files

    Returns:
        Configured Flask app instance
    """
    app = Flask(__name__)

    # Load model predictor
    print(f"Loading model from {model_dir}...")
    predictor = ResumePredictor(model_dir=model_dir)
    print("Model loaded successfully!")

    @app.route("/health", methods=["GET"])
    def health():
        """Health check endpoint."""
        return jsonify({"status": "healthy", "message": "API is running"})

    @app.route("/info", methods=["GET"])
    def info():
        """Get model information."""
        model_info = predictor.get_model_info()
        return jsonify(model_info)

    @app.route("/predict", methods=["POST"])
    def predict():
        """
        Predict category for a single resume.

        Request:
            {
                "resume_text": "Your resume text...",
                "return_details": true/false (optional)
            }

        Response:
            {
                "predicted_category": "Python Developer",
                "confidence": 0.85,
                "tier": "high",
                "requires_review": false,
                "hierarchy_path": ["Technology", "Software Development", "Python Developer"],
                "specific_category": "Python Developer"
            }
        """
        try:
            data = request.get_json()

            if not data or "resume_text" not in data:
                return jsonify({"error": "Missing resume_text in request body"}), 400

            resume_text = data["resume_text"]
            return_details = data.get("return_details", False)

            result = predictor.predict(resume_text, return_details=return_details)

            if return_details:
                return jsonify(result)
            else:
                return jsonify(
                    {
                        "predicted_category": result.category,
                        "confidence": result.confidence,
                        "tier": result.tier,
                        "requires_review": result.requires_review,
                        "hierarchy_path": result.path,
                        "specific_category": result.specific_category,
                    }
                )

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/predict_batch", methods=["POST"])
    def predict_batch():
        """
        Predict categories for multiple resumes.

        Request:
            {
                "resumes": ["resume 1 text...", "resume 2 text...", ...]
            }

        Response:
            {
                "predictions": [
                    {
                        "predicted_category": "Python Developer",
                        "confidence": 0.85,
                        ...
                    },
                    ...
                ]
            }
        """
        try:
            data = request.get_json()

            if not data or "resumes" not in data:
                return jsonify({"error": "Missing resumes in request body"}), 400

            resumes = data["resumes"]

            if not isinstance(resumes, list):
                return jsonify({"error": "resumes must be a list"}), 400

            results = predictor.predict_batch(resumes)

            predictions = [
                {
                    "predicted_category": r.category,
                    "confidence": r.confidence,
                    "tier": r.tier,
                    "requires_review": r.requires_review,
                    "hierarchy_path": r.path,
                    "specific_category": r.specific_category,
                }
                for r in results
            ]

            return jsonify({"predictions": predictions})

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app
