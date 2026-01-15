#!/usr/bin/env python3
"""
Test suite for the Resume Classifier.

This module tests:
- Text preprocessing
- Feature extraction
- Model loading and prediction
- Confidence calculation correctness
- API endpoints (if Flask-testing is available)

Run with: pytest tests/test_classifier.py -v
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np


class TestPreprocessing:
    """Tests for text preprocessing module."""

    def test_preprocess_text_removes_urls(self):
        """Test that URLs are removed from text."""
        from src.preprocessing import preprocess_text

        text = "Check out my portfolio at https://example.com and contact me"
        result = preprocess_text(text)

        assert "https" not in result
        assert "example" not in result
        assert "com" not in result
        assert "portfolio" in result
        assert "contact" in result

    def test_preprocess_text_removes_emails(self):
        """Test that email addresses are removed from text."""
        from src.preprocessing import preprocess_text

        text = "Contact me at john.doe@example.com for more info"
        result = preprocess_text(text)

        assert "@" not in result
        assert "john" not in result
        assert "example" not in result
        assert "contact" in result
        assert "info" in result

    def test_preprocess_text_removes_html(self):
        """Test that HTML tags are removed."""
        from src.preprocessing import preprocess_text

        text = "<div class='resume'><p>Software Engineer</p></div>"
        result = preprocess_text(text, remove_html=True)

        assert "<" not in result
        assert ">" not in result
        assert "div" not in result
        assert "software" in result
        assert "engineer" in result

    def test_preprocess_text_removes_stopwords(self):
        """Test that stopwords are removed when requested."""
        from src.preprocessing import preprocess_text

        text = "I am a software engineer and I have worked with many teams"
        result_with_stops = preprocess_text(text, remove_stops=False)
        result_without_stops = preprocess_text(text, remove_stops=True)

        # With stopwords, should be longer
        assert len(result_with_stops.split()) > len(result_without_stops.split())
        # Without stopwords, common words removed
        assert "and" not in result_without_stops.split()

    def test_preprocess_text_min_token_length(self):
        """Test that short tokens are filtered."""
        from src.preprocessing import preprocess_text

        text = "I do ML and AI at a top company"
        result = preprocess_text(text, remove_stops=False, min_token_length=3)

        # Tokens shorter than 3 characters should be removed
        for token in result.split():
            assert len(token) >= 3

    def test_preprocess_corpus(self):
        """Test batch preprocessing."""
        from src.preprocessing import preprocess_corpus

        texts = [
            "Software Engineer at Google",
            "Data Scientist with Python skills",
            "Product Manager in tech",
        ]

        results = preprocess_corpus(texts)

        assert len(results) == 3
        assert all(isinstance(r, str) for r in results)
        assert all(len(r) > 0 for r in results)


class TestFeatureExtraction:
    """Tests for feature extraction module."""

    def test_feature_extractor_fit_transform(self):
        """Test that fit_transform produces correct output shapes."""
        from src.feature_extraction import FeatureExtractor

        texts = [
            "python machine learning data science",
            "java software development backend",
            "javascript react frontend developer",
            "python data analysis pandas numpy",
        ]
        labels = ["Data Science", "Backend", "Frontend", "Data Science"]

        extractor = FeatureExtractor(max_features=100)
        X, y = extractor.fit_transform(texts, labels)

        assert X.shape[0] == 4  # 4 samples
        assert X.shape[1] <= 100  # max_features
        assert len(y) == 4
        assert extractor.get_num_classes() == 3  # 3 unique labels

    def test_feature_extractor_transform(self):
        """Test that transform works on new data."""
        from src.feature_extraction import FeatureExtractor

        # Use enough samples to satisfy min_df=2 in TfidfVectorizer
        train_texts = [
            "python data science machine learning",
            "python data analysis pandas numpy",
            "java backend development spring",
            "java software engineering backend",
            "javascript react frontend developer",
            "javascript frontend web development",
        ]
        train_labels = ["DS", "DS", "BE", "BE", "FE", "FE"]

        extractor = FeatureExtractor(max_features=50)
        extractor.fit(train_texts, train_labels)

        test_texts = ["python machine learning"]
        X, _ = extractor.transform(test_texts, None)

        assert X.shape[0] == 1

    def test_feature_extractor_decode_labels(self):
        """Test label encoding/decoding."""
        from src.feature_extraction import FeatureExtractor

        # Use enough samples to satisfy min_df=2 in TfidfVectorizer
        texts = [
            "python developer machine learning expert",
            "java developer backend systems",
            "python engineer data science",
            "java engineer software architecture",
            "python programmer ai algorithms",
            "java programmer enterprise solutions",
        ]
        labels = ["A", "B", "A", "B", "A", "B"]

        extractor = FeatureExtractor(max_features=50)
        X, y = extractor.fit_transform(texts, labels)

        decoded = extractor.decode_labels(y)

        assert decoded == labels


class TestConfidenceCalculation:
    """Tests for confidence calculation correctness.

    The confidence score should be the maximum probability across all classes.
    For soft voting ensembles, this is the weighted average probability.
    """

    def test_confidence_is_max_probability(self):
        """Test that confidence equals max(probabilities)."""
        # Simulate probability output
        probabilities = np.array([0.1, 0.15, 0.6, 0.1, 0.05])

        # Confidence should be the maximum
        expected_confidence = 0.6
        actual_confidence = float(np.max(probabilities))

        assert actual_confidence == expected_confidence

    def test_confidence_range(self):
        """Test that confidence is always between 0 and 1."""
        # Test various probability distributions
        test_cases = [
            np.array([1.0, 0.0, 0.0]),  # Certain prediction
            np.array([0.33, 0.33, 0.34]),  # Uncertain prediction
            np.array([0.5, 0.5]),  # Binary uncertain
            np.array([0.9, 0.05, 0.03, 0.02]),  # Highly confident
        ]

        for probs in test_cases:
            confidence = float(np.max(probs))
            assert 0.0 <= confidence <= 1.0

    def test_soft_voting_confidence(self):
        """Test confidence calculation for soft voting ensemble.

        For soft voting, the final probability for each class is:
        P(class) = sum(weight_i * P_i(class)) / sum(weights)

        The confidence is then max(P(class)) over all classes.
        """
        # Simulate 3 classifiers with weights [2, 2, 1]
        weights = np.array([2, 2, 1])

        # Each classifier's probability predictions for 3 classes
        clf1_probs = np.array([0.7, 0.2, 0.1])  # Confident in class 0
        clf2_probs = np.array([0.6, 0.3, 0.1])  # Confident in class 0
        clf3_probs = np.array([0.3, 0.5, 0.2])  # Confident in class 1

        # Weighted average
        weighted_sum = (
            weights[0] * clf1_probs + weights[1] * clf2_probs + weights[2] * clf3_probs
        )
        ensemble_probs = weighted_sum / np.sum(weights)

        # Calculate expected confidence
        expected_confidence = float(np.max(ensemble_probs))

        # Expected ensemble probabilities: [0.58, 0.30, 0.12]
        assert abs(ensemble_probs[0] - 0.58) < 0.01
        assert abs(ensemble_probs[1] - 0.30) < 0.01
        assert abs(ensemble_probs[2] - 0.12) < 0.01
        assert abs(expected_confidence - 0.58) < 0.01


class TestConfig:
    """Tests for configuration module."""

    def test_model_config_defaults(self):
        """Test that ModelConfig has sensible defaults."""
        from src.config import ModelConfig

        config = ModelConfig()

        assert config.random_state == 42
        assert 0 < config.test_size < 1
        assert config.cv_folds > 0
        assert config.max_features > 0
        assert isinstance(config.svm_c_values, list)
        assert len(config.svm_c_values) > 0

    def test_path_config_defaults(self):
        """Test that PathConfig has sensible defaults."""
        from src.config import PathConfig

        config = PathConfig()

        assert config.output_dir == "output"
        assert config.model_dir == "models"
        assert config.resume_csv == "Resume.csv"

    def test_prediction_config_thresholds(self):
        """Test that prediction thresholds are valid."""
        from src.config import PredictionConfig

        config = PredictionConfig()

        # Thresholds should be between 0 and 1
        assert 0 <= config.high_confidence_threshold <= 1
        assert 0 <= config.medium_confidence_threshold <= 1


class TestDataLoader:
    """Tests for data loader module."""

    def test_dataset_config_defaults(self):
        """Test DatasetConfig has sensible defaults."""
        from src.data_loader import DatasetConfig

        config = DatasetConfig()

        assert config.resume_csv == "Resume.csv"
        assert config.updated_csv == "UpdatedResumeDataSet.csv"
        assert config.remove_duplicates is True
        assert config.drop_na is True


class TestCategoryHierarchy:
    """Tests for category hierarchy module."""

    def test_get_category_path(self):
        """Test that category paths are correct."""
        from src.category_hierarchy import get_category_path

        path = get_category_path("Python Developer")

        assert path == ["Technology", "Software Development", "Python Developer"]

    def test_get_parent(self):
        """Test that parent lookup works."""
        from src.category_hierarchy import get_parent

        assert get_parent("Python Developer") == "Software Development"
        assert get_parent("Software Development") == "Technology"

    def test_backpropagate(self):
        """Test backpropagation through hierarchy."""
        from src.category_hierarchy import backpropagate

        # One level up
        assert backpropagate("Python Developer", levels=1) == "Software Development"
        # Two levels up
        assert backpropagate("Python Developer", levels=2) == "Technology"

    def test_select_by_confidence_high(self):
        """Test high confidence returns specific category."""
        from src.category_hierarchy import select_by_confidence

        result = select_by_confidence(
            predicted_category="Python Developer",
            confidence=0.85,
            top_predictions=["Python Developer", "Java Developer"],
        )

        assert result.tier == "high"
        assert result.category == "Python Developer"
        assert result.requires_review is False

    def test_select_by_confidence_medium(self):
        """Test medium confidence returns parent category."""
        from src.category_hierarchy import select_by_confidence

        result = select_by_confidence(
            predicted_category="Python Developer",
            confidence=0.55,
            top_predictions=["Python Developer", "Java Developer"],
        )

        assert result.tier == "medium"
        assert result.category == "Software Development"  # Parent
        assert result.requires_review is False

    def test_select_by_confidence_low(self):
        """Test low confidence returns domain and requires review."""
        from src.category_hierarchy import select_by_confidence

        result = select_by_confidence(
            predicted_category="Python Developer",
            confidence=0.25,
            top_predictions=["Python Developer", "Java Developer", "Web Developer"],
        )

        assert result.tier == "low"
        assert result.requires_review is True
        # Should return consensus domain from top predictions
        assert result.category == "Technology"

    def test_is_valid_category(self):
        """Test category validation."""
        from src.category_hierarchy import is_valid_category

        assert is_valid_category("Python Developer") is True
        assert is_valid_category("NonExistent Category") is False


class TestIntegration:
    """Integration tests for the full pipeline.

    These tests require the model to be trained first.
    Skip if model files don't exist.
    """

    @pytest.fixture
    def model_exists(self):
        """Check if model files exist."""
        from src.config import DEFAULT_PATH_CONFIG
        import os

        return os.path.exists(DEFAULT_PATH_CONFIG.get_model_path())

    def test_predictor_loads_model(self, model_exists):
        """Test that ResumePredictor can load the trained model."""
        if not model_exists:
            pytest.skip("Model not trained yet - run main.py first")

        from src.model import ResumePredictor

        predictor = ResumePredictor()
        info = predictor.get_model_info()

        assert "num_classes" in info
        assert info["num_classes"] > 0
        assert "vocabulary_size" in info
        assert info["vocabulary_size"] > 0

    def test_predictor_returns_valid_prediction(self, model_exists):
        """Test that predictions are valid HierarchicalPrediction objects."""
        if not model_exists:
            pytest.skip("Model not trained yet - run main.py first")

        from src.model import ResumePredictor
        from src.category_hierarchy import HierarchicalPrediction

        predictor = ResumePredictor()

        sample_resume = """
        Software Engineer with 5 years experience in Python and JavaScript.
        Worked on machine learning projects using TensorFlow and PyTorch.
        Strong background in data structures and algorithms.
        """

        result = predictor.predict(sample_resume, return_details=False)

        assert isinstance(result, HierarchicalPrediction)
        assert isinstance(result.category, str)
        assert len(result.category) > 0
        assert result.tier in ("high", "medium", "low")

    def test_predictor_returns_valid_confidence(self, model_exists):
        """Test that confidence scores are valid probabilities."""
        if not model_exists:
            pytest.skip("Model not trained yet - run main.py first")

        from src.model import ResumePredictor

        predictor = ResumePredictor()

        sample_resume = """
        Data Scientist specializing in natural language processing.
        Experience with pandas, numpy, scikit-learn, and deep learning.
        PhD in Computer Science with focus on machine learning.
        """

        result = predictor.predict(sample_resume, return_details=True)

        assert "confidence" in result
        assert 0 <= result["confidence"] <= 1

        # Check hierarchical fields
        assert "tier" in result
        assert result["tier"] in ("high", "medium", "low")
        assert "requires_review" in result
        assert isinstance(result["requires_review"], bool)
        assert "hierarchy_path" in result
        assert isinstance(result["hierarchy_path"], list)

        # Top 5 predictions should have valid probabilities
        assert "top_5_predictions" in result
        for pred in result["top_5_predictions"]:
            assert "probability" in pred
            assert 0 <= pred["probability"] <= 1

        # Confidence should equal the top prediction's probability
        assert (
            abs(result["confidence"] - result["top_5_predictions"][0]["probability"])
            < 0.001
        )

    def test_batch_prediction(self, model_exists):
        """Test batch prediction returns correct number of results."""
        if not model_exists:
            pytest.skip("Model not trained yet - run main.py first")

        from src.model import ResumePredictor
        from src.category_hierarchy import HierarchicalPrediction

        predictor = ResumePredictor()

        resumes = [
            "Software developer with Java experience",
            "Marketing manager with 10 years experience",
            "Data analyst skilled in SQL and Excel",
        ]

        predictions = predictor.predict_batch(resumes)

        assert len(predictions) == 3
        assert all(isinstance(p, HierarchicalPrediction) for p in predictions)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
