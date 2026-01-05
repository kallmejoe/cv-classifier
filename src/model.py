"""Model module for resume classification.

This module consolidates all model-related functionality:
- Model training and hyperparameter tuning
- Ensemble creation
- Model saving/loading
- Prediction with confidence calculation

The confidence calculation uses probability scores from the classifier.
For models that don't support predict_proba natively (like LinearSVC),
we use CalibratedClassifierCV to obtain probability estimates.
"""

import os
import pickle
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.multiclass import OneVsRestClassifier
import scipy.sparse

from src.config import ModelConfig, PathConfig, DEFAULT_MODEL_CONFIG, DEFAULT_PATH_CONFIG
from src.preprocessing import preprocess_text
from src.feature_extraction import FeatureExtractor
from src.category_hierarchy import (
    select_by_confidence,
    get_category_path,
    HierarchicalPrediction,
    ConfidenceThresholds,
    DEFAULT_THRESHOLDS,
)


def tune_hyperparameters(
    X_train: scipy.sparse.csr_matrix,
    y_train: np.ndarray,
    config: Optional[ModelConfig] = None
) -> Tuple[Any, Any, Any]:
    """
    Hyperparameter tuning with GridSearchCV.

    This function tunes hyperparameters for SVM, Logistic Regression, and Naive Bayes
    classifiers using cross-validation.

    Args:
        X_train: Training feature matrix
        y_train: Training labels
        config: Model configuration (uses defaults if not provided)

    Returns:
        Tuple of (best_svm, best_lr, best_nb) models
    """
    config = config or DEFAULT_MODEL_CONFIG

    print("\nHyperparameter Tuning...")

    # SVM tuning
    svm_grid = GridSearchCV(
        LinearSVC(
            random_state=config.random_state,
            class_weight='balanced',
            dual='auto'
        ),
        {
            'C': config.svm_c_values,
            'max_iter': [config.svm_max_iter]
        },
        cv=config.cv_folds,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    svm_grid.fit(X_train, y_train)
    print(f"  SVM best: C={svm_grid.best_params_['C']}, CV={svm_grid.best_score_:.4f}")

    # Logistic Regression tuning
    lr_grid = GridSearchCV(
        LogisticRegression(
            random_state=config.random_state,
            solver='lbfgs',
            class_weight='balanced'
        ),
        {
            'C': config.lr_c_values,
            'max_iter': [config.lr_max_iter]
        },
        cv=config.cv_folds,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    lr_grid.fit(X_train, y_train)
    print(f"  LR best: C={lr_grid.best_params_['C']}, CV={lr_grid.best_score_:.4f}")

    # Naive Bayes (no tuning needed, use configured alpha)
    nb = MultinomialNB(alpha=config.nb_alpha)
    nb_scores = cross_val_score(nb, X_train, y_train, cv=config.cv_folds, scoring='accuracy')
    print(f"  NB: alpha={config.nb_alpha}, CV={nb_scores.mean():.4f}")

    return svm_grid.best_estimator_, lr_grid.best_estimator_, nb


def tune_all_classifiers(
    X_train: scipy.sparse.csr_matrix,
    y_train: np.ndarray,
    config: Optional[ModelConfig] = None,
    fast_mode: bool = True
) -> Dict[str, Tuple[Any, float]]:
    """
    Tune multiple classifiers and return all with their CV scores.
    
    Args:
        X_train: Training feature matrix
        y_train: Training labels
        config: Model configuration
        fast_mode: If True, use reduced param grids for speed (default: True)
        
    Returns:
        Dict mapping classifier name to (fitted_model, cv_score)
    """
    config = config or DEFAULT_MODEL_CONFIG
    results: Dict[str, Tuple[Any, float]] = {}
    
    # Use fewer CV folds for faster tuning in fast mode
    cv_folds = 3 if fast_mode else config.cv_folds
    
    print("\n" + "="*60)
    print(f"HYPERPARAMETER TUNING {'(FAST MODE)' if fast_mode else ''}")
    print("="*60)
    
    # 1. LinearSVC - excellent for text classification
    print("\n[1/5] Tuning LinearSVC...")
    svm_grid = GridSearchCV(
        LinearSVC(random_state=config.random_state, class_weight='balanced', dual='auto', max_iter=5000),
        {'C': [0.5, 1.0, 2.0]},
        cv=cv_folds, scoring='accuracy', n_jobs=-1
    )
    svm_grid.fit(X_train, y_train)
    results['LinearSVC'] = (svm_grid.best_estimator_, svm_grid.best_score_)
    print(f"  Best C={svm_grid.best_params_['C']}, CV={svm_grid.best_score_:.4f}")
    
    # 2. Logistic Regression - robust and interpretable
    print("\n[2/5] Tuning LogisticRegression...")
    lr_grid = GridSearchCV(
        LogisticRegression(random_state=config.random_state, solver='lbfgs', class_weight='balanced', max_iter=2000),
        {'C': [0.5, 1.0, 2.0, 5.0]},
        cv=cv_folds, scoring='accuracy', n_jobs=-1
    )
    lr_grid.fit(X_train, y_train)
    results['LogisticRegression'] = (lr_grid.best_estimator_, lr_grid.best_score_)
    print(f"  Best C={lr_grid.best_params_['C']}, CV={lr_grid.best_score_:.4f}")
    
    # 3. MultinomialNB - fast and works well with TF-IDF
    print("\n[3/5] Tuning MultinomialNB...")
    nb_grid = GridSearchCV(
        MultinomialNB(),
        {'alpha': [0.01, 0.1, 0.5, 1.0]},
        cv=cv_folds, scoring='accuracy', n_jobs=-1
    )
    nb_grid.fit(X_train, y_train)
    results['MultinomialNB'] = (nb_grid.best_estimator_, nb_grid.best_score_)
    print(f"  Best alpha={nb_grid.best_params_['alpha']}, CV={nb_grid.best_score_:.4f}")
    
    # 4. SGDClassifier - very fast, scalable linear classifier
    print("\n[4/5] Tuning SGDClassifier...")
    sgd_grid = GridSearchCV(
        SGDClassifier(random_state=config.random_state, class_weight='balanced', max_iter=2000, tol=1e-3),
        {'alpha': [0.0001, 0.001], 'loss': ['hinge', 'log_loss']},
        cv=cv_folds, scoring='accuracy', n_jobs=-1
    )
    sgd_grid.fit(X_train, y_train)
    results['SGDClassifier'] = (sgd_grid.best_estimator_, sgd_grid.best_score_)
    print(f"  Best alpha={sgd_grid.best_params_['alpha']}, loss={sgd_grid.best_params_['loss']}, CV={sgd_grid.best_score_:.4f}")
    
    # 5. OneVsRest with MultinomialNB (from notebook approach)
    print("\n[5/5] Tuning OneVsRest(MultinomialNB)...")
    ovr_nb = OneVsRestClassifier(MultinomialNB(alpha=nb_grid.best_params_['alpha']))
    ovr_scores = cross_val_score(ovr_nb, X_train, y_train, cv=cv_folds, scoring='accuracy', n_jobs=-1)
    ovr_nb.fit(X_train, y_train)
    results['OneVsRest_NB'] = (ovr_nb, ovr_scores.mean())
    print(f"  CV={ovr_scores.mean():.4f}")
    
    # Summary
    print("\n" + "="*60)
    print("CLASSIFIER RANKING")
    print("="*60)
    sorted_results = sorted(results.items(), key=lambda x: x[1][1], reverse=True)
    for i, (name, (_, score)) in enumerate(sorted_results, 1):
        print(f"  {i}. {name}: {score:.4f}")
    
    return results


def create_enhanced_ensemble(
    classifiers: Dict[str, Tuple[Any, float]],
    X_train: scipy.sparse.csr_matrix,
    y_train: np.ndarray,
    config: Optional[ModelConfig] = None,
    top_n: int = 5
) -> VotingClassifier:
    """
    Create an enhanced voting ensemble from top classifiers.
    
    Args:
        classifiers: Dict from tune_all_classifiers
        X_train: Training feature matrix
        y_train: Training labels
        config: Model configuration
        top_n: Number of top classifiers to include
        
    Returns:
        Fitted VotingClassifier ensemble
    """
    config = config or DEFAULT_MODEL_CONFIG
    
    print("\n" + "="*60)
    print("CREATING ENHANCED ENSEMBLE")
    print("="*60)
    
    # Sort by CV score and take top N
    sorted_clf = sorted(classifiers.items(), key=lambda x: x[1][1], reverse=True)[:top_n]
    
    estimators = []
    weights = []
    
    for name, (clf, score) in sorted_clf:
        print(f"  Including: {name} (CV={score:.4f})")
        
        # Calibrate classifiers that don't support predict_proba
        if name in ['LinearSVC', 'SGDClassifier'] and hasattr(clf, 'decision_function'):
            clf_calibrated = CalibratedClassifierCV(clf, cv=3)
            estimators.append((name, clf_calibrated))
        else:
            estimators.append((name, clf))
        
        # Weight by CV score (higher score = higher weight)
        weights.append(score)
    
    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight * len(weights) for w in weights]
    
    print(f"\n  Weights: {[f'{w:.2f}' for w in weights]}")
    
    ensemble = VotingClassifier(
        estimators=estimators,
        voting='soft',
        weights=weights,
        n_jobs=-1
    )
    
    print("\n  Fitting ensemble...")
    ensemble.fit(X_train, y_train)
    
    scores = cross_val_score(ensemble, X_train, y_train, cv=config.cv_folds, scoring='accuracy')
    print(f"\n  ENSEMBLE CV: {scores.mean():.4f} (+/- {scores.std():.4f})")
    
    return ensemble


def create_ensemble(
    svm: Any,
    lr: Any,
    nb: Any,
    X_train: scipy.sparse.csr_matrix,
    y_train: np.ndarray,
    config: Optional[ModelConfig] = None
) -> VotingClassifier:
    """
    Create a voting ensemble from individual classifiers.

    The ensemble uses soft voting with calibrated probabilities.
    SVM is wrapped in CalibratedClassifierCV to enable probability estimation.

    Args:
        svm: Trained SVM classifier
        lr: Trained Logistic Regression classifier
        nb: Trained Naive Bayes classifier
        X_train: Training feature matrix
        y_train: Training labels
        config: Model configuration

    Returns:
        Fitted VotingClassifier ensemble
    """
    config = config or DEFAULT_MODEL_CONFIG

    print("\nCreating Ensemble...")

    # Calibrate SVM for soft voting (enables probability estimation)
    svm_calibrated = CalibratedClassifierCV(svm, cv=3)

    weights = config.ensemble_weights

    ensemble = VotingClassifier(
        estimators=[('svm', svm_calibrated), ('lr', lr), ('nb', nb)],
        voting='soft',
        weights=weights
    )

    ensemble.fit(X_train, y_train)
    scores = cross_val_score(
        ensemble, X_train, y_train,
        cv=config.cv_folds,
        scoring='accuracy'
    )
    print(f"  Ensemble CV: {scores.mean():.4f} (+/- {scores.std():.4f})")

    return ensemble


def save_model(
    model: Any,
    feature_extractor: FeatureExtractor,
    path_config: Optional[PathConfig] = None
) -> None:
    """
    Save model artifacts to disk.

    Saves:
    - The trained model (classifier or ensemble)
    - The feature extractor (vectorizer + label encoder)
    - Metadata (vocabulary size, class names, etc.)

    Args:
        model: Trained classifier
        feature_extractor: Fitted FeatureExtractor
        path_config: Path configuration
    """
    path_config = path_config or DEFAULT_PATH_CONFIG

    path_config.ensure_dirs()

    with open(path_config.get_model_path(), 'wb') as f:
        pickle.dump(model, f)

    with open(path_config.get_feature_extractor_path(), 'wb') as f:
        pickle.dump(feature_extractor, f)

    metadata = {
        'vocabulary_size': feature_extractor.get_vocabulary_size(),
        'num_classes': feature_extractor.get_num_classes(),
        'class_names': feature_extractor.get_label_names(),
        'model_type': 'ensemble' if hasattr(model, 'estimators') else 'single'
    }

    with open(path_config.get_metadata_path(), 'wb') as f:
        pickle.dump(metadata, f)

    print(f"\nModel saved to: {path_config.model_dir}/")


def load_model(
    path_config: Optional[PathConfig] = None
) -> Tuple[Any, FeatureExtractor, Dict[str, Any]]:
    """
    Load model artifacts from disk.

    Args:
        path_config: Path configuration

    Returns:
        Tuple of (model, feature_extractor, metadata)

    Raises:
        FileNotFoundError: If model files don't exist
    """
    path_config = path_config or DEFAULT_PATH_CONFIG

    model_path = path_config.get_model_path()
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}. Run 'python train.py' first to train a model."
        )

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    with open(path_config.get_feature_extractor_path(), 'rb') as f:
        feature_extractor = pickle.load(f)

    with open(path_config.get_metadata_path(), 'rb') as f:
        metadata = pickle.load(f)

    return model, feature_extractor, metadata


class ResumePredictor:
    """
    Resume classification predictor with hierarchical confidence-based output.

    This class handles:
    - Loading saved model artifacts
    - Text preprocessing
    - Prediction with probability/confidence scores
    - Hierarchical output based on confidence tiers

    Confidence-Based Hierarchy:
    ---------------------------
    - High confidence (>=0.7): Returns specific category (e.g., "Python Developer")
    - Medium confidence (0.4-0.7): Returns parent specialization (e.g., "Software Development")
    - Low confidence (<0.4): Returns domain (e.g., "Technology") + requires_review flag

    Confidence Calculation:
    -----------------------
    The confidence score is computed as the maximum probability across all classes.
    For an ensemble using soft voting, the probability for each class is the
    weighted average of the individual classifier probabilities.
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        path_config: Optional[PathConfig] = None,
        confidence_thresholds: Optional[ConfidenceThresholds] = None
    ):
        """
        Initialize the predictor.

        Args:
            model_dir: Directory containing model files (overrides path_config)
            path_config: Path configuration
            confidence_thresholds: Thresholds for confidence-based tier selection
        """
        self.path_config = path_config or DEFAULT_PATH_CONFIG
        if model_dir:
            self.path_config.model_dir = model_dir

        self.confidence_thresholds = confidence_thresholds or DEFAULT_THRESHOLDS

        self.model: Any = None
        self.feature_extractor: Optional[FeatureExtractor] = None
        self.metadata: Dict[str, Any] = {}

        self._load_model()

    def _load_model(self) -> None:
        """Load model artifacts from disk."""
        self.model, self.feature_extractor, self.metadata = load_model(self.path_config)
        print(f"Model loaded ({self.metadata['num_classes']} categories)")

    def predict(
        self,
        resume_text: str,
        return_details: bool = False
    ) -> Any:
        """
        Predict category for a resume with hierarchical confidence-based output.

        Confidence-Based Tiers:
        - High (>=0.7): Returns specific category (e.g., "Python Developer")
        - Medium (0.4-0.7): Returns specialization (e.g., "Software Development")
        - Low (<0.4): Returns domain (e.g., "Technology") + requires_review=True

        Args:
            resume_text: Raw resume text
            return_details: If True, return full HierarchicalPrediction object

        Returns:
            If return_details=False: HierarchicalPrediction object
            If return_details=True: dict with full prediction details
        """
        # Preprocess text
        processed_text = preprocess_text(resume_text, remove_stops=True, min_token_length=2)

        # Transform to features
        assert self.feature_extractor is not None
        X, _ = self.feature_extractor.transform([processed_text], None)

        # Get prediction and probabilities
        prediction = self.model.predict(X)[0]
        category = self.feature_extractor.decode_labels([prediction])[0]

        # Get probability scores for confidence calculation
        probabilities = self.model.predict_proba(X)[0]

        # Confidence is the maximum probability across all classes
        confidence = float(np.max(probabilities))

        # Get top-N predictions for low-confidence consensus
        class_names = self.metadata['class_names']
        prob_dict = dict(zip(class_names, probabilities))
        sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
        top_5_categories = [cat for cat, prob in sorted_probs[:5]]

        # Apply hierarchical selection based on confidence
        hierarchical_result = select_by_confidence(
            predicted_category=category,
            confidence=confidence,
            top_predictions=top_5_categories,
            thresholds=self.confidence_thresholds
        )

        if not return_details:
            return hierarchical_result

        # Build detailed response dict
        result: Dict[str, Any] = {
            # Primary output
            'predicted_category': hierarchical_result.category,
            'confidence': hierarchical_result.confidence,
            'tier': hierarchical_result.tier,
            'requires_review': hierarchical_result.requires_review,

            # Hierarchy info
            'specific_category': hierarchical_result.specific_category,
            'hierarchy_path': hierarchical_result.path,

            # Top predictions for context
            'top_5_predictions': [
                {'category': cat, 'probability': float(prob)}
                for cat, prob in sorted_probs[:5]
            ],
        }

        if hierarchical_result.top_suggestions:
            result['top_suggestions'] = hierarchical_result.top_suggestions

        return result

    def predict_batch(
        self,
        resume_texts: List[str]
    ) -> List[HierarchicalPrediction]:
        """
        Predict categories for multiple resumes.

        Args:
            resume_texts: List of raw resume texts

        Returns:
            List of HierarchicalPrediction objects
        """
        return [self.predict(text, return_details=False) for text in resume_texts]

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model metadata.

        Returns:
            Dict with num_classes, vocabulary_size, categories, model_type
        """
        return {
            'num_classes': self.metadata['num_classes'],
            'vocabulary_size': self.metadata['vocabulary_size'],
            'categories': self.metadata['class_names'],
            'model_type': self.metadata['model_type']
        }
