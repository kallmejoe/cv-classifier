"""Configuration module for the resume classifier.

This module centralizes all configurable parameters that were previously
hardcoded across multiple files. Import and use these instead of hardcoding values.
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any


@dataclass
class ModelConfig:
    """Configuration for model training."""
    
    # Random seed for reproducibility
    random_state: int = 42
    
    # Test/train split
    test_size: float = 0.2
    
    # Cross-validation
    cv_folds: int = 5
    
    # TF-IDF feature extraction
    max_features: int = 5000
    min_df: int = 3
    max_df: float = 0.90
    ngram_range: Tuple[int, int] = (1, 2)
    
    # Data augmentation
    use_augmentation: bool = True
    augmentation_factor: int = 2
    augmentation_methods: List[str] = field(
        default_factory=lambda: ['shuffle', 'delete', 'duplicate']
    )
    
    # Model type
    use_ensemble: bool = True
    
    # SVM hyperparameters
    svm_c_values: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.5, 1.0])
    svm_max_iter: int = 2000
    
    # Logistic Regression hyperparameters
    lr_c_values: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.5, 1.0])
    lr_max_iter: int = 1000
    
    # Naive Bayes hyperparameters
    nb_alpha: float = 0.1
    
    # Ensemble weights [SVM, LR, NB]
    ensemble_weights: List[int] = field(default_factory=lambda: [2, 2, 1])


@dataclass
class PathConfig:
    """Configuration for file paths."""
    
    # Output directories
    output_dir: str = 'output'
    model_dir: str = 'models'
    
    # Dataset paths
    resume_csv: str = 'Resume.csv'
    updated_csv: str = 'UpdatedResumeDataSet.csv'
    corpus_csv: str = 'ResumesCorpusDataSet.csv'
    corpus_dir: str = 'resumes_corpus'
    
    # Model artifacts
    model_file: str = 'resume_classifier.pkl'
    feature_extractor_file: str = 'feature_extractor.pkl'
    metadata_file: str = 'metadata.pkl'
    
    def get_model_path(self) -> str:
        """Get full path to saved model."""
        return os.path.join(self.model_dir, self.model_file)
    
    def get_feature_extractor_path(self) -> str:
        """Get full path to saved feature extractor."""
        return os.path.join(self.model_dir, self.feature_extractor_file)
    
    def get_metadata_path(self) -> str:
        """Get full path to saved metadata."""
        return os.path.join(self.model_dir, self.metadata_file)
    
    def ensure_dirs(self) -> None:
        """Create output and model directories if they don't exist."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)


@dataclass
class APIConfig:
    """Configuration for the Flask API."""
    
    host: str = '127.0.0.1'
    port: int = 5000
    debug: bool = False
    
    # Model directory (relative to project root)
    model_dir: str = 'models'


@dataclass 
class DataQualityConfig:
    """Configuration for data quality checks."""
    
    # Similarity threshold for duplicate detection
    similarity_threshold: float = 0.9
    
    # Minimum resume length (characters)
    min_resume_length: int = 100


@dataclass
class PredictionConfig:
    """
    Configuration for prediction confidence thresholds.
    
    Note: The old tech_patterns and rule-based overrides have been removed.
    We now use hierarchical category selection based on confidence tiers
    (see src/category_hierarchy.py).
    """
    
    # Legacy - kept for backward compatibility during transition
    # These are no longer used by the new hierarchical predictor
    high_confidence_threshold: float = 0.7
    medium_confidence_threshold: float = 0.4


@dataclass
class DatasetConfig:
    """Configuration for dataset loading."""
    resume_csv: str = 'Resume.csv'
    updated_csv: str = 'UpdatedResumeDataSet.csv'
    corpus_csv: str = 'ResumesCorpusDataSet.csv'
    corpus_dir: str = 'resumes_corpus'

    # Column mappings for different datasets
    resume_text_column: str = 'Resume_str'
    category_column: str = 'Category'

    # Cleaning options
    remove_duplicates: bool = True
    drop_na: bool = True

    # Verbose output
    verbose: bool = True


# Default configuration instances
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_PATH_CONFIG = PathConfig()
DEFAULT_API_CONFIG = APIConfig()
DEFAULT_DATA_QUALITY_CONFIG = DataQualityConfig()
DEFAULT_PREDICTION_CONFIG = PredictionConfig()
