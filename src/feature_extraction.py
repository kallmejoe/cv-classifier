"""Feature extraction using Bag-of-Words and TF-IDF with word and character n-grams."""

from typing import Tuple, Optional, List, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack, csr_matrix
import scipy.sparse
from .utils.validation_utils import require_fitted


class FeatureExtractor:
    """
    Convert text documents to numerical vectors using TF-IDF.
    
    Combines word-level and character-level n-grams for robust feature extraction.
    Character n-grams help capture:
    - Spelling patterns and partial words
    - Technical terms and abbreviations
    - Robustness to typos
    """

    def __init__(
        self,
        max_features: Optional[int] = 5000,
        min_df: int = 2,
        max_df: float = 0.95,
        use_tfidf: bool = True,
        ngram_range: Tuple[int, int] = (1, 2),
        use_char_ngrams: bool = True,
        char_ngram_range: Tuple[int, int] = (2, 5),
        char_max_features: int = 3000,
        sublinear_tf: bool = True
    ):
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.use_tfidf = use_tfidf
        self.ngram_range = ngram_range
        self.use_char_ngrams = use_char_ngrams
        self.char_ngram_range = char_ngram_range
        self.char_max_features = char_max_features
        self.sublinear_tf = sublinear_tf

        # Word-level TF-IDF vectorizer
        self.word_vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            lowercase=False,
            token_pattern=r'\b\w+\b',
            sublinear_tf=sublinear_tf,  # Use 1 + log(tf) instead of raw tf
            strip_accents='unicode'
        )
        
        # Character-level TF-IDF vectorizer (for spelling patterns)
        self.char_vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=char_ngram_range,
            max_features=char_max_features,
            min_df=min_df,
            sublinear_tf=sublinear_tf,
            strip_accents='unicode'
        ) if use_char_ngrams else None

        self.label_encoder = LabelEncoder()
        self._is_fitted = False
        
        # For backward compatibility with old pickled models
        self.vectorizer = self.word_vectorizer
    
    def __setstate__(self, state):
        """Handle unpickling old FeatureExtractor objects."""
        # Restore instance state
        self.__dict__.update(state)
        
        # Backward compatibility: if old model had 'vectorizer' but not 'word_vectorizer'
        if hasattr(self, 'vectorizer') and not hasattr(self, 'word_vectorizer'):
            self.word_vectorizer = self.vectorizer
            self.char_vectorizer = None
            self.use_char_ngrams = False
            self.char_ngram_range = (2, 5)
            self.char_max_features = 3000
            self.sublinear_tf = getattr(self, 'sublinear_tf', False)

    def fit(self, texts: List[str], labels: List[str]) -> 'FeatureExtractor':
        """Fit vectorizer and label encoder on training data."""
        self.word_vectorizer.fit(texts)
        if self.char_vectorizer is not None:
            self.char_vectorizer.fit(texts)
        self.label_encoder.fit(labels)
        self._is_fitted = True
        return self

    @require_fitted
    def transform(self, texts: List[str], labels: Optional[List[str]] = None) -> Tuple[Any, Any]:
        """Transform texts and labels to numerical representations."""
        word_features = self.word_vectorizer.transform(texts)
        
        # Combine with character features if enabled
        if self.char_vectorizer is not None:
            char_features = self.char_vectorizer.transform(texts)
            X = hstack([word_features, char_features])
        else:
            X = word_features
            
        y = self.label_encoder.transform(labels) if labels is not None else None

        return X, y

    def fit_transform(self, texts: List[str], labels: List[str]) -> Tuple[Any, Any]:
        """Fit and transform in one step."""
        self.fit(texts, labels)
        return self.transform(texts, labels)

    @require_fitted
    def get_feature_names(self) -> List[str]:
        """Get vocabulary feature names."""
        word_features = list(self.word_vectorizer.get_feature_names_out())
        if self.char_vectorizer is not None:
            char_features = [f"char_{f}" for f in self.char_vectorizer.get_feature_names_out()]
            return word_features + char_features
        return word_features

    @require_fitted
    def get_label_names(self) -> List[str]:
        """Get category names in order of encoded values."""
        return list(self.label_encoder.classes_)  # type: ignore

    @require_fitted
    def get_vocabulary_size(self) -> int:
        """Get total vocabulary size (word + char features)."""
        word_size = len(self.word_vectorizer.vocabulary_)  # type: ignore
        char_size = len(self.char_vectorizer.vocabulary_) if self.char_vectorizer else 0  # type: ignore
        return word_size + char_size

    @require_fitted
    def get_num_classes(self) -> int:
        """Get number of unique classes."""
        return len(self.label_encoder.classes_)  # type: ignore

    @require_fitted
    def decode_labels(self, encoded_labels: np.ndarray) -> List[str]:
        """Convert encoded labels back to category names."""
        return list(self.label_encoder.inverse_transform(encoded_labels))
