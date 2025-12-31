# Resume Classifier

A multiclass resume classification system using **Bag-of-Words** and classical ML algorithms (SVM, Logistic Regression, Naive Bayes ensemble).

## Quick Start

```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install pandas scikit-learn numpy matplotlib seaborn flask

# Train the model
python main.py

# Run the API
python api.py

# Run tests
python -m pytest tests/ -v
```

## Project Structure

```
cv-classifier/
├── main.py                     # Training pipeline (entry point)
├── api.py                      # Flask REST API for predictions
├── src/
│   ├── config.py               # All configuration (model params, paths, thresholds)
│   ├── preprocessing.py        # Text cleaning and normalization
│   ├── feature_extraction.py   # TF-IDF Bag-of-Words vectorization
│   ├── model.py                # Model training, ensemble, ResumePredictor
│   ├── augmentation.py         # Data augmentation (token-level)
│   ├── evaluation.py           # Metrics and visualization
│   ├── category_mapper.py      # Category normalization and mapping
│   └── data_quality.py         # Duplicate detection, data cleaning
├── importer/
│   ├── __init__.py
│   └── data_loader.py          # Unified dataset loading functions
├── tests/
│   └── test_classifier.py      # Pytest test suite
├── models/                     # Saved model artifacts
│   ├── resume_classifier.pkl
│   ├── feature_extractor.pkl
│   └── metadata.pkl
├── output/                     # Training outputs (plots, metrics)
├── archive/                    # Archived/unused code
├── Resume.csv                  # Primary dataset (2,482 samples)
├── UpdatedResumeDataSet.csv    # Additional dataset
└── ResumesCorpusDataSet.csv    # Large corpus dataset (~30K samples)
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/info` | GET | Model information (classes, vocabulary size) |
| `/predict` | POST | Classify single resume |
| `/predict_batch` | POST | Classify multiple resumes |

### Example API Usage

```bash
# Health check
curl http://localhost:5000/health

# Classify a resume
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"resume_text": "Software Engineer with 5 years Python experience..."}'

# Get detailed prediction with confidence
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"resume_text": "...", "return_probabilities": true}'
```

## Configuration

All configurable parameters are centralized in `src/config.py`:

```python
from src.config import ModelConfig, PathConfig, PredictionConfig

# Model training parameters
model_config = ModelConfig(
    random_state=42,
    test_size=0.2,
    cv_folds=5,
    max_features=5000,
    use_augmentation=True,
    use_ensemble=True
)

# File paths
path_config = PathConfig(
    model_dir='models',
    output_dir='output'
)

# Prediction thresholds
pred_config = PredictionConfig(
    high_confidence_threshold=0.5,
    weak_confidence_threshold=0.4
)
```

## Model Details

**Architecture:** Soft-voting ensemble of:
- Linear SVM (weight: 2)
- Logistic Regression (weight: 2)
- Multinomial Naive Bayes (weight: 1)

**Features:**
- TF-IDF weighted Bag-of-Words (5,000 features)
- Unigrams and bigrams
- Hyperparameter tuning via GridSearchCV

**Confidence Calculation:**
- Confidence = max(probability) across all classes
- For soft voting: P(class) = weighted average of classifier probabilities

## Datasets

The system combines multiple resume datasets:

| Dataset | Samples | Categories | Description |
|---------|---------|------------|-------------|
| Resume.csv | 2,482 | 24 | Industry categories |
| UpdatedResumeDataSet.csv | 166 | 25 | Job-specific roles |
| ResumesCorpusDataSet.csv | ~28K | 10 | Large technical corpus |

Combined dataset: ~30K samples, 43 normalized categories

## Testing

```bash
# Run all tests
python -m pytest tests/test_classifier.py -v

# Run specific test class
python -m pytest tests/test_classifier.py::TestPreprocessing -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=term-missing
```

## Usage Examples

### Training a New Model

```python
from main import run_pipeline
from src.config import ModelConfig, PathConfig

# Custom configuration
config = ModelConfig(
    use_augmentation=True,
    augmentation_factor=2,
    cv_folds=5
)

# Run training
run_pipeline(model_config=config)
```

### Making Predictions

```python
from src.model import ResumePredictor

# Load predictor
predictor = ResumePredictor(model_dir='models')

# Simple prediction
category = predictor.predict("Your resume text here...")
print(f"Category: {category}")

# Detailed prediction with probabilities
result = predictor.predict(
    "Your resume text...",
    return_probabilities=True
)
print(f"Category: {result['predicted_category']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Top 5: {result['top_5_predictions']}")

# Batch prediction
categories = predictor.predict_batch([
    "Resume 1...",
    "Resume 2...",
    "Resume 3..."
])
```

## Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | ~72-75% |
| F1-Score (macro) | ~0.72 |
| Categories | 43 |
| Training Time | ~10-15 min (with tuning) |

## Archived Files

Previous versions and utility scripts are preserved in `archive/`:
- `train_optimized.py` - Superseded by main.py
- `prepare_dataset.py` - Merged into importer/
- `predict.py`, `predict_improved.py` - Merged into src/model.py
- `analyze_categories.py`, `convert_corpus.py` - One-time utilities

## Requirements

- Python 3.10+
- pandas
- scikit-learn
- numpy
- matplotlib
- seaborn
- flask

## Academic Compliance

This implementation uses classical ML only:
- No deep learning or transformers
- No word embeddings  
- No NLP libraries (spaCy, NLTK for parsing)
- Pure TF-IDF statistical features
- Classical algorithms (SVM, LR, NB)
