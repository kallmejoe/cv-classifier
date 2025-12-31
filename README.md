# Resume Classifier

A multiclass resume classification system using **Bag-of-Words** and classical ML algorithms (SVM, Logistic Regression, Naive Bayes ensemble), with optional **GPU-accelerated neural network training**.

## Quick Start

```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install pandas scikit-learn numpy matplotlib seaborn flask

# (Optional) For GPU support - install PyTorch with CUDA
pip install torch  # CPU only
# OR for NVIDIA GPU:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Train the model (CPU - sklearn classifiers)
python main.py

# Train with GPU (neural network)
python main.py --gpu

# Run the API
python api.py

# Run tests
python -m pytest tests/ -v
```

## GPU Training

The classifier supports GPU-accelerated training using PyTorch neural networks:

```bash
# Automatic GPU detection (uses CUDA if available, falls back to CPU)
python main.py --gpu

# Specify device explicitly
python main.py --gpu --device cuda:0    # First NVIDIA GPU
python main.py --gpu --device cuda:1    # Second NVIDIA GPU
python main.py --gpu --device mps       # Apple Silicon GPU
python main.py --gpu --device cpu       # Force CPU

# Customize training parameters
python main.py --gpu --epochs 100 --batch-size 128

# Check GPU availability
python -c "from src.gpu_utils import print_device_info; print_device_info()"
```

### GPU Requirements

- **NVIDIA GPU**: CUDA 11.8+ and cuDNN
- **Apple Silicon**: macOS 12.3+ (MPS backend)
- **PyTorch**: 2.0+ recommended

The system automatically falls back to CPU-based sklearn classifiers if PyTorch or GPU is not available.

## Project Structure

```
cv-classifier/
├── main.py                     # Training pipeline (entry point)
├── api.py                      # Flask REST API for predictions
├── src/
│   ├── config.py               # All configuration (model params, paths, GPU settings)
│   ├── preprocessing.py        # Text cleaning and normalization
│   ├── feature_extraction.py   # TF-IDF Bag-of-Words vectorization
│   ├── model.py                # Model training, ensemble, ResumePredictor
│   ├── neural_model.py         # GPU-accelerated neural network classifier
│   ├── gpu_utils.py            # GPU detection and utilities
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

### CPU Mode (Default - sklearn)

**Architecture:** Soft-voting ensemble of:
- Linear SVM (weight: 2)
- Logistic Regression (weight: 2)
- Multinomial Naive Bayes (weight: 1)

**Features:**
- TF-IDF weighted Bag-of-Words (5,000 features)
- Unigrams and bigrams
- Hyperparameter tuning via GridSearchCV

### GPU Mode (Neural Network)

**Architecture:** Deep neural network:
- Input: TF-IDF features (8,000 dimensions)
- Hidden layers: 512 → 256 → 128 (ReLU, BatchNorm, Dropout)
- Output: Softmax over classes

**Features:**
- Automatic mixed precision (FP16) on supported GPUs
- Early stopping with validation monitoring
- Learning rate scheduling (ReduceLROnPlateau)
- Gradient clipping for stability

**Confidence Calculation:**
- Confidence = max(probability) across all classes
- For soft voting: P(class) = weighted average of classifier probabilities
- For neural net: P(class) = softmax output

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
