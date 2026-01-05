# Resume Classifier

A multiclass resume classification system using **Bag-of-Words** and classical ML algorithms.

## Quick Start

```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Train model
python train.py

# Start API
python serve.py
```

## Project Structure

```
cv-classifier/
├── train.py              # Train the model
├── serve.py              # Start API server
├── requirements.txt      # Dependencies
├── core/                 # Core logic
│   ├── trainer.py        # Training pipeline
│   └── api_server.py     # Flask app
├── src/                  # ML modules
├── importer/             # Data loaders
├── tests/                # Tests
└── models/               # Saved models (generated)
```

## Usage

### Train

```bash
python train.py
```

Models are saved to `models/`, plots to `output/`.

### Serve API

```bash
python serve.py
```

Server starts at http://localhost:5000

### Test API

```bash
# Health check
curl http://localhost:5000/health

# Classify resume
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"resume_text": "Python developer with ML experience..."}'
```

## Configuration

Edit [src/config.py](src/config.py) to customize model parameters.

## API Endpoints

- `GET  /health` - Health check
- `GET  /info` - Model information
- `POST /predict` - Classify single resume
- `POST /predict_batch` - Classify multiple resumes

## Datasets

- `Resume.csv` - Primary dataset (2,482 samples)
- `UpdatedResumeDataSet.csv` - Additional dataset
- `ResumesCorpusDataSet.csv` - Large corpus (~30K samples)

Combined dataset: ~30K samples, 43 normalized categories

## Model Details

- **Algorithm**: Ensemble of SVM, Logistic Regression, Naive Bayes
- **Features**: TF-IDF Bag-of-Words with n-grams
- **Accuracy**: ~72-75%

## Requirements

- Python 3.10+
- See [requirements.txt](requirements.txt) for dependencies
