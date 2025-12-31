# Resume Classification Pipeline - Complete Guide

## Overview

This project provides a machine learning pipeline for classifying resumes into job categories using classical ML algorithms with Bag-of-Words (TF-IDF) feature representation.

## 🚀 Quick Start

### 1. Convert the Corpus Dataset

First, convert the `resumes_corpus/` directory (containing .txt and .lab files) into CSV format:

```bash
python3 convert_corpus.py
```

This creates `ResumesCorpusDataSet.csv` with ~30,000 resumes.

### 2. Train the Model

Train the model using all three datasets:

```bash
python3 main.py
```

This will:
- Load all three datasets (Resume.csv, UpdatedResumeDataSet.csv, ResumesCorpusDataSet.csv)
- Remove duplicates
- Apply data augmentation
- Train an ensemble model (SVM + Logistic Regression + Naive Bayes)
- Save the model to `models/`

### 3. Make Predictions

Use the improved predictor with smart post-processing:

```bash
# Command line
python3 predict_improved.py --file path/to/resume.txt --probabilities

# Or as API
python3 api.py
# Then POST to http://127.0.0.1:5000/predict
```

## 📊 Available Datasets

| Dataset | Samples | Categories | Source |
|---------|---------|------------|--------|
| Resume.csv | ~2,484 | 24 | Large, clean dataset |
| UpdatedResumeDataSet.csv | ~155 | 25 | Small dataset (after dedup) |
| ResumesCorpusDataSet.csv | ~29,783 | ~10 | Corpus dataset (after conversion) |
| **Combined (ALL)** | **~32,000+** | **35+** | **All datasets merged** |

## 🎯 Dataset Modes

In `main.py`, you can choose different dataset modes:

```python
# Use specific dataset
main(dataset_mode='resume')      # Only Resume.csv
main(dataset_mode='updated')     # Only UpdatedResumeDataSet.csv  
main(dataset_mode='corpus')      # Only ResumesCorpusDataSet.csv

# Combine datasets
main(dataset_mode='both')        # Resume.csv + UpdatedResumeDataSet.csv
main(dataset_mode='all')         # All three datasets (RECOMMENDED)
```

## 📁 Project Structure

```
cv-classifier/
├── resumes_corpus/              # Raw corpus data (txt/lab files)
│   ├── 00001.txt
│   ├── 00001.lab
│   └── ...
├── src/                         # Core modules
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── model.py
│   ├── augmentation.py
│   └── evaluation.py
├── models/                      # Trained models (generated)
│   ├── resume_classifier.pkl
│   ├── feature_extractor.pkl
│   └── metadata.pkl
├── output/                      # Visualizations (generated)
│   ├── confusion_matrix.png
│   ├── class_distribution.png
│   └── learning_curve.png
├── convert_corpus.py            # Convert corpus to CSV
├── main.py                      # Training pipeline
├── predict.py                   # Basic prediction
├── predict_improved.py          # Improved prediction with rules
├── api.py                       # REST API
└── README.md                    # This file
```

## 🔧 Scripts

### convert_corpus.py
Converts `resumes_corpus/` directory to CSV format.

**Input:** 
- `resumes_corpus/*.txt` - Resume text files
- `resumes_corpus/*.lab` - Category label files

**Output:**
- `ResumesCorpusDataSet.csv` - CSV with Resume and Category columns

**Features:**
- Removes HTML tags
- Handles multiple labels (selects primary)
- Normalizes category names
- Deduplicates resumes

### main.py
Main training pipeline - train models on single or combined datasets.

**Key Features:**
- Multi-dataset support (5 modes)
- Data augmentation (optional)
- Hyperparameter tuning with GridSearchCV
- Ensemble models (voting classifier)
- Learning curve analysis
- Model persistence

**Configuration:**
```python
main(
    dataset_mode='all',          # Dataset selection
    use_augmentation=True,       # Enable text augmentation
    use_ensemble=True,           # Use ensemble vs single model
    augmentation_factor=2,       # 2x data increase
    test_size=0.2,              # 20% test set
    cv_folds=5                  # 5-fold cross-validation
)
```

### predict_improved.py
Improved prediction with intelligent post-processing rules.

**Features:**
- Technology pattern detection
- Smart category override for low-confidence predictions
- Detects modern web stack (Next.js, React, Node.js, etc.)
- Provides detailed explanations

**Example:**
```bash
python3 predict_improved.py --text "Full-stack developer with React, Node.js..." --probabilities
```

**Output:**
```
Predicted Category: Software Developer
Confidence: 35.23%

⚠️  OVERRIDE APPLIED
Original Prediction: Automation Testing
Reason: Low confidence (35.2%) + Strong fullstack pattern (4 keywords: react, node.js, next.js, typescript)

🔍 Detected Role: fullstack
Pattern Confidence: 45.0%
Keywords Found: react, node.js, next.js, typescript, express
```

### api.py
Flask REST API for resume classification.

**Endpoints:**
- `GET /health` - Health check
- `GET /info` - Model information
- `POST /predict` - Classify single resume
- `POST /predict_batch` - Classify multiple resumes

**Example:**
```bash
# Start API
python3 api.py

# Make prediction
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "resume_text": "Software developer with Python and Django experience...",
    "return_probabilities": true
  }'
```

## 🧠 Model Architecture

### Feature Extraction
- **Method:** TF-IDF (Term Frequency-Inverse Document Frequency)
- **Vocabulary Size:** 5,000 most important terms
- **N-grams:** Unigrams + Bigrams (1-2 word combinations)
- **Min Document Frequency:** 3 (term must appear in ≥3 resumes)
- **Max Document Frequency:** 90% (term can appear in ≤90% of resumes)

### Classifiers
Three classifiers combined in an ensemble:

1. **Linear SVM** (weight: 2)
   - Calibrated for probability estimation
   - Hyperparameter tuning: C ∈ {0.1, 0.3, 0.5, 1.0}

2. **Logistic Regression** (weight: 2)
   - L2 regularization
   - Hyperparameter tuning: C ∈ {0.1, 0.3, 0.5, 1.0}

3. **Naive Bayes** (weight: 1)
   - MultinomialNB
   - Alpha: 0.1

### Ensemble Method
- **Voting:** Soft voting (weighted probability averaging)
- **Weights:** [2, 2, 1] for [SVM, LR, NB]

## 📈 Performance

With all three datasets combined:
- **Training Samples:** ~32,000+ (after augmentation: ~64,000+)
- **Categories:** 35+
- **Expected Accuracy:** 75-85% (depends on category balance)

### Category Coverage

**Developer Roles:**
- Software Developer
- Python Developer
- Java Developer
- DotNet Developer
- Web Developer
- Front End Developer
- DevOps Engineer
- Database Administrator

**Other Roles:**
- Systems Administrator
- Network Administrator  
- Security Analyst
- Project Manager
- Data Science
- And 20+ more...

## 🔍 Improved Prediction Logic

The `predict_improved.py` script includes smart post-processing:

### Technology Pattern Detection

**Full-Stack Detection:**
- Keywords: next.js, react, node.js, express, vue, angular, mern, mean
- Min keywords: 2
- Target: "Software Developer"

**Frontend Detection:**
- Keywords: react, vue, angular, css, html, tailwind, webpack
- Min keywords: 3  
- Target: "Web Designing"

**Data Science Detection:**
- Keywords: machine learning, tensorflow, pytorch, pandas, numpy
- Min keywords: 3
- Target: "Data Science"

### Override Logic

Prediction is overridden if:
1. **Confidence < 40%**
2. **Strong technology pattern detected** (pattern confidence > 30%)
3. **Predicted category doesn't match** detected role

This prevents misclassification of modern web developers as "Automation Testing" or "Database" when the model has low confidence.

## 🚨 Common Issues

### Issue 1: ResumesCorpusDataSet.csv not found

**Solution:**
```bash
python3 convert_corpus.py
```

### Issue 2: Low prediction confidence

**Causes:**
- Resume uses modern technologies not well-represented in training data
- Resume spans multiple categories

**Solutions:**
1. Use `predict_improved.py` instead of `predict.py` for smart overrides
2. Retrain with more data using `dataset_mode='all'`
3. Add custom training samples for underrepresented categories

### Issue 3: Wrong category predicted

**Example:** Full-stack developer → "Automation Testing"

**Solution:** The improved predictor detects this and overrides:
```python
from predict_improved import ImprovedResumePredictor

predictor = ImprovedResumePredictor()
result = predictor.predict(resume_text, return_probabilities=True)

if result['override_applied']:
    print(f"Corrected: {result['original_prediction']} → {result['predicted_category']}")
```

## 📝 Training Workflow

### Complete Training Pipeline

```bash
# Step 1: Convert corpus dataset
python3 convert_corpus.py

# Step 2: Train model with all datasets
python3 main.py

# Step 3: Test predictions
python3 predict_improved.py --file test_resume.txt --probabilities

# Step 4: Start API (optional)
python3 api.py
```

### Custom Training

```python
from main import main

# Train with specific configuration
results, model, extractor = main(
    dataset_mode='all',
    use_augmentation=True,
    use_ensemble=True,
    augmentation_factor=3,  # 3x augmentation for small datasets
    test_size=0.2,
    cv_folds=5
)
```

## 🎓 Understanding the Results

### Confusion Matrix
Shows where the model makes mistakes:
- Diagonal: Correct predictions
- Off-diagonal: Misclassifications

### Learning Curve
- **Train-validation gap < 10%:** Good generalization
- **Train-validation gap > 10%:** Potential overfitting

### Class Distribution
- Balanced classes (similar counts): Better performance
- Imbalanced classes: May struggle with minority classes

## 🔮 Future Improvements

1. **Add More Categories:**
   - Mobile Developer (iOS/Android)
   - Full-Stack Developer (explicit category)
   - Cloud Engineer
   - AI/ML Engineer

2. **Better Feature Engineering:**
   - Add named entity recognition (NER) for skills
   - Extract years of experience
   - Detect certifications

3. **Deep Learning:**
   - Fine-tune BERT/RoBERTa for resume classification
   - Use transformer-based models for better context understanding

4. **Active Learning:**
   - Collect user feedback on predictions
   - Retrain with corrected labels

## 📚 Dependencies

```
pandas
numpy
scikit-learn
flask
matplotlib
```

Install all:
```bash
pip install pandas numpy scikit-learn flask matplotlib
```

## 📄 License

[Add your license here]

## 🤝 Contributing

[Add contribution guidelines]

---

**Need Help?** Open an issue or contact [your contact info]
