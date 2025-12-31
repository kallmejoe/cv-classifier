#!/usr/bin/env python3
"""
Resume Classification Pipeline - Unified Multi-Dataset Version

This script implements a comprehensive machine learning pipeline that can:
- Train on either Resume.csv or UpdatedResumeDataSet.csv
- Train on both datasets combined for maximum data utilization
- Handle duplicate removal and data cleaning
- Apply data augmentation when beneficial
- Use hyperparameter tuning and ensemble methods
- Generate comprehensive evaluation reports

Methodology:
- Bag-of-Words (TF-IDF) feature representation
- Classical ML classifiers (Logistic Regression, SVM, Naive Bayes)
- Optional ensemble voting for improved performance
- Proper train/test split to prevent overfitting
"""

import os
import sys
import warnings
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve

# Import from our modules
from src.config import ModelConfig, PathConfig, DEFAULT_MODEL_CONFIG, DEFAULT_PATH_CONFIG
from src.preprocessing import preprocess_corpus
from src.feature_extraction import FeatureExtractor
from src.augmentation import TextAugmenter
from src.evaluation import (
    evaluate_model,
    print_evaluation_summary,
    plot_confusion_matrix,
    plot_class_distribution
)
from src.dataset_cleaner import get_clean_dataset
from src.model import tune_hyperparameters, create_ensemble, save_model

from importer import load_resume_csv, load_updated_resume_csv, load_corpus_dataset, load_combined_datasets

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings('ignore')


def plot_learning_curve(
    estimator,
    X,
    y,
    title: str = "Learning Curve",
    cv: int = 5,
    save_path: Optional[str] = None,
    random_state: int = 42
):
    """Plot learning curve for overfitting detection."""
    print("\nGenerating learning curve...")

    result = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        random_state=random_state,
        shuffle=True
    )
    train_sizes, train_scores, val_scores = result[:3]

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, val_scores_mean - val_scores_std,
                     val_scores_mean + val_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, val_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.xlabel("Training Set Size", fontsize=12)
    plt.ylabel("Accuracy Score", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)

    gap = train_scores_mean[-1] - val_scores_mean[-1]
    if gap > 0.1:
        plt.text(0.5, 0.05, f"Large gap ({gap:.2%}) suggests overfitting",
                transform=plt.gca().transAxes, ha='center',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    else:
        plt.text(0.5, 0.05, f"Small gap ({gap:.2%}) - model generalizes well",
                transform=plt.gca().transAxes, ha='center',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Learning curve saved to: {save_path}")

    plt.close()

    return train_scores_mean, val_scores_mean


def main(
    dataset_mode: str = 'clean',
    model_config: Optional[ModelConfig] = None,
    path_config: Optional[PathConfig] = None
):
    """
    Main pipeline execution function.

    Args:
        dataset_mode: Dataset selection (default: 'clean')
            - 'clean': Use cleaned dataset with corpus (RECOMMENDED - 30K+ samples)
            - 'resume': Only Resume.csv (2,484 samples)
            - 'updated': Only UpdatedResumeDataSet.csv (~150 samples)
            - 'corpus': Only ResumesCorpusDataSet.csv (~30K samples)
            - 'both': Resume.csv + UpdatedResumeDataSet.csv
            - 'all': All three datasets combined (raw, no cleaning)
        model_config: Model configuration (uses defaults if not provided)
        path_config: Path configuration (uses defaults if not provided)
    """
    # Use provided configs or defaults
    config = model_config or DEFAULT_MODEL_CONFIG
    paths = path_config or DEFAULT_PATH_CONFIG

    # Set random seed
    np.random.seed(config.random_state)

    print("="*70)
    print("RESUME CLASSIFICATION PIPELINE - HIERARCHICAL VERSION")
    print(f"Dataset Mode: {dataset_mode.upper()}")
    print(f"Augmentation: {'ON' if config.use_augmentation else 'OFF'}")
    print(f"Ensemble: {'ON' if config.use_ensemble else 'OFF'}")
    print("="*70)

    # Ensure output directories exist
    paths.ensure_dirs()

    # =========================================================================
    # STEP 1: Load Dataset
    # =========================================================================
    print("\n" + "-"*70)
    print("STEP 1: Loading Dataset")
    print("-"*70)

    if dataset_mode == 'clean':
        # Use the new clean dataset with all data including corpus
        df = get_clean_dataset(include_corpus=True, verbose=True)
    elif dataset_mode == 'resume':
        df = load_resume_csv(paths.resume_csv)
    elif dataset_mode == 'updated':
        df = load_updated_resume_csv(paths.updated_csv)
    elif dataset_mode == 'corpus':
        df = load_corpus_dataset(paths.corpus_csv)
    elif dataset_mode == 'both':
        df = load_combined_datasets(include_corpus=False)
    elif dataset_mode == 'all':
        df = load_combined_datasets(include_corpus=True)
    else:
        raise ValueError(
            f"Invalid dataset_mode: {dataset_mode}. "
            f"Use 'clean', 'resume', 'updated', 'corpus', 'both', or 'all'"
        )

    print(f"\nDataset Statistics:")
    print(f"  - Total samples: {len(df)}")
    print(f"  - Categories: {len(df['Category'].unique())}")
    print(f"  - Average resume length: {df['Resume'].str.len().mean():.0f} characters")

    print(f"\nCategory Distribution:")
    category_counts = df['Category'].value_counts()
    for cat, count in category_counts.head(15).items():
        print(f"  - {cat}: {count}")
    if len(category_counts) > 15:
        print(f"  ... and {len(category_counts) - 15} more categories")

    # =========================================================================
    # STEP 2: Text Preprocessing
    # =========================================================================
    print("\n" + "-"*70)
    print("STEP 2: Preprocessing Text Data")
    print("-"*70)

    # Extract texts and labels from cleaned dataframe
    raw_texts = df['Resume'].tolist()
    labels = df['Category'].tolist()

    print("Applying preprocessing pipeline...")
    processed_texts = preprocess_corpus(raw_texts, remove_stops=True, min_token_length=2)

    print(f"\nPreprocessing complete:")
    print(f"  - Original avg length: {np.mean([len(t) for t in raw_texts]):.0f} chars")
    print(f"  - Processed avg length: {np.mean([len(t) for t in processed_texts]):.0f} chars")

    # =========================================================================
    # STEP 3: Train/Test Split
    # =========================================================================
    print("\n" + "-"*70)
    print("STEP 3: Train/Test Split")
    print("-"*70)

    texts_train, texts_test, labels_train, labels_test = train_test_split(
        processed_texts,
        labels,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=labels
    )

    print(f"\nData split (stratified):")
    print(f"  - Training set: {len(texts_train)} samples ({100*(1-config.test_size):.0f}%)")
    print(f"  - Test set: {len(texts_test)} samples ({100*config.test_size:.0f}%)")

    # =========================================================================
    # STEP 4: Optional Data Augmentation
    # =========================================================================
    if config.use_augmentation:
        print("\n" + "-"*70)
        print("STEP 4: Data Augmentation (Training Set Only)")
        print("-"*70)

        augmenter = TextAugmenter(random_state=config.random_state)

        print(f"Applying augmentation with factor={config.augmentation_factor}")
        print(f"Methods: {', '.join(config.augmentation_methods)}")

        texts_train, labels_train = augmenter.augment_dataset(
            texts_train,
            labels_train,
            augmentation_factor=config.augmentation_factor,
            methods=config.augmentation_methods
        )

        print(f"\nAugmented training set: {len(texts_train)} samples")
    else:
        print("\n" + "-"*70)
        print("STEP 4: Data Augmentation - SKIPPED")
        print("-"*70)

    # =========================================================================
    # STEP 5: Feature Extraction
    # =========================================================================
    print("\n" + "-"*70)
    print("STEP 5: Feature Extraction (Word + Character N-grams with TF-IDF)")
    print("-"*70)

    feature_extractor = FeatureExtractor(
        max_features=config.max_features,
        min_df=config.min_df,
        max_df=config.max_df,
        use_tfidf=True,
        ngram_range=config.ngram_range,
        use_char_ngrams=True,  # NEW: Add character n-grams
        char_ngram_range=(2, 5),  # Character 2-5 grams
        char_max_features=3000,  # Additional 3000 char features
        sublinear_tf=True  # NEW: Use 1 + log(tf) scaling
    )

    print("Feature extractor configuration:")
    print(f"  - Word max features: {feature_extractor.max_features}")
    print(f"  - Char max features: {feature_extractor.char_max_features}")
    print(f"  - Min document frequency: {feature_extractor.min_df}")
    print(f"  - Max document frequency: {feature_extractor.max_df}")
    print(f"  - Word N-gram range: {feature_extractor.ngram_range}")
    print(f"  - Char N-gram range: {feature_extractor.char_ngram_range}")
    print(f"  - Sublinear TF: {feature_extractor.sublinear_tf}")

    X_train, y_train = feature_extractor.fit_transform(texts_train, labels_train)
    X_test, y_test = feature_extractor.transform(texts_test, labels_test)

    vocab_size = feature_extractor.get_vocabulary_size()
    num_classes = feature_extractor.get_num_classes()
    class_names = feature_extractor.get_label_names()

    print(f"\nFeature space:")
    print(f"  - Total vocabulary size: {vocab_size} terms")
    print(f"  - Training matrix: {X_train.shape}")
    print(f"  - Test matrix: {X_test.shape}")
    print(f"  - Number of classes: {num_classes}")
    print(f"  - Sparsity: {100 * (1 - X_train.nnz / (X_train.shape[0] * X_train.shape[1])):.2f}%")

    plot_class_distribution(
        y_train,
        class_names,
        save_path=os.path.join(paths.output_dir, 'class_distribution.png'),
        title='Training Set Class Distribution'
    )

    # =========================================================================
    # STEP 6: Model Training
    # =========================================================================
    print("\n" + "-"*70)
    print("STEP 6: Model Training")
    print("-"*70)

    # Import enhanced training functions
    from src.model import tune_all_classifiers, create_enhanced_ensemble

    if config.use_ensemble:
        print(f"\nTraining ENHANCED ensemble with multiple classifiers...")
        
        # Tune all classifiers (fast mode for speed)
        all_classifiers = tune_all_classifiers(X_train, y_train, config, fast_mode=True)
        
        # Create enhanced ensemble from top 3 classifiers
        best_model = create_enhanced_ensemble(all_classifiers, X_train, y_train, config, top_n=3)
        model_name = "Enhanced Ensemble (Top 5 Classifiers)"
    else:
        print(f"\nTraining single model with hyperparameter tuning...")
        best_svm, best_lr, best_nb = tune_hyperparameters(X_train, y_train, config)

        # Select best single model based on CV score
        svm_score = cross_val_score(best_svm, X_train, y_train, cv=config.cv_folds).mean()
        lr_score = cross_val_score(best_lr, X_train, y_train, cv=config.cv_folds).mean()
        nb_score = cross_val_score(best_nb, X_train, y_train, cv=config.cv_folds).mean()

        scores = {'SVM': svm_score, 'LR': lr_score, 'NB': nb_score}
        models = {'SVM': best_svm, 'LR': best_lr, 'NB': best_nb}

        best_model_key = max(scores, key=lambda k: scores[k])
        best_model = models[best_model_key]
        model_name = best_model_key

        print(f"\nBest single model: {model_name} (CV: {scores[best_model_key]:.4f})")
        best_model.fit(X_train, y_train)

    # =========================================================================
    # STEP 7: Learning Curve Analysis
    # =========================================================================
    print("\n" + "-"*70)
    print("STEP 7: Learning Curve Analysis")
    print("-"*70)

    train_scores, val_scores = plot_learning_curve(
        best_model,
        X_train,
        y_train,
        title=f"Learning Curve - {model_name}",
        cv=config.cv_folds,
        save_path=os.path.join(paths.output_dir, 'learning_curve.png'),
        random_state=config.random_state
    )

    final_gap = train_scores[-1] - val_scores[-1]
    print(f"\nLearning Curve Analysis:")
    print(f"  - Final training score: {train_scores[-1]:.4f}")
    print(f"  - Final validation score: {val_scores[-1]:.4f}")
    print(f"  - Train-validation gap: {final_gap:.4f}")

    if final_gap > 0.1:
        print(f"  Warning: Large gap suggests potential overfitting")
    else:
        print(f"  Good generalization")

    # =========================================================================
    # STEP 8: Test Set Evaluation
    # =========================================================================
    print("\n" + "-"*70)
    print("STEP 8: Final Evaluation on Test Set")
    print("-"*70)

    y_pred = best_model.predict(X_test)
    results = evaluate_model(y_test, y_pred, class_names)

    print_evaluation_summary(results)

    # Use filtered class names from evaluation results
    eval_class_names = results.get('class_names', class_names)

    # Determine figure size based on number of classes
    figsize = (16, 14) if num_classes > 15 else (12, 10)

    plot_confusion_matrix(
        results['confusion_matrix'],
        eval_class_names,
        save_path=os.path.join(paths.output_dir, 'confusion_matrix.png'),
        title=f'Confusion Matrix - {model_name}',
        figsize=figsize
    )

    # =========================================================================
    # STEP 9: Save Model
    # =========================================================================
    print("\n" + "-"*70)
    print("STEP 9: Saving Model")
    print("-"*70)

    save_model(best_model, feature_extractor, paths)

    # =========================================================================
    # STEP 10: Summary
    # =========================================================================
    print("\n" + "-"*70)
    print("FINAL RESULTS SUMMARY")
    print("-"*70)

    print(f"\nDataset:")
    print(f"  - Mode: {dataset_mode}")
    print(f"  - Original samples: {len(df)}")
    print(f"  - After cleaning: {len(labels)}")
    print(f"  - Training size: {X_train.shape[0]}")
    print(f"  - Test size: {X_test.shape[0]}")
    print(f"  - Categories: {num_classes}")

    print(f"\nBest Model: {model_name}")
    print(f"  - Test Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"  - Weighted F1-Score: {results['f1_weighted']:.4f}")
    print(f"  - Macro F1-Score: {results['f1_macro']:.4f}")

    print(f"\nOutput files saved to:")
    print(f"  - {paths.output_dir}/confusion_matrix.png")
    print(f"  - {paths.output_dir}/class_distribution.png")
    print(f"  - {paths.output_dir}/learning_curve.png")
    print(f"  - {paths.model_dir}/resume_classifier.pkl")
    print(f"  - {paths.model_dir}/feature_extractor.pkl")
    print(f"  - {paths.model_dir}/metadata.pkl")

    print("\n" + "="*70)
    print("Pipeline completed successfully!")
    print("="*70)

    return results, best_model, feature_extractor


if __name__ == '__main__':
    # Run with cleaned dataset including corpus for maximum performance
    # The 'clean' mode uses the new dataset_cleaner which:
    # - Removes mislabeled samples (AUTOMOBILE, BPO issues)
    # - Normalizes category names to match hierarchy
    # - Includes 30K+ tech samples from the corpus
    
    # You can customize the configuration:
    # config = ModelConfig(
    #     use_augmentation=True,
    #     augmentation_factor=2,
    #     use_ensemble=True,
    #     test_size=0.2,
    #     cv_folds=5
    # )

    results, classifier, extractor = main(
        dataset_mode='clean',  # Use cleaned dataset with corpus (RECOMMENDED)
    )
