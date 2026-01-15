"""
Resume Classifier Training Pipeline

Core training logic separated from CLI interface.
Implements the complete ML pipeline from data loading to model saving.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    ModelConfig,
    PathConfig,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_PATH_CONFIG,
)
from src.preprocessing import preprocess_corpus
from src.feature_extraction import FeatureExtractor
from src.augmentation import TextAugmenter
from src.evaluation import (
    evaluate_model,
    print_evaluation_summary,
    plot_confusion_matrix,
    plot_class_distribution,
)
from src.dataset_cleaner import get_clean_dataset
from src.model import (
    tune_hyperparameters,
    tune_all_classifiers,
    create_enhanced_ensemble,
    save_model,
)
from src.data_loader import (
    load_resume_csv,
    load_updated_resume_csv,
    load_corpus_dataset,
    load_combined_datasets,
)
from src.utils.logging_utils import get_logger

logger = get_logger()


def plot_learning_curve(
    estimator,
    X,
    y,
    title: str = "Learning Curve",
    cv: int = 5,
    save_path: Optional[str] = None,
    random_state: int = 42,
):
    """Plot learning curve for overfitting detection."""
    logger.info("\nGenerating learning curve...")

    result = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        random_state=random_state,
        shuffle=True,
    )
    train_sizes, train_scores, val_scores = result[:3]

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    plt.fill_between(
        train_sizes,
        val_scores_mean - val_scores_std,
        val_scores_mean + val_scores_std,
        alpha=0.1,
        color="g",
    )
    plt.plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
    plt.plot(
        train_sizes, val_scores_mean, "o-", color="g", label="Cross-validation score"
    )

    plt.xlabel("Training Set Size", fontsize=12)
    plt.ylabel("Accuracy Score", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)

    gap = train_scores_mean[-1] - val_scores_mean[-1]
    if gap > 0.1:
        plt.text(
            0.5,
            0.05,
            f"Large gap ({gap:.2%}) suggests overfitting",
            transform=plt.gca().transAxes,
            ha="center",
            bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.5),
        )
    else:
        plt.text(
            0.5,
            0.05,
            f"Small gap ({gap:.2%}) - model generalizes well",
            transform=plt.gca().transAxes,
            ha="center",
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5),
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Learning curve saved to: {save_path}")

    plt.close()

    return train_scores_mean, val_scores_mean


def train_model(
    dataset_mode: str = "clean",
    model_config: Optional[ModelConfig] = None,
    path_config: Optional[PathConfig] = None,
) -> Tuple[dict, object, FeatureExtractor]:

    # Use provided configs or defaults
    config = model_config or DEFAULT_MODEL_CONFIG
    paths = path_config or DEFAULT_PATH_CONFIG

    # Set random seed
    np.random.seed(config.random_state)

    logger.info("=" * 70)
    logger.info("RESUME CLASSIFICATION PIPELINE")
    logger.info(f"Dataset Mode: {dataset_mode.upper()}")
    logger.info(f"Augmentation: {'ON' if config.use_augmentation else 'OFF'}")
    logger.info(f"Ensemble: {'ON' if config.use_ensemble else 'OFF'}")
    logger.info("=" * 70)

    # Ensure output directories exist
    paths.ensure_dirs()

    # =========================================================================
    # STEP 1: Load Dataset
    # =========================================================================
    logger.info("\n" + "-" * 70)
    logger.info("STEP 1: Loading Dataset")
    logger.info("-" * 70)

    if dataset_mode == "clean":
        df = get_clean_dataset(include_corpus=True, verbose=True)
    elif dataset_mode == "resume":
        df = load_resume_csv(paths.resume_csv)
    elif dataset_mode == "updated":
        df = load_updated_resume_csv(paths.updated_csv)
    elif dataset_mode == "corpus":
        df = load_corpus_dataset(paths.corpus_csv)
    elif dataset_mode == "both":
        df = load_combined_datasets(include_corpus=False)
    elif dataset_mode == "all":
        df = load_combined_datasets(include_corpus=True)
    else:
        raise ValueError(
            f"Invalid dataset_mode: {dataset_mode}. "
            f"Use 'clean', 'resume', 'updated', 'corpus', 'both', or 'all'"
        )

    logger.info(f"\nDataset Statistics:")
    logger.info(f"  - Total samples: {len(df)}")
    logger.info(f"  - Categories: {len(df['Category'].unique())}")
    logger.info(f"  - Average resume length: {df['Resume'].str.len().mean():.0f} characters")

    # =========================================================================
    # STEP 2: Text Preprocessing
    # =========================================================================
    logger.info("\n" + "-" * 70)
    logger.info("STEP 2: Preprocessing Text Data")
    logger.info("-" * 70)

    raw_texts = df["Resume"].tolist()
    labels = df["Category"].tolist()

    logger.info("Applying preprocessing pipeline...")
    processed_texts = preprocess_corpus(
        raw_texts, remove_stops=True, min_token_length=2
    )

    logger.info(f"\nPreprocessing complete:")
    logger.info(f"  - Original avg length: {np.mean([len(t) for t in raw_texts]):.0f} chars")
    logger.info(
        f"  - Processed avg length: {np.mean([len(t) for t in processed_texts]):.0f} chars"
    )

    # =========================================================================
    # STEP 3: Train/Test Split
    # =========================================================================
    logger.info("\n" + "-" * 70)
    logger.info("STEP 3: Train/Test Split")
    logger.info("-" * 70)

    texts_train, texts_test, labels_train, labels_test = train_test_split(
        processed_texts,
        labels,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=labels,
    )

    logger.info(f"\nData split (stratified):")
    logger.info(
        f"  - Training set: {len(texts_train)} samples ({100*(1-config.test_size):.0f}%)"
    )
    logger.info(f"  - Test set: {len(texts_test)} samples ({100*config.test_size:.0f}%)")

    # =========================================================================
    # STEP 4: Optional Data Augmentation
    # =========================================================================
    if config.use_augmentation:
        logger.info("\n" + "-" * 70)
        logger.info("STEP 4: Data Augmentation")
        logger.info("-" * 70)

        augmenter = TextAugmenter(random_state=config.random_state)
        texts_train, labels_train = augmenter.augment_dataset(
            texts_train,
            labels_train,
            augmentation_factor=config.augmentation_factor,
            methods=config.augmentation_methods,
        )

        logger.info(f"\nAugmented training set: {len(texts_train)} samples")
    else:
        logger.info("\n" + "-" * 70)
        logger.info("STEP 4: Data Augmentation - SKIPPED")
        logger.info("-" * 70)

    # =========================================================================
    # STEP 5: Feature Extraction
    # =========================================================================
    logger.info("\n" + "-" * 70)
    logger.info("STEP 5: Feature Extraction (TF-IDF + N-grams)")
    logger.info("-" * 70)

    feature_extractor = FeatureExtractor(
        max_features=config.max_features,
        min_df=config.min_df,
        max_df=config.max_df,
        use_tfidf=True,
        ngram_range=config.ngram_range,
        use_char_ngrams=True,
        char_ngram_range=(2, 5),
        char_max_features=3000,
        sublinear_tf=True,
    )

    X_train, y_train = feature_extractor.fit_transform(texts_train, labels_train)
    X_test, y_test = feature_extractor.transform(texts_test, labels_test)

    vocab_size = feature_extractor.get_vocabulary_size()
    num_classes = feature_extractor.get_num_classes()
    class_names = feature_extractor.get_label_names()

    logger.info(f"\nFeature space:")
    logger.info(f"  - Total vocabulary size: {vocab_size} terms")
    logger.info(f"  - Training matrix: {X_train.shape}")
    logger.info(f"  - Test matrix: {X_test.shape}")
    logger.info(f"  - Number of classes: {num_classes}")
    logger.info(
        f"  - Sparsity: {100 * (1 - X_train.nnz / (X_train.shape[0] * X_train.shape[1])):.2f}%"
    )

    plot_class_distribution(
        y_train,
        class_names,
        save_path=os.path.join(paths.output_dir, "class_distribution.png"),
        title="Training Set Class Distribution",
    )

    # =========================================================================
    # STEP 6: Model Training
    # =========================================================================
    logger.info("\n" + "-" * 70)
    logger.info("STEP 6: Model Training")
    logger.info("-" * 70)

    if config.use_ensemble:
        logger.info(f"\nTraining enhanced ensemble...")
        all_classifiers = tune_all_classifiers(X_train, y_train, config, fast_mode=True)
        best_model = create_enhanced_ensemble(
            all_classifiers, X_train, y_train, config, top_n=3
        )
        model_name = "Enhanced Ensemble"
    else:
        logger.info(f"\nTraining single model with hyperparameter tuning...")
        best_svm, best_lr, best_nb = tune_hyperparameters(X_train, y_train, config)

        svm_score = cross_val_score(
            best_svm, X_train, y_train, cv=config.cv_folds
        ).mean()
        lr_score = cross_val_score(best_lr, X_train, y_train, cv=config.cv_folds).mean()
        nb_score = cross_val_score(best_nb, X_train, y_train, cv=config.cv_folds).mean()

        scores = {"SVM": svm_score, "LR": lr_score, "NB": nb_score}
        models = {"SVM": best_svm, "LR": best_lr, "NB": best_nb}

        best_model_key = max(scores, key=lambda k: scores[k])
        best_model = models[best_model_key]
        model_name = best_model_key

        logger.info(f"\nBest single model: {model_name} (CV: {scores[best_model_key]:.4f})")
        best_model.fit(X_train, y_train)

    # =========================================================================
    # STEP 7: Learning Curve Analysis
    # =========================================================================
    logger.info("\n" + "-" * 70)
    logger.info("STEP 7: Learning Curve Analysis")
    logger.info("-" * 70)

    train_scores, val_scores = plot_learning_curve(
        best_model,
        X_train,
        y_train,
        title=f"Learning Curve - {model_name}",
        cv=config.cv_folds,
        save_path=os.path.join(paths.output_dir, "learning_curve.png"),
        random_state=config.random_state,
    )

    # =========================================================================
    # STEP 8: Test Set Evaluation
    # =========================================================================
    logger.info("\n" + "-" * 70)
    logger.info("STEP 8: Final Evaluation on Test Set")
    logger.info("-" * 70)

    y_pred = best_model.predict(X_test)
    results = evaluate_model(y_test, y_pred, class_names)

    print_evaluation_summary(results)

    eval_class_names = results.get("class_names", class_names)
    figsize = (16, 14) if num_classes > 15 else (12, 10)

    plot_confusion_matrix(
        results["confusion_matrix"],
        eval_class_names,
        save_path=os.path.join(paths.output_dir, "confusion_matrix.png"),
        title=f"Confusion Matrix - {model_name}",
        figsize=figsize,
    )

    # =========================================================================
    # STEP 9: Save Model
    # =========================================================================
    logger.info("\n" + "-" * 70)
    logger.info("STEP 9: Saving Model")
    logger.info("-" * 70)

    save_model(best_model, feature_extractor, paths)

    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("\n" + "-" * 70)
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("-" * 70)

    logger.info(f"\nBest Model: {model_name}")
    logger.info(
        f"  - Test Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)"
    )
    logger.info(f"  - Weighted F1-Score: {results['f1_weighted']:.4f}")
    logger.info(f"  - Macro F1-Score: {results['f1_macro']:.4f}")

    logger.info(f"\nOutput files saved to:")
    logger.info(f"  - {paths.output_dir}/")
    logger.info(f"  - {paths.model_dir}/")

    logger.info("\n" + "=" * 70)
    logger.info("Pipeline completed successfully!")
    logger.info("=" * 70)

    return results, best_model, feature_extractor
