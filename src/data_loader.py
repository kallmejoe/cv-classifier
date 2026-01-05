"""Unified data loader for resume datasets.

This module consolidates all dataset loading logic into a single place,
removing duplication across main.py, train_optimized.py, prepare_dataset.py, etc.
"""

import os
from dataclasses import dataclass
from typing import Optional, List, cast
import pandas as pd


@dataclass
class DatasetConfig:
    """Configuration for dataset loading."""
    resume_csv: str = 'Resume.csv'
    updated_csv: str = 'UpdatedResumeDataSet.csv'
    corpus_csv: str = 'ResumesCorpusDataSet.csv'
    corpus_dir: str = 'resumes_corpus'
    
    # Column mappings for different datasets
    resume_text_column: str = 'Resume_str'  # Resume.csv uses Resume_str
    category_column: str = 'Category'
    
    # Cleaning options
    remove_duplicates: bool = True
    drop_na: bool = True
    
    # Verbose output
    verbose: bool = True


def load_resume_csv(
    file_path: Optional[str] = None,
    config: Optional[DatasetConfig] = None
) -> pd.DataFrame:
    """
    Load Resume.csv dataset (large, clean dataset).
    
    Args:
        file_path: Path to Resume.csv (overrides config)
        config: Dataset configuration
        
    Returns:
        Cleaned DataFrame with 'Resume' and 'Category' columns
    """
    config = config or DatasetConfig()
    file_path = file_path or config.resume_csv
    
    if config.verbose:
        print(f"Loading Resume.csv from: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found: {file_path}")
        
    df: pd.DataFrame = pd.read_csv(file_path)
    
    # Use Resume_str column (plain text, not HTML)
    df = df[[config.resume_text_column, config.category_column]].copy()
    df.columns = pd.Index(['Resume', 'Category'])
    
    original_count = len(df)
    
    if config.remove_duplicates:
        df = df.drop_duplicates(subset='Resume', keep='first')
    
    if config.drop_na:
        df = df.dropna()
    
    if config.verbose:
        duplicates = original_count - len(df)
        print(f"  - Original: {original_count} samples")
        print(f"  - Duplicates removed: {duplicates}")
        num_categories = len(df['Category'].unique())
        print(f"  - Cleaned: {len(df)} samples, {num_categories} categories")
    
    return df


def load_updated_resume_csv(
    file_path: Optional[str] = None,
    config: Optional[DatasetConfig] = None
) -> pd.DataFrame:
    """
    Load UpdatedResumeDataSet.csv (smaller dataset with many duplicates).
    
    Args:
        file_path: Path to UpdatedResumeDataSet.csv (overrides config)
        config: Dataset configuration
        
    Returns:
        Cleaned DataFrame with 'Resume' and 'Category' columns
    """
    config = config or DatasetConfig()
    file_path = file_path or config.updated_csv
    
    if config.verbose:
        print(f"Loading UpdatedResumeDataSet.csv from: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    
    df: pd.DataFrame = pd.read_csv(file_path)
    
    original_count = len(df)
    
    if config.remove_duplicates:
        duplicates = df.duplicated(subset='Resume', keep='first')
        dup_count = int(duplicates.sum())
        df = df.drop_duplicates(subset='Resume', keep='first')
    else:
        dup_count = 0
    
    if config.drop_na:
        df = df.dropna()
    
    if config.verbose:
        print(f"  - Original: {original_count} samples")
        if dup_count > 0:
            print(f"  - Duplicates: {dup_count} ({100 * dup_count / original_count:.1f}%)")
        num_categories = len(df['Category'].unique())
        print(f"  - Cleaned: {len(df)} samples, {num_categories} categories")
    
    return df


def load_corpus_dataset(
    file_path: Optional[str] = None,
    config: Optional[DatasetConfig] = None
) -> pd.DataFrame:
    """
    Load ResumesCorpusDataSet.csv (large corpus dataset).
    
    Args:
        file_path: Path to ResumesCorpusDataSet.csv (overrides config)
        config: Dataset configuration
        
    Returns:
        Cleaned DataFrame with 'Resume' and 'Category' columns
    """
    config = config or DatasetConfig()
    file_path = file_path or config.corpus_csv
    
    if config.verbose:
        print(f"Loading ResumesCorpusDataSet.csv from: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"{file_path} not found. Run archive/convert_corpus.py first to create it."
        )
    
    df: pd.DataFrame = pd.read_csv(file_path)
    
    # Ensure correct columns exist
    if 'Category' in df.columns and 'Resume' in df.columns:
        df = df[['Category', 'Resume']].copy()
    else:
        raise ValueError(f"Expected 'Category' and 'Resume' columns in {file_path}")
    
    original_count = len(df)
    
    if config.remove_duplicates:
        df = df.drop_duplicates(subset='Resume', keep='first')
    
    if config.drop_na:
        df = df.dropna()
    
    if config.verbose:
        duplicates = original_count - len(df)
        print(f"  - Original: {original_count} samples")
        print(f"  - Duplicates removed: {duplicates}")
        num_categories = len(df['Category'].unique())
        print(f"  - Cleaned: {len(df)} samples, {num_categories} categories")
    
    return df


def load_combined_datasets(
    include_resume: bool = True,
    include_updated: bool = True,
    include_corpus: bool = False,
    config: Optional[DatasetConfig] = None
) -> pd.DataFrame:
    """
    Load and combine multiple datasets, removing duplicates across all.
    
    Args:
        include_resume: Include Resume.csv
        include_updated: Include UpdatedResumeDataSet.csv
        include_corpus: Include ResumesCorpusDataSet.csv
        config: Dataset configuration
        
    Returns:
        Combined and deduplicated DataFrame
    """
    config = config or DatasetConfig()
    
    if config.verbose:
        print("Loading and combining datasets...")
    
    datasets: List[pd.DataFrame] = []
    
    if include_resume and os.path.exists(config.resume_csv):
        df1 = load_resume_csv(config=config)
        datasets.append(df1)
    
    if include_updated and os.path.exists(config.updated_csv):
        df2 = load_updated_resume_csv(config=config)
        datasets.append(df2)
    
    if include_corpus and os.path.exists(config.corpus_csv):
        df3 = load_corpus_dataset(config=config)
        datasets.append(df3)
    
    if not datasets:
        raise ValueError("No datasets found to load")
    
    # Combine datasets
    df_combined: pd.DataFrame = pd.concat(datasets, ignore_index=True)
    
    if config.verbose:
        print(f"\nCombined: {len(df_combined)} samples")
    
    # Remove duplicates across all datasets
    if config.remove_duplicates:
        before_count = len(df_combined)
        df_combined = df_combined.drop_duplicates(subset='Resume', keep='first')
        cross_dups = before_count - len(df_combined)
        
        if config.verbose:
            print(f"  - Cross-dataset duplicates: {cross_dups}")
    
    if config.verbose:
        num_categories = len(df_combined['Category'].unique())
        print(f"  - Final combined: {len(df_combined)} samples, {num_categories} categories")
    
    return df_combined
