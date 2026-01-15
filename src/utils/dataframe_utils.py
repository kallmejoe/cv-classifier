"""
DataFrame utilities for resume classification data loading.

This module provides common DataFrame cleaning and processing functions
to eliminate code duplication across data loading modules.
"""

import pandas as pd
from typing import Optional
from src.config import DatasetConfig


def clean_dataframe(
    df: pd.DataFrame, 
    config: Optional[DatasetConfig] = None,
    resume_column: str = "Resume",
    category_column: str = "Category",
    verbose: bool = None
) -> pd.DataFrame:
    """
    Clean a DataFrame by removing duplicates and NaN values.
    
    Consolidates repeated cleaning logic from data_loader.py functions.
    
    Args:
        df: DataFrame to clean
        config: Dataset configuration (takes precedence over individual params)
        resume_column: Name of resume text column for duplicate detection
        category_column: Name of category column for reporting
        verbose: Whether to print cleaning statistics (overrides config if specified)
        
    Returns:
        Cleaned DataFrame
    """
    config = config or DatasetConfig()
    verbose = verbose if verbose is not None else config.verbose
    
    original_count = len(df)
    dup_count = 0
    
    # Remove duplicates if requested
    if config.remove_duplicates:
        # Calculate duplicate count before removal
        if verbose:
            duplicates = df.duplicated(subset=resume_column, keep='first')
            dup_count = int(duplicates.sum())
        
        df = df.drop_duplicates(subset=resume_column, keep='first')
    
    # Drop NaN values if requested  
    if config.drop_na:
        df = df.dropna()
    
    # Print cleaning statistics
    if verbose:
        print(f"  - Original: {original_count} samples")
        if dup_count > 0:
            print(f"  - Duplicates: {dup_count} ({100 * dup_count / original_count:.1f}%)")
        
        # Only print category stats if category column exists
        if category_column in df.columns:
            num_categories = len(df[category_column].unique())
            print(f"  - Cleaned: {len(df)} samples, {num_categories} categories")
        else:
            print(f"  - Cleaned: {len(df)} samples")
    
    return df


def validate_file_exists(file_path: str, description: str = "File") -> None:
    """
    Validate that a file exists, raising FileNotFoundError if not.
    
    Consolidates repeated file validation logic.
    
    Args:
        file_path: Path to validate
        description: Description for error message
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    import os
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{description} not found: {file_path}")


def validate_dataframe_structure(
    df: pd.DataFrame, 
    required_columns: list[str],
    name: str = "DataFrame"
) -> None:
    """
    Validate that a DataFrame has the required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        name: Name of DataFrame for error messages
        
    Raises:
        ValueError: If required columns are missing
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        available = list(df.columns)
        raise ValueError(
            f"{name} missing required columns: {missing_columns}. "
            f"Available columns: {available}"
        )


def print_dataframe_info(df: pd.DataFrame, name: str, category_column: str = "Category") -> None:
    """
    Print standardized information about a DataFrame.
    
    Args:
        df: DataFrame to describe
        name: Name/description of the DataFrame
        category_column: Name of category column for unique count
    """
    print(f"Loading {name}")
    print(f"  - Shape: {df.shape}")
    if category_column in df.columns:
        print(f"  - Categories: {df[category_column].nunique()} unique")
    print(f"  - Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")