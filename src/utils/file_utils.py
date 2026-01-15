"""
File operation utilities for safe file handling.

This module provides utilities for file validation, safe operations,
and model file management.
"""

import os
from typing import Optional
from pathlib import Path


def validate_file_exists(file_path: str, description: str = "File") -> None:
    """
    Validate that a file exists, raising FileNotFoundError if not.
    
    Consolidates repeated file validation logic throughout the codebase.
    
    Args:
        file_path: Path to validate
        description: Description for error message
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{description} not found: {file_path}")


def validate_directory_exists(dir_path: str, description: str = "Directory") -> None:
    """
    Validate that a directory exists, raising FileNotFoundError if not.
    
    Args:
        dir_path: Directory path to validate
        description: Description for error message
        
    Raises:
        FileNotFoundError: If directory doesn't exist
    """
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"{description} not found: {dir_path}")


def ensure_directory_exists(dir_path: str, create: bool = True) -> None:
    """
    Ensure a directory exists, optionally creating it.
    
    Args:
        dir_path: Directory path to ensure exists
        create: Whether to create the directory if it doesn't exist
        
    Raises:
        FileNotFoundError: If directory doesn't exist and create=False
    """
    if not os.path.exists(dir_path):
        if create:
            os.makedirs(dir_path, exist_ok=True)
        else:
            raise FileNotFoundError(f"Directory not found: {dir_path}")


def ensure_model_exists(model_dir: str, description: str = "Model") -> None:
    """
    Ensure model files exist in the specified directory.
    
    Common pattern for model loading validation.
    
    Args:
        model_dir: Directory containing model files
        description: Description for error message
        
    Raises:
        FileNotFoundError: If model directory or files don't exist
    """
    if not os.path.exists(model_dir):
        raise FileNotFoundError(
            f"{description} not found: {model_dir}. "
            "Run 'python train.py' first to train the model."
        )
    
    # Check for essential model files
    required_files = ['model.pkl', 'feature_extractor.pkl', 'metadata.pkl']
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_dir, f))]
    
    if missing_files:
        raise FileNotFoundError(
            f"{description} files missing in {model_dir}: {missing_files}. "
            "Run 'python train.py' first to train the model."
        )


def get_file_size_mb(file_path: str) -> float:
    """
    Get file size in megabytes.
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in MB
    """
    if not os.path.exists(file_path):
        return 0.0
    return os.path.getsize(file_path) / (1024 * 1024)


def safe_remove_file(file_path: str, silent: bool = True) -> bool:
    """
    Safely remove a file, handling errors gracefully.
    
    Args:
        file_path: Path to file to remove
        silent: If True, suppress errors
        
    Returns:
        True if file was removed, False otherwise
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        return False
    except Exception as e:
        if not silent:
            raise
        return False


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path object pointing to project root
    """
    # Assuming utils is in src/utils/, go up two levels
    return Path(__file__).parent.parent.parent


def get_data_dir() -> Path:
    """
    Get the data directory path.
    
    Returns:
        Path object pointing to data directory
    """
    return get_project_root() / "data"


def get_models_dir() -> Path:
    """
    Get the models directory path.
    
    Returns:
        Path object pointing to models directory
    """
    return get_project_root() / "models"


def list_csv_files(directory: str) -> list[str]:
    """
    List all CSV files in a directory.
    
    Args:
        directory: Directory to search
        
    Returns:
        List of CSV file paths
    """
    if not os.path.exists(directory):
        return []
    
    return [
        os.path.join(directory, f) 
        for f in os.listdir(directory) 
        if f.endswith('.csv')
    ]


def backup_file(file_path: str, backup_suffix: str = ".bak") -> Optional[str]:
    """
    Create a backup copy of a file.
    
    Args:
        file_path: Path to file to backup
        backup_suffix: Suffix to add to backup file
        
    Returns:
        Path to backup file, or None if source doesn't exist
    """
    import shutil
    
    if not os.path.exists(file_path):
        return None
    
    backup_path = file_path + backup_suffix
    shutil.copy2(file_path, backup_path)
    return backup_path