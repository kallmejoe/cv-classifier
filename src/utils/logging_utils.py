"""
Logging utilities for configurable output management.

This module provides a simple logging system to replace the 167+ print statements
throughout the codebase with configurable logging levels and output formatting.
"""

import sys
from typing import Optional, TextIO
from enum import Enum
from .display_utils import print_header, print_step, print_distribution_table


class LogLevel(Enum):
    """Log level enumeration."""
    SILENT = 0
    ERROR = 1
    WARNING = 2
    INFO = 3
    DEBUG = 4


class Logger:
    """
    Simple configurable logger for cv-classifier project.
    
    Provides a clean interface to replace scattered print statements
    with configurable verbosity levels and consistent formatting.
    """
    
    def __init__(
        self, 
        level: LogLevel = LogLevel.INFO, 
        output: TextIO = sys.stdout,
        name: str = "cv-classifier"
    ):
        """
        Initialize logger.
        
        Args:
            level: Minimum log level to output
            output: Output stream (stdout, stderr, or file)
            name: Logger name for identification
        """
        self.level = level
        self.output = output
        self.name = name
    
    def set_level(self, level: LogLevel) -> None:
        """Set the logging level."""
        self.level = level
    
    def _should_log(self, level: LogLevel) -> bool:
        """Check if message should be logged at given level."""
        return level.value <= self.level.value
    
    def _log(self, level: LogLevel, message: str, end: str = "\n") -> None:
        """Internal logging method."""
        if self._should_log(level):
            print(message, file=self.output, end=end, flush=True)
    
    def error(self, message: str) -> None:
        """Log an error message."""
        self._log(LogLevel.ERROR, f"ERROR: {message}")
    
    def warning(self, message: str) -> None:
        """Log a warning message.""" 
        self._log(LogLevel.WARNING, f"WARNING: {message}")
    
    def info(self, message: str) -> None:
        """Log an info message."""
        self._log(LogLevel.INFO, message)
    
    def debug(self, message: str) -> None:
        """Log a debug message."""
        self._log(LogLevel.DEBUG, f"DEBUG: {message}")
    
    def progress(self, message: str) -> None:
        """Log a progress message (same as info but semantically different)."""
        self._log(LogLevel.INFO, message)
    
    def header(self, title: str, width: int = 70) -> None:
        """Log a formatted header (only if INFO level or higher)."""
        if self._should_log(LogLevel.INFO):
            print_header(title, width)
    
    def step(self, step_num: int, total_steps: int, description: str) -> None:
        """Log a formatted step indicator."""
        if self._should_log(LogLevel.INFO):
            print_step(step_num, total_steps, description)
    
    def stats(self, stats_dict: dict, title: str = "Statistics") -> None:
        """Log formatted statistics."""
        if self._should_log(LogLevel.INFO):
            self.info(f"\n{title}:")
            for key, value in stats_dict.items():
                if isinstance(value, float):
                    self.info(f"  {key:25s}: {value:.2f}")
                elif isinstance(value, int):
                    self.info(f"  {key:25s}: {value:,}")
                else:
                    self.info(f"  {key:25s}: {value}")
    
    def file_saved(self, file_path: str, description: str = "File") -> None:
        """Log file save notification."""
        self.info(f"{description} saved to: {file_path}")


# Global logger instance
_global_logger: Optional[Logger] = None


def get_logger() -> Logger:
    """Get or create the global logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = Logger()
    return _global_logger


def set_log_level(level: LogLevel) -> None:
    """Set the global logging level."""
    get_logger().set_level(level)


def configure_logging(verbose: bool = True, debug: bool = False) -> Logger:
    """
    Configure logging based on common verbosity flags.
    
    Args:
        verbose: Enable verbose output (INFO level)
        debug: Enable debug output (DEBUG level)
        
    Returns:
        Configured logger instance
    """
    global _global_logger
    
    if debug:
        level = LogLevel.DEBUG
    elif verbose:
        level = LogLevel.INFO
    else:
        level = LogLevel.WARNING
    
    _global_logger = Logger(level=level)
    return _global_logger


# Convenience functions for global logger
def error(message: str) -> None:
    """Log error message using global logger."""
    get_logger().error(message)


def warning(message: str) -> None:
    """Log warning message using global logger."""
    get_logger().warning(message)


def info(message: str) -> None:
    """Log info message using global logger."""
    get_logger().info(message)


def debug(message: str) -> None:
    """Log debug message using global logger."""
    get_logger().debug(message)


def progress(message: str) -> None:
    """Log progress message using global logger."""
    get_logger().progress(message)


def header(title: str, width: int = 70) -> None:
    """Log formatted header using global logger."""
    get_logger().header(title, width)


def step(step_num: int, total_steps: int, description: str) -> None:
    """Log formatted step using global logger."""
    get_logger().step(step_num, total_steps, description)


def stats(stats_dict: dict, title: str = "Statistics") -> None:
    """Log formatted statistics using global logger."""
    get_logger().stats(stats_dict, title)


def file_saved(file_path: str, description: str = "File") -> None:
    """Log file save notification using global logger."""
    get_logger().file_saved(file_path, description)