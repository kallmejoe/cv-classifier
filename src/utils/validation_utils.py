"""
Validation decorators and utilities.

This module provides decorators for common validation patterns,
reducing repetitive validation code throughout the codebase.
"""

from functools import wraps
from typing import Any, Callable, Optional


def require_fitted(method: Callable) -> Callable:
    """
    Decorator to ensure a model or feature extractor is fitted before use.
    
    Replaces repeated validation checks like:
        if not self._is_fitted:
            raise ValueError("FeatureExtractor must be fitted first")
    
    Usage:
        @require_fitted
        def transform(self, texts):
            # Method implementation
    
    Args:
        method: Method to decorate
        
    Returns:
        Wrapped method with fitted check
        
    Raises:
        ValueError: If object is not fitted (no _is_fitted attribute or False)
    """
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, '_is_fitted'):
            raise AttributeError(
                f"{self.__class__.__name__} doesn't have _is_fitted attribute. "
                "Add self._is_fitted = False in __init__"
            )
        
        if not self._is_fitted:
            raise ValueError(
                f"{self.__class__.__name__} must be fitted before calling {method.__name__}(). "
                "Call fit() first."
            )
        
        return method(self, *args, **kwargs)
    
    return wrapper


def validate_not_empty(param_name: str = "value"):
    """
    Decorator factory to validate that a parameter is not None or empty.
    
    Usage:
        @validate_not_empty("text")
        def process_text(self, text):
            # text is guaranteed to be non-empty
    
    Args:
        param_name: Name of parameter to validate (first arg after self)
        
    Returns:
        Decorator function
    """
    def decorator(method: Callable) -> Callable:
        @wraps(method)
        def wrapper(self, value, *args, **kwargs):
            if value is None or (isinstance(value, str) and not value.strip()):
                raise ValueError(
                    f"{param_name} cannot be None or empty in {method.__name__}()"
                )
            return method(self, value, *args, **kwargs)
        return wrapper
    return decorator


def validate_type(*expected_types):
    """
    Decorator factory to validate parameter types.
    
    Usage:
        @validate_type(str, int)
        def process(self, value):
            # value is guaranteed to be str or int
    
    Args:
        *expected_types: Expected type(s) for first parameter after self
        
    Returns:
        Decorator function
    """
    def decorator(method: Callable) -> Callable:
        @wraps(method)
        def wrapper(self, value, *args, **kwargs):
            if not isinstance(value, expected_types):
                type_names = [t.__name__ for t in expected_types]
                raise TypeError(
                    f"Expected {' or '.join(type_names)}, got {type(value).__name__} "
                    f"in {method.__name__}()"
                )
            return method(self, value, *args, **kwargs)
        return wrapper
    return decorator


def validate_positive(param_name: str = "value"):
    """
    Decorator to validate that a numeric parameter is positive.
    
    Usage:
        @validate_positive("count")
        def set_count(self, count):
            # count is guaranteed to be > 0
    
    Args:
        param_name: Name of parameter to validate
        
    Returns:
        Decorator function
    """
    def decorator(method: Callable) -> Callable:
        @wraps(method)
        def wrapper(self, value, *args, **kwargs):
            if not isinstance(value, (int, float)):
                raise TypeError(
                    f"{param_name} must be numeric in {method.__name__}()"
                )
            if value <= 0:
                raise ValueError(
                    f"{param_name} must be positive in {method.__name__}(), got {value}"
                )
            return method(self, value, *args, **kwargs)
        return wrapper
    return decorator


def validate_range(min_val: Optional[float] = None, max_val: Optional[float] = None):
    """
    Decorator factory to validate that a numeric parameter is in a range.
    
    Usage:
        @validate_range(0.0, 1.0)
        def set_threshold(self, threshold):
            # threshold is guaranteed to be in [0.0, 1.0]
    
    Args:
        min_val: Minimum value (inclusive), or None for no minimum
        max_val: Maximum value (inclusive), or None for no maximum
        
    Returns:
        Decorator function
    """
    def decorator(method: Callable) -> Callable:
        @wraps(method)
        def wrapper(self, value, *args, **kwargs):
            if not isinstance(value, (int, float)):
                raise TypeError(
                    f"Value must be numeric in {method.__name__}()"
                )
            
            if min_val is not None and value < min_val:
                raise ValueError(
                    f"Value must be >= {min_val} in {method.__name__}(), got {value}"
                )
            
            if max_val is not None and value > max_val:
                raise ValueError(
                    f"Value must be <= {max_val} in {method.__name__}(), got {value}"
                )
            
            return method(self, value, *args, **kwargs)
        return wrapper
    return decorator


def handle_empty_input(default_return: Any = None):
    """
    Decorator to handle empty/None input gracefully.
    
    Returns default value if input is None or empty string.
    
    Usage:
        @handle_empty_input(default_return=[])
        def process_list(self, items):
            # Returns [] if items is None or empty
    
    Args:
        default_return: Value to return for empty input
        
    Returns:
        Decorator function
    """
    def decorator(method: Callable) -> Callable:
        @wraps(method)
        def wrapper(self, value, *args, **kwargs):
            if value is None or (isinstance(value, str) and not value.strip()):
                return default_return
            return method(self, value, *args, **kwargs)
        return wrapper
    return decorator


def cache_result(method: Callable) -> Callable:
    """
    Simple caching decorator for methods.
    
    Caches the result in self._cache[method_name].
    
    Usage:
        @cache_result
        def expensive_computation(self):
            # Result is cached after first call
    
    Args:
        method: Method to decorate
        
    Returns:
        Wrapped method with caching
    """
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        # Initialize cache if needed
        if not hasattr(self, '_cache'):
            self._cache = {}
        
        cache_key = (method.__name__, args, tuple(sorted(kwargs.items())))
        
        if cache_key not in self._cache:
            self._cache[cache_key] = method(self, *args, **kwargs)
        
        return self._cache[cache_key]
    
    return wrapper


def deprecated(message: str = ""):
    """
    Decorator to mark methods as deprecated.
    
    Usage:
        @deprecated("Use new_method() instead")
        def old_method(self):
            pass
    
    Args:
        message: Deprecation message
        
    Returns:
        Decorator function
    """
    def decorator(method: Callable) -> Callable:
        @wraps(method)
        def wrapper(*args, **kwargs):
            import warnings
            warning_msg = f"{method.__name__}() is deprecated."
            if message:
                warning_msg += f" {message}"
            warnings.warn(warning_msg, DeprecationWarning, stacklevel=2)
            return method(*args, **kwargs)
        return wrapper
    return decorator