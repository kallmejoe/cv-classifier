"""Text preprocessing module for resume classification.

This module provides text cleaning and normalization functions for resume text.
Consolidates preprocessing logic that was previously scattered across multiple files.
"""

import re
from typing import List, Set

# Common English stopwords (expanded set)
STOPWORDS: Set[str] = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
    'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
    'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
    'she', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your',
    'his', 'our', 'their', 'what', 'which', 'who', 'whom', 'when', 'where',
    'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
    'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
    'so', 'than', 'too', 'very', 'just', 'also', 'now', 'here', 'there',
    'then', 'once', 'if', 'about', 'into', 'through', 'during', 'before',
    'after', 'above', 'below', 'between', 'under', 'again', 'further',
    'while', 'any', 'because', 'being', 'having', 'doing', 'am', 'up',
    'down', 'out', 'off', 'over', 'under', 'get', 'got', 'etc', 'ie',
    'eg', 'via', 'vs', 'per', 'de', 'la', 'le', 'les'
}


def clean_html_tags(text: str) -> str:
    """
    Remove HTML tags from text.

    Args:
        text: Input text that may contain HTML tags

    Returns:
        Text with HTML tags removed
    """
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Decode common HTML entities
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&quot;', '"')
    text = text.replace('&#39;', "'")
    return text


def preprocess_text(
    text: str,
    remove_stops: bool = True,
    min_token_length: int = 2,
    remove_html: bool = True
) -> str:
    """
    Clean and normalize text for feature extraction.

    This function performs the following transformations:
    1. Remove HTML tags (optional)
    2. Remove URLs
    3. Remove email addresses
    4. Convert to lowercase
    5. Remove non-alphabetic characters
    6. Remove stopwords
    7. Remove short tokens 

    Args:
        text: Raw resume text
        remove_stops: Whether to remove stopwords (default: True)
        min_token_length: Minimum token length to keep (default: 2)
        remove_html: Whether to remove HTML tags (default: True)

    Returns:
        Cleaned and normalized text
    """
    if remove_html:
        text = clean_html_tags(text)

    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

    text = re.sub(r'\S+@\S+\.\S+', ' ', text)

    text = text.lower()

    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    text = ' '.join(text.split())

    tokens = text.split()

    if remove_stops:
        tokens = [t for t in tokens if t not in STOPWORDS]

    tokens = [t for t in tokens if len(t) >= min_token_length]

    return ' '.join(tokens)


def preprocess_corpus(
    texts: List[str],
    remove_stops: bool = True,
    min_token_length: int = 2,
    remove_html: bool = True
) -> List[str]:
    """
    Apply preprocessing to multiple texts.

    Args:
        texts: List of raw resume texts
        remove_stops: Whether to remove stopwords (default: True)
        min_token_length: Minimum token length to keep (default: 2)
        remove_html: Whether to remove HTML tags (default: True)

    Returns:
        List of cleaned and normalized texts
    """
    return [
        preprocess_text(text, remove_stops, min_token_length, remove_html)
        for text in texts
    ]
