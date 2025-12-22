"""
Query Utilities Module
=====================

Shared utility functions for query modules.

This module provides:
- TF-IDF score computation helpers
- Score normalization utilities
- Test file detection
"""

from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..minicolumn import Minicolumn


def get_tfidf_score(col: 'Minicolumn', doc_id: Optional[str] = None) -> float:
    """
    Get TF-IDF score for a term, optionally per-document.

    This helper encapsulates the common pattern of getting per-document
    TF-IDF when available, falling back to global TF-IDF otherwise.

    Args:
        col: Minicolumn for the term
        doc_id: Optional document ID for per-document TF-IDF

    Returns:
        TF-IDF score (per-document if doc_id provided and available, else global)
    """
    if doc_id and hasattr(col, 'tfidf_per_doc'):
        return col.tfidf_per_doc.get(doc_id, col.tfidf)
    return col.tfidf


def normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize scores to 0-1 range by dividing by maximum.

    Args:
        scores: Dictionary mapping keys to scores

    Returns:
        New dictionary with normalized scores (0-1 range).
        If all scores are 0 or dict is empty, returns original scores.
    """
    if not scores:
        return scores

    max_score = max(scores.values())
    if max_score == 0:
        return scores

    return {key: score / max_score for key, score in scores.items()}


def is_test_file(doc_id: str) -> bool:
    """
    Detect if a document ID represents a test file.

    Checks for common test file patterns:
    - Path contains 'tests/' or 'test/'
    - Filename starts with 'test_' or ends with '_test.py'
    - Path contains 'mock' or 'fixture'

    Args:
        doc_id: Document identifier (typically a file path)

    Returns:
        True if the document appears to be a test file

    Example:
        >>> is_test_file('tests/test_processor.py')
        True
        >>> is_test_file('cortical/processor.py')
        False
        >>> is_test_file('tests/fixtures/data.json')
        True
    """
    doc_lower = doc_id.lower()

    # Check path components
    if '/tests/' in doc_lower or '/test/' in doc_lower:
        return True

    # Check filename patterns
    filename = doc_lower.split('/')[-1] if '/' in doc_lower else doc_lower
    if filename.startswith('test_') or filename.endswith('_test.py'):
        return True
    if 'mock' in filename or 'fixture' in filename:
        return True

    return False
