"""
Test Fixtures
=============

Shared test data and utilities used across test categories.

Available fixtures:
- small_corpus: Synthetic 25-document corpus for fast tests
- shared_processor: Singleton processor with full sample corpus

Usage:
    from tests.fixtures.small_corpus import get_small_corpus, get_small_processor
    from tests.fixtures.shared_processor import get_shared_processor
"""

from .small_corpus import get_small_corpus, get_small_processor, SMALL_CORPUS_DOCS
from .shared_processor import get_shared_processor

__all__ = [
    'get_small_corpus',
    'get_small_processor',
    'get_shared_processor',
    'SMALL_CORPUS_DOCS',
]
