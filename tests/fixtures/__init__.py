"""
Test Fixtures
=============

Shared test data and utilities used across test categories.

Available fixtures:
- test_bootstrap: Container wiring for tests (DI/IoC)
- small_corpus: Synthetic 25-document corpus for fast tests
- shared_processor: Singleton processor with full sample corpus

Usage:
    # Container for DI tests
    from tests.fixtures.test_bootstrap import create_test_container

    # Corpus fixtures
    from tests.fixtures.small_corpus import get_small_corpus, get_small_processor
    from tests.fixtures.shared_processor import get_shared_processor
"""

from .test_bootstrap import create_test_container
from .small_corpus import get_small_corpus, get_small_processor, SMALL_CORPUS_DOCS
from .shared_processor import get_shared_processor

__all__ = [
    'create_test_container',
    'get_small_corpus',
    'get_small_processor',
    'get_shared_processor',
    'SMALL_CORPUS_DOCS',
]
