"""
Pytest Configuration and Shared Fixtures
=========================================

This module configures pytest for the Cortical Text Processor test suite.
It provides:
- Path setup for importing cortical modules
- Custom markers for test categorization
- Shared fixtures available to all tests

Test Categories (markers):
- @pytest.mark.unit: Fast, isolated unit tests
- @pytest.mark.integration: Component interaction tests
- @pytest.mark.smoke: Quick sanity checks
- @pytest.mark.performance: Timing-based tests (skip under coverage)
- @pytest.mark.regression: Bug-specific regression tests
- @pytest.mark.behavioral: User workflow and quality tests
- @pytest.mark.slow: Tests that take > 5 seconds

Usage:
    # Run only unit tests
    pytest -m unit

    # Run everything except slow tests
    pytest -m "not slow"

    # Run performance tests without coverage
    pytest -m performance --no-cov
"""

import os
import sys

import pytest


# =============================================================================
# PATH SETUP
# =============================================================================

# Ensure the cortical package is importable from any test directory
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


# =============================================================================
# PYTEST MARKERS
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "unit: Fast, isolated unit tests (< 1s each)"
    )
    config.addinivalue_line(
        "markers", "integration: Component interaction tests"
    )
    config.addinivalue_line(
        "markers", "smoke: Quick sanity checks (< 10s total)"
    )
    config.addinivalue_line(
        "markers", "performance: Timing-based tests (run without coverage)"
    )
    config.addinivalue_line(
        "markers", "regression: Bug-specific regression tests"
    )
    config.addinivalue_line(
        "markers", "behavioral: User workflow and quality tests"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take > 5 seconds"
    )


# =============================================================================
# SHARED FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def small_processor():
    """
    Session-scoped fixture providing a processor with small synthetic corpus.

    This is fast to create (~1s) and suitable for most tests.
    """
    from tests.fixtures.small_corpus import get_small_processor
    return get_small_processor()


@pytest.fixture(scope="session")
def shared_processor():
    """
    Session-scoped fixture providing a processor with full sample corpus.

    This is slower to create (~10-20s) but provides realistic test data.
    Use sparingly - prefer small_processor for most tests.
    """
    from tests.fixtures.shared_processor import get_shared_processor
    return get_shared_processor()


@pytest.fixture
def fresh_processor():
    """
    Function-scoped fixture providing a fresh, empty processor.

    Use when tests need to modify processor state.
    """
    from cortical import CorticalTextProcessor
    return CorticalTextProcessor()


@pytest.fixture
def small_corpus_docs():
    """
    Fixture providing the raw small corpus document dictionary.
    """
    from tests.fixtures.small_corpus import SMALL_CORPUS_DOCS
    return SMALL_CORPUS_DOCS.copy()


# =============================================================================
# TEST COLLECTION HOOKS
# =============================================================================

def pytest_collection_modifyitems(config, items):
    """
    Automatically mark tests based on their location.

    Tests in tests/unit/ get @pytest.mark.unit, etc.
    """
    for item in items:
        # Get the test file path relative to tests/
        test_path = str(item.fspath)

        if '/unit/' in test_path or '\\unit\\' in test_path:
            item.add_marker(pytest.mark.unit)
        elif '/integration/' in test_path or '\\integration\\' in test_path:
            item.add_marker(pytest.mark.integration)
        elif '/smoke/' in test_path or '\\smoke\\' in test_path:
            item.add_marker(pytest.mark.smoke)
        elif '/performance/' in test_path or '\\performance\\' in test_path:
            item.add_marker(pytest.mark.performance)
            # Performance tests should skip under coverage
            if 'coverage' in sys.modules:
                item.add_marker(pytest.mark.skip(
                    reason="Performance tests skip under coverage (10x+ overhead)"
                ))
        elif '/regression/' in test_path or '\\regression\\' in test_path:
            item.add_marker(pytest.mark.regression)
        elif '/behavioral/' in test_path or '\\behavioral\\' in test_path:
            item.add_marker(pytest.mark.behavioral)


# =============================================================================
# COVERAGE DETECTION
# =============================================================================

@pytest.fixture(scope="session")
def running_under_coverage():
    """Fixture indicating whether tests are running under coverage."""
    return 'coverage' in sys.modules
