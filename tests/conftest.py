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
# GOT (GRAPH OF THOUGHT) FIXTURES
# =============================================================================
# These shared fixtures prevent the common anti-pattern of each test creating
# its own GoTManager with expensive disk I/O (~5s per creation).
#
# GUIDELINES:
# - Use fresh_got_manager when your test MODIFIES state
# - Use got_manager_with_sample_tasks for read-mostly tests (class-scoped)
# - Never create GoTManager directly in tests - use these fixtures!

import tempfile
import shutil
from pathlib import Path


@pytest.fixture
def fresh_got_manager(tmp_path):
    """
    Function-scoped fixture for a fresh, empty GoT manager.

    Use when your test needs to create/modify tasks and needs isolation.
    Each test gets its own empty manager.

    Example:
        def test_create_task(fresh_got_manager):
            task = fresh_got_manager.create_task("My task")
            assert task.title == "My task"
    """
    from cortical.got import GoTManager
    got_dir = tmp_path / ".got"
    return GoTManager(got_dir)


@pytest.fixture(scope="class")
def got_manager_with_sample_tasks(tmp_path_factory):
    """
    Class-scoped fixture with pre-populated sample tasks.

    Creates 20 tasks with edges - shared across all tests in a class.
    ~2s to create, but shared across many tests = huge time savings.

    Use for read-mostly tests that don't modify critical state.

    Provides:
        manager: GoTManager with 20 tasks
        tasks: List of created Task objects

    Example:
        class TestQueryFeatures:
            def test_filter(self, got_manager_with_sample_tasks):
                manager, tasks = got_manager_with_sample_tasks
                results = Query(manager).tasks().where(status="pending").execute()
    """
    from cortical.got import GoTManager

    temp_dir = tmp_path_factory.mktemp("got_sample")
    got_dir = temp_dir / ".got"
    manager = GoTManager(got_dir)

    # Create sample tasks with variety
    tasks = []
    priorities = ["critical", "high", "medium", "low"]
    statuses = ["pending", "in_progress", "completed"]

    for i in range(20):
        task = manager.create_task(
            f"Sample Task {i}",
            priority=priorities[i % len(priorities)]
        )
        if i % 3 == 0:
            manager.update_task(task.id, status="in_progress")
        elif i % 5 == 0:
            manager.update_task(task.id, status="completed")
        tasks.append(task)

    # Add some edges for graph tests
    for i in range(0, 15, 3):
        manager.add_edge(tasks[i].id, tasks[i+1].id, "DEPENDS_ON")
        manager.add_edge(tasks[i+1].id, tasks[i+2].id, "BLOCKS")

    return manager, tasks


@pytest.fixture(scope="class")
def got_manager_large(tmp_path_factory):
    """
    Class-scoped fixture with 100 tasks for performance testing.

    ~5s to create, but shared across all tests in a class.
    Use for performance tests that need larger datasets.

    Example:
        class TestPerformance:
            def test_query_scales(self, got_manager_large):
                manager, tasks = got_manager_large
                # Test with 100 tasks
    """
    from cortical.got import GoTManager

    temp_dir = tmp_path_factory.mktemp("got_large")
    got_dir = temp_dir / ".got"
    manager = GoTManager(got_dir)

    tasks = []
    priorities = ["critical", "high", "medium", "low"]

    for i in range(100):
        task = manager.create_task(
            f"Task {i}",
            priority=priorities[i % len(priorities)]
        )
        tasks.append(task)

    # Create chain of dependencies
    for i in range(0, 90, 3):
        manager.add_edge(tasks[i].id, tasks[i+1].id, "DEPENDS_ON")

    return manager, tasks


# =============================================================================
# COVERAGE DETECTION
# =============================================================================

@pytest.fixture(scope="session")
def running_under_coverage():
    """Fixture indicating whether tests are running under coverage."""
    return 'coverage' in sys.modules
