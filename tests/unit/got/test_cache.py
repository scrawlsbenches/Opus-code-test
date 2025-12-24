"""
Tests for GoTManager entity caching layer.

Validates:
- Cache is enabled by default
- Cache can be disabled
- Cache hits/misses are tracked correctly
- Cache is populated on first read
- Cache is used on subsequent reads
- Cache is invalidated on writes (transaction commit)
- Cache is invalidated on delete
- Cache can be cleared manually
- Cache statistics are accurate
"""

import pytest
import tempfile
import shutil
from pathlib import Path

from cortical.got import GoTManager
from cortical.got.types import Task


class TestCacheBasics:
    """Test basic cache functionality."""

    @pytest.fixture
    def temp_got_dir(self):
        """Create a temporary GoT directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir) / ".got"
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_cache_enabled_by_default(self, temp_got_dir):
        """Cache should be enabled by default."""
        manager = GoTManager(temp_got_dir)
        assert manager._cache_enabled is True
        assert manager.cache_stats()['enabled'] is True

    def test_cache_can_be_disabled(self, temp_got_dir):
        """Cache can be disabled via constructor."""
        manager = GoTManager(temp_got_dir, cache_enabled=False)
        assert manager._cache_enabled is False
        assert manager.cache_stats()['enabled'] is False

    def test_initial_cache_stats(self, temp_got_dir):
        """Initial cache stats should be zeros."""
        manager = GoTManager(temp_got_dir)
        stats = manager.cache_stats()
        assert stats['hits'] == 0
        assert stats['misses'] == 0
        assert stats['size'] == 0
        assert stats['hit_rate'] == 0.0

    def test_cache_clear(self, temp_got_dir):
        """Cache clear should reset all stats."""
        manager = GoTManager(temp_got_dir)

        # Create a task to populate cache
        task = manager.create_task("Test task", priority="high")

        # Force cache population via find_tasks
        manager.find_tasks()
        stats_before = manager.cache_stats()
        assert stats_before['size'] > 0

        # Clear cache
        manager.cache_clear()
        stats_after = manager.cache_stats()
        assert stats_after['hits'] == 0
        assert stats_after['misses'] == 0
        assert stats_after['size'] == 0


class TestCacheHitsAndMisses:
    """Test cache hit/miss tracking."""

    @pytest.fixture
    def manager_with_tasks(self):
        """Create a manager with some tasks."""
        temp_dir = tempfile.mkdtemp()
        got_dir = Path(temp_dir) / ".got"
        manager = GoTManager(got_dir)

        # Create several tasks
        for i in range(5):
            manager.create_task(f"Task {i}", priority="medium")

        # Clear cache to start fresh
        manager.cache_clear()

        yield manager
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_first_read_is_cache_miss(self, manager_with_tasks):
        """First read should be a cache miss."""
        # Read all tasks (first time)
        tasks = manager_with_tasks.find_tasks()
        stats = manager_with_tasks.cache_stats()

        assert stats['misses'] == 5
        assert stats['hits'] == 0
        assert stats['size'] == 5

    def test_second_read_is_cache_hit(self, manager_with_tasks):
        """Second read should hit cache."""
        # First read
        manager_with_tasks.find_tasks()

        # Clear miss count but keep cache populated
        initial_misses = manager_with_tasks._cache_misses

        # Second read
        manager_with_tasks.find_tasks()
        stats = manager_with_tasks.cache_stats()

        # All should be hits now
        assert stats['hits'] == 5
        assert stats['misses'] == initial_misses  # Same as after first read

    def test_hit_rate_calculation(self, manager_with_tasks):
        """Hit rate should be calculated correctly."""
        # First read (5 misses)
        manager_with_tasks.find_tasks()

        # Second read (5 hits)
        manager_with_tasks.find_tasks()

        stats = manager_with_tasks.cache_stats()
        # 5 hits / 10 total = 0.5
        assert stats['hit_rate'] == 0.5


class TestCacheInvalidation:
    """Test cache invalidation on writes."""

    @pytest.fixture
    def manager_with_task(self):
        """Create a manager with one task."""
        temp_dir = tempfile.mkdtemp()
        got_dir = Path(temp_dir) / ".got"
        manager = GoTManager(got_dir)

        task = manager.create_task("Test task", priority="high")
        task_id = task.id

        # Clear cache
        manager.cache_clear()

        yield manager, task_id
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_cache_populated_by_find_tasks(self, manager_with_task):
        """find_tasks should populate cache."""
        manager, task_id = manager_with_task

        # Read via find_tasks
        tasks = manager.find_tasks()
        stats = manager.cache_stats()

        assert stats['size'] == 1
        assert task_id in manager._entity_cache

    def test_update_invalidates_cache(self, manager_with_task):
        """Updating a task should invalidate its cache entry."""
        manager, task_id = manager_with_task

        # Populate cache
        manager.find_tasks()
        assert task_id in manager._entity_cache

        # Update task
        manager.update_task(task_id, description="Updated")

        # Cache entry should be gone
        assert task_id not in manager._entity_cache

    def test_delete_invalidates_cache(self, manager_with_task):
        """Deleting a task should invalidate its cache entry."""
        manager, task_id = manager_with_task

        # Populate cache
        manager.find_tasks()
        assert task_id in manager._entity_cache

        # Delete task
        manager.delete_task(task_id, force=True)

        # Cache entry should be gone
        assert task_id not in manager._entity_cache


class TestCacheDisabled:
    """Test behavior when cache is disabled."""

    @pytest.fixture
    def manager_no_cache(self):
        """Create a manager with caching disabled."""
        temp_dir = tempfile.mkdtemp()
        got_dir = Path(temp_dir) / ".got"
        manager = GoTManager(got_dir, cache_enabled=False)

        # Create a task
        manager.create_task("Test task", priority="high")

        yield manager
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_no_cache_population(self, manager_no_cache):
        """Cache should not be populated when disabled."""
        # Read tasks
        manager_no_cache.find_tasks()
        stats = manager_no_cache.cache_stats()

        assert stats['size'] == 0
        assert stats['enabled'] is False

    def test_no_cache_hits(self, manager_no_cache):
        """No cache hits when disabled."""
        # Multiple reads
        manager_no_cache.find_tasks()
        manager_no_cache.find_tasks()
        stats = manager_no_cache.cache_stats()

        # Both hits and misses should be 0 (no tracking when disabled)
        assert stats['hits'] == 0
        assert stats['misses'] == 0


class TestCacheWithEdges:
    """Test cache works with edge entities."""

    @pytest.fixture
    def manager_with_edges(self):
        """Create a manager with tasks and edges."""
        temp_dir = tempfile.mkdtemp()
        got_dir = Path(temp_dir) / ".got"
        manager = GoTManager(got_dir)

        # Create tasks and edge
        task1 = manager.create_task("Task 1", priority="high")
        task2 = manager.create_task("Task 2", priority="medium")
        edge = manager.add_edge(task1.id, task2.id, "DEPENDS_ON")

        # Clear cache
        manager.cache_clear()

        yield manager, task1.id, task2.id, edge.id
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_edge_caching(self, manager_with_edges):
        """Edges should be cached."""
        manager, t1, t2, edge_id = manager_with_edges

        # Read edges
        edges = manager.list_edges()
        stats = manager.cache_stats()

        assert stats['size'] >= 1
        assert edge_id in manager._entity_cache

    def test_edge_second_read_hits_cache(self, manager_with_edges):
        """Second edge read should hit cache."""
        manager, t1, t2, edge_id = manager_with_edges

        # First read
        manager.list_edges()
        first_misses = manager._cache_misses

        # Second read
        manager.list_edges()

        # Should have hits
        assert manager._cache_hits > 0


class TestCacheWithMultipleEntityTypes:
    """Test cache works with different entity types."""

    @pytest.fixture
    def manager_with_entities(self):
        """Create a manager with various entity types."""
        temp_dir = tempfile.mkdtemp()
        got_dir = Path(temp_dir) / ".got"
        manager = GoTManager(got_dir)

        # Create different entity types
        task = manager.create_task("Test task", priority="high")
        decision = manager.create_decision("Test decision", rationale="Testing")
        sprint = manager.create_sprint("Test sprint", number=99)

        # Clear cache
        manager.cache_clear()

        yield manager
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_tasks_cached(self, manager_with_entities):
        """Tasks should be cached correctly."""
        manager_with_entities.find_tasks()
        manager_with_entities.find_tasks()

        stats = manager_with_entities.cache_stats()
        assert stats['hits'] > 0

    def test_decisions_cached(self, manager_with_entities):
        """Decisions should be cached correctly."""
        manager_with_entities.list_decisions()
        manager_with_entities.list_decisions()

        stats = manager_with_entities.cache_stats()
        assert stats['hits'] > 0

    def test_sprints_cached(self, manager_with_entities):
        """Sprints should be cached correctly."""
        manager_with_entities.list_sprints()
        manager_with_entities.list_sprints()

        stats = manager_with_entities.cache_stats()
        assert stats['hits'] > 0
