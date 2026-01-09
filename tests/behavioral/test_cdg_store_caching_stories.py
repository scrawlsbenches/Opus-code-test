"""
Behavioral tests for CDGStore caching.

These tests define the expected caching behavior at the storage layer.
Caching at CDGStore means ALL entity access benefits from caching,
not just specific domain methods.

Uses a minimal test entity - CDGStore is generic and shouldn't
know about domain-specific types like Task.

Uses InMemoryFileSystem for fast tests - no disk I/O.
"""

import pytest
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

from cortical.cdg.storage import CDGStore
from cortical.common.filesystem import InMemoryFileSystem


@dataclass
class SampleEntity:
    """Minimal entity for CDGStore cache testing."""
    id: str
    name: str
    entity_type: str = "sample_entity"
    version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "entity_type": self.entity_type,
            "version": self.version,
        }

    def bump_version(self) -> None:
        """Increment entity version (required by CDGStore.write)."""
        self.version += 1

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SampleEntity":
        return cls(
            id=data["id"],
            name=data["name"],
            entity_type=data.get("entity_type", "sample_entity"),
            version=data.get("version", 1),
        )


def sample_entity_factory(data: Dict[str, Any]) -> SampleEntity:
    """Factory function for CDGStore to create SampleEntity instances."""
    return SampleEntity.from_dict(data)


class TestCDGStoreCacheReads:
    """Cache behavior for read operations."""

    @pytest.fixture
    def store(self):
        """Create CDGStore with in-memory filesystem and caching enabled."""
        fs = InMemoryFileSystem(Path("/store"))
        return CDGStore(
            filesystem=fs,
            entity_factory=sample_entity_factory,
            cache_enabled=True,
        )

    def test_first_read_is_cache_miss(self, store):
        """First read of an entity should be a cache miss."""
        store.write(SampleEntity(id="TE-001", name="Test"))
        store.cache_clear()

        store.read("TE-001")

        stats = store.cache_stats()
        assert stats['misses'] == 1
        assert stats['hits'] == 0

    def test_second_read_is_cache_hit(self, store):
        """Second read of the same entity should be a cache hit."""
        store.write(SampleEntity(id="TE-001", name="Test"))
        store.cache_clear()

        store.read("TE-001")  # Miss
        store.read("TE-001")  # Hit

        stats = store.cache_stats()
        assert stats['hits'] == 1
        assert stats['misses'] == 1

    def test_cached_reads_return_equal_isolated_copies(self, store):
        """Cached reads return equal data but isolated copies for transaction safety.

        This ensures proper isolation - modifications to one copy
        should not affect other reads or the cached original.
        """
        store.write(SampleEntity(id="TE-001", name="Test"))
        store.cache_clear()

        first = store.read("TE-001")
        second = store.read("TE-001")

        # Data should be equal
        assert first.id == second.id
        assert first.name == second.name
        assert first.version == second.version

        # But objects should be isolated copies (not same reference)
        assert first is not second

        # Modifying one should not affect the other (isolation)
        first.name = "Modified"
        assert second.name == "Test"


class TestCDGStoreCacheIterEntities:
    """Cache behavior for iter_entities."""

    @pytest.fixture
    def store(self):
        fs = InMemoryFileSystem(Path("/store"))
        return CDGStore(
            filesystem=fs,
            entity_factory=sample_entity_factory,
            cache_enabled=True,
        )

    def test_iter_entities_populates_cache(self, store):
        """iter_entities should populate the cache."""
        for i in range(3):
            store.write(SampleEntity(id=f"TE-00{i}", name=f"Test {i}"))
        store.cache_clear()

        list(store.iter_entities(prefix="TE-"))

        stats = store.cache_stats()
        assert stats['size'] == 3

    def test_read_after_iter_is_cache_hit(self, store):
        """Reading an entity after iter_entities should be a cache hit."""
        store.write(SampleEntity(id="TE-001", name="Test"))
        store.cache_clear()

        list(store.iter_entities(prefix="TE-"))
        hits_before = store.cache_stats()['hits']

        store.read("TE-001")

        assert store.cache_stats()['hits'] == hits_before + 1


class TestCDGStoreCacheInvalidation:
    """Cache invalidation on writes and deletes."""

    @pytest.fixture
    def store(self):
        fs = InMemoryFileSystem(Path("/store"))
        return CDGStore(
            filesystem=fs,
            entity_factory=sample_entity_factory,
            cache_enabled=True,
        )

    def test_write_invalidates_cache(self, store):
        """Writing an entity should invalidate its cache entry."""
        store.write(SampleEntity(id="TE-001", name="Original"))
        store.read("TE-001")  # Cache it

        store.write(SampleEntity(id="TE-001", name="Updated"))
        result = store.read("TE-001")

        assert result.name == "Updated"

    def test_delete_invalidates_cache(self, store):
        """Deleting an entity should invalidate its cache entry."""
        store.write(SampleEntity(id="TE-001", name="Test"))
        store.read("TE-001")  # Cache it
        assert store.cache_stats()['size'] == 1

        store.delete("TE-001")

        assert store.cache_stats()['size'] == 0
        assert store.read("TE-001") is None


class TestCDGStoreCacheStatistics:
    """Cache statistics and observability."""

    @pytest.fixture
    def store(self):
        fs = InMemoryFileSystem(Path("/store"))
        return CDGStore(
            filesystem=fs,
            entity_factory=sample_entity_factory,
            cache_enabled=True,
        )

    def test_cache_stats_initial_state(self, store):
        """Cache stats should show zero activity initially."""
        stats = store.cache_stats()

        assert stats['hits'] == 0
        assert stats['misses'] == 0
        assert stats['size'] == 0
        assert stats['enabled'] is True

    def test_cache_clear_resets_stats(self, store):
        """Clearing cache should reset statistics."""
        store.write(SampleEntity(id="TE-001", name="Test"))
        store.read("TE-001")
        store.read("TE-001")

        store.cache_clear()

        stats = store.cache_stats()
        assert stats['hits'] == 0
        assert stats['misses'] == 0
        assert stats['size'] == 0

    def test_hit_rate_calculation(self, store):
        """Hit rate should be calculated correctly."""
        store.write(SampleEntity(id="TE-001", name="Test"))
        store.cache_clear()

        store.read("TE-001")  # Miss
        store.read("TE-001")  # Hit
        store.read("TE-001")  # Hit
        store.read("TE-001")  # Hit

        assert store.cache_stats()['hit_rate'] == 0.75


class TestCDGStoreCacheDisabled:
    """Behavior when caching is disabled."""

    def test_cache_disabled_no_caching(self):
        """With cache disabled, reads should not be cached."""
        fs = InMemoryFileSystem(Path("/store"))
        store = CDGStore(
            filesystem=fs,
            entity_factory=sample_entity_factory,
            cache_enabled=False,
        )
        store.write(SampleEntity(id="TE-001", name="Test"))

        store.read("TE-001")
        store.read("TE-001")

        stats = store.cache_stats()
        assert stats['enabled'] is False
        assert stats['size'] == 0

    def test_cache_disabled_reads_still_work(self):
        """With cache disabled, reads should still return correct data."""
        fs = InMemoryFileSystem(Path("/store"))
        store = CDGStore(
            filesystem=fs,
            entity_factory=sample_entity_factory,
            cache_enabled=False,
        )
        store.write(SampleEntity(id="TE-001", name="Test"))

        result = store.read("TE-001")

        assert result.id == "TE-001"
        assert result.name == "Test"
