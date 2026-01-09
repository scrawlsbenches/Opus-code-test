"""
Unit tests for BTreeIndex range query support.

Tests the btree index implementation for:
- Basic insert/remove operations
- Equality lookups
- Range queries (GT, GTE, LT, LTE)
- Serialization/deserialization
- Integration with CDGIndexManager

See: docs/design/cdg-query-language.md
See: cortical/cdg/btree.py
"""

import pytest
from cortical.cdg.btree import BTreeIndex


class TestBTreeIndexBasics:
    """Test basic BTreeIndex operations."""

    def test_create_empty_index(self):
        """Empty index should have length 0."""
        btree = BTreeIndex()
        assert len(btree) == 0
        assert not btree  # Empty is falsy

    def test_insert_single_entry(self):
        """Insert a single entry."""
        btree = BTreeIndex()
        btree.insert("2026-01-01", "E-001")

        assert len(btree) == 1
        assert btree

    def test_insert_multiple_entries_same_key(self):
        """Multiple entries with same key."""
        btree = BTreeIndex()
        btree.insert("2026-01-01", "E-001")
        btree.insert("2026-01-01", "E-002")
        btree.insert("2026-01-01", "E-003")

        assert len(btree) == 3
        result = btree.lookup_eq("2026-01-01")
        assert result == {"E-001", "E-002", "E-003"}

    def test_insert_multiple_entries_different_keys(self):
        """Multiple entries with different keys."""
        btree = BTreeIndex()
        btree.insert("2026-01-01", "E-001")
        btree.insert("2026-01-05", "E-002")
        btree.insert("2026-01-10", "E-003")

        assert len(btree) == 3

    def test_remove_entry(self):
        """Remove an entry."""
        btree = BTreeIndex()
        btree.insert("2026-01-01", "E-001")
        btree.insert("2026-01-01", "E-002")

        result = btree.remove("2026-01-01", "E-001")
        assert result is True
        assert len(btree) == 1
        assert btree.lookup_eq("2026-01-01") == {"E-002"}

    def test_remove_nonexistent_entry(self):
        """Remove a nonexistent entry returns False."""
        btree = BTreeIndex()
        btree.insert("2026-01-01", "E-001")

        result = btree.remove("2026-01-01", "E-999")
        assert result is False
        assert len(btree) == 1

    def test_remove_last_entry_for_key(self):
        """Removing last entry for a key removes the key."""
        btree = BTreeIndex()
        btree.insert("2026-01-01", "E-001")

        btree.remove("2026-01-01", "E-001")
        assert len(btree) == 0
        assert btree.lookup_eq("2026-01-01") == set()

    def test_clear(self):
        """Clear removes all entries."""
        btree = BTreeIndex()
        btree.insert("2026-01-01", "E-001")
        btree.insert("2026-01-05", "E-002")

        btree.clear()
        assert len(btree) == 0


class TestBTreeIndexEqualityLookup:
    """Test equality lookups."""

    def test_lookup_eq_found(self):
        """Lookup existing key."""
        btree = BTreeIndex()
        btree.insert("pending", "E-001")
        btree.insert("pending", "E-002")
        btree.insert("completed", "E-003")

        result = btree.lookup_eq("pending")
        assert result == {"E-001", "E-002"}

    def test_lookup_eq_not_found(self):
        """Lookup nonexistent key returns empty set."""
        btree = BTreeIndex()
        btree.insert("pending", "E-001")

        result = btree.lookup_eq("completed")
        assert result == set()

    def test_lookup_eq_returns_copy(self):
        """Lookup returns a copy, not the original set."""
        btree = BTreeIndex()
        btree.insert("pending", "E-001")

        result = btree.lookup_eq("pending")
        result.add("E-999")

        # Original should be unchanged
        assert btree.lookup_eq("pending") == {"E-001"}


class TestBTreeIndexRangeQueries:
    """Test range query operations."""

    @pytest.fixture
    def date_btree(self):
        """BTree with date keys."""
        btree = BTreeIndex()
        btree.insert("2026-01-01", "E-001")
        btree.insert("2026-01-05", "E-002")
        btree.insert("2026-01-10", "E-003")
        btree.insert("2026-01-15", "E-004")
        btree.insert("2026-01-20", "E-005")
        return btree

    def test_lookup_gt(self, date_btree):
        """Greater than lookup."""
        result = date_btree.lookup_gt("2026-01-10")
        assert result == {"E-004", "E-005"}

    def test_lookup_gt_none_match(self, date_btree):
        """Greater than with no matches."""
        result = date_btree.lookup_gt("2026-01-25")
        assert result == set()

    def test_lookup_gte(self, date_btree):
        """Greater than or equal lookup."""
        result = date_btree.lookup_gte("2026-01-10")
        assert result == {"E-003", "E-004", "E-005"}

    def test_lookup_lt(self, date_btree):
        """Less than lookup."""
        result = date_btree.lookup_lt("2026-01-10")
        assert result == {"E-001", "E-002"}

    def test_lookup_lt_none_match(self, date_btree):
        """Less than with no matches."""
        result = date_btree.lookup_lt("2025-12-01")
        assert result == set()

    def test_lookup_lte(self, date_btree):
        """Less than or equal lookup."""
        result = date_btree.lookup_lte("2026-01-10")
        assert result == {"E-001", "E-002", "E-003"}

    def test_lookup_range_inclusive(self, date_btree):
        """Range query with inclusive bounds."""
        result = date_btree.lookup_range(
            start_key="2026-01-05",
            end_key="2026-01-15",
            start_inclusive=True,
            end_inclusive=True
        )
        assert result == {"E-002", "E-003", "E-004"}

    def test_lookup_range_exclusive(self, date_btree):
        """Range query with exclusive bounds."""
        result = date_btree.lookup_range(
            start_key="2026-01-05",
            end_key="2026-01-15",
            start_inclusive=False,
            end_inclusive=False
        )
        assert result == {"E-003"}

    def test_lookup_range_no_lower_bound(self, date_btree):
        """Range query with no lower bound."""
        result = date_btree.lookup_range(
            start_key=None,
            end_key="2026-01-10",
            end_inclusive=True
        )
        assert result == {"E-001", "E-002", "E-003"}

    def test_lookup_range_no_upper_bound(self, date_btree):
        """Range query with no upper bound."""
        result = date_btree.lookup_range(
            start_key="2026-01-10",
            end_key=None,
            start_inclusive=True
        )
        assert result == {"E-003", "E-004", "E-005"}

    def test_lookup_range_all(self, date_btree):
        """Range query for all entries."""
        result = date_btree.lookup_range()
        assert result == {"E-001", "E-002", "E-003", "E-004", "E-005"}

    def test_get_all(self, date_btree):
        """Get all entity IDs."""
        result = date_btree.get_all()
        assert result == {"E-001", "E-002", "E-003", "E-004", "E-005"}


class TestBTreeIndexNumericKeys:
    """Test with numeric keys (integers)."""

    @pytest.fixture
    def numeric_btree(self):
        """BTree with integer keys (priority levels)."""
        btree = BTreeIndex()
        btree.insert(1, "E-LOW1")
        btree.insert(1, "E-LOW2")
        btree.insert(5, "E-MED")
        btree.insert(10, "E-HIGH")
        btree.insert(10, "E-HIGH2")
        return btree

    def test_numeric_equality(self, numeric_btree):
        """Equality lookup with numeric keys."""
        result = numeric_btree.lookup_eq(10)
        assert result == {"E-HIGH", "E-HIGH2"}

    def test_numeric_gt(self, numeric_btree):
        """Greater than with numeric keys."""
        result = numeric_btree.lookup_gt(1)
        assert result == {"E-MED", "E-HIGH", "E-HIGH2"}

    def test_numeric_lt(self, numeric_btree):
        """Less than with numeric keys."""
        result = numeric_btree.lookup_lt(10)
        assert result == {"E-LOW1", "E-LOW2", "E-MED"}


class TestBTreeIndexSerialization:
    """Test serialization/deserialization."""

    def test_to_dict_empty(self):
        """Serialize empty index."""
        btree = BTreeIndex()
        data = btree.to_dict()

        assert data["keys"] == []
        assert data["entries"] == {}
        assert data["count"] == 0

    def test_to_dict_with_entries(self):
        """Serialize index with entries."""
        btree = BTreeIndex()
        btree.insert("2026-01-01", "E-001")
        btree.insert("2026-01-01", "E-002")
        btree.insert("2026-01-05", "E-003")

        data = btree.to_dict()

        assert "2026-01-01" in data["keys"]
        assert "2026-01-05" in data["keys"]
        assert set(data["entries"]["2026-01-01"]) == {"E-001", "E-002"}
        assert data["count"] == 3

    def test_from_dict_empty(self):
        """Deserialize empty index."""
        data = {"keys": [], "entries": {}, "count": 0}
        btree = BTreeIndex.from_dict(data)

        assert len(btree) == 0

    def test_from_dict_with_entries(self):
        """Deserialize index with entries."""
        data = {
            "keys": ["2026-01-01", "2026-01-05"],
            "entries": {
                "2026-01-01": ["E-001", "E-002"],
                "2026-01-05": ["E-003"]
            },
            "count": 3
        }
        btree = BTreeIndex.from_dict(data)

        assert len(btree) == 3
        assert btree.lookup_eq("2026-01-01") == {"E-001", "E-002"}
        assert btree.lookup_eq("2026-01-05") == {"E-003"}

    def test_roundtrip(self):
        """Serialize and deserialize preserves data."""
        btree1 = BTreeIndex()
        btree1.insert("2026-01-01", "E-001")
        btree1.insert("2026-01-05", "E-002")
        btree1.insert("2026-01-10", "E-003")

        data = btree1.to_dict()
        btree2 = BTreeIndex.from_dict(data)

        assert btree2.lookup_eq("2026-01-01") == {"E-001"}
        assert btree2.lookup_gt("2026-01-01") == {"E-002", "E-003"}
        assert len(btree2) == 3


class TestBTreeIndexStats:
    """Test statistics."""

    def test_stats_empty(self):
        """Stats for empty index."""
        btree = BTreeIndex()
        stats = btree.stats()

        assert stats["distinct_keys"] == 0
        assert stats["total_entries"] == 0
        assert stats["min_key"] is None
        assert stats["max_key"] is None

    def test_stats_with_entries(self):
        """Stats for index with entries."""
        btree = BTreeIndex()
        btree.insert("2026-01-01", "E-001")
        btree.insert("2026-01-01", "E-002")
        btree.insert("2026-01-20", "E-003")

        stats = btree.stats()

        assert stats["distinct_keys"] == 2
        assert stats["total_entries"] == 3
        assert stats["min_key"] == "2026-01-01"
        assert stats["max_key"] == "2026-01-20"
        assert stats["avg_entries_per_key"] == 1.5

    def test_get_distinct_keys(self):
        """Get distinct keys in sorted order."""
        btree = BTreeIndex()
        btree.insert("2026-01-10", "E-002")
        btree.insert("2026-01-01", "E-001")
        btree.insert("2026-01-20", "E-003")

        keys = btree.get_distinct_keys()
        assert keys == ["2026-01-01", "2026-01-10", "2026-01-20"]

    def test_get_min_max_key(self):
        """Get min and max keys."""
        btree = BTreeIndex()
        btree.insert("2026-01-10", "E-002")
        btree.insert("2026-01-01", "E-001")
        btree.insert("2026-01-20", "E-003")

        assert btree.get_min_key() == "2026-01-01"
        assert btree.get_max_key() == "2026-01-20"


class TestBTreeIndexKeyNormalization:
    """Test key normalization for different types."""

    def test_string_keys(self):
        """String keys are stored as-is."""
        btree = BTreeIndex()
        btree.insert("alpha", "E-001")
        btree.insert("beta", "E-002")
        btree.insert("gamma", "E-003")

        assert btree.lookup_eq("alpha") == {"E-001"}
        keys = btree.get_distinct_keys()
        assert keys == ["alpha", "beta", "gamma"]

    def test_integer_keys_sorted_correctly(self):
        """Integer keys sort numerically, not lexicographically."""
        btree = BTreeIndex()
        btree.insert(1, "E-001")
        btree.insert(10, "E-002")
        btree.insert(2, "E-003")
        btree.insert(100, "E-004")

        # Range queries should work correctly
        result = btree.lookup_lt(10)
        assert result == {"E-001", "E-003"}

        result = btree.lookup_gt(2)
        assert result == {"E-002", "E-004"}

    def test_negative_integers_sort_correctly(self):
        """Negative integers sort before positive integers."""
        btree = BTreeIndex()
        btree.insert(-10, "E-NEG10")
        btree.insert(-1, "E-NEG1")
        btree.insert(0, "E-ZERO")
        btree.insert(1, "E-POS1")
        btree.insert(10, "E-POS10")

        # Negative numbers should sort before positive
        result = btree.lookup_lt(0)
        assert result == {"E-NEG10", "E-NEG1"}

        result = btree.lookup_lte(0)
        assert result == {"E-NEG10", "E-NEG1", "E-ZERO"}

        result = btree.lookup_gt(-1)
        assert result == {"E-ZERO", "E-POS1", "E-POS10"}

        # Check full ordering
        result = btree.lookup_range(start_key=-5, end_key=5)
        assert result == {"E-NEG1", "E-ZERO", "E-POS1"}

    def test_none_key(self):
        """None key is handled specially."""
        btree = BTreeIndex()
        btree.insert(None, "E-NULL")
        btree.insert("value", "E-001")

        assert btree.lookup_eq(None) == {"E-NULL"}
        assert btree.lookup_eq("value") == {"E-001"}

    def test_boolean_keys(self):
        """Boolean keys are normalized to strings."""
        btree = BTreeIndex()
        btree.insert(True, "E-TRUE")
        btree.insert(False, "E-FALSE")

        assert btree.lookup_eq(True) == {"E-TRUE"}
        assert btree.lookup_eq(False) == {"E-FALSE"}
