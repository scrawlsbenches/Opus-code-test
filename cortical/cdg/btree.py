"""
B-tree index for range queries in CDG.

Provides sorted key-value storage enabling:
- O(log n) point lookups
- O(log n + k) range queries (k = result count)
- Efficient updates and deletes

This is a practical in-memory implementation using Python's bisect module.
For persistence, indexes are serialized to JSON like hash indexes.

The design aligns with docs/architecture/DISTRIBUTED_GRAPH_SPECIFICATION.md
Section 7: Index Structures.

Index types supported by CDG:
- "hash": O(1) exact match lookups (CDGIndexManager._indexes)
- "btree": Range queries, ordering (this module)
- "fulltext": Text search (future)

Usage:
    from cortical.cdg.btree import BTreeIndex

    # Create index
    btree = BTreeIndex()

    # Insert entries
    btree.insert("2026-01-01", "T-001")
    btree.insert("2026-01-05", "T-002")
    btree.insert("2026-01-01", "T-003")  # Same key, different value

    # Point lookup
    ids = btree.lookup_eq("2026-01-01")  # {"T-001", "T-003"}

    # Range queries
    ids = btree.lookup_gt("2026-01-01")   # {"T-002"}
    ids = btree.lookup_gte("2026-01-01")  # {"T-001", "T-002", "T-003"}
    ids = btree.lookup_lt("2026-01-05")   # {"T-001", "T-003"}
    ids = btree.lookup_range("2026-01-01", "2026-01-10")  # All

    # Serialization
    data = btree.to_dict()
    btree2 = BTreeIndex.from_dict(data)

See: docs/design/cdg-query-language.md (Range query operators)
"""

from __future__ import annotations

import bisect
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class BTreeIndex:
    """
    In-memory B-tree index for range queries.

    Stores sorted (key, entity_id) pairs enabling efficient range queries.
    Keys are normalized to strings for consistent sorting.

    Implementation uses sorted lists with bisect for O(log n) operations.
    This is simpler than a full B-tree with pages but provides the same
    query semantics for moderate data sizes (< 1M entries).

    Thread Safety:
        This class is NOT thread-safe. External synchronization required.
        CDGIndexManager provides thread safety via self._lock.

    Attributes:
        _keys: Sorted list of unique keys
        _entries: Dict mapping key -> set of entity IDs
        _count: Total number of (key, entity_id) pairs
    """
    _keys: List[str] = field(default_factory=list)
    _entries: Dict[str, Set[str]] = field(default_factory=dict)
    _count: int = 0

    def insert(self, key: Any, entity_id: str) -> None:
        """
        Insert a (key, entity_id) pair into the index.

        If the key already exists, adds entity_id to the existing set.

        Args:
            key: The index key (will be normalized to string)
            entity_id: The entity ID to associate with this key

        Complexity: O(log n) for new keys, O(1) for existing keys
        """
        key_str = self._normalize_key(key)

        if key_str in self._entries:
            # Key exists, just add to set
            if entity_id not in self._entries[key_str]:
                self._entries[key_str].add(entity_id)
                self._count += 1
        else:
            # New key, insert in sorted order
            idx = bisect.bisect_left(self._keys, key_str)
            self._keys.insert(idx, key_str)
            self._entries[key_str] = {entity_id}
            self._count += 1

    def remove(self, key: Any, entity_id: str) -> bool:
        """
        Remove a (key, entity_id) pair from the index.

        Args:
            key: The index key
            entity_id: The entity ID to remove

        Returns:
            True if the entry was found and removed, False otherwise

        Complexity: O(log n) if key is removed, O(1) otherwise
        """
        key_str = self._normalize_key(key)

        if key_str not in self._entries:
            return False

        if entity_id not in self._entries[key_str]:
            return False

        self._entries[key_str].discard(entity_id)
        self._count -= 1

        # Remove key if no more entries
        if not self._entries[key_str]:
            del self._entries[key_str]
            idx = bisect.bisect_left(self._keys, key_str)
            if idx < len(self._keys) and self._keys[idx] == key_str:
                self._keys.pop(idx)

        return True

    def lookup_eq(self, key: Any) -> Set[str]:
        """
        Lookup entity IDs with exactly this key.

        Args:
            key: The key to match

        Returns:
            Set of entity IDs (copy)

        Complexity: O(1)
        """
        key_str = self._normalize_key(key)
        return self._entries.get(key_str, set()).copy()

    def lookup_gt(self, key: Any) -> Set[str]:
        """
        Lookup entity IDs with key > given key.

        Args:
            key: The lower bound (exclusive)

        Returns:
            Set of entity IDs

        Complexity: O(log n + k) where k is result count
        """
        key_str = self._normalize_key(key)
        idx = bisect.bisect_right(self._keys, key_str)

        result: Set[str] = set()
        for k in self._keys[idx:]:
            result.update(self._entries[k])
        return result

    def lookup_gte(self, key: Any) -> Set[str]:
        """
        Lookup entity IDs with key >= given key.

        Args:
            key: The lower bound (inclusive)

        Returns:
            Set of entity IDs

        Complexity: O(log n + k) where k is result count
        """
        key_str = self._normalize_key(key)
        idx = bisect.bisect_left(self._keys, key_str)

        result: Set[str] = set()
        for k in self._keys[idx:]:
            result.update(self._entries[k])
        return result

    def lookup_lt(self, key: Any) -> Set[str]:
        """
        Lookup entity IDs with key < given key.

        Args:
            key: The upper bound (exclusive)

        Returns:
            Set of entity IDs

        Complexity: O(log n + k) where k is result count
        """
        key_str = self._normalize_key(key)
        idx = bisect.bisect_left(self._keys, key_str)

        result: Set[str] = set()
        for k in self._keys[:idx]:
            result.update(self._entries[k])
        return result

    def lookup_lte(self, key: Any) -> Set[str]:
        """
        Lookup entity IDs with key <= given key.

        Args:
            key: The upper bound (inclusive)

        Returns:
            Set of entity IDs

        Complexity: O(log n + k) where k is result count
        """
        key_str = self._normalize_key(key)
        idx = bisect.bisect_right(self._keys, key_str)

        result: Set[str] = set()
        for k in self._keys[:idx]:
            result.update(self._entries[k])
        return result

    def lookup_range(
        self,
        start_key: Optional[Any] = None,
        end_key: Optional[Any] = None,
        start_inclusive: bool = True,
        end_inclusive: bool = True,
        limit: Optional[int] = None
    ) -> Set[str]:
        """
        Lookup entity IDs within a key range.

        Args:
            start_key: Lower bound (None = no lower bound)
            end_key: Upper bound (None = no upper bound)
            start_inclusive: Include start_key in results
            end_inclusive: Include end_key in results
            limit: Maximum number of entity IDs to return (None = no limit)

        Returns:
            Set of entity IDs within the range

        Complexity: O(log n + k) where k is result count

        Example:
            # All entries from 2026-01-01 to 2026-01-31
            btree.lookup_range("2026-01-01", "2026-01-31")

            # All entries after 2026-01-01 (exclusive)
            btree.lookup_range("2026-01-01", None, start_inclusive=False)
        """
        # Determine start index
        if start_key is None:
            start_idx = 0
        else:
            start_str = self._normalize_key(start_key)
            if start_inclusive:
                start_idx = bisect.bisect_left(self._keys, start_str)
            else:
                start_idx = bisect.bisect_right(self._keys, start_str)

        # Determine end index
        if end_key is None:
            end_idx = len(self._keys)
        else:
            end_str = self._normalize_key(end_key)
            if end_inclusive:
                end_idx = bisect.bisect_right(self._keys, end_str)
            else:
                end_idx = bisect.bisect_left(self._keys, end_str)

        # Collect results
        result: Set[str] = set()
        for k in self._keys[start_idx:end_idx]:
            result.update(self._entries[k])
            if limit is not None and len(result) >= limit:
                break

        return result

    def get_all(self) -> Set[str]:
        """
        Get all entity IDs in the index.

        Returns:
            Set of all entity IDs
        """
        result: Set[str] = set()
        for entity_ids in self._entries.values():
            result.update(entity_ids)
        return result

    def get_distinct_keys(self) -> List[str]:
        """
        Get all distinct keys in sorted order.

        Returns:
            List of keys (sorted)
        """
        return self._keys.copy()

    def get_min_key(self) -> Optional[str]:
        """Get the minimum key, or None if empty."""
        return self._keys[0] if self._keys else None

    def get_max_key(self) -> Optional[str]:
        """Get the maximum key, or None if empty."""
        return self._keys[-1] if self._keys else None

    def __len__(self) -> int:
        """Return total number of (key, entity_id) pairs."""
        return self._count

    def __bool__(self) -> bool:
        """Return True if index has any entries."""
        return self._count > 0

    def clear(self) -> None:
        """Remove all entries from the index."""
        self._keys.clear()
        self._entries.clear()
        self._count = 0

    def _normalize_key(self, key: Any) -> str:
        """
        Normalize a key to string for consistent sorting.

        Numeric types are zero-padded for correct lexicographic ordering.
        None is represented as a special string that sorts first.
        """
        if key is None:
            return "\x00__NULL__"  # Sorts before all other strings
        if isinstance(key, bool):
            # Must check bool before int (bool is subclass of int)
            return "true" if key else "false"
        if isinstance(key, int):
            # Zero-pad integers for correct sorting
            # Negative numbers get special handling
            if key >= 0:
                return f"+{key:020d}"  # + prefix for positive
            else:
                # For negative: invert to maintain order
                return f"-{key + 10**20:020d}"
        if isinstance(key, float):
            # Convert to string with enough precision
            # This isn't perfect for all floats but works for typical use
            return f"{key:+025.10f}"
        if isinstance(key, str):
            return key
        # Fallback: JSON serialization for complex types
        return json.dumps(key, sort_keys=True)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize index to dictionary for JSON persistence.

        Returns:
            Dictionary with keys and entries
        """
        return {
            "keys": self._keys,
            "entries": {k: sorted(v) for k, v in self._entries.items()},
            "count": self._count
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BTreeIndex":
        """
        Deserialize index from dictionary.

        Args:
            data: Dictionary from to_dict()

        Returns:
            Reconstructed BTreeIndex
        """
        index = cls()
        index._keys = data.get("keys", [])
        index._entries = {k: set(v) for k, v in data.get("entries", {}).items()}
        index._count = data.get("count", 0)

        # Validate count
        actual_count = sum(len(v) for v in index._entries.values())
        if actual_count != index._count:
            index._count = actual_count

        return index

    def stats(self) -> Dict[str, Any]:
        """
        Get index statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "distinct_keys": len(self._keys),
            "total_entries": self._count,
            "min_key": self.get_min_key(),
            "max_key": self.get_max_key(),
            "avg_entries_per_key": (
                self._count / len(self._keys) if self._keys else 0
            )
        }


__all__ = ['BTreeIndex']
