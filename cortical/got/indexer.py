"""
Query Index Manager for Graph of Thought.

This module provides index management for fast query lookups.
Indexes are stored as JSON files in .got/indexes/ and are
automatically updated when entities are created/updated/deleted.

SUPPORTED INDEXES
-----------------
- by_status.json: Tasks grouped by status (pending, in_progress, completed, etc.)
- by_priority.json: Tasks grouped by priority (low, medium, high, critical)
- by_sprint.json: Tasks grouped by sprint ID

PERFORMANCE IMPACT
------------------
Without indexes: O(n) scan of all entities (~5ms per query)
With indexes: O(1) lookup from pre-computed groups (~0.5ms per query)

Expected speedup: 5-10x for indexed queries.
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class IndexStats:
    """Statistics about index usage."""
    hits: int = 0
    misses: int = 0
    rebuilds: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


@dataclass
class IndexEntry:
    """A single index mapping field values to entity IDs."""
    field_name: str
    values: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    version: int = 0

    def add(self, entity_id: str, value: Any) -> None:
        """Add an entity ID to a value group."""
        str_value = str(value) if value is not None else "__null__"
        self.values[str_value].add(entity_id)
        self.version += 1

    def remove(self, entity_id: str, value: Any = None) -> None:
        """Remove an entity ID from index."""
        str_value = str(value) if value is not None else "__null__"
        if value is not None:
            # Remove from specific value
            if str_value in self.values:
                self.values[str_value].discard(entity_id)
                if not self.values[str_value]:
                    del self.values[str_value]
        else:
            # Remove from all values (entity deleted)
            for val_set in self.values.values():
                val_set.discard(entity_id)
        self.version += 1

    def get(self, value: Any) -> Set[str]:
        """Get entity IDs for a value."""
        str_value = str(value) if value is not None else "__null__"
        return self.values.get(str_value, set())

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "field_name": self.field_name,
            "values": {k: list(v) for k, v in self.values.items()},
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IndexEntry":
        """Deserialize from dictionary."""
        entry = cls(field_name=data["field_name"], version=data.get("version", 0))
        for key, ids in data.get("values", {}).items():
            entry.values[key] = set(ids)
        return entry


class QueryIndexManager:
    """
    Manages query indexes for fast lookups.

    Indexes are automatically maintained when entities change
    and are persisted to .got/indexes/ directory.

    Usage:
        manager = QueryIndexManager(got_dir)

        # Check if index exists for a field
        if manager.has_index("status"):
            ids = manager.lookup("status", "pending")

        # Update index when entity changes
        manager.update_task(task_id, old_status, new_status)

        # Rebuild all indexes
        manager.rebuild_all(all_tasks)
    """

    # Standard indexes to maintain
    TASK_INDEXES = ["status", "priority"]
    SPRINT_INDEXES = ["status"]

    def __init__(self, got_dir: Path):
        """
        Initialize index manager.

        Args:
            got_dir: Path to .got directory
        """
        self._got_dir = Path(got_dir)
        self._index_dir = self._got_dir / "indexes"
        self._indexes: Dict[str, IndexEntry] = {}
        self._sprint_index: Dict[str, Set[str]] = defaultdict(set)  # sprint_id -> task_ids
        self._stats = IndexStats()
        self._dirty = False

        # Ensure index directory exists
        self._index_dir.mkdir(parents=True, exist_ok=True)

        # Load existing indexes
        self._load_indexes()

    def _load_indexes(self) -> None:
        """Load indexes from disk."""
        for field_name in self.TASK_INDEXES:
            index_file = self._index_dir / f"by_{field_name}.json"
            if index_file.exists():
                try:
                    with open(index_file, "r") as f:
                        data = json.load(f)
                    self._indexes[field_name] = IndexEntry.from_dict(data)
                    logger.debug(f"Loaded index: {field_name}")
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Failed to load index {field_name}: {e}")
                    self._indexes[field_name] = IndexEntry(field_name=field_name)
            else:
                self._indexes[field_name] = IndexEntry(field_name=field_name)

        # Load sprint index
        sprint_file = self._index_dir / "by_sprint.json"
        if sprint_file.exists():
            try:
                with open(sprint_file, "r") as f:
                    data = json.load(f)
                for sprint_id, task_ids in data.get("sprints", {}).items():
                    self._sprint_index[sprint_id] = set(task_ids)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to load sprint index: {e}")

    def _save_indexes(self) -> None:
        """Save all indexes to disk."""
        if not self._dirty:
            return

        for field_name, index in self._indexes.items():
            index_file = self._index_dir / f"by_{field_name}.json"
            try:
                with open(index_file, "w") as f:
                    json.dump(index.to_dict(), f, indent=2)
            except IOError as e:
                logger.error(f"Failed to save index {field_name}: {e}")

        # Save sprint index
        sprint_file = self._index_dir / "by_sprint.json"
        try:
            data = {
                "sprints": {k: list(v) for k, v in self._sprint_index.items()},
                "version": 1,
            }
            with open(sprint_file, "w") as f:
                json.dump(data, f, indent=2)
        except IOError as e:
            logger.error(f"Failed to save sprint index: {e}")

        self._dirty = False

    def has_index(self, field_name: str) -> bool:
        """Check if an index exists for the given field."""
        return field_name in self._indexes or field_name == "sprint"

    def lookup(self, field_name: str, value: Any) -> Set[str]:
        """
        Look up entity IDs by field value.

        Args:
            field_name: Field to look up (status, priority, sprint)
            value: Value to match

        Returns:
            Set of entity IDs matching the value
        """
        if field_name == "sprint":
            result = self._sprint_index.get(str(value), set())
            if result:
                self._stats.hits += 1
            else:
                self._stats.misses += 1
            return result.copy()

        if field_name in self._indexes:
            result = self._indexes[field_name].get(value)
            if result:
                self._stats.hits += 1
            else:
                self._stats.misses += 1
            return result.copy()

        self._stats.misses += 1
        return set()

    def lookup_multi(self, field_name: str, values: List[Any]) -> Set[str]:
        """Look up entity IDs matching any of the given values."""
        result: Set[str] = set()
        for value in values:
            result |= self.lookup(field_name, value)
        return result

    def index_task(
        self,
        task_id: str,
        status: Optional[str] = None,
        priority: Optional[str] = None,
        sprint_id: Optional[str] = None,
    ) -> None:
        """
        Add or update a task in all indexes.

        Args:
            task_id: Task ID to index
            status: Task status value
            priority: Task priority value
            sprint_id: Sprint ID if task is in a sprint
        """
        if status is not None:
            self._indexes["status"].add(task_id, status)
        if priority is not None:
            self._indexes["priority"].add(task_id, priority)
        if sprint_id is not None:
            self._sprint_index[sprint_id].add(task_id)
        self._dirty = True

    def update_task(
        self,
        task_id: str,
        old_status: Optional[str] = None,
        new_status: Optional[str] = None,
        old_priority: Optional[str] = None,
        new_priority: Optional[str] = None,
        old_sprint: Optional[str] = None,
        new_sprint: Optional[str] = None,
    ) -> None:
        """
        Update a task's index entries when values change.

        Args:
            task_id: Task ID to update
            old_*: Previous values (for removal)
            new_*: New values (for addition)
        """
        # Update status index
        if old_status != new_status:
            if old_status is not None:
                self._indexes["status"].remove(task_id, old_status)
            if new_status is not None:
                self._indexes["status"].add(task_id, new_status)

        # Update priority index
        if old_priority != new_priority:
            if old_priority is not None:
                self._indexes["priority"].remove(task_id, old_priority)
            if new_priority is not None:
                self._indexes["priority"].add(task_id, new_priority)

        # Update sprint index
        if old_sprint != new_sprint:
            if old_sprint is not None:
                self._sprint_index[old_sprint].discard(task_id)
            if new_sprint is not None:
                self._sprint_index[new_sprint].add(task_id)

        self._dirty = True

    def remove_task(self, task_id: str) -> None:
        """
        Remove a task from all indexes.

        Args:
            task_id: Task ID to remove
        """
        for index in self._indexes.values():
            index.remove(task_id)
        for sprint_tasks in self._sprint_index.values():
            sprint_tasks.discard(task_id)
        self._dirty = True

    def link_task_to_sprint(self, task_id: str, sprint_id: str) -> None:
        """Link a task to a sprint in the index."""
        self._sprint_index[sprint_id].add(task_id)
        self._dirty = True

    def unlink_task_from_sprint(self, task_id: str, sprint_id: str) -> None:
        """Unlink a task from a sprint in the index."""
        if sprint_id in self._sprint_index:
            self._sprint_index[sprint_id].discard(task_id)
        self._dirty = True

    def rebuild_all(self, tasks: List[Any], edges: Optional[List[Any]] = None) -> None:
        """
        Rebuild all indexes from scratch.

        Args:
            tasks: List of all tasks
            edges: Optional list of edges (for sprint membership)
        """
        # Clear existing indexes
        for index in self._indexes.values():
            index.values.clear()
            index.version = 0
        self._sprint_index.clear()

        # Rebuild from tasks
        for task in tasks:
            task_id = task.id if hasattr(task, "id") else task.get("id")
            status = getattr(task, "status", None) or task.get("status")
            priority = getattr(task, "priority", None) or task.get("priority")

            if task_id:
                self.index_task(task_id, status=status, priority=priority)

        # Rebuild sprint index from edges
        if edges:
            for edge in edges:
                edge_type = getattr(edge, "edge_type", None) or edge.get("edge_type")
                if edge_type == "CONTAINS":
                    from_id = getattr(edge, "from_id", None) or edge.get("from_id")
                    to_id = getattr(edge, "to_id", None) or edge.get("to_id")
                    # Sprint CONTAINS Task
                    if from_id and to_id and from_id.startswith("S-"):
                        self._sprint_index[from_id].add(to_id)

        self._dirty = True
        self._stats.rebuilds += 1
        self._save_indexes()
        logger.info(f"Rebuilt indexes: {len(tasks)} tasks indexed")

    def save(self) -> None:
        """Persist indexes to disk."""
        self._save_indexes()

    def get_stats(self) -> Dict[str, Any]:
        """Get index usage statistics."""
        return {
            "hits": self._stats.hits,
            "misses": self._stats.misses,
            "hit_rate": f"{self._stats.hit_rate:.1%}",
            "rebuilds": self._stats.rebuilds,
            "indexes": list(self._indexes.keys()) + ["sprint"],
            "index_sizes": {
                name: sum(len(v) for v in idx.values.values())
                for name, idx in self._indexes.items()
            },
            "sprint_index_size": sum(len(v) for v in self._sprint_index.values()),
        }

    def get_all_indexed_values(self, field_name: str) -> List[str]:
        """Get all indexed values for a field."""
        if field_name == "sprint":
            return list(self._sprint_index.keys())
        if field_name in self._indexes:
            return list(self._indexes[field_name].values.keys())
        return []
