# Foreign Key Patterns: Practical Implementation for GoT

**Date:** 2025-12-23
**Target:** Cortical Text Processor GoT (Graph of Thought) system
**Goal:** Implement hybrid reference patterns with no external dependencies

---

## Overview: GoT's Current State & Needs

### What GoT Has
- **Entity types**: Task, Decision, Sprint, Epic, Edge
- **Edge types**: DEPENDS_ON, BLOCKS, CONTAINS, PART_OF, etc.
- **Persistence**: WAL-based with event sourcing
- **References**: Currently ID-based with no explicit FK strategy

### What's Missing
- Explicit reference lifecycle management
- Safe deletion semantics (when is a delete cascading vs. orphaning?)
- ID reservation for distributed creation
- Cross-domain transaction coordination
- Soft reference handling (broken FK tolerance)

---

## Phase 1: Core Reference Patterns

### Step 1: Add Reference Strength Classification

**File: `cortical/got/reference_strength.py`** (NEW)

```python
"""
Reference strength classification for semantic deletion handling.

Determines whether a reference should cause CASCADE DELETE or allow orphaning.
"""

from enum import Enum
from typing import List


class ReferenceStrength(Enum):
    """Semantic strength of a reference edge."""

    STRONG = "strong"        # CASCADE DELETE: child cannot exist without parent
    WEAK = "weak"            # ORPHAN: child can exist without parent
    CONDITIONAL = "conditional"  # Depends on business rules


class EdgeTypeClassification:
    """Classify edge types by reference strength."""

    STRONG_EDGES = {
        "PART_OF",       # Task is part of Sprint → delete sprint, delete task
        "REQUIRES",      # Task requires Decision → if decision deleted, task blocked
        "DEPENDS_ON",    # Sprint depends on Epic → if epic deleted, sprint affected
        "CONTAINS"       # Container-content relationship
    }

    WEAK_EDGES = {
        "MENTIONS",      # Task mentions Decision → decision can disappear
        "SUGGESTS",      # Task suggests approach → approach can change
        "INFLUENCES",    # X influences Y → not a hard dependency
        "REFERENCES",    # Task references other task → can work around
        "MOTIVATES"      # Decision motivates task → task can exist without decision
    }

    @classmethod
    def get_strength(cls, edge_type: str) -> ReferenceStrength:
        """Determine reference strength from edge type."""

        if edge_type in cls.STRONG_EDGES:
            return ReferenceStrength.STRONG

        elif edge_type in cls.WEAK_EDGES:
            return ReferenceStrength.WEAK

        # Default: ask when uncertain
        return ReferenceStrength.CONDITIONAL

    @classmethod
    def is_cascade_delete(cls, edge_type: str) -> bool:
        """Should this edge cascade delete?"""
        return cls.get_strength(edge_type) == ReferenceStrength.STRONG


# Usage in GoT:
# 1. User tries to delete Task
# 2. Check all incoming edges to this Task
# 3. If any STRONG edges exist → ask user to resolve first
# 4. If only WEAK edges → delete without cascade
```

### Step 2: Add Deletion Safety Checks

**File: `cortical/got/deletion_safety.py`** (NEW)

```python
"""
Safety checks before entity deletion.

Prevents accidental cascades and data loss.
"""

from typing import Tuple, List
from .reference_strength import EdgeTypeClassification, ReferenceStrength


class DeletionValidator:
    """Validate if entity can be safely deleted."""

    def __init__(self, got_manager):
        self.got_manager = got_manager

    def can_delete_entity(
        self,
        entity_id: str,
        allow_cascade: bool = False
    ) -> Tuple[bool, List[str]]:
        """
        Check if entity can be deleted.

        Returns:
            (can_delete, blocking_reasons)

        Raises:
            EntityNotFoundError if entity doesn't exist
        """
        reasons = []

        entity = self.got_manager.get_entity(entity_id)
        if not entity:
            raise ValueError(f"Entity {entity_id} not found")

        # Get all INCOMING edges (entities that reference this one)
        incoming_edges = self.got_manager.find_edges_by_target(entity_id)

        # Classify by strength
        strong_refs = []
        weak_refs = []

        for edge in incoming_edges:
            strength = EdgeTypeClassification.get_strength(edge.edge_type)

            if strength == ReferenceStrength.STRONG:
                strong_refs.append(edge)
            else:
                weak_refs.append(edge)

        # Check strong references
        if strong_refs:
            if not allow_cascade:
                source_names = [
                    self.got_manager.get_entity(e.source_id).title or e.source_id
                    for e in strong_refs[:3]
                ]
                reasons.append(
                    f"Has {len(strong_refs)} STRONG incoming references (CASCADE DELETE required): "
                    f"{', '.join(source_names)}"
                )
                return False, reasons

            # Recursive check: can we delete the referencing entities?
            for edge in strong_refs:
                source_can_delete, source_reasons = self.can_delete_entity(
                    edge.source_id,
                    allow_cascade=True
                )
                if not source_can_delete:
                    reasons.extend(source_reasons)

        # Check weak references (informational only)
        if weak_refs:
            reasons.append(
                f"Has {len(weak_refs)} WEAK incoming references "
                f"(will become orphaned, safe to delete)"
            )

        return (True, reasons) if not strong_refs else (False, reasons)

    def get_cascade_scope(self, entity_id: str) -> dict:
        """
        If CASCADE DELETE is chosen, what gets deleted?

        Returns:
            {
                'entities_to_delete': ['T-001', 'T-002', ...],
                'edges_to_delete': ['E-...', ...],
                'depth': 3,
                'total_count': 25
            }
        """
        to_delete = {'entities': [], 'edges': []}
        visited = set()

        def traverse(eid):
            if eid in visited:
                return
            visited.add(eid)

            # Find STRONG incoming edges
            for edge in self.got_manager.find_edges_by_target(eid):
                if EdgeTypeClassification.is_cascade_delete(edge.edge_type):
                    to_delete['entities'].append(edge.source_id)
                    to_delete['edges'].append(edge.id)
                    traverse(edge.source_id)  # Recurse

        traverse(entity_id)
        to_delete['entities'].insert(0, entity_id)
        to_delete['depth'] = len(visited)
        to_delete['total_count'] = len(to_delete['entities']) + len(to_delete['edges'])

        return to_delete


# Usage:
# validator = DeletionValidator(got_manager)
#
# can_delete, reasons = validator.can_delete_entity("T-001")
# for reason in reasons:
#     print(f"  ⚠️  {reason}")
#
# if can_delete:
#     got_manager.delete_entity("T-001")
# else:
#     scope = validator.get_cascade_scope("T-001")
#     print(f"Would delete {scope['total_count']} items")
#     print(f"Entities: {scope['entities_to_delete']}")
```

### Step 3: Add Soft Reference Support

**File: `cortical/got/soft_references.py`** (NEW)

```python
"""
Soft reference handling for orphan-tolerant edges.

Allows references to deleted entities (with graceful degradation).
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path


@dataclass
class BrokenReference:
    """Record of a reference to a deleted entity."""

    source_id: str
    target_id: str
    edge_type: str
    discovered_at: str  # ISO timestamp
    attempts_to_resolve: int = 0


class SoftReferenceHandler:
    """Handle references to entities that may have been deleted."""

    def __init__(self, got_manager, orphan_log_dir: Path = Path(".got/orphans")):
        self.got_manager = got_manager
        self.orphan_log_dir = Path(orphan_log_dir)
        self.orphan_log_dir.mkdir(parents=True, exist_ok=True)
        self.broken_refs_file = self.orphan_log_dir / "broken_references.jsonl"

    def resolve_reference(
        self,
        source_id: str,
        target_id: str,
        fallback: Optional[Any] = None,
        strict: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Resolve reference, handling deleted targets gracefully.

        Args:
            source_id: Entity making the reference
            target_id: Entity being referenced
            fallback: Value to return if target deleted (strict=False)
            strict: Raise error if target missing (vs return fallback)

        Returns:
            Target entity or fallback
        """
        target = self.got_manager.get_entity(target_id)

        if target is not None:
            return target.to_dict() if hasattr(target, 'to_dict') else target

        # Target is missing
        if strict:
            raise ReferenceError(
                f"{source_id} references deleted entity {target_id}"
            )

        # Log broken reference
        self._log_broken_reference(
            BrokenReference(
                source_id=source_id,
                target_id=target_id,
                edge_type="unknown",  # Would need edge lookup
                discovered_at=datetime.now(timezone.utc).isoformat()
            )
        )

        return fallback

    def validate_all_references(self) -> List[BrokenReference]:
        """
        Scan all edges and find broken references.

        Returns:
            List of references to deleted entities
        """
        broken = []

        # Get all edges
        all_edges = self.got_manager.list_edges()

        for edge in all_edges:
            # Check if target still exists
            target = self.got_manager.get_entity(edge.target_id)

            if target is None:
                broken_ref = BrokenReference(
                    source_id=edge.source_id,
                    target_id=edge.target_id,
                    edge_type=edge.edge_type,
                    discovered_at=datetime.now(timezone.utc).isoformat()
                )
                broken.append(broken_ref)

        return broken

    def cleanup_orphaned_edges(
        self,
        older_than_days: int = 7,
        dry_run: bool = False
    ) -> int:
        """
        Delete edges pointing to entities that were deleted > N days ago.

        Returns:
            Count of edges deleted
        """
        cutoff_time = datetime.now(timezone.utc).timestamp() - (older_than_days * 86400)

        edges_to_delete = []

        # Find edges to deleted entities
        for edge in self.got_manager.list_edges():
            if self.got_manager.get_entity(edge.target_id) is None:
                # Target was deleted, check how long ago
                if hasattr(edge, 'created_at'):
                    if edge.created_at < cutoff_time:
                        edges_to_delete.append(edge.id)

        # Delete them
        if not dry_run:
            for edge_id in edges_to_delete:
                self.got_manager.delete_edge(edge_id)

        return len(edges_to_delete)

    def _log_broken_reference(self, broken_ref: BrokenReference) -> None:
        """Append broken reference to log."""
        data = {
            'source_id': broken_ref.source_id,
            'target_id': broken_ref.target_id,
            'edge_type': broken_ref.edge_type,
            'discovered_at': broken_ref.discovered_at
        }

        with open(self.broken_refs_file, 'a') as f:
            f.write(json.dumps(data) + '\n')


# Usage:
# handler = SoftReferenceHandler(got_manager)
#
# # Safe reference resolution
# decision = handler.resolve_reference(
#     source_id="T-001",
#     target_id="D-123",
#     fallback={'title': '(Decision deleted)', 'status': 'unknown'},
#     strict=False
# )
#
# # Find all broken references
# broken = handler.validate_all_references()
# print(f"Found {len(broken)} broken references")
#
# # Clean up old orphaned edges
# cleaned = handler.cleanup_orphaned_edges(older_than_days=7, dry_run=True)
# print(f"Would delete {cleaned} orphaned edges")
```

---

## Phase 2: Safe Deletion Operation

### Step 4: Unified Deletion Operation

**File: `cortical/got/safe_deletion.py`** (NEW)

```python
"""
Safe entity deletion with CASCADE/ORPHAN semantics.

Unified interface for deleting entities with proper FK handling.
"""

from typing import Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timezone

from .deletion_safety import DeletionValidator
from .soft_references import SoftReferenceHandler
from .reference_strength import EdgeTypeClassification


class DeletionMode(Enum):
    """How to handle cascades during deletion."""

    SAFE = "safe"              # Fail if cascade needed
    CASCADE = "cascade"         # Recursively delete
    ORPHAN = "orphan"          # Leave orphaned references


@dataclass
class DeletionResult:
    """Result of a deletion operation."""

    entity_id: str
    success: bool
    mode: DeletionMode
    entities_deleted: int = 0
    edges_deleted: int = 0
    errors: list = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class SafeEntityDeleter:
    """Safely delete entities with FK handling."""

    def __init__(self, got_manager):
        self.got_manager = got_manager
        self.validator = DeletionValidator(got_manager)
        self.soft_ref_handler = SoftReferenceHandler(got_manager)

    def delete_entity(
        self,
        entity_id: str,
        mode: DeletionMode = DeletionMode.SAFE,
        user_id: str = "",
        reason: str = ""
    ) -> DeletionResult:
        """
        Delete entity with safety checks.

        Args:
            entity_id: Entity to delete
            mode: SAFE | CASCADE | ORPHAN
            user_id: User requesting deletion (for audit)
            reason: Why deletion was requested

        Returns:
            DeletionResult with details
        """

        result = DeletionResult(
            entity_id=entity_id,
            success=False,
            mode=mode
        )

        # Validate entity exists
        entity = self.got_manager.get_entity(entity_id)
        if not entity:
            result.errors.append(f"Entity {entity_id} not found")
            return result

        # Check if deletion is safe
        can_delete, reasons = self.validator.can_delete_entity(
            entity_id,
            allow_cascade=(mode == DeletionMode.CASCADE)
        )

        if not can_delete and mode != DeletionMode.CASCADE:
            result.errors = reasons
            return result

        # Perform deletion based on mode
        if mode == DeletionMode.CASCADE:
            result = self._cascade_delete(entity_id, result)

        elif mode == DeletionMode.ORPHAN:
            result = self._orphan_delete(entity_id, result)

        else:  # SAFE mode
            if not can_delete:
                result.errors = reasons
                return result
            result = self._safe_delete(entity_id, result)

        # Audit log
        if result.success:
            self._log_deletion(
                entity_id=entity_id,
                user_id=user_id,
                reason=reason,
                result=result
            )

        return result

    def _cascade_delete(
        self,
        entity_id: str,
        result: DeletionResult
    ) -> DeletionResult:
        """Recursively delete entity and all strong dependents."""

        to_delete = self.validator.get_cascade_scope(entity_id)

        try:
            # Delete entities in reverse topological order
            for eid in reversed(to_delete['entities_to_delete']):
                # Delete all edges first
                edges = self.got_manager.find_edges_by_source(eid)
                for edge in edges:
                    self.got_manager.delete_edge(edge.id)
                    result.edges_deleted += 1

                # Then delete entity
                self.got_manager.delete_entity(eid)
                result.entities_deleted += 1

            result.success = True

        except Exception as e:
            result.errors.append(f"Cascade delete failed: {str(e)}")

        return result

    def _orphan_delete(
        self,
        entity_id: str,
        result: DeletionResult
    ) -> DeletionResult:
        """Delete entity, leave weak references orphaned."""

        try:
            # Get weak references (log them)
            weak_edges = [
                e for e in self.got_manager.find_edges_by_target(entity_id)
                if not EdgeTypeClassification.is_cascade_delete(e.edge_type)
            ]

            for edge in weak_edges:
                self.soft_ref_handler._log_broken_reference(
                    source_id=edge.source_id,
                    target_id=entity_id,
                    edge_type=edge.edge_type,
                    discovered_at=datetime.now(timezone.utc).isoformat()
                )

            # Delete all edges TO this entity
            edges = self.got_manager.find_edges_by_target(entity_id)
            for edge in edges:
                self.got_manager.delete_edge(edge.id)
                result.edges_deleted += 1

            # Delete the entity
            self.got_manager.delete_entity(entity_id)
            result.entities_deleted += 1

            result.success = True

        except Exception as e:
            result.errors.append(f"Orphan delete failed: {str(e)}")

        return result

    def _safe_delete(
        self,
        entity_id: str,
        result: DeletionResult
    ) -> DeletionResult:
        """Delete entity only if no strong references exist."""

        try:
            # Delete all edges
            edges = self.got_manager.find_edges_by_target(entity_id)
            for edge in edges:
                self.got_manager.delete_edge(edge.id)
                result.edges_deleted += 1

            # Delete entity
            self.got_manager.delete_entity(entity_id)
            result.entities_deleted += 1

            result.success = True

        except Exception as e:
            result.errors.append(f"Safe delete failed: {str(e)}")

        return result

    def _log_deletion(
        self,
        entity_id: str,
        user_id: str,
        reason: str,
        result: DeletionResult
    ) -> None:
        """Log deletion for audit trail."""

        log_dir = self.got_manager.storage_dir / ".got" / "deletion_log"
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / "deletions.jsonl"

        import json
        data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'entity_id': entity_id,
            'user_id': user_id,
            'reason': reason,
            'mode': result.mode.value,
            'entities_deleted': result.entities_deleted,
            'edges_deleted': result.edges_deleted
        }

        with open(log_file, 'a') as f:
            f.write(json.dumps(data) + '\n')


# Usage:
# deleter = SafeEntityDeleter(got_manager)
#
# # Try safe delete first
# result = deleter.delete_entity(
#     "T-001",
#     mode=DeletionMode.SAFE,
#     user_id="admin",
#     reason="Duplicate task"
# )
#
# if not result.success:
#     print("Cannot safely delete:")
#     for error in result.errors:
#         print(f"  - {error}")
#
#     # Option 1: Check cascade scope
#     scope = deleter.validator.get_cascade_scope("T-001")
#     print(f"\nWould cascade delete {scope['total_count']} items:")
#     print(f"  Entities: {scope['entities_to_delete']}")
#
#     # Option 2: Do cascade delete
#     result = deleter.delete_entity(
#         "T-001",
#         mode=DeletionMode.CASCADE,
#         user_id="admin",
#         reason="Cleaning up experiment"
#     )
#     print(f"Deleted {result.entities_deleted} entities, {result.edges_deleted} edges")
```

---

## Phase 3: Integration with GoT Manager

### Step 5: Extend GoT Manager

**Modify: `cortical/got/manager.py`**

```python
# Add to existing GoTManager class:

def delete_entity_safe(
    self,
    entity_id: str,
    mode: str = "safe",  # "safe" | "cascade" | "orphan"
    user_id: str = "",
    reason: str = ""
):
    """
    Delete entity with FK safety checks.

    Args:
        entity_id: Entity to delete
        mode: Deletion mode
        user_id: Audit trail
        reason: Audit trail

    Returns:
        DeletionResult
    """
    from .safe_deletion import SafeEntityDeleter, DeletionMode

    deleter = SafeEntityDeleter(self)
    return deleter.delete_entity(
        entity_id,
        mode=DeletionMode(mode),
        user_id=user_id,
        reason=reason
    )

def validate_references(self) -> list:
    """
    Find all broken references in the system.

    Returns:
        List of BrokenReference objects
    """
    from .soft_references import SoftReferenceHandler

    handler = SoftReferenceHandler(self)
    return handler.validate_all_references()

def cleanup_orphaned_references(self, older_than_days: int = 7) -> int:
    """
    Delete edges to deleted entities that are > N days old.

    Returns:
        Count of edges deleted
    """
    from .soft_references import SoftReferenceHandler

    handler = SoftReferenceHandler(self)
    return handler.cleanup_orphaned_edges(older_than_days=older_than_days)
```

---

## Usage Examples

### Example 1: Safe Task Deletion

```python
from cortical.got import GoTManager
from cortical.got.safe_deletion import DeletionMode

got_mgr = GoTManager(Path(".got"))

# User tries to delete a task
result = got_mgr.delete_entity_safe(
    "T-20251223-123456-a1b2",
    mode="safe",
    user_id="admin@company.com",
    reason="Duplicate with T-20251223-123457-c3d4"
)

if not result.success:
    # Deletion blocked
    print("Cannot delete task:")
    for error in result.errors:
        print(f"  - {error}")

    # Show what would cascade
    from cortical.got.deletion_safety import DeletionValidator
    validator = DeletionValidator(got_mgr)
    scope = validator.get_cascade_scope("T-20251223-123456-a1b2")
    print(f"\nCascade delete would affect {scope['total_count']} items")
else:
    print(f"Deleted {result.entities_deleted} entities")
```

### Example 2: Soft Reference Handling

```python
from cortical.got.soft_references import SoftReferenceHandler

handler = SoftReferenceHandler(got_mgr)

# Get reference, with fallback
decision = handler.resolve_reference(
    source_id="T-001",
    target_id="D-not-exist",
    fallback={'title': '(Decision was deleted)', 'status': 'unknown'},
    strict=False
)

# Will return fallback without error

# Find all broken references
broken = handler.validate_all_references()
print(f"Found {len(broken)} broken references")

for ref in broken[:5]:
    print(f"  {ref.source_id} -> {ref.target_id} (via {ref.edge_type})")

# Clean up old orphaned edges
cleaned = handler.cleanup_orphaned_edges(older_than_days=7, dry_run=False)
print(f"Cleaned {cleaned} orphaned edges")
```

### Example 3: Decision Making

```python
from cortical.got.deletion_safety import DeletionValidator
from cortical.got.reference_strength import EdgeTypeClassification

validator = DeletionValidator(got_mgr)

# Check if entity can be deleted
can_delete, reasons = validator.can_delete_entity(
    "S-sprint-001",
    allow_cascade=False
)

if not can_delete:
    print("Cannot delete sprint (safe mode):")
    for reason in reasons:
        print(f"  ℹ️  {reason}")

    # Show what would happen with cascade
    scope = validator.get_cascade_scope("S-sprint-001")
    print(f"\n⚠️  CASCADE DELETE would affect {scope['total_count']} items:")
    for entity_id in scope['entities_to_delete']:
        entity = got_mgr.get_entity(entity_id)
        print(f"  - {entity_id} ({entity.entity_type}: {entity.title})")
else:
    print("✅ Safe to delete")
```

---

## Testing

### Test File: `tests/unit/test_fk_patterns.py`

```python
import pytest
from pathlib import Path
from datetime import datetime, timezone

from cortical.got import GoTManager
from cortical.got.types import Task, Sprint, Decision
from cortical.got.safe_deletion import SafeEntityDeleter, DeletionMode
from cortical.got.deletion_safety import DeletionValidator
from cortical.got.soft_references import SoftReferenceHandler
from cortical.got.reference_strength import EdgeTypeClassification


class TestReferenceStrength:
    """Test reference strength classification."""

    def test_strong_edges(self):
        """Strong edges should be CASCADE DELETE."""
        assert EdgeTypeClassification.is_cascade_delete("PART_OF")
        assert EdgeTypeClassification.is_cascade_delete("REQUIRES")

    def test_weak_edges(self):
        """Weak edges should allow orphaning."""
        assert not EdgeTypeClassification.is_cascade_delete("MENTIONS")
        assert not EdgeTypeClassification.is_cascade_delete("SUGGESTS")


class TestDeletionValidation:
    """Test deletion safety checks."""

    @pytest.fixture
    def got_mgr(self, tmp_path):
        """Create temporary GoT manager."""
        return GoTManager(tmp_path / ".got")

    def test_can_delete_no_references(self, got_mgr):
        """Entity with no references should be deletable."""
        task = Task(id="T-001", title="Orphan task")
        got_mgr.create_task(task)

        validator = DeletionValidator(got_mgr)
        can_delete, reasons = validator.can_delete_entity("T-001")

        assert can_delete
        assert len(reasons) == 0

    def test_cannot_delete_strong_reference(self, got_mgr):
        """Entity with strong reference should block deletion."""
        sprint = Sprint(id="S-001", title="Sprint 1")
        task = Task(id="T-001", title="Task 1")

        got_mgr.create_sprint(sprint)
        got_mgr.create_task(task)

        # Create STRONG reference: task PART_OF sprint
        got_mgr.add_edge("T-001", "S-001", "PART_OF")

        validator = DeletionValidator(got_mgr)
        can_delete, reasons = validator.can_delete_entity("S-001")

        assert not can_delete
        assert any("STRONG" in r for r in reasons)

    def test_cascade_scope(self, got_mgr):
        """Cascade scope should include all dependents."""
        sprint = Sprint(id="S-001", title="Sprint 1")
        task1 = Task(id="T-001", title="Task 1")
        task2 = Task(id="T-002", title="Task 2")

        got_mgr.create_sprint(sprint)
        got_mgr.create_task(task1)
        got_mgr.create_task(task2)

        # Both tasks part of sprint
        got_mgr.add_edge("T-001", "S-001", "PART_OF")
        got_mgr.add_edge("T-002", "S-001", "PART_OF")

        validator = DeletionValidator(got_mgr)
        scope = validator.get_cascade_scope("S-001")

        assert scope['total_count'] >= 3  # Sprint + 2 tasks


class TestSafeDeleteOperation:
    """Test safe deletion."""

    @pytest.fixture
    def got_mgr(self, tmp_path):
        return GoTManager(tmp_path / ".got")

    def test_safe_delete_succeeds(self, got_mgr):
        """Safe delete should succeed with no references."""
        task = Task(id="T-001", title="Task 1")
        got_mgr.create_task(task)

        deleter = SafeEntityDeleter(got_mgr)
        result = deleter.delete_entity("T-001", mode=DeletionMode.SAFE)

        assert result.success
        assert result.entities_deleted == 1

    def test_cascade_delete_succeeds(self, got_mgr):
        """Cascade delete should delete dependents."""
        sprint = Sprint(id="S-001", title="Sprint 1")
        task = Task(id="T-001", title="Task 1")

        got_mgr.create_sprint(sprint)
        got_mgr.create_task(task)
        got_mgr.add_edge("T-001", "S-001", "PART_OF")

        deleter = SafeEntityDeleter(got_mgr)
        result = deleter.delete_entity("S-001", mode=DeletionMode.CASCADE)

        assert result.success
        assert result.entities_deleted >= 2  # Sprint + task


class TestSoftReferences:
    """Test orphan-tolerant reference handling."""

    @pytest.fixture
    def got_mgr(self, tmp_path):
        return GoTManager(tmp_path / ".got")

    def test_resolve_deleted_reference(self, got_mgr):
        """Resolving deleted reference returns fallback."""
        handler = SoftReferenceHandler(got_mgr)

        # Decision doesn't exist
        result = handler.resolve_reference(
            source_id="T-001",
            target_id="D-not-exist",
            fallback={'title': 'Unknown'},
            strict=False
        )

        assert result == {'title': 'Unknown'}

    def test_strict_mode_raises(self, got_mgr):
        """Strict mode should raise error for missing reference."""
        handler = SoftReferenceHandler(got_mgr)

        with pytest.raises(ReferenceError):
            handler.resolve_reference(
                source_id="T-001",
                target_id="D-not-exist",
                strict=True
            )

    def test_validate_references(self, got_mgr):
        """Should find broken references."""
        # Create task and reference deleted decision
        task = Task(id="T-001", title="Task 1")
        got_mgr.create_task(task)

        # Manually add edge to non-existent decision
        got_mgr.add_edge("T-001", "D-not-exist", "MENTIONS")

        handler = SoftReferenceHandler(got_mgr)
        broken = handler.validate_all_references()

        assert len(broken) > 0
        assert broken[0].target_id == "D-not-exist"
```

---

## Deployment Checklist

- [ ] Add `reference_strength.py` with EdgeTypeClassification
- [ ] Add `deletion_safety.py` with DeletionValidator
- [ ] Add `soft_references.py` with SoftReferenceHandler
- [ ] Add `safe_deletion.py` with SafeEntityDeleter
- [ ] Update `got/manager.py` with new deletion methods
- [ ] Add tests in `tests/unit/test_fk_patterns.py`
- [ ] Add documentation link to `CLAUDE.md`
- [ ] Update GoT command line interface to use safe delete by default
- [ ] Add periodic cleanup of orphaned edges (background task or cron)
- [ ] Test in staging with real GoT usage
- [ ] Train team on new deletion semantics

---

## Configuration (Optional)

Create `cortical/got/fk_config.py` for customizable settings:

```python
"""Foreign key configuration."""

# Edge type classification (customizable)
FK_STRONG_EDGES = {
    "PART_OF", "REQUIRES", "DEPENDS_ON", "CONTAINS"
}

FK_WEAK_EDGES = {
    "MENTIONS", "SUGGESTS", "INFLUENCES", "REFERENCES"
}

# Cleanup settings
ORPHAN_CLEANUP_DAYS = 7
BROKEN_REF_LOG_ENABLED = True

# Deletion settings
DEFAULT_DELETION_MODE = "safe"  # Require explicit cascade
REQUIRE_REASON_FOR_DELETION = True
ALLOW_CASCADE_DELETE = True
```

---

## Summary

This implementation provides:

1. **Clear semantics**: Strong vs. weak references codified
2. **Safety first**: Safe deletion by default, cascade requires consent
3. **Eventual consistency**: Soft references and orphan cleanup
4. **Auditability**: Deletion logs and broken reference tracking
5. **Flexibility**: SAFE/CASCADE/ORPHAN modes for different needs

Works with existing GoT system, no external dependencies, pure Python.

