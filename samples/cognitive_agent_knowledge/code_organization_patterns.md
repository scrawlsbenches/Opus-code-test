# Code Organization Patterns

This document teaches AI agents how to organize code for maintainability, using patterns from the Cortical codebase.

## Module Structure: When to Split, When to Keep Together

### When to Split a Module

Split a module when:

1. **It exceeds ~500-800 lines**: Large files are hard to navigate and understand
2. **It has distinct responsibilities**: If you can name different "jobs" it does, split them
3. **Different parts change at different rates**: Stable vs. volatile code should separate
4. **Tests become unwieldy**: If testing one aspect requires mocking another, split them

### When to Keep Together

Keep code together when:

1. **Tight cohesion**: The pieces must change together
2. **Small enough to understand**: Under 300 lines with clear purpose
3. **Splitting would create circular imports**: A sign of true cohesion
4. **Performance**: Sometimes hot paths need to be co-located

### Pattern: Mixin Decomposition

The `cortical/processor/` module demonstrates mixin-based decomposition:

```
cortical/processor/
    __init__.py      # Composes the final class
    core.py          # Initialization, staleness tracking
    documents.py     # Document processing, add/remove
    compute.py       # Analysis, clustering, embeddings
    query_api.py     # Search and retrieval
    introspection.py # State inspection
    persistence_api.py # Save/load operations
    spark_api.py     # SparkSLM integration
```

The `__init__.py` composes them:

```python
class CorticalTextProcessor(
    SparkMixin,
    CoreMixin,
    DocumentsMixin,
    ComputeMixin,
    QueryMixin,
    IntrospectionMixin,
    PersistenceMixin
):
    """Combined processor from focused mixins."""
    pass
```

Benefits:
- Each file has a single responsibility
- Easy to find relevant code
- Can test mixins in isolation
- Full backwards compatibility maintained

### Pattern: Subpackage for CLI Commands

The `cortical/got/cli/` directory separates CLI commands from business logic:

```
cortical/got/cli/
    __init__.py      # Router and shared utilities
    task.py          # Task commands
    decision.py      # Decision commands
    sprint.py        # Sprint commands
    handoff.py       # Handoff commands
    knowledge_transfer.py # KT commands
    query.py         # Query commands
    shared.py        # Common CLI utilities
```

Each command module handles:
- Argument parsing
- Output formatting
- Calling the API layer

Business logic lives in `cortical/got/api.py`, not in CLI modules.

## Naming Conventions

### Files

| Type | Convention | Examples |
|------|------------|----------|
| Module | `snake_case.py` | `transaction_manager.py`, `graph_walker.py` |
| Package | `snake_case/` | `cortical/got/`, `cortical/cdg/` |
| Test | `test_*.py` | `test_got_modules.py`, `test_transaction.py` |
| Private | `_prefix.py` | `_internal.py` (rarely needed) |

### Classes

| Type | Convention | Examples |
|------|------------|----------|
| Regular class | `PascalCase` | `TransactionManager`, `GoTManager` |
| Mixin | `*Mixin` suffix | `CoreMixin`, `QueryMixin` |
| Protocol/Interface | `*Protocol` or base name | `StorageBackend`, `GoTBackend` |
| Exception | `*Error` suffix | `ValidationError`, `ConflictError` |
| TypedDict | `*Schema` or `*Metadata` | `TaskSchema`, `TaskMetadata` |

### Functions and Methods

| Type | Convention | Examples |
|------|------------|----------|
| Public method | `snake_case` | `create_task()`, `compute_all()` |
| Private method | `_prefix` | `_validate_entity()`, `_sync_indexes()` |
| Factory | `create_*` or `make_*` | `create_container()`, `make_edge()` |
| Boolean | `is_*`, `has_*`, `can_*` | `is_valid()`, `has_changes()` |
| Getter | Just the noun | `status()`, `metadata()` |

### Variables

| Type | Convention | Examples |
|------|------------|----------|
| Local | `snake_case` | `task_id`, `entity_data` |
| Constant | `UPPER_SNAKE_CASE` | `MAX_RETRIES`, `DEFAULT_TIMEOUT` |
| Private | `_prefix` | `_cache`, `_dirty_flag` |
| TypeVar | Single uppercase or `T*` | `T`, `TEntity`, `TResult` |

## The Seven Pillars Architecture

This codebase follows a "Seven Pillars" architecture. Each pillar has a distinct responsibility:

```
cortical/
    cdg/        # Pillar 1: Foundation - Storage, Transactions, WAL
    reasoning/
        prism_*.py      # Pillar 2: PRISM - Hebbian learning
        woven_mind.py   # Pillar 5: Woven Mind - Dual-process cognition
        cognitive_loop.py # Pillar 7: QAPV - Verification cycle
    cel/        # Pillar 3: CEL - Event sourcing, Merkle DAG
    got/        # Pillar 4: GoT - Tasks, decisions, knowledge
    spark/      # Pillar 6: Spark - Fast language model
```

### Layer Dependencies

```
     CLI Layer
         |
         v
     API Layer (got/api.py, processor/__init__.py)
         |
         v
     Logic Layer (reasoning/, query/)
         |
         v
     Storage Layer (cdg/, persistence.py)
         |
         v
     Foundation (common/, utils/)
```

Rules:
- Upper layers may depend on lower layers
- Lower layers NEVER import from upper layers
- Sibling modules at the same level avoid direct imports

## Separation of Concerns

### The Four-Layer Pattern

```
1. CLI Layer (cortical/got/cli/, cortical/cli/)
   - Argument parsing
   - Output formatting
   - User interaction
   - NEVER contains business logic

2. API Layer (cortical/got/api.py, cortical/processor/)
   - Public interface
   - Orchestration
   - Transaction boundaries
   - Input validation

3. Logic Layer (cortical/reasoning/, cortical/query/)
   - Business rules
   - Algorithms
   - Pure functions where possible
   - No I/O

4. Storage Layer (cortical/cdg/, cortical/persistence.py)
   - Data persistence
   - Caching
   - Index management
   - ACID guarantees
```

### Example: Creating a Task

```
CLI (got/cli/task.py)
    |-- Parses: python -m cortical.got task create "Title"
    |-- Calls: api.create_task(title)
    |
API (got/api.py)
    |-- Validates: title is non-empty
    |-- Starts: transaction
    |-- Calls: storage.write(entity)
    |
Storage (cdg/storage.py)
    |-- Serializes: entity to JSON
    |-- Writes: to disk with checksum
    |-- Updates: indexes
```

### What Belongs Where

| Component | Location | Rationale |
|-----------|----------|-----------|
| Entity types | `got/types.py`, `cdg/types.py` | Core data structures |
| Validation | `got/validation.py`, `cdg/schema/` | Separate from storage |
| CLI commands | `got/cli/*.py` | One file per command group |
| Business logic | `got/api.py`, `reasoning/` | Testable without I/O |
| Storage ops | `cdg/storage.py` | All I/O in one place |
| Error types | `got/errors.py`, `cdg/errors.py` | Per-module errors |
| Config | `got/config.py`, `cdg/config.py` | Separate from logic |

## Avoiding "Utils" Dumping Grounds

The `utils/` directory is NOT a dumping ground. It has specific, focused modules:

```
cortical/utils/
    checksums.py     # Checksum computation and verification
    id_generation.py # ID generation for entities
    locking.py       # Process and thread locking
    persistence.py   # Low-level persistence helpers
    text.py          # Text manipulation utilities
```

### Rules for Utils

1. **Each file has ONE responsibility**: `checksums.py` only does checksums
2. **No business logic**: Utils are pure, reusable functions
3. **No dependencies on higher layers**: Utils don't import from `got/` or `cdg/`
4. **Well-documented**: Each function explains its purpose
5. **Well-tested**: Utils need thorough unit tests

### When to Add to Utils

Add to utils when:
- The function is used by 3+ modules
- It has no business logic
- It could exist as a standalone library
- It's truly general-purpose

Don't add to utils:
- Module-specific helpers (keep in that module)
- Business logic (belongs in API/Logic layer)
- I/O operations (belongs in Storage layer)

### The Common Pattern

For cross-cutting infrastructure, use `common/`:

```
cortical/common/
    __init__.py        # Public exports
    container.py       # Dependency injection
    filesystem.py      # FileSystem abstraction
    recovery_types.py  # Shared recovery types
```

`common/` is for:
- DI container and modules
- Abstract base classes/protocols
- Types shared across pillars
- Infrastructure that all pillars need

## Import Organization

### Import Order

Follow this order, with blank lines between groups:

```python
# 1. Standard library
from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# 2. Third-party (if any - rare in this codebase)
import pytest  # Only in tests

# 3. Local package (absolute imports)
from cortical.common import Container
from cortical.cdg import CDGStore, Entity
from cortical.got import GoTManager

# 4. Relative imports (same package only)
from .types import Task, Decision
from .errors import ValidationError
```

### Avoiding Circular Imports

**Pattern 1: Type-only imports**

```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cortical.got.api import GoTManager  # Only for type hints
```

**Pattern 2: Import in function**

```python
def late_binding_example():
    # Import when needed, not at module level
    from cortical.got.api import GoTManager
    return GoTManager()
```

**Pattern 3: Protocol/Interface**

```python
# In cortical/got/protocol.py
class GoTBackend(Protocol):
    def read(self, entity_id: str) -> Entity: ...
    def write(self, entity: Entity) -> None: ...

# Other modules depend on Protocol, not implementation
```

### The `__init__.py` Contract

The `__init__.py` defines the public API:

```python
# cortical/got/__init__.py
from .types import Task, Decision, Edge
from .api import GoTManager
from .errors import GoTError, ValidationError

__all__ = [
    'Task', 'Decision', 'Edge',
    'GoTManager',
    'GoTError', 'ValidationError',
]
```

Rules:
- Only export what users need
- Keep implementation details private
- Use `__all__` to make exports explicit
- Document the module's purpose in the docstring

## Refactoring for Clarity

### Safe Refactoring Steps

1. **Write tests first**: Before changing, ensure test coverage
2. **One change at a time**: Don't combine refactors
3. **Run tests after each change**: Catch regressions immediately
4. **Preserve the interface**: Don't break callers

### Common Refactoring Patterns

**Extract Method**
```python
# Before
def process_entity(entity):
    # 50 lines of validation
    # 50 lines of transformation
    # 50 lines of persistence

# After
def process_entity(entity):
    validated = _validate_entity(entity)
    transformed = _transform_entity(validated)
    return _persist_entity(transformed)
```

**Replace Conditional with Polymorphism**
```python
# Before
def handle_entity(entity):
    if entity.type == 'task':
        # task-specific code
    elif entity.type == 'decision':
        # decision-specific code

# After
class EntityHandler(Protocol):
    def handle(self, entity: Entity) -> None: ...

class TaskHandler:
    def handle(self, entity: Entity) -> None: ...

class DecisionHandler:
    def handle(self, entity: Entity) -> None: ...

handlers: Dict[str, EntityHandler] = {
    'task': TaskHandler(),
    'decision': DecisionHandler(),
}
```

**Introduce Parameter Object**
```python
# Before
def create_task(title, description, priority, category, tags, parent_id):
    ...

# After
@dataclass
class TaskCreateRequest:
    title: str
    description: str = ""
    priority: str = "medium"
    category: str = "task"
    tags: List[str] = field(default_factory=list)
    parent_id: Optional[str] = None

def create_task(request: TaskCreateRequest):
    ...
```

### When NOT to Refactor

- In the middle of a feature (finish first)
- Without tests (add tests first)
- For "cleanup" with no clear benefit
- When it would break the public API

### The Refactoring Checklist

Before refactoring:
1. [ ] Tests exist and pass
2. [ ] The change has clear motivation
3. [ ] I understand the current behavior

During refactoring:
1. [ ] One small change at a time
2. [ ] Tests pass after each change
3. [ ] No behavior changes

After refactoring:
1. [ ] All tests still pass
2. [ ] Code is more readable
3. [ ] No new warnings or errors
4. [ ] Commit with clear message

## Summary

| Principle | Implementation |
|-----------|----------------|
| Single Responsibility | One module = one job |
| Layered Architecture | CLI -> API -> Logic -> Storage |
| Clean Interfaces | `__init__.py` defines public API |
| Dependency Injection | Container manages wiring |
| Avoid Utils Dumping | Focused, well-named utility files |
| Import Discipline | Order, TYPE_CHECKING, protocols |
| Safe Refactoring | Tests first, one change at a time |

The goal is code that future developers (including AI agents) can understand, modify, and extend without fear.
