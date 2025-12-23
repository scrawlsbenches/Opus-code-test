# Client API Design Patterns for Purpose-Built Databases

## Overview

This guide provides concrete Python API patterns for databases that are used during their own development (dogfooding). The focus is on making the API intuitive for developers who are **both building and testing the system**.

Key principles:
- **Fluent APIs** - Method chaining for natural, readable operations
- **Context managers** - Python idioms for transactions and sessions
- **Progressive disclosure** - Simple for common cases, powerful options for experts
- **Self-diagnosing errors** - Errors that suggest fixes, not just what went wrong
- **Testability-first** - Design APIs that are easy to mock and test
- **Graceful evolution** - API versioning without breaking changes

---

## 1. Fluent API Patterns

### 1.1 The Fluent Builder Pattern

The fluent API uses method chaining to build readable, natural-language-like operations. Each method returns `self`, enabling the builder pattern.

**Pattern:**
```python
class FluentAPI:
    """Builder with method chaining."""

    def operation_one(self, param: str) -> 'FluentAPI':
        """Perform operation 1 (chainable)."""
        # Perform operation
        self._state = param
        return self  # Return self for chaining

    def operation_two(self, value: int) -> 'FluentAPI':
        """Perform operation 2 (chainable)."""
        # Perform operation
        self._value = value
        return self

    # Terminal operations (return results, not self)
    def execute(self) -> Dict[str, Any]:
        """Execute and return results."""
        return {"state": self._state, "value": self._value}

    def save_to(self, path: str) -> str:
        """Save and return path."""
        # Implementation
        return path
```

**Usage:**
```python
# Natural, readable flow
result = (FluentAPI()
    .operation_one("step1")
    .operation_two(42)
    .execute())
```

**Real-world example from cortical/fluent.py:**
```python
from cortical import FluentProcessor

# Simple usage - readable like English
processor = (FluentProcessor()
    .add_document("doc1", "Neural networks process information")
    .add_document("doc2", "Deep learning uses neural architectures")
    .build(verbose=False)
    .search("neural processing", top_n=5))

# Handles common case with minimal boilerplate
results = processor[0]  # Most relevant result

# Progressive disclosure - more options available
processor = (FluentProcessor()
    .add_documents({
        "doc1": "content 1",
        "doc2": "content 2"
    })
    .with_config(CorticalConfig(
        pagerank_damping=0.9,
        cluster_strictness=0.8
    ))
    .build(
        verbose=False,
        build_concepts=True,
        pagerank_method='semantic',
        cluster_strictness=0.8
    )
    .save("corpus.json"))
```

### 1.2 Distinguishing Chainable vs Terminal Operations

**Key distinction**: Some methods return `self` (chainable), while others return results (terminal).

```python
class DataQuery:
    """Example showing chainable vs terminal operations."""

    # Chainable methods (configure, transform, add)
    def filter(self, condition: Callable) -> 'DataQuery':
        """Filter records (chainable)."""
        self._filters.append(condition)
        return self

    def sort_by(self, field: str) -> 'DataQuery':
        """Sort results (chainable)."""
        self._sort_field = field
        return self

    def limit(self, n: int) -> 'DataQuery':
        """Limit result count (chainable)."""
        self._limit = n
        return self

    # Terminal methods (consume, execute, return results)
    def get(self) -> List[Dict]:
        """Execute query and return results."""
        # Implementation
        return results

    def count(self) -> int:
        """Get count without returning full results."""
        return len(self.get())

    def first(self) -> Optional[Dict]:
        """Get first result or None."""
        results = self.limit(1).get()
        return results[0] if results else None

    def __str__(self) -> str:
        """String representation (non-terminal)."""
        return f"Query(filters={len(self._filters)}, sort={self._sort_field})"
```

**Usage pattern:**
```python
# Build query with chainable methods
query = (DataQuery()
    .filter(lambda r: r['status'] == 'active')
    .sort_by('created_at')
    .limit(10))

# Terminal operation - actually runs query
results = query.get()

# Each terminal operation can be called independently
count = query.count()
first = query.first()

# Query is reusable - each call executes independently
more_results = query.limit(20).get()
```

### 1.3 Factory Methods for Progressive Disclosure

Use class methods to handle different initialization patterns:

```python
class Processor:
    """Database processor with multiple construction paths."""

    def __init__(self, config: Optional[Config] = None):
        """Basic initialization (simple case)."""
        self.config = config or Config()
        self._data = {}

    @classmethod
    def from_files(cls, paths: List[str]) -> 'Processor':
        """Load from files (common case)."""
        processor = cls()
        for path in paths:
            processor._load_file(path)
        return processor

    @classmethod
    def from_directory(
        cls,
        directory: str,
        pattern: str = "*.txt",
        recursive: bool = False
    ) -> 'Processor':
        """Load from directory (batch case)."""
        processor = cls()
        processor._load_directory(directory, pattern, recursive)
        return processor

    @classmethod
    def from_existing(cls, other: 'Processor') -> 'Processor':
        """Clone/wrap existing processor (integration case)."""
        processor = cls(other.config)
        processor._data = dict(other._data)
        return processor

    @classmethod
    def load(cls, path: str) -> 'Processor':
        """Restore from saved state (persistence case)."""
        # Load from disk
        return processor
```

**Usage:**
```python
# Simple case
proc1 = Processor()

# Load from files
proc2 = Processor.from_files(["file1.txt", "file2.txt"])

# Load from directory
proc3 = Processor.from_directory("./data", pattern="*.md")

# Clone/wrap
proc4 = Processor.from_existing(proc1)

# Restore
proc5 = Processor.load("saved_state.pkl")
```

---

## 2. Context Managers for Transactions and Sessions

### 2.1 The Context Manager Pattern

Use `__enter__` and `__exit__` for automatic resource management, transactions, and sessions.

```python
from contextlib import contextmanager
from typing import Generator

class Transaction:
    """Explicit transaction context manager."""

    def __init__(self, database):
        self.database = database
        self.is_active = False
        self.changes = []

    def __enter__(self) -> 'Transaction':
        """Begin transaction."""
        self.is_active = True
        self.changes = []
        self.database._log("Transaction started")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Commit or rollback on exit."""
        if exc_type is not None:
            # Exception occurred - rollback
            self.database._rollback(self.changes)
            self.database._log(f"Transaction rolled back: {exc_type.__name__}")
            return False  # Re-raise exception
        else:
            # Success - commit
            self.database._commit(self.changes)
            self.database._log("Transaction committed")
            return True  # Suppress any exception

class Database:
    """Database with transactional support."""

    def transaction(self) -> Transaction:
        """Create a transaction context."""
        return Transaction(self)

    def execute(self, query: str) -> Any:
        """Execute query within implicit transaction."""
        # Implementation
        pass

    def _log(self, message: str):
        """Internal logging."""
        pass

    def _commit(self, changes: list):
        """Commit changes."""
        pass

    def _rollback(self, changes: list):
        """Rollback changes."""
        pass
```

**Usage:**
```python
db = Database()

# Automatic rollback on error
try:
    with db.transaction() as tx:
        db.execute("INSERT INTO users VALUES (...)")
        db.execute("INSERT INTO logs VALUES (...)")
        # If second insert fails, first is rolled back
except Exception as e:
    print(f"Transaction failed: {e}")

# Automatic commit on success
with db.transaction():
    db.execute("UPDATE account SET balance = balance - 100")
    db.execute("UPDATE account SET balance = balance + 100")
    # Both committed together, or both rolled back
```

### 2.2 Context Manager Generator Pattern

Use `@contextmanager` decorator for simpler cases:

```python
from contextlib import contextmanager

@contextmanager
def database_session(connection_string: str) -> Generator:
    """Create a database session context."""
    connection = None
    try:
        # Setup
        connection = create_connection(connection_string)
        connection.begin()
        yield connection
        # Teardown (success)
        connection.commit()
    except Exception:
        # Teardown (failure)
        if connection:
            connection.rollback()
        raise
    finally:
        # Always cleanup
        if connection:
            connection.close()

# Usage
with database_session("postgresql://...") as conn:
    result = conn.execute("SELECT * FROM users")
    # Auto-commits on success, rolls back on error, always closes
```

### 2.3 Session Context with State Management

For more complex session management:

```python
from enum import Enum
from typing import Optional

class SessionState(Enum):
    """Session lifecycle states."""
    CREATED = "created"
    ACTIVE = "active"
    COMMITTING = "committing"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"

class Session:
    """Session with explicit state tracking (for GoT-style systems)."""

    def __init__(self, session_id: str, store: 'DataStore'):
        self.session_id = session_id
        self.store = store
        self.state = SessionState.CREATED
        self.changes: Dict[str, Any] = {}
        self.reads: Dict[str, int] = {}  # Track read versions for conflict detection

    def __enter__(self) -> 'Session':
        """Enter session context."""
        self.state = SessionState.ACTIVE
        self.store._register_session(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Exit session context - commit or rollback."""
        try:
            if exc_type is None:
                # No exception - attempt commit
                self.state = SessionState.COMMITTING
                result = self.store._commit(self)
                if result.success:
                    self.state = SessionState.COMMITTED
                else:
                    self.state = SessionState.ROLLED_BACK
                    # Conflict during commit - don't suppress error
                    raise ConflictError(
                        f"Commit conflict: {result.conflict_info}",
                        version_mismatch=result.conflict_info
                    )
            else:
                # Exception occurred - rollback
                self.state = SessionState.ROLLED_BACK
                self.store._rollback(self)
                # Don't suppress the exception
                return False
        finally:
            self.store._unregister_session(self)

    def write(self, entity_id: str, entity: Any) -> None:
        """Buffer a write (visible to reads within this session)."""
        if not self.state == SessionState.ACTIVE:
            raise RuntimeError(f"Session is not active: {self.state.value}")
        self.changes[entity_id] = entity

    def read(self, entity_id: str) -> Optional[Any]:
        """Read entity, checking own writes first."""
        if not self.state == SessionState.ACTIVE:
            raise RuntimeError(f"Session is not active: {self.state.value}")

        # Read your own writes (uncommitted changes)
        if entity_id in self.changes:
            return self.changes[entity_id]

        # Read from store
        entity, version = self.store._get(entity_id)
        self.reads[entity_id] = version  # Track for conflict detection
        return entity

class DataStore:
    """Store with session support."""

    def __init__(self):
        self._data: Dict[str, tuple] = {}  # id -> (entity, version)
        self._sessions: set = set()
        self._version = 0

    def session(self) -> Session:
        """Create a new session."""
        return Session(f"S-{self._version}", self)

    def _commit(self, session: Session) -> Dict[str, Any]:
        """Commit a session's changes."""
        # Check for conflicts (simplified)
        for entity_id in session.changes:
            if entity_id in session.reads:
                current_version = self._data.get(entity_id, (None, 0))[1]
                read_version = session.reads[entity_id]
                if current_version != read_version:
                    return {
                        'success': False,
                        'conflict_info': {
                            'entity_id': entity_id,
                            'read_version': read_version,
                            'current_version': current_version
                        }
                    }

        # Apply writes
        self._version += 1
        for entity_id, entity in session.changes.items():
            self._data[entity_id] = (entity, self._version)

        return {'success': True, 'version': self._version}

    def _rollback(self, session: Session):
        """Rollback a session."""
        session.changes.clear()

    def _register_session(self, session: Session):
        """Register active session."""
        self._sessions.add(session.session_id)

    def _unregister_session(self, session: Session):
        """Unregister session."""
        self._sessions.discard(session.session_id)

    def _get(self, entity_id: str) -> tuple:
        """Get entity and version."""
        if entity_id in self._data:
            return self._data[entity_id]
        return (None, 0)
```

**Usage:**
```python
store = DataStore()

# Session with automatic conflict detection
try:
    with store.session() as s:
        # Read triggers version capture
        user = s.read("user:123")

        # Modify
        user['email'] = 'new@example.com'

        # Write
        s.write("user:123", user)

        # Auto-commit on exit (or rollback if conflict detected)
except ConflictError as e:
    print(f"Cannot commit: {e.context['version_mismatch']}")
```

---

## 3. Progressive Disclosure

Progressive disclosure makes simple operations require minimal code, while allowing power users to access advanced options.

### 3.1 Config Objects with Sensible Defaults

```python
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class Config:
    """Configuration with three disclosure levels."""

    # Level 1: Most common parameters
    # (defaults work for 80% of use cases)
    threshold: float = 0.5
    max_results: int = 10

    # Level 2: Intermediate tuning
    # (for users who know their domain)
    damping_factor: float = 0.85
    iteration_limit: int = 100
    convergence_tolerance: float = 1e-6

    # Level 3: Expert configuration
    # (for power users and researchers)
    relation_weights: Dict[str, float] = field(default_factory=lambda: {
        'strong': 1.5,
        'weak': 0.5,
        'semantic': 1.2,
        'syntactic': 0.8,
    })
    algorithm_variants: Dict[str, bool] = field(default_factory=lambda: {
        'use_cache': True,
        'parallel_execution': False,
        'approximate_mode': False,
    })
    internal_buffers: Dict[str, int] = field(default_factory=lambda: {
        'max_pending_writes': 1000,
        'max_read_cache_size': 10000,
    })

    def get_simple_config(self) -> Dict[str, Any]:
        """Get Level 1 parameters (most common usage)."""
        return {
            'threshold': self.threshold,
            'max_results': self.max_results,
        }

    def get_intermediate_config(self) -> Dict[str, Any]:
        """Get Levels 1 + 2 parameters."""
        return {
            **self.get_simple_config(),
            'damping_factor': self.damping_factor,
            'iteration_limit': self.iteration_limit,
            'convergence_tolerance': self.convergence_tolerance,
        }

    def get_all_config(self) -> Dict[str, Any]:
        """Get all parameters (expert level)."""
        return {
            **self.get_intermediate_config(),
            'relation_weights': self.relation_weights,
            'algorithm_variants': self.algorithm_variants,
            'internal_buffers': self.internal_buffers,
        }
```

**Usage at different levels:**
```python
# Level 1: Completely default (newbies)
config = Config()

# Level 2: Basic tuning (intermediate users)
config = Config(threshold=0.3, max_results=20)

# Level 3: Full control (power users)
config = Config(
    threshold=0.3,
    max_results=20,
    damping_factor=0.9,
    iteration_limit=50,
    relation_weights={'strong': 2.0, 'weak': 0.3},
    algorithm_variants={'parallel_execution': True}
)
```

### 3.2 Optional Parameters with Sensible Grouping

```python
class Search:
    """Search with progressive disclosure of options."""

    def find(
        self,
        query: str,
        # Level 1: Basic (required + most common)
        top_n: int = 5,
        # Level 2: Refinement (optional, domain-specific)
        min_score: float = 0.0,
        boost_recent: bool = True,
        # Level 3: Advanced (expert tweaking)
        use_cache: bool = True,
        custom_weights: Optional[Dict[str, float]] = None,
        debug_scoring: bool = False,
    ) -> List[Dict]:
        """
        Find items matching query.

        Args:
            query: Search string (Level 1)
            top_n: Number of results (Level 1)
            min_score: Filter results below threshold (Level 2)
            boost_recent: Prioritize recent items (Level 2)
            use_cache: Use cached query results (Level 3)
            custom_weights: Weights for scoring algorithm (Level 3)
            debug_scoring: Log scoring details (Level 3)

        Returns:
            List of matching items sorted by relevance
        """
        # Implementation
        pass
```

### 3.3 Method Variants for Different Use Cases

```python
class Database:
    """Database with method variants for different use cases."""

    def query(self, sql: str) -> List[Dict]:
        """Simple query (Level 1)."""
        # Basic, synchronous execution
        pass

    def query_detailed(
        self,
        sql: str,
        explain_plan: bool = False,
        timeout_ms: int = 30000,
        cache_ttl_seconds: int = 300,
    ) -> Dict[str, Any]:
        """Query with detailed options (Level 2)."""
        result = self.query(sql)
        return {
            'results': result,
            'explain_plan': self._explain(sql) if explain_plan else None,
            'timing_ms': self._get_timing(),
        }

    async def query_async(self, sql: str) -> List[Dict]:
        """Async query for non-blocking I/O (Level 2)."""
        return await self._async_execute(sql)

    def query_streaming(
        self,
        sql: str,
        batch_size: int = 1000,
    ) -> Iterator[List[Dict]]:
        """Stream results for large datasets (Level 3)."""
        # Yield results in batches
        pass
```

---

## 4. Error Messages That Help

Good error messages teach the user how to fix the problem. They should:
- State what went wrong (clearly)
- Suggest what to do about it (specific fix)
- Provide context for debugging (what were you doing)
- Point to documentation if needed

### 4.1 Context-Aware Exception Hierarchy

```python
class DatabaseError(Exception):
    """Base exception with context-aware messaging."""

    def __init__(self, message: str, **context):
        """
        Initialize error with message and debugging context.

        Args:
            message: User-friendly error description
            **context: Debug information (operation, entity_id, version, etc)
        """
        super().__init__(message)
        self.message = message
        self.context = context

    def __str__(self) -> str:
        """Detailed error message with suggestions."""
        lines = [f"Error: {self.message}"]

        # Add context details
        if self.context:
            lines.append("\nContext:")
            for key, value in self.context.items():
                if value is not None:
                    lines.append(f"  {key}: {value}")

        # Add suggestions from subclass
        suggestions = self.get_suggestions()
        if suggestions:
            lines.append("\nHow to fix:")
            for suggestion in suggestions:
                lines.append(f"  • {suggestion}")

        return "\n".join(lines)

    def get_suggestions(self) -> List[str]:
        """Override in subclasses to provide helpful suggestions."""
        return []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable format."""
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'context': self.context,
            'suggestions': self.get_suggestions(),
        }

class ConflictError(DatabaseError):
    """Concurrent write conflict during commit."""

    def get_suggestions(self) -> List[str]:
        """Provide conflict resolution strategies."""
        entity_id = self.context.get('entity_id')
        read_version = self.context.get('read_version')
        current_version = self.context.get('current_version')

        suggestions = [
            f"Reload entity {entity_id} to see current state",
            "Merge changes manually or retry the operation",
        ]

        if current_version and read_version:
            suggestions.append(
                f"Entity changed since you read it "
                f"(was v{read_version}, now v{current_version})"
            )

        return suggestions

class ValidationError(DatabaseError):
    """Data validation failed."""

    def get_suggestions(self) -> List[str]:
        """Provide fix suggestions."""
        field = self.context.get('field')
        constraint = self.context.get('constraint')

        suggestions = []

        if field:
            suggestions.append(f"Check field '{field}'")

        if constraint == 'required':
            suggestions.append(f"Provide a value for {field}")
        elif constraint == 'type_mismatch':
            expected = self.context.get('expected_type')
            actual = self.context.get('actual_type')
            suggestions.append(
                f"Expected {expected}, got {actual}"
            )
        elif constraint == 'range':
            min_val = self.context.get('min')
            max_val = self.context.get('max')
            suggestions.append(
                f"Value must be between {min_val} and {max_val}"
            )

        return suggestions

class NotFoundError(DatabaseError):
    """Entity not found."""

    def get_suggestions(self) -> List[str]:
        """Suggest how to find the entity."""
        entity_id = self.context.get('entity_id')
        entity_type = self.context.get('entity_type')

        suggestions = [
            f"Verify {entity_type} ID: {entity_id}",
            f"Use list() to see available {entity_type}s",
            "Check if it was deleted by another session",
        ]

        # If we have similar entities, suggest them
        similar = self.context.get('similar_ids', [])
        if similar:
            suggestions.append(
                f"Did you mean: {', '.join(similar[:3])}?"
            )

        return suggestions
```

**Example error message output:**
```
Error: Cannot commit transaction TX-20250101-120000-a1b2c3d4

Context:
  entity_id: task:123
  read_version: 5
  current_version: 7
  read_time: 2025-01-01T12:00:00Z
  conflict_time: 2025-01-01T12:00:05Z

How to fix:
  • Reload entity task:123 to see current state
  • Merge changes manually or retry the operation
  • Entity changed since you read it (was v5, now v7)
```

### 4.2 Self-Diagnosing Validators

```python
class Validator:
    """Validate inputs and provide detailed diagnostic messages."""

    @staticmethod
    def validate_entity_id(entity_id: str) -> None:
        """
        Validate entity ID format.

        Raises:
            ValidationError with helpful context if invalid.
        """
        if not entity_id:
            raise ValidationError(
                "Entity ID cannot be empty",
                field='entity_id',
                constraint='required',
            )

        if len(entity_id) > 255:
            raise ValidationError(
                f"Entity ID too long: {len(entity_id)} characters",
                field='entity_id',
                constraint='max_length',
                max_length=255,
                actual_length=len(entity_id),
            )

        if not entity_id[0].isalpha():
            raise ValidationError(
                "Entity ID must start with a letter",
                field='entity_id',
                constraint='format',
                example='task:123 (not 123:task)',
                got=entity_id,
            )

    @staticmethod
    def validate_version(version: int, context: str = 'unknown') -> None:
        """Validate version is positive integer."""
        if not isinstance(version, int):
            raise ValidationError(
                f"Version must be an integer, got {type(version).__name__}",
                field='version',
                constraint='type_mismatch',
                expected_type='int',
                actual_type=type(version).__name__,
                context=context,
            )

        if version < 0:
            raise ValidationError(
                f"Version must be non-negative, got {version}",
                field='version',
                constraint='range',
                min=0,
                actual=version,
                context=context,
            )
```

---

## 5. Testability-First Design

### 5.1 Dependency Injection for Easy Mocking

```python
from abc import ABC, abstractmethod
from unittest.mock import MagicMock

class StorageBackend(ABC):
    """Abstract storage interface for easy mocking."""

    @abstractmethod
    def get(self, key: str) -> Any:
        """Get value by key."""
        pass

    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        """Set value by key."""
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete value by key."""
        pass

class FileStorage(StorageBackend):
    """Real file-based storage."""

    def get(self, key: str) -> Any:
        # Real implementation
        pass

    def set(self, key: str, value: Any) -> None:
        # Real implementation
        pass

    def delete(self, key: str) -> None:
        # Real implementation
        pass

class Database:
    """Database with injected storage backend."""

    def __init__(self, storage: StorageBackend):
        """
        Initialize with storage backend.

        Args:
            storage: StorageBackend implementation (real or mock)
        """
        self.storage = storage

    def get_user(self, user_id: str) -> Optional[Dict]:
        """Get user by ID."""
        return self.storage.get(f"user:{user_id}")

    def save_user(self, user: Dict) -> None:
        """Save user."""
        self.storage.set(f"user:{user['id']}", user)
```

**Testing with mocks:**
```python
import pytest

def test_database_with_mock():
    """Test database using mock storage."""
    # Create mock storage
    mock_storage = MagicMock(spec=StorageBackend)
    mock_storage.get.return_value = {'id': '123', 'name': 'Alice'}

    # Create database with mock
    db = Database(mock_storage)

    # Test
    user = db.get_user('123')
    assert user['name'] == 'Alice'

    # Verify storage was called correctly
    mock_storage.get.assert_called_once_with('user:123')

def test_database_with_real_storage(tmp_path):
    """Integration test with real storage."""
    storage = FileStorage(tmp_path)
    db = Database(storage)

    user = {'id': '456', 'name': 'Bob'}
    db.save_user(user)

    retrieved = db.get_user('456')
    assert retrieved == user
```

### 5.2 Fixture Patterns for Shared Test Data

```python
import pytest
from unittest.mock import MagicMock

@pytest.fixture
def mock_storage():
    """Create a mock storage backend."""
    return MagicMock(spec=StorageBackend)

@pytest.fixture
def database(mock_storage):
    """Create a database with mock storage."""
    return Database(mock_storage)

@pytest.fixture
def sample_user():
    """Sample user data for testing."""
    return {
        'id': 'user:123',
        'name': 'Test User',
        'email': 'test@example.com',
        'status': 'active',
    }

@pytest.fixture
def populated_database(database, sample_user):
    """Database with sample data pre-loaded."""
    database.storage.get.return_value = sample_user
    return database

class TestDatabase:
    """Test suite for Database class."""

    def test_get_user_exists(self, database, sample_user):
        """Test getting existing user."""
        database.storage.get.return_value = sample_user

        user = database.get_user('123')

        assert user == sample_user

    def test_get_user_not_found(self, database):
        """Test getting non-existent user."""
        database.storage.get.return_value = None

        user = database.get_user('not-exists')

        assert user is None

    def test_save_user(self, database, sample_user):
        """Test saving user."""
        database.save_user(sample_user)

        # Verify storage.set was called
        database.storage.set.assert_called_once_with(
            'user:user:123',
            sample_user
        )
```

### 5.3 Test Doubles Pattern

```python
class TestableProcessor:
    """Processor designed for easy testing."""

    def __init__(self,
                 storage: StorageBackend,
                 clock: 'Clock' = None):
        """
        Initialize processor with dependencies.

        Args:
            storage: StorageBackend for persistence
            clock: Clock for timestamp injection (default: system clock)
        """
        self.storage = storage
        self.clock = clock or SystemClock()

    def process(self, item: Dict) -> Dict:
        """Process item with timestamp."""
        item['processed_at'] = self.clock.now()
        self.storage.set(f"item:{item['id']}", item)
        return item

# In tests: inject a fake clock to control time
class FakeClock:
    """Fake clock for testing time-dependent logic."""

    def __init__(self):
        self.current_time = '2025-01-01T12:00:00Z'

    def now(self) -> str:
        return self.current_time

    def advance(self, seconds: int) -> None:
        """Advance time for testing."""
        # Implementation
        pass

def test_processor_with_fake_clock():
    """Test processor with controlled time."""
    mock_storage = MagicMock(spec=StorageBackend)
    fake_clock = FakeClock()

    processor = TestableProcessor(mock_storage, fake_clock)

    fake_clock.current_time = '2025-01-01T12:00:00Z'
    item1 = processor.process({'id': '1', 'data': 'test'})
    assert item1['processed_at'] == '2025-01-01T12:00:00Z'

    fake_clock.advance(3600)  # Advance 1 hour
    item2 = processor.process({'id': '2', 'data': 'test'})
    assert item2['processed_at'] == '2025-01-01T13:00:00Z'
```

---

## 6. API Evolution Without Breaking Changes

### 6.1 Versioned Transactions and Entities

```python
class Entity:
    """Base entity with version tracking."""

    # Schema version for backward compatibility
    SCHEMA_VERSION = 2

    def __init__(self, id: str, version: int = 1):
        self.id = id
        self.version = version
        self._schema_version = self.SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        """Serialize with schema version."""
        return {
            '__schema_version__': self._schema_version,
            'id': self.id,
            'version': self.version,
            # ... other fields
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Entity':
        """Deserialize with schema migration."""
        schema_version = data.get('__schema_version__', 1)

        # Migrate old schemas to current schema
        if schema_version == 1:
            data = cls._migrate_v1_to_v2(data)
        elif schema_version > cls.SCHEMA_VERSION:
            raise ValueError(
                f"Data schema v{schema_version} is newer than "
                f"supported schema v{cls.SCHEMA_VERSION}"
            )

        return cls(
            id=data['id'],
            version=data['version'],
        )

    @staticmethod
    def _migrate_v1_to_v2(data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate v1 schema to v2."""
        # Add new required fields with defaults
        if 'new_field' not in data:
            data['new_field'] = 'default_value'

        # Rename fields if needed
        if 'old_name' in data:
            data['new_name'] = data.pop('old_name')

        return data
```

### 6.2 Deprecated Methods with Clear Migration Path

```python
import warnings
from functools import wraps

def deprecated(message: str):
    """Decorator to mark methods as deprecated."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"'{func.__name__}' is deprecated. {message}",
                category=DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)
        return wrapper
    return decorator

class Database:
    """Database with evolution support."""

    # Old API (deprecated)
    @deprecated(
        "Use find_entities(type='user') instead. "
        "See https://docs.example.com/migration-guide"
    )
    def get_all_users(self) -> List[Dict]:
        """Get all users (deprecated)."""
        return self.find_entities(type='user')

    # New API (recommended)
    def find_entities(self, type: str, **filters) -> List[Dict]:
        """Find entities by type."""
        # Implementation
        pass

    # Backward-compatible change: add optional parameter
    def query(self,
              sql: str,
              timeout_ms: int = 30000,  # New optional parameter
              cache: bool = True,  # New optional parameter with default
              ) -> List[Dict]:
        """Execute query."""
        # Implementation
        pass

    # Breaking change with explicit version gating
    def query_v2(self,
                 sql: str,
                 config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute query (v2 API).

        Breaking changes from v1:
        - Returns dict with results + metadata
        - Requires config parameter
        - Async by default

        Migration: Use find_query_v2() helper
        """
        # Implementation
        pass

    @classmethod
    def find_query_v2(cls, sql: str, **options) -> Dict[str, Any]:
        """Helper to migrate from query() to query_v2()."""
        config = {
            'timeout_ms': options.get('timeout_ms', 30000),
            'cache': options.get('cache', True),
            # New options specific to v2
            'explain': options.get('explain', False),
        }
        return cls.query_v2(sql, config)
```

### 6.3 Feature Flags for Gradual Rollout

```python
from enum import Enum

class FeatureFlag(Enum):
    """Feature flags for controlling API changes."""
    # Experimental features (may change or be removed)
    PARALLEL_QUERIES = "parallel_queries"
    SEMANTIC_CACHING = "semantic_caching"

    # New but stable features
    QUERY_V2_API = "query_v2_api"
    ASYNC_TRANSACTIONS = "async_transactions"

class Database:
    """Database with feature flags."""

    def __init__(self, features: set = None):
        """
        Initialize database with feature flags.

        Args:
            features: Set of enabled FeatureFlag enums
        """
        self.features = features or set()

    def is_feature_enabled(self, flag: FeatureFlag) -> bool:
        """Check if feature is enabled."""
        return flag in self.features

    def query(self, sql: str, **kwargs) -> List[Dict]:
        """Query with optional v2 behavior based on feature flag."""
        if self.is_feature_enabled(FeatureFlag.QUERY_V2_API):
            # Use new v2 implementation
            result = self._query_v2(sql, kwargs)
            return result['results']
        else:
            # Use legacy v1 implementation
            return self._query_v1(sql, kwargs)

    def parallel_search(self, queries: List[str]) -> Dict[str, List[Dict]]:
        """Search multiple queries in parallel (experimental)."""
        if not self.is_feature_enabled(FeatureFlag.PARALLEL_QUERIES):
            # Fallback to sequential
            return {q: self.query(q) for q in queries}

        # Use parallel implementation
        return self._parallel_search(queries)
```

**Usage:**
```python
# Start with stable features only
db = Database()

# Opt-in to new features
db = Database(features={
    FeatureFlag.QUERY_V2_API,
    FeatureFlag.ASYNC_TRANSACTIONS,
})

# Experimental features require opt-in
if db.is_feature_enabled(FeatureFlag.PARALLEL_QUERIES):
    results = db.parallel_search([...])
```

---

## 7. Real-World Examples from Cortical

### 7.1 Fluent API for Text Processing

From `cortical/fluent.py`:

```python
# Simple case (Level 1)
processor = (FluentProcessor()
    .add_document("doc1", "text")
    .build()
    .search("query"))

# Intermediate (Level 2)
processor = (FluentProcessor()
    .add_documents({
        "doc1": "content 1",
        "doc2": "content 2"
    })
    .build(verbose=False)
    .save("corpus.json"))

# Advanced (Level 3)
processor = (FluentProcessor()
    .from_directory("./docs", pattern="*.md", recursive=True)
    .with_config(CorticalConfig(
        pagerank_damping=0.9,
        cluster_strictness=0.8,
        relation_weights={'IsA': 2.0}
    ))
    .build(
        verbose=True,
        pagerank_method='semantic',
        build_concepts=True
    )
    .save("corpus.json"))
```

### 7.2 Transaction Pattern from GoT

From `cortical/got/transaction.py`:

```python
from cortical.got import TransactionManager, Task

manager = TransactionManager(got_dir)

# Transaction-like pattern (manual)
tx = manager.begin()
try:
    task = Task(id="T-001", title="Test", status="pending")
    manager.write(tx, task)
    result = manager.commit(tx)
    if result.success:
        print(f"Committed at version {result.version}")
except ConflictError as e:
    manager.rollback(tx)
    print(f"Conflict: {e.context}")
```

Or with context manager wrapper:

```python
# Context manager pattern (idiomatic Python)
with manager.transaction() as tx:
    task = Task(id="T-001", title="Test", status="pending")
    manager.write(tx, task)
    # Auto-commits on success, auto-rollbacks on error
```

### 7.3 Error Handling from GoT

From `cortical/got/errors.py`:

```python
from cortical.got import ConflictError, NotFoundError

try:
    task = manager.get_task("T-123")
except NotFoundError as e:
    print(e)
    # Error: Task not found
    # Context:
    #   task_id: T-123
    #   searched_at: 2025-01-01T12:00:00Z
    # How to fix:
    #   • Check if task ID is correct
    #   • Use list_tasks() to see available tasks
    #   • Task may have been deleted
```

---

## 8. Patterns Summary Table

| Pattern | Purpose | Example | Benefit |
|---------|---------|---------|---------|
| **Fluent Builder** | Chainable configuration | `.add_doc().build().search()` | Readable, natural flow |
| **Factory Methods** | Multiple construction paths | `.from_files()`, `.from_directory()` | Flexible, discoverability |
| **Context Managers** | Resource cleanup & transactions | `with db.transaction():` | Idiomatic Python, safe |
| **Config Objects** | Progressive disclosure | `CorticalConfig()` with defaults | Simple defaults, power user options |
| **Custom Exceptions** | Self-diagnosing errors | `ConflictError` with suggestions | Users learn how to fix issues |
| **Dependency Injection** | Testable design | `Database(storage)` | Easy to mock, test without side effects |
| **Feature Flags** | Gradual rollout | `FeatureFlag.QUERY_V2_API` | Gradual migration, A/B testing |
| **Schema Versioning** | Forward/backward compatibility | `from_dict()` with migration | API evolution without breaking changes |
| **Test Fixtures** | Shared test data | `@pytest.fixture` | DRY tests, consistent setup |
| **Deprecated Decorator** | Clear migration paths | `@deprecated("Use new_func()")` | Users know what changed |

---

## 9. Design Checklist

When designing a dogfooding API, verify:

### ✓ Fluent/Progressive
- [ ] Simple case requires minimal boilerplate (2-3 lines)
- [ ] Advanced options available without breaking simplicity
- [ ] Clear distinction between chainable and terminal operations
- [ ] Multiple factory methods for common scenarios

### ✓ Context-Aware
- [ ] Context managers for automatic cleanup
- [ ] Transactions with ACID guarantees
- [ ] Session management with state tracking
- [ ] Read-your-own-writes semantics

### ✓ Self-Diagnosing
- [ ] Exceptions state what went wrong
- [ ] Exceptions suggest how to fix it
- [ ] Error context is JSON-serializable
- [ ] Validation errors point to documentation

### ✓ Testable
- [ ] Dependency injection for all external services
- [ ] Mockable interfaces (ABC + duck typing)
- [ ] Test fixtures for shared setup
- [ ] Test doubles (fake clock, etc.) for time-dependent logic

### ✓ Evolvable
- [ ] Schema versioning for entities
- [ ] Deprecated methods with migration guides
- [ ] Feature flags for gradual rollout
- [ ] Backward compatibility layer for old code

---

## 10. Further Reading

- **Fluent Interfaces**: https://martinfowler.com/bliki/FluentInterface.html
- **Builder Pattern**: Gang of Four Design Patterns
- **Context Managers**: PEP 343 – The "with" Statement
- **ACID Transactions**: Database Systems: The Complete Book
- **Python Dataclasses**: PEP 557
- **Feature Flags**: Feature Toggles (Martin Fowler)

