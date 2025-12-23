"""
Concrete Python examples for dogfooding database API design patterns.

This file contains runnable examples for each pattern discussed in
client-api-design-patterns.md. Use this as a reference implementation
when building your own database client APIs.

Run individual examples:
    python api-design-examples.py --example fluent_api
    python api-design-examples.py --example transactions
    python api-design-examples.py --example progressive_disclosure

Or run all:
    python api-design-examples.py --all
"""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Generator, Callable
from unittest.mock import MagicMock
import json


# ============================================================================
# EXAMPLE 1: FLUENT API PATTERNS
# ============================================================================

class DocumentProcessor:
    """Simple document processor (Level 1 - basic functionality)."""

    def __init__(self):
        self.documents: Dict[str, str] = {}

    def add_document(self, doc_id: str, content: str) -> 'DocumentProcessor':
        """Add document (chainable)."""
        self.documents[doc_id] = content
        return self

    def search(self, query: str) -> List[tuple]:
        """Search documents (terminal operation)."""
        results = []
        for doc_id, content in self.documents.items():
            if query.lower() in content.lower():
                results.append((doc_id, 1.0))
        return results


def example_fluent_api():
    """Example: Fluent API with method chaining."""
    print("=== FLUENT API EXAMPLE ===\n")

    # Simple usage - readable flow
    results = (DocumentProcessor()
               .add_document("doc1", "The quick brown fox")
               .add_document("doc2", "The lazy dog")
               .search("quick"))

    print(f"Search results: {results}")
    print()


# ============================================================================
# EXAMPLE 2: CONTEXT MANAGERS FOR TRANSACTIONS
# ============================================================================

class Transaction:
    """Explicit transaction context manager."""

    def __init__(self, database: 'SimpleDatabase'):
        self.database = database
        self.is_active = False
        self.changes: Dict[str, Any] = {}

    def __enter__(self) -> 'Transaction':
        """Begin transaction."""
        self.is_active = True
        self.changes = {}
        print("  [TX] Transaction started")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Commit or rollback on exit."""
        if exc_type is not None:
            # Exception occurred - rollback
            print(f"  [TX] Rolling back: {exc_type.__name__}")
            self.database._rollback(self.changes)
            return False  # Re-raise exception
        else:
            # Success - commit
            print("  [TX] Committing changes")
            self.database._commit(self.changes)
            return True

    def write(self, key: str, value: Any) -> 'Transaction':
        """Write within transaction (chainable for setup)."""
        self.changes[key] = value
        print(f"  [TX] Buffered write: {key}")
        return self


class SimpleDatabase:
    """Database with transactional support."""

    def __init__(self):
        self.data: Dict[str, Any] = {}
        self.version = 0

    def transaction(self) -> Transaction:
        """Create a transaction context."""
        return Transaction(self)

    def _commit(self, changes: Dict[str, Any]):
        """Commit changes."""
        self.data.update(changes)
        self.version += 1

    def _rollback(self, changes: Dict[str, Any]):
        """Rollback changes."""
        changes.clear()

    def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        return self.data.get(key)


def example_transactions():
    """Example: Context managers for transactions."""
    print("=== TRANSACTION CONTEXT MANAGER EXAMPLE ===\n")

    db = SimpleDatabase()

    # Automatic commit on success
    print("Scenario 1: Successful transaction")
    with db.transaction() as tx:
        tx.write("user:1", {"name": "Alice"})
        tx.write("user:2", {"name": "Bob"})
    print(f"  DB State: {db.data}")
    print()

    # Automatic rollback on error
    print("Scenario 2: Failed transaction (automatic rollback)")
    try:
        with db.transaction() as tx:
            tx.write("user:3", {"name": "Charlie"})
            raise ValueError("Validation error!")
            tx.write("user:4", {"name": "David"})  # Never reaches here
    except ValueError as e:
        print(f"  Error caught: {e}")
    print(f"  DB State (user:3 was not committed): {db.data}")
    print()


# ============================================================================
# EXAMPLE 3: PROGRESSIVE DISCLOSURE WITH CONFIG
# ============================================================================

@dataclass
class SearchConfig:
    """Configuration with three disclosure levels."""

    # Level 1: Most common (defaults work for 80% of cases)
    query: str = ""
    max_results: int = 10

    # Level 2: Intermediate tuning (for domain experts)
    min_score: float = 0.0
    boost_recent: bool = True
    use_cache: bool = True

    # Level 3: Expert configuration (for researchers)
    custom_weights: Dict[str, float] = field(default_factory=dict)
    debug_mode: bool = False
    algorithm_variant: str = "default"

    @property
    def simple_params(self) -> Dict[str, Any]:
        """Level 1 parameters (most common)."""
        return {'query': self.query, 'max_results': self.max_results}

    @property
    def intermediate_params(self) -> Dict[str, Any]:
        """Level 1 + Level 2 parameters."""
        return {
            **self.simple_params,
            'min_score': self.min_score,
            'boost_recent': self.boost_recent,
            'use_cache': self.use_cache,
        }

    @property
    def all_params(self) -> Dict[str, Any]:
        """All parameters (expert level)."""
        return {
            **self.intermediate_params,
            'custom_weights': self.custom_weights,
            'debug_mode': self.debug_mode,
            'algorithm_variant': self.algorithm_variant,
        }


def example_progressive_disclosure():
    """Example: Progressive disclosure with config objects."""
    print("=== PROGRESSIVE DISCLOSURE EXAMPLE ===\n")

    # Level 1: Completely default
    config1 = SearchConfig(query="test")
    print("Level 1 (Beginner):")
    print(f"  Config: {config1.simple_params}\n")

    # Level 2: Basic tuning
    config2 = SearchConfig(
        query="neural networks",
        max_results=20,
        min_score=0.3,
        boost_recent=True
    )
    print("Level 2 (Intermediate):")
    print(f"  Config: {config2.intermediate_params}\n")

    # Level 3: Full control
    config3 = SearchConfig(
        query="deep learning",
        max_results=50,
        min_score=0.5,
        boost_recent=False,
        use_cache=False,
        custom_weights={'neural': 2.0, 'network': 1.5},
        debug_mode=True,
        algorithm_variant='semantic'
    )
    print("Level 3 (Expert):")
    print(f"  Config: {config3.all_params}\n")


# ============================================================================
# EXAMPLE 4: SELF-DIAGNOSING ERROR MESSAGES
# ============================================================================

class DatabaseError(Exception):
    """Base exception with context-aware messaging."""

    def __init__(self, message: str, **context):
        super().__init__(message)
        self.message = message
        self.context = context

    def __str__(self) -> str:
        """Detailed error message with suggestions."""
        lines = [f"Error: {self.message}"]

        if self.context:
            lines.append("\nContext:")
            for key, value in self.context.items():
                if value is not None:
                    lines.append(f"  {key}: {value}")

        suggestions = self.get_suggestions()
        if suggestions:
            lines.append("\nHow to fix:")
            for suggestion in suggestions:
                lines.append(f"  • {suggestion}")

        return "\n".join(lines)

    def get_suggestions(self) -> List[str]:
        """Override in subclasses."""
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
    """Concurrent write conflict."""

    def get_suggestions(self) -> List[str]:
        """Provide conflict resolution strategies."""
        entity_id = self.context.get('entity_id')
        return [
            f"Reload entity {entity_id} to see current state",
            "Merge changes manually or retry the operation",
            "Check if another session modified this entity",
        ]


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
            suggestions.append(f"Expected {expected}, got {actual}")

        return suggestions


def example_error_messages():
    """Example: Self-diagnosing error messages."""
    print("=== SELF-DIAGNOSING ERROR MESSAGES ===\n")

    # Example 1: Conflict error
    print("Scenario 1: Conflict error with suggestions")
    try:
        raise ConflictError(
            "Cannot commit transaction",
            entity_id="task:123",
            read_version=5,
            current_version=7,
            read_time="2025-01-01T12:00:00Z",
        )
    except ConflictError as e:
        print(str(e))
        print()

    # Example 2: Validation error
    print("Scenario 2: Validation error with suggestions")
    try:
        raise ValidationError(
            "Invalid entity ID",
            field='entity_id',
            constraint='type_mismatch',
            expected_type='str',
            actual_type='int',
        )
    except ValidationError as e:
        print(str(e))
        print()

    # Example 3: JSON serialization (for APIs)
    print("Scenario 3: Error as JSON (for REST APIs)")
    error = ValidationError(
        "Invalid email format",
        field='email',
        constraint='format',
    )
    print(json.dumps(error.to_dict(), indent=2))
    print()


# ============================================================================
# EXAMPLE 5: TESTABILITY WITH DEPENDENCY INJECTION
# ============================================================================

class StorageBackend(ABC):
    """Abstract storage interface for easy mocking."""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        pass

    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        """Set value by key."""
        pass


class InMemoryStorage(StorageBackend):
    """Real in-memory storage implementation."""

    def __init__(self):
        self.data: Dict[str, Any] = {}

    def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        return self.data.get(key)

    def set(self, key: str, value: Any) -> None:
        """Set value by key."""
        self.data[key] = value


class UserRepository:
    """Repository with injected storage backend."""

    def __init__(self, storage: StorageBackend):
        """Initialize with storage backend."""
        self.storage = storage

    def get_user(self, user_id: str) -> Optional[Dict]:
        """Get user by ID."""
        return self.storage.get(f"user:{user_id}")

    def save_user(self, user: Dict) -> None:
        """Save user."""
        self.storage.set(f"user:{user['id']}", user)


def example_testability():
    """Example: Testing with dependency injection."""
    print("=== TESTABILITY WITH DEPENDENCY INJECTION ===\n")

    # Test 1: With real storage
    print("Test 1: Real storage")
    real_storage = InMemoryStorage()
    repo = UserRepository(real_storage)

    user = {'id': 'user123', 'name': 'Alice'}
    repo.save_user(user)
    retrieved = repo.get_user('user123')
    print(f"  Saved: {user}")
    print(f"  Retrieved: {retrieved}")
    assert retrieved == user
    print("  ✓ Test passed\n")

    # Test 2: With mock storage (for unit testing)
    print("Test 2: Mock storage (unit testing)")
    mock_storage = MagicMock(spec=StorageBackend)
    mock_storage.get.return_value = {'id': 'user456', 'name': 'Bob'}

    repo_mock = UserRepository(mock_storage)
    user = repo_mock.get_user('user456')

    print(f"  Retrieved: {user}")
    print(f"  Mock was called with: {mock_storage.get.call_args}")
    assert user['name'] == 'Bob'
    print("  ✓ Test passed (no database needed)\n")


# ============================================================================
# EXAMPLE 6: API EVOLUTION WITH DEPRECATION
# ============================================================================

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


class LegacyAPI:
    """API with old and new methods for evolution."""

    # Old API (deprecated)
    @deprecated(
        "Use find_by_type(type='user') instead. "
        "See https://docs.example.com/migration-guide"
    )
    def get_all_users(self) -> List[Dict]:
        """Get all users (deprecated)."""
        return self.find_by_type(type='user')

    # New API (recommended)
    def find_by_type(self, type: str) -> List[Dict]:
        """Find entities by type."""
        return [{'type': type, 'id': '1', 'name': f'{type}:1'}]

    # Backward-compatible change: add optional parameter
    def search(self,
               query: str,
               limit: int = 10,  # Existing param
               offset: int = 0,  # New optional param with default
               ) -> List[Dict]:
        """Search with new optional parameter."""
        return []


def example_api_evolution():
    """Example: API evolution without breaking changes."""
    print("=== API EVOLUTION AND DEPRECATION ===\n")

    api = LegacyAPI()

    # Old API still works but warns
    print("Calling deprecated method:")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        users = api.get_all_users()
        if w:
            print(f"  Warning: {w[0].message}\n")

    # New API (recommended)
    print("Using new API:")
    users = api.find_by_type(type='user')
    print(f"  Result: {users}\n")


# ============================================================================
# EXAMPLE 7: COMBINED REAL-WORLD SCENARIO
# ============================================================================

class SearchProcessor:
    """Real-world example combining multiple patterns."""

    def __init__(self, storage: StorageBackend, config: Optional[SearchConfig] = None):
        """Initialize with dependency injection and config."""
        self.storage = storage
        self.config = config or SearchConfig(query="")

    def with_config(self, config: SearchConfig) -> 'SearchProcessor':
        """Set configuration (chainable)."""
        self.config = config
        return self

    @contextmanager
    def search_session(self) -> Generator['SearchSession', None, None]:
        """Create a search session context."""
        session = SearchSession(self)
        try:
            yield session
        finally:
            session.close()

    def execute(self, query: str) -> List[Dict]:
        """Execute search (terminal operation)."""
        documents = self.storage.get("documents") or []
        results = [d for d in documents if query.lower() in d['content'].lower()]
        return results[:self.config.max_results]


class SearchSession:
    """Session for multi-step searches."""

    def __init__(self, processor: SearchProcessor):
        self.processor = processor
        self.history: List[str] = []

    def search(self, query: str) -> List[Dict]:
        """Search and track history."""
        self.history.append(query)
        return self.processor.execute(query)

    def close(self):
        """Cleanup session."""
        if self.history:
            print(f"  [Session] Searched for: {', '.join(self.history)}")


def example_real_world():
    """Example: Real-world scenario combining patterns."""
    print("=== REAL-WORLD SCENARIO ===\n")

    # Setup
    storage = InMemoryStorage()
    storage.set("documents", [
        {'id': '1', 'content': 'Neural networks are powerful'},
        {'id': '2', 'content': 'Deep learning models'},
        {'id': '3', 'content': 'Machine learning basics'},
    ])

    # Scenario 1: Simple usage
    print("Scenario 1: Simple search")
    processor = SearchProcessor(storage)
    results = processor.execute("neural")
    print(f"  Found: {len(results)} results\n")

    # Scenario 2: With configuration
    print("Scenario 2: Configured search")
    config = SearchConfig(query="learning", max_results=2)
    results = (SearchProcessor(storage)
               .with_config(config)
               .execute("learning"))
    print(f"  Found: {len(results)} results (limited to {config.max_results})\n")

    # Scenario 3: With session context manager
    print("Scenario 3: Search session with context manager")
    with SearchProcessor(storage).search_session() as session:
        results1 = session.search("neural")
        results2 = session.search("learning")
        print(f"  First search: {len(results1)} results")
        print(f"  Second search: {len(results2)} results")
    print()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all examples."""
    examples = {
        'fluent_api': example_fluent_api,
        'transactions': example_transactions,
        'progressive_disclosure': example_progressive_disclosure,
        'error_messages': example_error_messages,
        'testability': example_testability,
        'api_evolution': example_api_evolution,
        'real_world': example_real_world,
    }

    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Run API design pattern examples')
    parser.add_argument('--example', choices=list(examples.keys()),
                        help='Run specific example')
    parser.add_argument('--all', action='store_true',
                        help='Run all examples')
    args = parser.parse_args()

    if args.example:
        examples[args.example]()
    elif args.all or len(sys.argv) == 1:
        for name, func in examples.items():
            func()
            print("-" * 70)
            print()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
