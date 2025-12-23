# API Design Patterns - Quick Reference

## When to Use Each Pattern

### 1. **When building your API layer...**

| Scenario | Pattern | Reason |
|----------|---------|--------|
| Users need to chain multiple operations | **Fluent Builder** | Readable, natural flow; easier to learn |
| Multiple ways to initialize objects | **Factory Methods** | Each path documents a use case |
| Need resource cleanup (files, connections) | **Context Managers** | Idiomatic Python; automatic cleanup |
| Configuration is optional/complex | **Config Objects** | Progressive disclosure; decouples API from implementation |
| Errors are common operation outcomes | **Custom Exceptions** | Self-diagnosing; users learn how to handle them |
| Testing is important | **Dependency Injection** | Easy to mock; no side effects in tests |
| You're releasing incrementally | **Feature Flags** | Gradual rollout; A/B testing support |
| Data structure may change | **Schema Versioning** | Forward/backward compatibility |

---

## Decision Tree: Choosing Your Primary API Style

```
Do you want method chaining?
├─ YES → Use FLUENT BUILDER
│        (.add().configure().build().search())
│
└─ NO → Use BUILDER PATTERN
         (object construction with with_* methods)

Do you need automatic cleanup?
├─ YES → Use CONTEXT MANAGERS
│        (with db.transaction(): ... )
│
└─ NO → Use METHOD CHAINING alone

Is configuration complex?
├─ YES → Use CONFIG OBJECTS with PROGRESSIVE DISCLOSURE
│        (ConfigLevel1 for basics, advanced for power users)
│
└─ NO → Use OPTIONAL PARAMETERS

Do users need to handle errors?
├─ YES → Use CUSTOM EXCEPTIONS with SUGGESTIONS
│        (self-diagnosing errors guide users to solutions)
│
└─ NO → Use standard exceptions + logging

Is your API tested?
├─ YES → Use DEPENDENCY INJECTION
│        (inject real/mock backends; easy unit tests)
│
└─ NO → Use direct instantiation (add DI later)
```

---

## Code Patterns by Use Case

### Simple CRUD Operations
```python
# ✓ Good: Direct methods
db.create(entity)
db.read(id)
db.update(entity)
db.delete(id)

# ✗ Avoid: Over-engineered
db.with_id(id).with_type('user').with_fields(...).execute()
```

### Complex Query Building
```python
# ✓ Good: Fluent builder with terminal operation
(QueryBuilder()
    .filter(field='status', value='active')
    .sort_by('created_at')
    .limit(10)
    .execute())

# ✗ Avoid: Magic methods
query.filter.status = 'active'  # Confusing
results = query()  # What gets called?
```

### Multi-Step Transactions
```python
# ✓ Good: Context manager (automatic commit/rollback)
with db.transaction() as tx:
    tx.write('entity1', data1)
    tx.write('entity2', data2)
    # Auto-commits on success, auto-rollbacks on error

# ✗ Avoid: Manual management
tx = db.begin()
try:
    tx.write('entity1', data1)
    tx.write('entity2', data2)
    db.commit(tx)
except Exception:
    db.rollback(tx)
    raise
```

### Configuration Management
```python
# ✓ Good: Config object with sensible defaults
config = ProcessorConfig(
    # Only specify what you need to change
    threshold=0.3,
    # Everything else uses defaults
)
processor = Processor(config)

# ✗ Avoid: Many optional parameters
processor = Processor(
    threshold=0.3,
    max_results=10,
    damping=0.85,
    iterations=100,
    # ... 20 more parameters
)
```

### Error Handling
```python
# ✓ Good: Self-diagnosing errors with context
try:
    db.update(entity)
except ConflictError as e:
    print(e)  # Error message includes what to do
    # Error: Cannot update entity:123
    # Context: read_version=5, current_version=7
    # How to fix: Reload and retry

# ✗ Avoid: Opaque errors
try:
    db.update(entity)
except Exception as e:
    print(f"Error: {e}")  # Unclear what went wrong
```

### Testing
```python
# ✓ Good: Dependency injection
storage = InMemoryStorage()  # Or mock
repo = Repository(storage)
results = repo.find(...)

# ✗ Avoid: Hard-coded dependencies
class Repository:
    def __init__(self):
        self.storage = PostgresStorage()  # Can't mock!
```

---

## Pattern Maturity Model

### Phase 1: MVP (Minimum Viable Product)
- Focus: Working functionality
- Patterns: Basic fluent API + exceptions
- Config: Simple constructor parameters
- Testing: Integration tests only
- Errors: Standard exceptions

### Phase 2: Growing (Adding Features)
- Focus: Usability + correctness
- Patterns: Add context managers for resources
- Config: Move to config objects
- Testing: Add unit tests with mocks
- Errors: Custom exceptions with suggestions

### Phase 3: Mature (Stable Release)
- Focus: Backward compatibility + evolution
- Patterns: Full fluent API + transactions + sessions
- Config: Progressive disclosure with all levels
- Testing: Full test coverage with fixtures
- Errors: Self-diagnosing errors with context

### Phase 4: Evolution (API Growth)
- Focus: Non-breaking changes
- Patterns: Feature flags for new features
- Config: Schema versioning for data
- Testing: Acceptance tests for migrations
- Errors: Deprecation warnings for old APIs

---

## Common Mistakes to Avoid

### ❌ Mistake 1: Mixing Chainable and Non-Chainable Methods
```python
# BAD: Users don't know which methods return self
processor = (FluentProcessor()
    .add_document("doc1", "text")
    .build()
    .search("query"))  # Which of these return self?

# GOOD: Separate chainable from terminal
processor = (FluentProcessor()  # Chainable setup
    .add_document("doc1", "text")
    .build())
results = processor.search("query")  # Terminal operation
```

### ❌ Mistake 2: Configuration Object with No Defaults
```python
# BAD: Users must specify everything
config = Config(
    threshold=0.5,
    max_results=10,
    # ... required even if sensible default exists
)

# GOOD: Sensible defaults, override what you need
config = Config()  # Works out of the box
config = Config(threshold=0.3)  # Override one param
```

### ❌ Mistake 3: Generic Exception Messages
```python
# BAD: User doesn't know what to do
raise ValueError("Invalid input")

# GOOD: User knows exactly how to fix it
raise ValidationError(
    "Invalid threshold value",
    field='threshold',
    constraint='range',
    min=0.0,
    max=1.0,
    actual=1.5
)
```

### ❌ Mistake 4: Hard-Coded Dependencies (Can't Test)
```python
# BAD: Can't unit test without real database
class Service:
    def __init__(self):
        self.db = RealDatabase()

# GOOD: Testable with real or mock
class Service:
    def __init__(self, db: Database):
        self.db = db
```

### ❌ Mistake 5: Context Managers That Don't Clean Up
```python
# BAD: Exception in exit handler masks original error
def __exit__(self, exc_type, exc_val, exc_tb):
    try:
        self.cleanup()
    except Exception as e:
        raise  # Masks original error!

# GOOD: Cleanup never masks original exception
def __exit__(self, exc_type, exc_val, exc_tb):
    try:
        if exc_type is None:
            self.commit()
        else:
            self.rollback()
    finally:
        self._cleanup()
    return False  # Don't suppress original exception
```

### ❌ Mistake 6: Breaking Changes Without Deprecation
```python
# BAD: Users' code breaks without warning
# Old API removed in v2.0
db.get_users()  # Now raises AttributeError

# GOOD: Deprecation period before removal
@deprecated("Use find_by_type(type='user') instead")
def get_users(self):
    return self.find_by_type(type='user')

# v1.0-v1.5: Both methods work (old issues warning)
# v2.0: Old method raises deprecation error with link
# v2.5+: Old method removed
```

---

## Testing Patterns for APIs

### Unit Testing (No Real Dependencies)
```python
def test_search_with_mock():
    """Unit test using mock storage."""
    mock_storage = MagicMock()
    mock_storage.get.return_value = [...]

    searcher = Searcher(mock_storage)
    results = searcher.find("query")

    assert len(results) > 0
    mock_storage.get.assert_called_once()
```

### Integration Testing (With Real Dependencies)
```python
def test_search_with_real_storage(tmp_path):
    """Integration test using real storage."""
    storage = FileStorage(tmp_path)
    storage.set("documents", [...])

    searcher = Searcher(storage)
    results = searcher.find("query")

    assert len(results) > 0
```

### Context Manager Testing
```python
def test_transaction_commits_on_success():
    """Test transaction commits when no error."""
    db = SimpleDatabase()

    with db.transaction() as tx:
        tx.write('key', 'value')

    assert db.get('key') == 'value'

def test_transaction_rollback_on_error():
    """Test transaction rolls back on error."""
    db = SimpleDatabase()

    with pytest.raises(ValueError):
        with db.transaction() as tx:
            tx.write('key', 'value')
            raise ValueError("Test error")

    assert db.get('key') is None  # Rolled back
```

### Error Testing
```python
def test_error_message_includes_suggestions():
    """Test that errors guide users to solutions."""
    try:
        raise ValidationError(
            "Invalid threshold",
            constraint='range',
            min=0.0,
            max=1.0,
        )
    except ValidationError as e:
        assert "How to fix:" in str(e)
        suggestions = e.get_suggestions()
        assert len(suggestions) > 0
```

---

## Architecture Decision Records (ADRs)

### ADR 1: Use Fluent Builder for Complex Queries
**Decision:** Queries use fluent API with method chaining

**Rationale:**
- Readable: reads like English sentences
- Discoverable: IDE autocomplete guides users
- Familiar: Python developers know this pattern

**Consequences:**
- Terminal operations must be clearly marked
- Chain depth limited (avoid >5 methods)
- Every method returns self (except terminal ops)

---

### ADR 2: Context Managers for Transactions
**Decision:** Database operations use `with` statement for transactions

**Rationale:**
- Idiomatic Python (PEP 343)
- Automatic cleanup (commit/rollback)
- Clear scope (changes scoped to block)

**Consequences:**
- Users must understand context managers
- Nested transactions need special handling
- Exception handling required for recovery

---

### ADR 3: Custom Exceptions with Suggestions
**Decision:** Exceptions include context and recovery suggestions

**Rationale:**
- Developers learn how to fix issues
- Reduces support requests
- Works well with logging/monitoring

**Consequences:**
- Exception classes have more logic
- Context data must be JSON-serializable
- Suggestions must be kept current

---

### ADR 4: Dependency Injection for Testability
**Decision:** All external dependencies are injected via constructor

**Rationale:**
- Easy to unit test with mocks
- No global state or singletons
- Flexible for different environments

**Consequences:**
- Slightly verbose initialization
- Users must understand DI pattern
- Testing frameworks must support mocking

---

## Checklist for New APIs

- [ ] Fluent methods return `self`, terminal methods return results
- [ ] Config objects have sensible defaults
- [ ] All custom exceptions have `get_suggestions()` method
- [ ] Context managers properly handle exceptions without masking them
- [ ] All external dependencies are injected
- [ ] Error messages suggest how to fix the problem
- [ ] Feature flags for experimental features
- [ ] Schema versioning for persisted data
- [ ] Tests use fixtures and mocks, not real dependencies
- [ ] Deprecated methods have replacement suggestions

---

## Further Examples

For runnable examples of each pattern, see: `docs/api-design-examples.py`

```bash
# Run all examples
python docs/api-design-examples.py --all

# Run specific example
python docs/api-design-examples.py --example fluent_api
python docs/api-design-examples.py --example transactions
python docs/api-design-examples.py --example error_messages
```

---

## References

**Books:**
- *Design Patterns* - Gang of Four
- *API Design Rulebook* - Langlois
- *Release It!* - Nygard (architecture patterns)

**PEPs:**
- PEP 343 – The "with" Statement (context managers)
- PEP 557 – Data Classes
- PEP 3151 – Reworking the OS and IO exception hierarchy

**Articles:**
- [Fluent Interface](https://martinfowler.com/bliki/FluentInterface.html) - Martin Fowler
- [Feature Toggles](https://martinfowler.com/articles/feature-toggles.html) - Martin Fowler
- [Python Context Managers](https://docs.python.org/3/library/contextlib.html) - Python Docs

