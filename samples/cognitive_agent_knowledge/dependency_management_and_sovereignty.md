# Dependency Management and Sovereignty

## The Sovereignty Principle

This codebase follows a strict philosophy: **we build what we can, we depend only when necessary**.

### Core Tenets

```
WE BUILD. WE MAINTAIN. WE CONTROL.

This project does not depend on what it cannot own.
We do not adopt third-party components.
We do not integrate external libraries we cannot rebuild.
We do not inherit dependencies we cannot maintain.
```

### Why This Matters

**Problem**: External dependencies create hidden costs:
- Security vulnerabilities outside your control (CVEs in transitive deps)
- Breaking changes in minor versions
- Abandoned projects leaving you stranded
- License changes that conflict with your usage
- Supply chain attacks through compromised packages

**Solution**: Build from first principles when feasible.

**Acceptable Dependencies**:
- Python stdlib: Maintained by Python core team, stable API
- Pytest: Meta-tooling for testing, not runtime dependency
- Anything else: Must document WHY we cannot build it ourselves

### Evaluating External Dependencies

Before adding ANY external dependency, ask these questions:

| Question | Why It Matters |
|----------|---------------|
| Can we build this ourselves in reasonable time? | If yes, build it. You'll understand it better. |
| Is the maintenance active? | Check: commits in last 6 months, issues addressed? |
| What's the bus factor? | One maintainer = high risk |
| What does the dependency graph look like? | Pulls in 50 transitive deps? Hard no. |
| What's the license? | GPL in your MIT project? Problem. |
| Can we fork and maintain if abandoned? | If not, you're at mercy of maintainers |
| Does it solve a genuinely hard problem? | Crypto, compression, parsing complex formats - reasonable. String manipulation - no. |

### Real Example: Why We Built Our Own Tokenizer

We needed text tokenization. Options:
1. Use NLTK (popular, feature-rich)
2. Use spaCy (fast, ML-based)
3. Build our own BPE tokenizer

We chose option 3 because:
- NLTK pulls in numpy, scipy, and 20+ other packages
- spaCy requires trained models (300MB+)
- Our use case needs simple BPE, not linguistic analysis
- We can understand, debug, and optimize our own code
- Total implementation: ~400 lines of Python

---

## Internal Dependency Management

### Avoiding Circular Imports

**Problem**: Module A imports Module B, Module B imports Module A. Python raises ImportError.

**Solution**: Proper layering.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     DEPENDENCY DIRECTION                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Higher layers depend on lower layers. NEVER the reverse.               │
│                                                                          │
│  Application Layer (CLI, API)                                           │
│       │                                                                  │
│       ▼                                                                  │
│  Service Layer (GoT, CEL, Cognitive)                                    │
│       │                                                                  │
│       ▼                                                                  │
│  Foundation Layer (CDG, Storage, Transactions)                          │
│       │                                                                  │
│       ▼                                                                  │
│  Common Layer (Container, Utilities, Protocols)                         │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**Rules**:
1. Lower layers NEVER import from higher layers
2. Siblings MAY import from each other (carefully)
3. Use protocols/interfaces at boundaries
4. If you need to import "up", you have a design problem

### Fixing Circular Imports

When you encounter a circular import:

```python
# WRONG: Direct import creates cycle
# file: got/manager.py
from cortical.cdg.storage import CDGStore  # CDGStore imports GoT types

# RIGHT: Import inside function (lazy)
def save_entity(self, entity):
    from cortical.cdg.storage import CDGStore
    store = self.container.resolve(CDGStore)
    store.save(entity)

# BETTER: Use protocol/interface
from typing import Protocol

class StorageBackend(Protocol):
    def save(self, entity: dict) -> None: ...
    def load(self, entity_id: str) -> dict: ...

# Both layers depend on the protocol, not each other
```

### Package Structure in Cortical

```
cortical/
├── common/          # NO DEPENDENCIES on other cortical modules
│   ├── container.py # DI container
│   ├── filesystem.py # FileSystem protocol
│   └── protocols.py # Shared protocols
│
├── cdg/             # Depends only on common/
│   ├── storage.py   # Entity storage
│   └── transaction_manager.py
│
├── got/             # Depends on common/, cdg/
│   ├── api.py       # GoT operations
│   └── indexer.py   # Entity indexing
│
├── cognitive/       # Depends on common/, may use cdg/
│   └── graph.py     # Cognitive graph
│
└── core/            # Orchestration, depends on all
    ├── bootstrap.py # Container wiring
    └── modules/     # Module registrations
```

---

## Dependency Injection for Testability

### The Problem with Hardcoded Dependencies

```python
# WRONG: Hardcoded dependency
class UserService:
    def __init__(self):
        self.db = PostgresDatabase("production_connection_string")
        self.cache = RedisCache("redis://localhost")

    def get_user(self, user_id):
        if cached := self.cache.get(user_id):
            return cached
        return self.db.query(user_id)

# How do you test this without a real Postgres and Redis?
# Answer: You can't. You need integration tests for everything.
```

### The Solution: Constructor Injection

```python
# RIGHT: Dependencies injected
class UserService:
    def __init__(self, db: Database, cache: Cache):
        self.db = db
        self.cache = cache

    def get_user(self, user_id):
        if cached := self.cache.get(user_id):
            return cached
        return self.db.query(user_id)

# Now testing is easy:
def test_get_user_from_cache():
    mock_db = MockDatabase()
    mock_cache = MockCache(data={"user-1": {"name": "Alice"}})

    service = UserService(db=mock_db, cache=mock_cache)

    result = service.get_user("user-1")

    assert result["name"] == "Alice"
    assert mock_db.query_count == 0  # DB was never called
```

### Benefits of Dependency Injection

| Benefit | Description |
|---------|-------------|
| Testability | Inject mocks/stubs without modifying code |
| Flexibility | Swap implementations at runtime |
| Explicit dependencies | Constructor shows what's needed |
| Single Responsibility | Classes do one thing, receive collaborators |
| Configuration | Different configs for dev/test/prod |

---

## The Container Pattern in Cortical

### What Is the Container?

The Container is a **dependency injection container** that manages:
- Service registration (what implements what)
- Lifecycle management (singleton vs transient)
- Dependency resolution (wiring things together)
- Child containers (for test isolation)

### Core Concepts

```python
from cortical.common import Container, Lifecycle

container = Container()

# Register implementation for interface
container.register(StorageBackend, FileSystemStorage)

# Register with specific lifecycle
container.register(Cache, LRUCache, lifecycle=Lifecycle.TRANSIENT)

# Register pre-created instance
container.register_instance(Config, my_config)

# Auto-wire constructor dependencies
container.register_auto(MyService)  # Inspects type hints

# Resolve to get instance
storage = container.resolve(StorageBackend)
```

### Lifecycle Options

```python
class Lifecycle(Enum):
    SINGLETON = auto()   # One instance, shared everywhere
    TRANSIENT = auto()   # New instance each time
    SCOPED = auto()      # One per scope (e.g., request)
```

**When to use each**:
- SINGLETON: Stateless services, expensive-to-create objects
- TRANSIENT: Stateful objects that shouldn't be shared
- SCOPED: Per-request isolation (web apps, batch jobs)

### Child Containers for Testing

```python
# Production container
container = create_container()

# Test container inherits but can override
test_container = container.create_child()
test_container.register(StorageBackend, InMemoryStorage)
test_container.register(Cache, NoOpCache)

# Production code uses real implementations
prod_service = container.resolve(MyService)

# Test code uses mocks
test_service = test_container.resolve(MyService)
```

### Module Pattern

Group related registrations into modules:

```python
from cortical.common import ContainerModule

class CDGModule(ContainerModule):
    def __init__(self, base_dir: Path, use_memory: bool = False):
        self.base_dir = base_dir
        self.use_memory = use_memory

    def register(self, container: Container) -> None:
        # Register CDG services
        container.register(CDGStore, create_store)
        container.register(CDGTransactionManager, create_tx_manager)
        container.register(CDGRecoveryManager, create_recovery)

# Apply module
container = Container()
container.apply_module(CDGModule(base_dir=Path(".got")))
```

### Bootstrap: Where It All Comes Together

```python
# cortical/core/bootstrap.py

def create_container(
    got_dir: Optional[Path] = None,
    use_memory: bool = False,
) -> Container:
    container = Container()

    # Register FileSystem strategy
    filesystem = InMemoryFileSystem() if use_memory else RealFileSystem()
    container.register_instance(FileSystem, filesystem)

    # Apply subsystem modules (order matters!)
    container.apply_module(SchemaModule())      # Foundation
    container.apply_module(CDGModule(...))      # Storage layer
    container.apply_module(GoTModule(...))      # Application layer

    return container
```

---

## When to Create Abstractions

### Create Abstractions When

1. **Multiple implementations exist or are planned**
   - FileSystemStorage vs InMemoryStorage vs S3Storage
   - ConsoleLogger vs FileLogger vs RemoteLogger

2. **Testing requires substitution**
   - Real database vs mock database
   - Real network vs fake network

3. **Boundary between subsystems**
   - Storage interface between CDG and GoT
   - Event interface between CEL and consumers

4. **External system integration**
   - Wrap external APIs for easier mocking
   - Isolate vendor-specific code

### Use Direct Implementation When

1. **Only one implementation will ever exist**
   - Internal utility functions
   - Simple data transformations

2. **Abstraction adds complexity without benefit**
   - Wrapping stdlib functions
   - Single-use helpers

3. **Performance is critical**
   - Hot paths where indirection costs matter
   - Low-level operations

### Protocol Pattern (Preferred in Python)

```python
from typing import Protocol

class StorageBackend(Protocol):
    """Protocol for entity storage backends."""

    def save(self, entity_id: str, data: dict) -> None:
        """Save entity to storage."""
        ...

    def load(self, entity_id: str) -> dict:
        """Load entity from storage."""
        ...

    def delete(self, entity_id: str) -> None:
        """Delete entity from storage."""
        ...

# Implementations don't need to inherit - structural typing
class FileSystemStorage:
    def save(self, entity_id: str, data: dict) -> None:
        # writes to disk
        pass

    def load(self, entity_id: str) -> dict:
        # reads from disk
        pass

    def delete(self, entity_id: str) -> None:
        # deletes from disk
        pass
```

---

## Upgrading and Maintaining Dependencies

### Safe Upgrade Process

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     DEPENDENCY UPGRADE PROTOCOL                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. READ THE CHANGELOG                                                  │
│     What changed? Breaking changes? Security fixes?                     │
│                                                                          │
│  2. CHECK COMPATIBILITY                                                 │
│     Does it still work with our Python version?                         │
│     Do our transitive dependencies conflict?                            │
│                                                                          │
│  3. UPGRADE IN ISOLATION                                                │
│     Create a branch. Upgrade ONE dependency at a time.                  │
│                                                                          │
│  4. RUN FULL TEST SUITE                                                 │
│     Not just unit tests. Integration tests. Performance tests.          │
│                                                                          │
│  5. REVIEW DIFF CAREFULLY                                               │
│     What new dependencies were pulled in?                               │
│     Did any existing deps change version?                               │
│                                                                          │
│  6. TEST IN STAGING                                                     │
│     Before production. Always.                                          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Pinning Versions

```toml
# pyproject.toml

[project]
dependencies = [
    # Pin exact versions for reproducibility
    "pytest==7.4.0",

    # Or use compatible release (allows patch updates)
    "coverage~=7.3",

    # Upper bound for known compatibility
    "hypothesis>=6.0,<7.0",
]
```

### Lock Files

Use `pip freeze > requirements.lock` or poetry.lock to capture exact versions:
- Reproducible builds
- Same versions across team/CI
- Known-good configuration

### Security Monitoring

Regularly check for vulnerabilities:

```bash
# Using pip-audit (if installed)
pip-audit

# Using safety (if installed)
safety check

# Manual check on PyPI/GitHub Security Advisories
```

---

## Long-Term Maintainability Principles

### The "Can I Delete This?" Test

For every dependency, ask: "If this disappeared tomorrow, how bad would it be?"

| Impact | Example | Action |
|--------|---------|--------|
| Catastrophic | Database driver | Keep, monitor actively |
| Significant | Testing framework | Keep, have migration plan |
| Minor | Formatting utility | Consider building in-house |
| None | Unused import | Delete immediately |

### Technical Debt from Dependencies

Dependencies accumulate debt:
- API changes require code updates
- Security patches require upgrades
- Version conflicts block other upgrades
- Deprecated features need replacement

**Minimize debt by**:
- Wrapping third-party code in thin adapters
- Keeping dependency count low
- Preferring stable, boring libraries over cutting-edge
- Building simple things yourself

### The Cost Calculation

Before adding a dependency, calculate:

```
Cost = (Integration Time) +
       (Learning Curve * Team Size) +
       (Maintenance Hours/Year * Expected Lifetime) +
       (Risk of Abandonment * Replacement Cost)
```

Often, building it yourself has lower total cost than the "free" library.

---

## Summary: Key Takeaways

1. **Default to building** - External dependencies have hidden costs
2. **Layer your architecture** - Higher depends on lower, never reverse
3. **Inject dependencies** - Makes testing possible, coupling explicit
4. **Use the Container** - Central wiring point, child containers for tests
5. **Abstract at boundaries** - Protocols for interfaces between subsystems
6. **Direct for internals** - Don't over-abstract simple internal code
7. **Upgrade carefully** - Read changelogs, test thoroughly, pin versions
8. **Think long-term** - Convenience today is maintenance tomorrow

The goal is a codebase that remains comprehensible, testable, and maintainable for years, not just days.
