# Migration Plan: Remove GOT_DIR from GoT Layer

**Status:** Draft
**Author:** Claude Code
**Created:** 2026-01-09
**Task:** Consolidate all storage path management into CDG layer

---

## Executive Summary

The GoT (Graph of Thought) layer currently duplicates path management that should be owned exclusively by the CDG (Cortical Distributed Graph) layer. This creates:

1. **Architectural confusion** - Two layers managing the same paths
2. **Tight coupling** - GoT directly depends on file system structure
3. **Testing friction** - Hard to mock storage without mocking paths
4. **Maintenance burden** - Path changes require updates in multiple places

**Goal:** GoT should use CDG abstractions exclusively. No `got_dir` parameter, no direct path construction.

---

## Current Architecture (Problem)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Bootstrap                                       │
│  got_dir = Path(".got")                                                 │
│       │                                                                 │
│       ├──► CDGModule(got_dir)                                           │
│       │    ├── CDGStore(store_dir=got_dir/entities)                     │
│       │    ├── CDGTransactionManager(store_dir=got_dir/entities)        │
│       │    ├── CDGWALManager(wal_dir=got_dir/wal)                       │
│       │    └── CDGIndexManager(store_dir=got_dir/entities)              │
│       │                                                                 │
│       └──► GoTModule(got_dir)  ← ALSO receives got_dir ❌               │
│            └── GoTManager(got_dir, tx_manager)                          │
│                ├── self.got_dir = got_dir  ← Stores it ❌               │
│                ├── self.got_dir / "entities"  ← Direct paths ❌         │
│                └── SyncManager(got_dir)  ← Passes it down ❌            │
└─────────────────────────────────────────────────────────────────────────┘
```

**Problem:** GoT layer bypasses CDG abstractions and directly manipulates paths.

---

## Target Architecture (Solution)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Bootstrap                                       │
│  base_dir = Path(".got")                                                │
│       │                                                                 │
│       └──► CDGModule(base_dir)  ← Single source of truth                │
│            ├── CDGStore (owns entities_dir)                             │
│            ├── CDGTransactionManager (uses CDGStore)                    │
│            ├── CDGWALManager (owns wal_dir)                             │
│            ├── CDGRecoveryManager (uses CDGStore)                       │
│            └── CDGConfig (exposes paths if needed)                      │
│                    │                                                    │
│                    ▼                                                    │
│            GoTModule (no path config)                                   │
│            └── GoTManager(tx_manager, schema_registry)                  │
│                ├── Uses tx_manager.store for all I/O ✓                  │
│                ├── No self.got_dir ✓                                    │
│                └── No direct path construction ✓                        │
└─────────────────────────────────────────────────────────────────────────┘
```

**Principle:** GoT sees CDG abstractions, never raw paths.

---

## Impact Analysis

### Files Requiring Changes

| File | `got_dir` Count | Severity | Notes |
|------|-----------------|----------|-------|
| `cortical/got/cli/doc.py` | 27 | HIGH | Heavy path usage for document management |
| `cortical/got/cli/failure.py` | 25 | HIGH | Failures stored in `got_dir/failures/` |
| `cortical/got/claudemd.py` | 14 | MEDIUM | ClaudeMd layer management |
| `cortical/got/cli/task.py` | 11 | MEDIUM | History access, learning bridge |
| `cortical/got/api.py` | 9 | HIGH | Core GoTManager implementation |
| `cortical/got/learning_integration.py` | 8 | MEDIUM | Learning storage in `got_dir/learning/` |
| `cortical/got/sync.py` | 6 | MEDIUM | Git sync operations |
| `cortical/got/adapter.py` | 4 | LOW | CLI adapter, creates container |
| `cortical/got/cli/backup.py` | 4 | LOW | Snapshot path access |
| `cortical/got/factory.py` | 3 | LOW | Factory for adapter |
| `cortical/got/query_api.py` | 2 | LOW | Delegates to manager |
| `cortical/got/query_builder.py` | 1 | LOW | Entity iteration |
| `cortical/got/cli/query.py` | 1 | LOW | Entity directory access |
| `cortical/got/__main__.py` | 1 | LOW | Debug output |

**Total: ~115 references across 14 files**

### Storage Locations Currently Managed by GoT

| Path | Purpose | Should Move To |
|------|---------|----------------|
| `.got/entities/` | Entity JSON files | CDGStore (already owns) |
| `.got/entities/_history/` | Entity history | CDGStore.history_dir |
| `.got/entities/_indexes/` | Query indexes | CDGIndexManager |
| `.got/wal/` | Write-ahead log | CDGWALManager |
| `.got/wal/snapshots/` | Backup snapshots | CDGWALManager or new CDGBackupManager |
| `.got/failures/` | Failed approach tracking | **New:** CDGStore partition or separate manager |
| `.got/learning/` | Learning data | **New:** CDGStore partition or separate manager |
| `.got/claude-md/` | ClaudeMd preferences | **New:** CDGStore partition or separate manager |
| `.got/generated/` | Generated output | **New:** CDGStore partition or keep separate |

---

## Migration Strategy

### Phase 1: CDG Layer Enhancements (Foundation)

**Goal:** Ensure CDG exposes all necessary abstractions before changing GoT.

#### 1.1 Add Path Accessors to CDGStore

```python
# cortical/cdg/storage.py
class CDGStore:
    @property
    def base_dir(self) -> Path:
        """Base directory for all CDG storage."""
        return self.store_dir.parent  # .got/

    @property
    def entities_dir(self) -> Path:
        """Directory for entity JSON files."""
        return self.store_dir  # .got/entities/

    @property
    def history_dir(self) -> Path:
        """Directory for entity history."""
        return self._history_dir  # .got/entities/_history/
```

#### 1.2 Add Partition Support to CDGStore

For `failures/`, `learning/`, `claude-md/`:

```python
# Option A: Named partitions
store.get_partition("failures")  # Returns CDGStore for .got/failures/
store.get_partition("learning")  # Returns CDGStore for .got/learning/

# Option B: Separate managers registered in container
container.resolve(FailureStore)  # Dedicated store for failures
container.resolve(LearningStore)  # Dedicated store for learning
```

#### 1.3 Expose Paths via CDGConfig (if needed)

```python
# cortical/cdg/config.py
@dataclass
class CDGConfig:
    base_dir: Path = field(default_factory=lambda: Path(".got"))

    @property
    def entities_dir(self) -> Path:
        return self.base_dir / "entities"

    @property
    def wal_dir(self) -> Path:
        return self.base_dir / "wal"
```

---

### Phase 2: GoT Core Refactoring

#### 2.1 Remove `got_dir` from GoTManager

**Before:**
```python
class GoTManager:
    def __init__(
        self,
        got_dir: Path,
        tx_manager: CDGTransactionManager,
        schema_registry: SchemaRegistry,
    ):
        self.got_dir = Path(got_dir)
        self.tx_manager = tx_manager
```

**After:**
```python
class GoTManager:
    def __init__(
        self,
        tx_manager: CDGTransactionManager,
        schema_registry: SchemaRegistry,
    ):
        self.tx_manager = tx_manager
        # No self.got_dir - use tx_manager.store for all storage
```

#### 2.2 Update GoTModule

**Before:**
```python
def create_got_manager() -> GoTManager:
    tx_manager = container.resolve(CDGTransactionManager)
    registry = container.resolve(SchemaRegistry)
    return GoTManager(
        self.config.got_dir,  # ← Remove this
        tx_manager=tx_manager,
        schema_registry=registry,
    )
```

**After:**
```python
def create_got_manager() -> GoTManager:
    tx_manager = container.resolve(CDGTransactionManager)
    registry = container.resolve(SchemaRegistry)
    return GoTManager(
        tx_manager=tx_manager,
        schema_registry=registry,
    )
```

#### 2.3 Replace Direct Path Access

**Pattern: Entity iteration**

Before:
```python
entities_dir = self.got_dir / "entities"
for f in entities_dir.glob("T-*.json"):
    ...
```

After:
```python
for entity in self.tx_manager.store.iter_entities(prefix="T-"):
    ...
```

**Pattern: Recovery manager**

Before:
```python
self._recovery_manager = CDGRecoveryManager(
    store_dir=self.got_dir / "entities",
    ...
)
```

After:
```python
# Resolve from container (already configured by CDGModule)
self._recovery_manager = container.resolve(CDGRecoveryManager)
```

---

### Phase 3: CLI Layer Updates

#### 3.1 TransactionalGoTAdapter

**Before:**
```python
GOT_DIR = Path(os.environ.get("GOT_DIR", _PROJECT_ROOT / ".got"))

class TransactionalGoTAdapter:
    def __init__(self, got_dir: Path = GOT_DIR):
        self.got_dir = Path(got_dir)
        container = create_container(got_dir=self.got_dir)
```

**After:**
```python
class TransactionalGoTAdapter:
    def __init__(self, base_dir: Optional[Path] = None):
        effective_dir = base_dir or Path(os.environ.get("CDG_BASE_DIR", ".got"))
        container = create_container(base_dir=effective_dir)
        self._manager = container.resolve(GoTManager)
        # Access paths through CDG if needed:
        # self._store = container.resolve(CDGStore)
```

#### 3.2 CLI Commands Using `manager.got_dir`

Pattern for `cli/backup.py`, `cli/failure.py`, `cli/doc.py`, etc.:

**Before:**
```python
snapshots_dir = manager.got_dir / "wal" / "snapshots"
```

**After (Option A - CDGStore accessor):**
```python
snapshots_dir = manager.tx_manager.store.base_dir / "wal" / "snapshots"
```

**After (Option B - Dedicated manager):**
```python
backup_manager = container.resolve(CDGBackupManager)
snapshots = backup_manager.list_snapshots()
```

---

### Phase 4: Ancillary Components

#### 4.1 SyncManager

Currently: `SyncManager(got_dir)`

Options:
1. Inject CDGStore/CDGConfig instead of raw path
2. Keep path but receive from CDG layer, not GoT

#### 4.2 ClaudeMdManager

Currently stores in `.got/claude-md/` and `.got/generated/`

Options:
1. Register as CDG partition
2. Keep separate but receive base_dir from CDGConfig

#### 4.3 LearningIntegration

Currently stores in `.got/learning/`

Options:
1. Register as CDG partition with its own entity schemas
2. Keep separate filesystem access but inject path from CDG

#### 4.4 FailureTracking

Currently stores in `.got/failures/`

Recommendation: This is entity-like data, should be a CDG entity type:
```python
class FailedApproach(Entity):
    task_id: str
    description: str
    timestamp: datetime
    context: Dict[str, Any]
```

---

## Migration Order

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Sprint 1: CDG Foundation (Non-breaking)                                │
│  ─────────────────────────────────────────                              │
│  1. Add path accessors to CDGStore                                      │
│  2. Add base_dir to CDGConfig                                           │
│  3. Update CDGModule to expose paths                                    │
│  4. Add tests for new accessors                                         │
│                                                                         │
│  Deliverable: CDG exposes all paths GoT needs                           │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Sprint 2: GoT Core (Breaking changes contained)                        │
│  ───────────────────────────────────────────────                        │
│  1. Remove got_dir from GoTManager.__init__                             │
│  2. Update GoTModule to not pass got_dir                                │
│  3. Replace self.got_dir usage with tx_manager.store accessors          │
│  4. Update api.py, query_api.py, query_builder.py                       │
│  5. Update sync.py to receive path from CDG                             │
│                                                                         │
│  Deliverable: GoTManager has no got_dir                                 │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Sprint 3: CLI Adapter (User-facing changes)                            │
│  ──────────────────────────────────────────                             │
│  1. Update TransactionalGoTAdapter                                      │
│  2. Rename GOT_DIR env var to CDG_BASE_DIR                              │
│  3. Update factory.py                                                   │
│  4. Update __main__.py                                                  │
│                                                                         │
│  Deliverable: CLI uses CDG paths                                        │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Sprint 4: CLI Commands (Bulk updates)                                  │
│  ─────────────────────────────────────                                  │
│  1. Update cli/backup.py (4 refs)                                       │
│  2. Update cli/query.py (1 ref)                                         │
│  3. Update cli/task.py (11 refs)                                        │
│  4. Update cli/failure.py (25 refs)                                     │
│  5. Update cli/doc.py (27 refs)                                         │
│                                                                         │
│  Deliverable: All CLI commands use CDG paths                            │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Sprint 5: Ancillary Systems (Architectural decisions)                  │
│  ─────────────────────────────────────────────────────                  │
│  1. Decide: failures as CDG entity vs separate storage                  │
│  2. Migrate FailureTracking (25 refs in failure.py)                     │
│  3. Decide: learning as CDG partition vs separate                       │
│  4. Migrate LearningIntegration (8 refs)                                │
│  5. Migrate ClaudeMd (14 refs)                                          │
│                                                                         │
│  Deliverable: All storage managed by CDG                                │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Sprint 6: Cleanup & Documentation                                      │
│  ─────────────────────────────────                                      │
│  1. Remove GOT_DIR constant from codebase                               │
│  2. Update CLAUDE.md                                                    │
│  3. Update architecture docs                                            │
│  4. Add migration guide for external users                              │
│  5. Delete got_dir from all signatures                                  │
│                                                                         │
│  Deliverable: Clean architecture, no legacy paths                       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Testing Strategy

### Unit Tests

Each sprint should include:
1. Tests for new CDG accessors
2. Tests that GoT works without got_dir
3. Tests that CLI commands work with new path resolution

### Integration Tests

1. Full workflow: create task → query → complete
2. Backup/restore cycle
3. Recovery from corruption

### Regression Prevention

Add test that fails if `got_dir` is used in GoT:

```python
def test_got_layer_does_not_use_got_dir():
    """Ensure GoT layer doesn't bypass CDG for path access."""
    import ast
    import cortical.got

    got_files = Path(cortical.got.__file__).parent.glob("**/*.py")

    forbidden_patterns = ['got_dir', 'GOT_DIR', '".got"', "'.got'"]
    violations = []

    for f in got_files:
        content = f.read_text()
        for pattern in forbidden_patterns:
            if pattern in content:
                violations.append(f"{f.name}: contains '{pattern}'")

    assert not violations, f"GoT layer should not use paths directly:\n" + "\n".join(violations)
```

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Breaking existing scripts | HIGH | Keep `GOT_DIR` env var as deprecated alias |
| Performance regression | MEDIUM | Profile before/after, ensure no extra I/O |
| Incomplete migration | MEDIUM | Automated test that detects `got_dir` usage |
| Test coverage gaps | MEDIUM | Run full test suite after each sprint |
| Container complexity | LOW | Document new resolution patterns |

---

## Success Criteria

1. **Zero `got_dir` in GoT layer** (except deprecated aliases)
2. **All storage paths derived from CDG**
3. **Tests pass** with in-memory and disk storage
4. **No performance regression** in common operations
5. **Documentation updated** to reflect new architecture

---

## Open Questions

1. **Failures as entities?** Should `FailedApproach` become a first-class CDG entity with schema?
2. **Learning partition?** Should `.got/learning/` become a CDG partition or stay separate?
3. **ClaudeMd storage?** Is ClaudeMd data entity-like or configuration-like?
4. **Backward compatibility?** How long to support `GOT_DIR` env var?

---

## Appendix: Reference Count by Pattern

```
Pattern                          | Count | Files
---------------------------------|-------|------
self.got_dir                     | 45    | 8
manager.got_dir                  | 32    | 6
got_dir: Path                    | 18    | 10
got_dir /                        | 15    | 7
GOT_DIR                          | 4     | 2
Path(".got")                     | 8     | 4
```

---

## Appendix: Files to Modify (Complete List)

### High Priority (Core)
- `cortical/got/api.py` - GoTManager, remove got_dir param
- `cortical/got/adapter.py` - TransactionalGoTAdapter
- `cortical/got/factory.py` - Remove GOT_DIR constant
- `cortical/core/modules/got_module.py` - Update GoTManager creation

### Medium Priority (Features)
- `cortical/got/query_api.py` - Remove got_dir property
- `cortical/got/query_builder.py` - Use store.iter_entities
- `cortical/got/sync.py` - Receive path from CDG
- `cortical/got/claudemd.py` - Migrate to CDG partition
- `cortical/got/learning_integration.py` - Migrate to CDG partition

### Lower Priority (CLI)
- `cortical/got/cli/task.py` - Update history access
- `cortical/got/cli/failure.py` - Migrate failures storage
- `cortical/got/cli/doc.py` - Update document access
- `cortical/got/cli/backup.py` - Use CDG backup manager
- `cortical/got/cli/query.py` - Use store.iter_entities
- `cortical/got/__main__.py` - Remove debug got_dir output

### CDG Enhancements
- `cortical/cdg/storage.py` - Add path accessors
- `cortical/cdg/config.py` - Add base_dir property
- `cortical/core/modules/cdg_module.py` - Expose paths
