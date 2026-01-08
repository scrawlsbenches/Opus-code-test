# Cortical Codebase Mental Model

*Generated: 2026-01-07*
*Branch: claude/refactor-cortical-codebase-OZ8em*

---

## Executive Summary

**Cortical** is a 106K+ line, 213-file Python system implementing a neocortex-inspired text processor with:
- ACID-compliant transactional graph storage (CDG)
- Task/decision management (GoT)
- Semantic analysis and cognitive reasoning
- Event sourcing and dual-process cognition

**Current State**: Active architectural refactoring - consolidating GoT infrastructure into CDG foundation layer.

---

## Architecture Layers (Bottom-Up)

```
┌─────────────────────────────────────────────────────────────────┐
│ Layer 7: Applications (CorticalTextProcessor, CLI, Scripts)     │
├─────────────────────────────────────────────────────────────────┤
│ Layer 6: Reasoning (Woven Mind, QAPV, PRISM, Thought Graphs)    │
├─────────────────────────────────────────────────────────────────┤
│ Layer 5: Analysis (PageRank, TF-IDF, Clustering, Embeddings)    │
├─────────────────────────────────────────────────────────────────┤
│ Layer 4: Query/Retrieval (Search, Expansion, Ranking)           │
├─────────────────────────────────────────────────────────────────┤
│ Layer 3: GoT (Tasks, Decisions, Edges, Sprints, Query DSL)      │
├─────────────────────────────────────────────────────────────────┤
│ Layer 2: CDG (Storage, Transactions, WAL, Recovery, Indexing)   │
├─────────────────────────────────────────────────────────────────┤
│ Layer 1: Foundation (Container/DI, FileSystem, Utils)           │
└─────────────────────────────────────────────────────────────────┘
```

**Rule**: Dependencies flow upward only. No circular imports.

---

## The Two Core Layers

### CDG (Cortical Distributed Graph) - Foundation

**Location**: `cortical/cdg/`
**Purpose**: ACID-compliant storage, transactions, crash recovery

| Component | File | Responsibility |
|-----------|------|----------------|
| CDGStore | storage.py | Entity persistence, checksums, versioning |
| CDGTransactionManager | transaction_manager.py | ACID transactions, snapshot isolation |
| CDGWALManager | wal.py | Write-ahead logging, durability |
| CDGRecoveryManager | recovery.py | Crash recovery, checksum verification |
| CDGIndexManager | index_manager.py | Schema-based automatic indexing |

**Key Contract**: WAL-first commit (write to WAL → fsync → then modify entities)

### GoT (Graph of Thought) - Domain Layer

**Location**: `cortical/got/`
**Purpose**: Task/decision management built ON TOP of CDG

| Component | File | Responsibility |
|-----------|------|----------------|
| GoTManager | api.py | High-level API for tasks, decisions, edges |
| Query/QueryBuilder | query_builder.py | Fluent query DSL |
| Entity Types | types.py | Task, Decision, Edge, Sprint, etc. |
| CLI | cli/*.py | Command-line interface |

**Key Principle**: GoT delegates ALL storage to CDG. No file I/O in GoT.

---

## Dependency Injection System

**Entry Point**: `cortical/core/bootstrap.py`

```python
container = create_container(got_dir=Path(".got"), use_memory=False)
manager = container.resolve(GoTManager)
```

**Module Application Order** (dependencies):
1. SchemaModule - Schema registry
2. CDGModule - Storage, transactions, WAL, indexing
3. GoTModule - GoTManager wrapping CDG

**Test Isolation**: Use `create_container(use_memory=True)` for 10x faster tests.

---

## Entity Types (GoT)

| Type | ID Prefix | Purpose |
|------|-----------|---------|
| Task | T- | Work items with status, priority |
| Decision | D- | Choices with rationale |
| Edge | E- | Relationships (DEPENDS_ON, BLOCKS, etc.) |
| Sprint | S- | Time-boxed work periods |
| Epic | EPIC- | Large initiatives |
| Handoff | H- | Agent-to-agent transfers |
| KnowledgeTransfer | KT- | Session learnings |

**19 Edge Types**: DEPENDS_ON, BLOCKS, CONTAINS, PART_OF, IMPLEMENTS, REFERENCES, etc.

---

## Test Structure

| Tier | Location | Time | Purpose |
|------|----------|------|---------|
| Smoke | tests/smoke/ | ~1s | Quick sanity checks |
| Unit | tests/unit/ | ~30s | Isolated component tests |
| Behavioral | tests/behavioral/ | ~2m | BDD user stories |
| Integration | tests/integration/ | ~2m | Component interaction |
| Performance | tests/performance/ | ~5m | Latency/throughput contracts |

**Key Fixtures**: `memory_got_manager`, `fresh_got_manager`, `memory_container`

**Coverage Minimum**: 86%

---

## Recent Refactoring (Git History)

**Pattern**: Consolidating GoT infrastructure into CDG

**Deleted from GoT**:
- `got/wal.py` → CDGWALManager
- `got/tx_manager.py` → CDGTransactionManager
- `got/versioned_store.py` → CDGStore
- `got/recovery.py` → CDGRecoveryManager
- `got/indexer.py` → CDGIndexManager

**Result**: ~10,000 lines deleted, ~2,000 added (net -8,000 LOC)

---

## Key Design Principles

1. **Container First** - All services via DI, no hardcoded paths
2. **WAL Before Storage** - Crash safety through write-ahead logging
3. **Schema-Based Validation** - Entity types defined in schemas
4. **Snapshot Isolation** - Transactions see consistent views
5. **Sovereignty** - No external dependencies for core functionality

---

## Critical File Reference

| Purpose | File |
|---------|------|
| Bootstrap | cortical/core/bootstrap.py |
| DI Container | cortical/common/container.py |
| CDG Storage | cortical/cdg/storage.py |
| CDG Transactions | cortical/cdg/transaction_manager.py |
| GoT API | cortical/got/api.py |
| Entity Schemas | cortical/got/entity_schemas.py |
| Test Fixtures | tests/conftest.py |
| Project Guidelines | CLAUDE.md |

---

## Quick Commands

```bash
# Health check
python scripts/got_utils.py validate

# Tests
pytest tests/smoke/ -v              # ~1s
pytest tests/unit/ -v               # ~30s
pytest tests/ -v -m "not slow"      # Skip slow tests

# Task management
python scripts/got_utils.py task create "Title"
python scripts/got_utils.py task list --status in_progress
python scripts/got_utils.py query "blocked tasks"
```

---

## Current Work

**Active Refactoring**: CDG Layer Migration (Phase 3 complete)

**Remaining**:
1. Fix test imports (4 files with broken imports)
2. Validation rule extraction (technical debt)
3. Graph traversal consolidation (deferred)

See: `docs/sessions/file-access-audit-scratchpad.md`
