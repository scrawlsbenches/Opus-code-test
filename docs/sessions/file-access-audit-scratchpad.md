# Working Scratchpad - Refactoring Session

*Purpose: Working state of mind for context recovery. Git tracks what's done.*

---

## THE BIG PICTURE

**CDG (Cortical Distributed Graph)** = Future general graph storage
- Unified API for all graphs (GoT, ThoughtGraph, Knowledge Graph)
- ACID transactions, partitioning, WAL-based durability
- Spec: `docs/architecture/DISTRIBUTED_GRAPH_SPECIFICATION.md`

**GoT (Graph of Thought)** = Current system with NL query enhancements
- Domain entities: Task, Decision, Sprint, Edge, Handoff, KnowledgeTransfer
- Recent NL query additions worth salvaging
- Design: `docs/design/got-query-audit-and-design.md`

**Current Refactoring Goal:**
Salvage best parts of GoT while implementing CDG as storage foundation.
GoT becomes thin domain layer on CDG.

---

## ARCHITECTURAL PRINCIPLES

```
┌─────────────────────────────────────────────────────────────────────────┐
│  1. CDG = FOUNDATION — GoT = THIN DOMAIN LAYER                          │
│  2. CONTAINER-FIRST — DI/IoC via cortical/core/bootstrap.py             │
│  3. NO TWO LAYERS — Delete wrappers, use CDG directly                   │
│  4. NO BACKWARD COMPAT — Fix directly, don't maintain fallbacks         │
│  5. SOVEREIGNTY — No external deps, build from first principles         │
│  6. NO TEST MAINTENANCE NOW — Scope too large, tests will break         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## KEY FILES / ENTRY POINTS

| What | Where |
|------|-------|
| Container bootstrap | `cortical/core/bootstrap.py` |
| CDG storage | `cortical/cdg/` |
| GoT domain layer | `cortical/got/` |
| GoT query enhancement | `cortical/got/expression/` (half-baked but good ideas) |
| Schema registry | `cortical/got/entity_schemas.py` |
| Query builder | `cortical/got/query_builder.py` |

---

## WHAT GoT UNIQUELY PROVIDES (keep in GoT)

1. **Entity types** - Task, Decision, Edge, Sprint (domain models)
2. **Entity factory** - `create_entity_from_dict()` dispatches to correct type
3. **QueryIndexManager** - GoT-specific indexing
4. **GoTManager** - high-level domain API
5. **NL Query enhancements** - expression parser, function registry pattern

---

## WHAT SHOULD BE IN CDG (generalize from GoT)

- Transaction management → CDGTransactionManager ✓
- Versioned storage → CDGStore ✓
- WAL → CDGWALManager ✓
- Recovery → CDGRecoveryManager (GoT has domain-specific index logic)
- Generic schema → CDG schema (GoT has entity_schemas.py pattern)

---

## GIT HANDLING FOR SESSION CONTINUATIONS

**Problem:** Each new session gets NEW branch name from system.

**Solution - ALWAYS do this first:**
```bash
git fetch --all
git checkout claude/fix-file-access-issues-1zUM9  # IGNORE system branch
git pull origin claude/fix-file-access-issues-1zUM9
git log --oneline -5  # Verify recent work
# ASK USER before proceeding
```

---

## REFERENCE DOCS

- CDG Spec: `docs/architecture/DISTRIBUTED_GRAPH_SPECIFICATION.md`
- GoT Query Design: `docs/design/got-query-audit-and-design.md`
- GoT Query Future: `docs/design/got-query-future-enhancements.md`
- Forensic Analysis: `docs/FORENSIC_ANALYSIS_2026-01-05.md`
- Old Handoffs: `docs/handoffs/` (check if relevant)

---

## CURRENT FOCUS

[Update this section when starting new work]

Branch: `claude/fix-file-access-issues-1zUM9`
Last context: File access audit completed, removed GoT wrappers for tx_manager, wal, versioned_store.
Next: User will specify next task.
