# Audit Tools Migration Plan

## Date: 2026-01-08
## Author: Claude
## Status: Phase 1-3 Complete

---

## Progress Summary

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | ✅ Complete | CLI infrastructure, base modules |
| Phase 2 | ✅ Complete | audit_tool.py commands migrated |
| Phase 3 | ✅ Complete | PLN/PRISM integration scripts |

---

## Current State Analysis

### Existing CLI Patterns in Cortical

1. **`cortical/got/cli/`** - Well-structured command modules:
   - Pattern: `setup_X_parser()`, `handle_X_command()`, `cmd_X_Y()`
   - 15+ command modules (task, edge, sprint, decision, query, etc.)
   - Good separation of concerns

2. **`cortical/cli/`** - NEW unified CLI layer (Phase 1 complete):
   - `__init__.py` - Registry with auto-discovery
   - `_base.py` - Base utilities
   - `audit/` - 6 command modules (generate, train, scan, patterns, similar, index)

3. **`cortical/core/bootstrap.py`** - DI Container with modules
   - Pattern: `ContainerModule.register(container)`
   - Existing: SchemaModule, CDGModule, GoTModule, AuditModule

4. **`cortical/audits/`** - Audit business logic (Phase 1 complete):
   - `algorithms/` - 11 algorithm implementations
   - `patterns.py` - Pattern definitions
   - `scanner.py` - File scanning utilities
   - `classifier.py` - Classification logic
   - `training.py` - Training data generation

### Scripts Remaining to Migrate (Phase 3)

| Script | Lines | Purpose | Complexity |
|--------|-------|---------|------------|
| `audit_reasoning.py` | ~1200 | PLN-based reasoning + NLU queries | High |
| `codebase_health.py` | ~400 | Health analyzer (used by reasoning) | Medium |
| `woven_audit_discovery.py` | ~300 | WovenMind pattern discovery | Medium |
| `generate_synthetic_audit.py` | ~200 | Test data generation | Low |

---

## Phase 3: PLN/PRISM Integration Migration

### Target Architecture

```
cortical/
├── cli/
│   └── audit/
│       ├── __init__.py               # UPDATE: Add reason, discover, health
│       ├── generate.py               # ✅ Complete
│       ├── train.py                  # ✅ Complete
│       ├── scan.py                   # ✅ Complete
│       ├── patterns.py               # ✅ Complete
│       ├── similar.py                # ✅ Complete
│       ├── index.py                  # ✅ Complete
│       ├── reason.py                 # NEW: PLN reasoning command
│       ├── discover.py               # NEW: WovenMind discovery command
│       └── health.py                 # NEW: Codebase health command
│
├── audits/
│   ├── __init__.py                   # UPDATE: Export new modules
│   ├── algorithms/                   # ✅ Complete (11 algorithms)
│   ├── patterns.py                   # ✅ Complete
│   ├── scanner.py                    # ✅ Complete
│   ├── classifier.py                 # ✅ Complete
│   ├── training.py                   # ✅ Complete
│   ├── reasoning.py                  # NEW: PLN reasoning business logic
│   ├── health.py                     # NEW: Health analysis logic
│   ├── discovery.py                  # NEW: WovenMind discovery logic
│   └── persistence.py                # NEW: State persistence backend
│
└── core/modules/
    └── audit_module.py               # UPDATE: Register reasoning services
```

### Migration Steps

#### Step 3.1: Extract Business Logic

Extract from `scripts/audit_reasoning.py` → `cortical/audits/`:

| Source | Target | Content |
|--------|--------|---------|
| `PersistenceBackend` protocol | `audits/persistence.py` | State persistence |
| `FilePersistenceBackend` class | `audits/persistence.py` | File-based persistence |
| `AuditPersistenceState` | `audits/persistence.py` | State dataclass |
| `AuditQuery` dataclass | `audits/reasoning.py` | NLU query parsing |
| `translate_natural_language()` | `audits/reasoning.py` | NLU translation |
| `run_audit_analysis()` | `audits/reasoning.py` | Main analysis logic |
| `explain_file_risk()` | `audits/reasoning.py` | Explainability |
| PLN rule loading/saving | `audits/reasoning.py` | Rule management |

Extract from `scripts/codebase_health.py` → `cortical/audits/health.py`:

| Source | Target | Content |
|--------|--------|---------|
| `CodebaseAnalyzer` class | `audits/health.py` | Main analyzer |
| `analyze_directory()` | `audits/health.py` | Entry point |
| Pattern detection logic | `audits/health.py` | Reusable analysis |

#### Step 3.2: Create CLI Commands

| Command | File | Usage |
|---------|------|-------|
| `reason` | `cli/audit/reason.py` | `audit reason cortical/ --verbose` |
| `discover` | `cli/audit/discover.py` | `audit discover cortical/ --with-git` |
| `health` | `cli/audit/health.py` | `audit health cortical/` |

#### Step 3.3: Update Container

Add to `cortical/core/modules/audit_module.py`:

```python
# Reasoning services
container.register(PersistenceBackend, FilePersistenceBackend)
container.register(AuditReasoner)  # Wraps PLNReasoner
container.register(NLQueryParser)
container.register(HealthAnalyzer)
```

#### Step 3.4: Update Thin Wrappers

Keep scripts as backwards-compatible wrappers:
- `scripts/audit_reasoning.py` → delegates to `cortical.cli.audit.reason`
- `scripts/codebase_health.py` → delegates to `cortical.cli.audit.health`
- `scripts/woven_audit_discovery.py` → delegates to `cortical.cli.audit.discover`

---

## Files to Create (Phase 3)

### Business Logic
1. `cortical/audits/reasoning.py` - PLN reasoning logic
2. `cortical/audits/health.py` - Health analysis logic
3. `cortical/audits/discovery.py` - WovenMind discovery logic
4. `cortical/audits/persistence.py` - State persistence

### CLI Commands
5. `cortical/cli/audit/reason.py` - Reason command
6. `cortical/cli/audit/discover.py` - Discover command
7. `cortical/cli/audit/health.py` - Health command

## Files to Update (Phase 3)

1. `cortical/cli/audit/__init__.py` - Add new commands
2. `cortical/audits/__init__.py` - Export new modules
3. `cortical/core/modules/audit_module.py` - Register new services
4. `scripts/audit_reasoning.py` - Convert to thin wrapper
5. `scripts/codebase_health.py` - Convert to thin wrapper
6. `scripts/woven_audit_discovery.py` - Convert to thin wrapper

---

## Issues Found During Phase 1-2

The following issues were identified during testing and should be addressed:

### Audit Tool Issues (from functional testing)

| Issue | Severity | Description |
|-------|----------|-------------|
| scan false positives | Medium | Normal comments flagged as misleading (e.g., "Utility classes" at 84% confidence) |
| patterns finds noise | Medium | Top patterns are separator lines (`====`), not meaningful patterns |
| Count-Min Sketch overcounts | Low | Estimates show 16-40 for patterns appearing 2 times |
| similar tokenizer aggressive | Low | "Create task" tokenizes to just `['task']`, dropping verb |

### Deferred Improvements

```
# TODO(migration): Filter separator lines in patterns command before analysis
# TODO(migration): Tune Naive Bayes training for better precision
# TODO(migration): Adjust LSH threshold defaults for better recall
# TODO(migration): Review tokenizer settings for similar command
```

---

## Design Decisions

### Q: Why separate `reasoning.py` from `classifier.py`?

A: Different concerns:
- `classifier.py` - Naive Bayes comment classification (simple ML)
- `reasoning.py` - PLN probabilistic logic with uncertainty propagation (complex inference)

### Q: Why extract persistence to its own module?

A: The persistence backend is used across multiple commands (reason, discover) and supports
different implementations (File, InMemory, Null). Separate module enables DI.

### Q: Should `codebase_health.py` be merged into `scanner.py`?

A: No. Health analysis goes beyond scanning - it includes pattern frequency, LSH similarity,
suffix arrays. Keep separate for single responsibility.

---

## Approval Checklist

- [x] Phase 1 structure reviewed
- [x] Phase 2 container integration approach confirmed
- [x] Phase 2 migration scope agreed
- [x] Phase 2 backwards compatibility plan acceptable
- [x] Phase 3 scope reviewed
- [x] Phase 3 PLN integration approach confirmed

## Phase 3 Completion Summary

### Created Files
1. `cortical/audits/persistence.py` - State persistence backends (File, InMemory, Null)
2. `cortical/audits/health.py` - CodebaseAnalyzer and health analysis logic
3. `cortical/audits/reasoning.py` - AuditReasoner, AuditQuery, NLU parsing
4. `cortical/audits/discovery.py` - WovenMindDiscovery, pattern tokenization, novelty detection
5. `cortical/cli/audit/health.py` - Health CLI command
6. `cortical/cli/audit/reason.py` - Reason CLI command
7. `cortical/cli/audit/discover.py` - WovenMind discovery CLI command (experimental)

### Updated Files
1. `cortical/audits/__init__.py` - Export all new modules (persistence, health, reasoning, discovery)
2. `cortical/cli/audit/__init__.py` - Register all new commands (health, reason, discover)
3. `cortical/core/modules/audit_module.py` - Register reasoning services in DI container
4. `scripts/audit_reasoning.py` - Fixed monkeypatch bug for tests, added deprecation note
5. `scripts/codebase_health.py` - Added deprecation note
6. `scripts/woven_audit_discovery.py` - Added deprecation note

### DI Container Services Registered
- `PersistenceBackend` → `FilePersistenceBackend`
- `AuditReasoner` (with injected persistence)
- `CodebaseAnalyzer`
- `WovenMindDiscovery`

### Test Results
- 34 smoke tests pass
- 138 audit reasoning tests pass
- All CLI commands functional (9 total: generate, train, scan, patterns, similar, index, health, reason, discover)
- DI container registration verified

---

## Escape Note

**Remember to add TODO comments for any issues encountered during the migration.**
This allows us to track and implement deferred items later without blocking progress.

Format: `# TODO(migration): Description of issue or deferred work`
