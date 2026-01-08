# Audit Tools Migration Plan

## Date: 2026-01-08
## Author: Claude
## Status: Planning

---

## Current State Analysis

### Existing CLI Patterns in Cortical

1. **`cortical/got/cli/`** - Well-structured command modules:
   - Pattern: `setup_X_parser()`, `handle_X_command()`, `cmd_X_Y()`
   - 15+ command modules (task, edge, sprint, decision, query, etc.)
   - Good separation of concerns

2. **`cortical/cli_wrapper.py`** - Shell command wrapper framework
   - Not relevant for subcommand pattern

3. **`cortical/core/bootstrap.py`** - DI Container with modules
   - Pattern: `ContainerModule.register(container)`
   - Existing: SchemaModule, CDGModule, GoTModule

4. **`cortical/audits/algorithms/`** - 11 algorithm implementations
   - Already well-placed, should stay here

### Scripts to Migrate

1. `scripts/audit_tool.py` - 6 commands (generate, train, scan, patterns, similar, index)
2. `scripts/audit_reasoning.py` - PLN-based reasoning (complex, ~2000 lines)
3. `scripts/codebase_health.py` - Health analyzer
4. `scripts/causal_audit_analyzer.py` - Git correlation analysis
5. `scripts/woven_audit_discovery.py` - WovenMind pattern discovery

---

## Proposed Architecture

```
cortical/
├── cli/                              # NEW: Unified CLI layer
│   ├── __init__.py                   # CLI registry, auto-discovery
│   ├── _base.py                      # Base utilities, decorators
│   └── audit/                        # Audit commands
│       ├── __init__.py               # Exports setup_audit_parser, handle_audit_command
│       ├── generate.py               # generate command
│       ├── train.py                  # train command
│       ├── scan.py                   # scan command
│       ├── patterns.py               # patterns command
│       ├── similar.py                # similar command
│       ├── index.py                  # index command
│       └── reason.py                 # reasoning command (from audit_reasoning.py)
│
├── audits/                           # KEEP: Audit business logic
│   ├── __init__.py                   # Public API
│   ├── algorithms/                   # 11 algorithm implementations (KEEP)
│   ├── classifier.py                 # NEW: Comment classification logic
│   ├── patterns.py                   # NEW: Pattern definitions (MISLEADING, ACCURATE, etc.)
│   ├── training.py                   # NEW: Training data generation logic
│   ├── scanner.py                    # NEW: Scanning logic
│   └── reasoning.py                  # NEW: PLN reasoning integration
│
├── core/
│   ├── bootstrap.py                  # UPDATE: Add AuditModule
│   └── modules/
│       └── audit_module.py           # NEW: DI registration for audit services
│
└── got/
    └── cli/                          # KEEP: Existing GoT CLI (unchanged)
```

---

## Container Integration

### New AuditModule

```python
# cortical/core/modules/audit_module.py
class AuditModule(ContainerModule):
    """Register audit-related services."""

    def register(self, container: Container) -> None:
        # Register algorithms
        container.register(SuspiciousCommentFilter)
        container.register(CommentClassifier)
        container.register(SimilarCommentFinder)
        # etc.

        # Register services
        container.register(AuditScanner)
        container.register(TrainingDataGenerator)
        container.register(CommentPatternMatcher)
```

### Why Container Makes Sense Here

1. **Testability** - Can inject mock classifiers, mock file systems
2. **Configuration** - Pattern lists, thresholds can be injected
3. **Reusability** - Algorithms shared across multiple commands
4. **Consistency** - Same pattern as GoT, CDG modules

---

## Migration Steps

### Phase 1: Infrastructure (This PR)

1. Create `cortical/cli/` with base infrastructure
2. Create `cortical/cli/audit/` with command modules
3. Create `cortical/core/modules/audit_module.py`
4. Extract patterns/utilities to `cortical/audits/` modules
5. Update `cortical/core/bootstrap.py` to include AuditModule

### Phase 2: Script Migration

1. Migrate `audit_tool.py` commands to `cortical/cli/audit/`
2. Keep `scripts/audit_tool.py` as thin wrapper (backwards compat)
3. Run tests to verify functionality

### Phase 3: Extended Tools (Future)

1. Migrate `audit_reasoning.py`
2. Migrate `codebase_health.py`
3. Migrate `causal_audit_analyzer.py`
4. Migrate `woven_audit_discovery.py`

---

## Design Decisions

### Q: Why `cortical/cli/audit/` not `cortical/audits/cli/`?

A: Grouping by capability (cli/) makes it easier to:
- Add new CLI domains (cli/spark/, cli/cel/, etc.)
- Share CLI infrastructure across domains
- Keep business logic separate from presentation

### Q: Why keep algorithms in `cortical/audits/algorithms/`?

A: They are business logic, not CLI concerns. Commands use them via the container.

### Q: Why use existing bootstrap pattern?

A: Consistency with the codebase. All major subsystems (CDG, GoT, CEL) use modules.
   Adding AuditModule follows established patterns.

### Q: What about the prototype in `scripts/audit_commands/`?

A: Delete it after migration. It was a proof-of-concept.

---

## Files to Create

1. `cortical/cli/__init__.py`
2. `cortical/cli/_base.py`
3. `cortical/cli/audit/__init__.py`
4. `cortical/cli/audit/generate.py`
5. `cortical/cli/audit/train.py`
6. `cortical/cli/audit/scan.py`
7. `cortical/cli/audit/patterns.py`
8. `cortical/cli/audit/similar.py`
9. `cortical/cli/audit/index.py`
10. `cortical/audits/classifier.py`
11. `cortical/audits/patterns.py`
12. `cortical/audits/training.py`
13. `cortical/audits/scanner.py`
14. `cortical/core/modules/audit_module.py`

## Files to Update

1. `cortical/core/bootstrap.py` - Import and apply AuditModule
2. `cortical/core/modules/__init__.py` - Export AuditModule
3. `scripts/audit_tool.py` - Thin wrapper calling cortical.cli.audit

## Files to Delete

1. `scripts/audit_commands/__init__.py`
2. `scripts/audit_commands/_base.py`
3. `scripts/audit_commands/generate.py`
4. `scripts/audit_tool_new.py`

---

## Approval Checklist

- [ ] Structure reviewed
- [ ] Container integration approach confirmed
- [ ] Migration scope agreed
- [ ] Backwards compatibility plan acceptable

---

## Escape Note

**Remember to add TODO comments for any issues encountered during the migration.**
This allows us to track and implement deferred items later without blocking progress.

Format: `# TODO(migration): Description of issue or deferred work`
