# File Access Audit - Working Scratchpad

*Session: 2026-01-06*
*Branch: claude/fix-file-access-issues-1zUM9*

---

## Goal

Hunt down code accessing files incorrectly (bypassing CDG transactional storage API).

**Architecture principle:**
- CDG = foundation layer (handles all file I/O, storage, transactions)
- GoT = thin facade on CDG (domain logic only, no direct file access)

---

## Schema Refactoring Status

### Completed
- [x] Removed singleton from SchemaRegistry
- [x] Added ReferenceRule and OnDeleteAction for referential integrity
- [x] CDGStore receives SchemaRegistry via constructor injection
- [x] CDGStore._validate_entity() uses SchemaRegistry when available
- [x] Behavioral tests for CDG schema (16 tests)

### Needs Cleanup (backward compat cruft to remove)
- [ ] `cortical/cdg/schema/__init__.py`: Remove global registry functions
  - `_registry` global variable
  - `get_registry()`
  - `set_registry()`
  - `register_schema()`
  - `validate_entity()`
  - `migrate_entity()`
- [ ] `cortical/got/schema.py`: Delete re-export shim entirely
- [ ] `cortical/got/entity_schemas.py`: Remove `reset_schema_registration()` hack
- [ ] Update all code using `get_registry()` to use Container injection
- [ ] Update all imports from `cortical.got.schema` to `cortical.cdg.schema`

---

## Direct File Access Violations Found

### From sub-agent research (25+ violations in cortical/got/)

**api.py - Entity reader methods bypassing transactional API:**
1. `get_task()` - direct file read
2. `get_decision()` - direct file read
3. `get_sprint()` - direct file read
4. `get_edge()` - direct file read
5. `get_handoff()` - direct file read
6. `get_knowledge_transfer()` - direct file read
7. `get_epic()` - direct file read
8. `get_goal()` - direct file read
9. `get_plan()` - direct file read

**api.py - Direct Path operations:**
- `Path.exists()` calls for entity checking
- `delete_sprint()` using direct file operations instead of transactional API

**Other files with direct file access:**
- `cortical/got/claudemd.py` - direct entity scanning
- `cortical/got/cli/query.py` - Path.exists() checks
- `cortical/got/query_builder.py` - direct file access
- `cortical/got/indexer.py` - direct file I/O (open() calls)

**WAL/Recovery (may be legitimate):**
- `.unlink()` calls in WAL files
- Recovery code file operations

---

## Questions to Resolve

1. How should entity reader methods work? Should CDGStore have a generic `read()` that GoTManager wraps?

2. For entity existence checks, should CDGStore expose `exists(entity_id)` method?

3. The indexer builds search indexes - should this be a CDG responsibility or remain in GoT?

---

## Next Steps

1. Clean up schema backward compat cruft
2. Find all usages of removed functions and fix them
3. Return to direct file access list and fix systematically

---

## Notes

- Don't run tests until user says to
- Go slow, check in before making changes
- We don't care about backward compatibility - fix the problems
