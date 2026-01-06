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

### Completed (this session)
- [x] Removed singleton from SchemaRegistry
- [x] Added ReferenceRule and OnDeleteAction for referential integrity
- [x] CDGStore receives SchemaRegistry via constructor injection
- [x] CDGStore._validate_entity() uses SchemaRegistry when available
- [x] Behavioral tests for CDG schema (16 tests)
- [x] Removed global registry functions from `cortical/cdg/schema/__init__.py`
- [x] Deleted `cortical/got/schema.py` re-export shim
- [x] Changed `ensure_schemas_registered()` → `register_all_schemas(registry)`
- [x] Updated SchemaModule to use new function
- [x] Updated `cortical/cdg/__init__.py` exports

### Code Still Using Removed Functions (BREAKS)

Files that import from deleted `cortical.got.schema`:
```
cortical/got/__init__.py - line 109-118: imports schema functions to re-export
cortical/got/api.py - line 49, 202-203: imports SchemaRegistry, get_registry()
```

Files using `get_registry()` that needs Container injection:
```
cortical/got/query_builder.py - line 890, 894
cortical/got/versioned_store.py - line 33, 307
cortical/got/cli/query.py - line 470, 476
cortical/got/expression/validator.py - line 14, 43
cortical/got/entity_schemas.py - line 762, 783 (in helper functions)
```

### Next to Fix (in order)
1. `cortical/got/__init__.py` - remove schema re-exports
2. `cortical/got/api.py` - import SchemaRegistry from CDG
3. Files using `get_registry()` - need to receive SchemaRegistry via constructor

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

## Design Decisions Made

1. **No backward compatibility** - fix problems directly, don't create shims
2. **Container-first** - SchemaRegistry injected via Container, no globals
3. **Single registration** - schemas registered once at startup, no live changes
4. **CDG is foundation** - all storage goes through CDG, GoT is facade

---

## Notes

- Don't run tests until user says to
- Go slow, check in before making changes
- Keep this scratchpad updated before context compaction
