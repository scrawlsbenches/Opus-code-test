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
- [x] Behavioral tests for CDG schema (16 tests → 15 after removing global test)
- [x] Removed global registry functions from `cortical/cdg/schema/__init__.py`
- [x] Deleted `cortical/got/schema.py` re-export shim
- [x] Changed `ensure_schemas_registered()` → `register_all_schemas(registry)`
- [x] Updated SchemaModule to use new function
- [x] Updated `cortical/cdg/__init__.py` exports
- [x] Fixed `cortical/got/__init__.py` - imports from CDG, removed deleted exports
- [x] Deleted test_global_registry_can_be_replaced (tested removed pattern)

### In Progress - Classes Needing Constructor Injection

These classes call `get_registry()` and need SchemaRegistry passed via constructor:

| Class | File | What it uses registry for |
|-------|------|---------------------------|
| Query | query_builder.py:890 | Validates entity types in `.entities()` |
| VersionedStore | versioned_store.py:307 | Validates entities in `_validate_entity()` |
| FieldValidator | expression/validator.py:43 | Gets valid fields for entity types |

**Note:** No backward compat - SchemaRegistry will be REQUIRED parameter.

### Pending - Module Functions

These standalone functions call `get_registry()`:

| Function | File | What it does |
|----------|------|--------------|
| get_entity_type_for_prefix() | entity_schemas.py:762 | Maps ID prefix to type |
| list_id_prefixes() | entity_schemas.py:783 | Lists all prefixes |

**Decision:** Accept registry as required parameter.

### Pending - CLI Handler

| Function | File |
|----------|------|
| _cmd_list_fields() | cli/query.py:470 |

**Decision:** Resolve SchemaRegistry from Container.

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

## Design Decisions Made

1. **No backward compatibility** - fix problems directly, don't create shims
2. **Container-first** - SchemaRegistry injected via Container, no globals
3. **Single registration** - schemas registered once at startup, no live changes
4. **CDG is foundation** - all storage goes through CDG, GoT is facade
5. **Required parameters** - no optional with fallback to globals

---

## Commits This Session

1. `bcf16a0a` - refactor(cdg): Remove singleton from SchemaRegistry, add referential integrity
2. `b36f97ac` - test(cdg): Add behavioral tests for CDG schema infrastructure
3. `737227d8` - docs: Add file access audit scratchpad
4. `ea8532b8` - refactor(schema): Remove global registry functions, delete GoT schema shim
5. `9e7e5d5b` - fix: Update imports after schema.py deletion

---

## Notes

- Don't run tests until user says to
- Go slow, check in before making changes
- Keep this scratchpad updated before context compaction
