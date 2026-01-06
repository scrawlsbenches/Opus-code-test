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

## Current State

Schema constructor injection is DONE. All classes now receive SchemaRegistry via constructor or resolve from Container:
- VersionedStore, GoTManager, Query, FieldValidator - all fixed
- CLI handlers resolve from Container
- Module functions accept registry parameter

**Next focus:** Direct file access violations in GoT (see below).

---

## Direct File Access Violations (not yet addressed)

### api.py - bypassing transactional API
- get_task(), get_decision(), get_sprint(), etc. - direct file reads
- Path.exists() calls for entity checking
- delete_sprint() using direct file operations

### Other files
- claudemd.py - direct entity scanning
- indexer.py - direct file I/O (open() calls)
- query_builder.py - direct file access

### May be legitimate
- WAL/Recovery code file operations (low-level by design)

---

## Design Decisions

1. **No backward compatibility** - fix problems directly, don't create shims
2. **Container-first** - dependencies injected via Container, no globals
3. **CDG is foundation** - all storage goes through CDG, GoT is facade
4. **Required parameters** - no optional with fallback to globals

---

## Notes

- Don't run tests until user says to
- Go slow, check in before making changes
