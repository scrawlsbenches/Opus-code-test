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

Schema constructor injection is DONE.

**Next:** Investigate if VersionedStore is redundant with CDG.

---

## Architectural Insights (from user)

### Durability Mode
- GoT has NO business with durability mode
- Must be configured centrally (in Container/CDG)
- Push back on any durability config in GoT

### VersionedStore vs CDG
- Question: Is versioned_store.py redundant now that CDG exists?
- Need to investigate what it does vs what CDGStore does
- If redundant, the whole file may be unnecessary

### got/expression/*
- Contains half-baked but FANTASTIC ideas
- Will need to utilize schema more but in a GENERAL way
- This is harder than it sounds - be careful going down this path!

### got/cli/* Pattern
- Should have a member variable for Container
- Query container from member variable in functions
- DO NOT import cortical.core.bootstrap in functions

### Schema Validation on Missing
- When schema not found: should we throw exception instead of silent return?
- Current: `if not registry.has_schema(entity_type): return`
- Need to decide proper behavior

---

## Design Decisions

1. **No backward compatibility** - fix problems directly, don't create shims
2. **Container-first** - dependencies injected via Container, no globals
3. **CDG is foundation** - all storage goes through CDG, GoT is facade
4. **Required parameters** - no optional with fallback to globals
5. **Centralized configuration** - durability, paths, etc. configured at Container level

---

## Notes

- Don't run tests until user says to
- Go slow, check in before making changes
- Store architectural insights in this scratchpad
