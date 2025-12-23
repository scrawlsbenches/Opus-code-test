# Knowledge Transfer: CLAUDE.md Auto-Generation System

**Date:** 2025-12-23
**Branch:** `claude/auto-generate-claude-md-LwdTl`
**Status:** Ready for PR, rebased on main
**Epic:** T-20251222-204058-3675a72c (completed)

---

## Summary

Implemented a 5-layer CLAUDE.md auto-generation system integrated with Graph of Thought (GoT). The system allows context-aware, persona-customized generation of CLAUDE.md content with freshness tracking and fault tolerance.

---

## What Was Built

### 1. Core Entities (`cortical/got/types.py`)

| Entity | Purpose | Key Methods |
|--------|---------|-------------|
| `ClaudeMdLayer` | Content layer with freshness tracking | `is_stale()`, `mark_fresh()`, `mark_stale()` |
| `ClaudeMdVersion` | Version snapshots for audit trail | `compute_content_hash()` |
| `PersonaProfile` | Role-based layer preferences | `should_include_layer()`, `get_effective_preferences()` |
| `Team` | Organizational hierarchy | `matches_branch()`, `is_in_scope()`, `get_setting()` |

### 2. 5-Layer Architecture

```
Layer 0 (Core)        → Quick Session Start - ALWAYS included
Layer 1 (Operational) → Dev Workflow - ALWAYS included
Layer 2 (Contextual)  → GoT Guide - When working on cortical/got/*
Layer 3 (Persona)     → ML Guide - For ml-engineer persona
Layer 4 (Ephemeral)   → Session Context - Per-session, 1-day decay
```

### 3. Generator Module (`cortical/got/claudemd.py` - 619 lines)

- `ContextAnalyzer` - Detects branch, sprint, active files
- `LayerSelector` - Context-based layer filtering
- `ClaudeMdComposer` - Content assembly with ordering
- `ClaudeMdValidator` - Required section checking
- `ClaudeMdGenerator` - Fault-tolerant generation with fallback
- `ClaudeMdManager` - High-level API

### 4. ID Generation (`cortical/utils/id_generation.py`)

- `generate_claudemd_layer_id(layer_number, section_id)` → `CML{N}-{section}-{timestamp}-{hex}`
- `generate_claudemd_version_id(layer_id, version)` → `CMV-{layer_id}-v{N}`
- `generate_persona_profile_id()` → `PP-{timestamp}-{hex}`
- `generate_team_id()` → `TEAM-{timestamp}-{hex}`

### 5. VersionedStore Updates

Added deserialization support for all new entity types in `_entity_from_dict()`.

---

## Files Changed

### New Files
- `cortical/got/claudemd.py` (619 lines)
- `docs/claude-md-generation-design.md` (~1100 lines)
- `scripts/claudemd_generation_demo.py` (669 lines)
- `tests/unit/test_claudemd_layer.py` (24 tests)
- `tests/unit/test_claudemd_generator.py` (23 tests)
- `tests/unit/test_persona_profile.py` (32 tests)
- `tests/unit/test_team.py` (42 tests)
- `tests/integration/test_claudemd_pipeline.py` (14 tests)

### Modified Files
- `cortical/got/types.py` (+281 lines - 4 new entities)
- `cortical/got/api.py` (CRUD methods for ClaudeMdLayer)
- `cortical/got/versioned_store.py` (deserialization)
- `cortical/utils/id_generation.py` (+40 lines - 2 new ID generators)

---

## Tests

**135 new tests added**, all passing:
- Unit tests for all 4 new entities
- Generator component tests
- Integration pipeline tests
- Full test suite: 7,846+ tests passing

---

## Pending Work (Future Sessions)

### High Priority
- `T-20251222-204135-781dad64` - Context detection for branch/module awareness
- `T-20251222-204138-5915d70a` - Regression tests for CLAUDE.md stability
- `T-20251222-204138-51a38502` - Validate fault tolerance and recovery

### Medium Priority
- `T-20251222-204134-140ee2d3` - Knowledge freshness and decay logic
- `T-20251222-204137-56f13aae` - Behavioral tests for user workflows
- `T-20251222-204137-a80b5842` - Performance tests for generation
- `T-20251222-204133-da5a8f58` - Versioning and persona evolution

### Ideas for Future
- `T-20251223-003108-9872bb95` - Multi-branch layer inheritance (SDLC pipelines)
- `T-20251223-003121-067f7ddd` - Layer pull/merge mechanism (cross-team)

---

## How to Resume

### 1. Run the Demo
```bash
python scripts/claudemd_generation_demo.py
```

### 2. Check Task Status
```bash
python scripts/got_utils.py task list --status pending | grep -i claude
```

### 3. Read Design Doc
```bash
cat docs/claude-md-generation-design.md
```

### 4. Key Files to Understand
- `cortical/got/types.py` - Entity definitions (lines 469-978)
- `cortical/got/claudemd.py` - Generator implementation
- `scripts/claudemd_generation_demo.py` - Working demo with real content

---

## Design Decisions Made

1. **Keep original CLAUDE.md unchanged** - Acts as fallback, never modified
2. **Store layers in GoT** - Uses existing transactional infrastructure
3. **5-layer architecture** - Core → Operational → Contextual → Persona → Ephemeral
4. **Freshness decay** - Each layer has configurable decay period (1-30 days)
5. **Persona inheritance** - PersonaProfile can inherit from parent profile
6. **Team hierarchy** - Teams can have parent teams for SDLC pipelines

---

## Potential Issues to Watch

1. **GoT _version.json conflicts** - Will occur when parallel branches modify GoT
   - Resolution: Take highest version number during rebase
   - Future fix: `T-20251223-005914-f298d276` (make merge-conflict free)

2. **Module scope matching** - `Team.is_in_scope()` uses simple string matching
   - May need glob pattern support for complex module structures

3. **Circular inheritance** - PersonaProfile has `inherits_from` field
   - Validation prevents self-reference but not longer cycles

---

## Commands Reference

```bash
# Run demo
python scripts/claudemd_generation_demo.py

# Run tests for new features
python -m pytest tests/unit/test_claudemd_*.py tests/unit/test_persona_profile.py tests/unit/test_team.py -v

# Check branch status
git log --oneline origin/main..HEAD
```

---

**Next Session Priority:** Complete the remaining pending tasks, especially context detection (T-20251222-204135-781dad64) and fault tolerance validation (T-20251222-204138-51a38502).
