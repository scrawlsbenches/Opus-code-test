# Knowledge Transfer: Sprint System Migration to GoT

**Date:** 2025-12-22
**Session:** claude/add-state-management-ROOCK
**Tags:** `sprint`, `got`, `migration`, `architecture`, `refactor`

## Executive Summary

Migrated sprint/epic tracking from a markdown file (`tasks/CURRENT_SPRINT.md`) to the GoT (Graph of Thought) transactional system. The TX backend is now the default, simplifying the developer experience.

## What Changed

### Before
- Sprints tracked in `tasks/CURRENT_SPRINT.md` (markdown)
- Two GoT backends: event-sourced (`.got/`) and transactional (`.got-tx/`)
- Required `GOT_USE_TX=1` environment variable for TX backend
- Manual parsing of markdown for sprint status

### After
- Sprints stored as JSON entities in `.got/entities/S-*.json`
- Epics stored as JSON entities in `.got/entities/EPIC-*.json`
- Single `.got/` directory for all data
- TX backend is default (no env var needed)
- `CURRENT_SPRINT.md` marked deprecated with migration instructions

## New Entity Types

### Sprint (`cortical/got/types.py`)
```python
@dataclass
class Sprint(Entity):
    title: str = ""
    status: str = "available"  # available, in_progress, completed, blocked
    epic_id: str = ""          # Parent epic
    number: int = 0            # Sprint number
    session_id: str = ""       # Claude session working on it
    isolation: List[str] = []  # File paths for isolation
    goals: List[Dict] = []     # Goals with completion status
    notes: List[str] = []
```

### Epic (`cortical/got/types.py`)
```python
@dataclass
class Epic(Entity):
    title: str = ""
    status: str = "active"  # active, completed, on_hold
    phase: int = 1
    phases: List[Dict] = []  # Phase definitions
```

## CLI Commands

```bash
# Sprint commands
python scripts/got_utils.py sprint list      # List all sprints
python scripts/got_utils.py sprint status    # Show current sprint
python scripts/got_utils.py sprint create "Title" --number N

# Task commands (unchanged)
python scripts/got_utils.py task list
python scripts/got_utils.py task create "Title" --priority high
python scripts/got_utils.py validate
```

## Files Modified

| File | Change |
|------|--------|
| `cortical/got/types.py` | Added Sprint, Epic entity types |
| `cortical/got/api.py` | Added sprint/epic CRUD methods to GoTManager |
| `cortical/got/__init__.py` | Exported Sprint, Epic |
| `cortical/utils/id_generation.py` | Added generate_epic_id() |
| `scripts/got_utils.py` | TX backend now default, added adapter methods |
| `scripts/migrate_sprints_to_got.py` | New migration script |
| `tasks/CURRENT_SPRINT.md` | Added deprecation notice |
| `CLAUDE.md` | Updated sprint documentation |
| `docs/sprint-migration-guide.md` | New migration guide |

## Migration Details

### Data Migrated
- 3 Epics (Hubris MoE, Cortical Core, NLU Enhancement)
- 20 Sprints with full metadata (goals, notes, isolation paths)
- 14 PART_OF edges (sprint → epic relationships)

### Status Mapping
| Markdown | GoT Status |
|----------|------------|
| 🟢 Available | `available` |
| 🟡 In Progress | `in_progress` |
| ✅ Complete | `completed` |
| 🔴 Blocked | `blocked` |

## Architecture Decisions

### Why TX Backend as Default?
1. **ACID transactions** - Concurrent agent safety
2. **Simpler code** - No event log management
3. **Sprint support** - Full CRUD for sprints/epics
4. **Single directory** - No `.got` vs `.got-tx` confusion

### Why Keep Event-Sourced Backend?
1. **Backward compatibility** - Set `GOT_USE_LEGACY=1`
2. **Audit trail** - Events still in `.got/events/`
3. **Git-friendly** - Event logs merge better

## Open Work Items

| Task ID | Description | Priority |
|---------|-------------|----------|
| T-20251222-114148-01df7b7e | Add sprint start command | Medium |
| T-20251222-114159-9301a173 | Add sprint complete command | Medium |
| T-20251222-114203-6991f2c8 | Add task-to-sprint linking CLI | Low |
| T-20251222-114208-1aa8752e | Add sprint goal tracking CLI | Low |
| T-20251222-114214-6fa641da | Deprecate event-sourced backend | Medium |
| T-20251222-114219-4369f190 | Auto-migration on first run | Low |

## Lessons Learned

### What Worked Well
- Sub-agents for parallel implementation (types, API, migration)
- Single directory consolidation reduced confusion
- TX backend auto-commit eliminates manual save() calls

### What Was Confusing
- Two backends with different directories
- Environment variable switching
- `self.manager` vs `self._manager` inconsistency
- Migration script defaulting to wrong directory

### Recommendations for Future
1. Default to TX backend for new entity types
2. Keep single `.got/` directory
3. Add deprecation warnings to event-sourced code
4. Test CLI commands after adapter changes

## Quick Reference

```bash
# Check sprint status (new session start)
python scripts/got_utils.py sprint status

# List all sprints
python scripts/got_utils.py sprint list

# Validate GoT health
python scripts/got_utils.py validate

# Fall back to legacy backend (if needed)
GOT_USE_LEGACY=1 python scripts/got_utils.py task list
```

## Related Documents
- `docs/sprint-migration-guide.md` - Detailed migration instructions
- `tasks/CURRENT_SPRINT.md` - Deprecated, historical reference
- `CLAUDE.md` - Updated Quick Session Start section
