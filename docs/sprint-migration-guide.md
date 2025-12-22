# Sprint Migration Guide

## Overview

This guide documents the migration of sprint data from `tasks/CURRENT_SPRINT.md` to the GoT (Graph of Thought) transactional backend.

## Migration Script

**Location:** `/home/user/Opus-code-test/scripts/migrate_sprints_to_got.py`

### Features

- Parses CURRENT_SPRINT.md to extract Epics and Sprints
- Creates Epic and Sprint entities in GoT with proper structure
- Links Sprints to Epics via PART_OF edges
- Handles multiple epic definition patterns
- Supports dry-run mode for preview
- Validates all data before creating entities

### Usage

```bash
# Preview what would be migrated (recommended first step)
python scripts/migrate_sprints_to_got.py --dry-run

# Perform actual migration
python scripts/migrate_sprints_to_got.py

# Migrate from custom file
python scripts/migrate_sprints_to_got.py --source custom_sprints.md

# Use custom GoT directory
python scripts/migrate_sprints_to_got.py --got-dir /path/to/.got
```

## Data Structure Mapping

### Epic Parsing

The script handles two epic definition patterns:

**Pattern 1: Epics Section**
```markdown
# Epics
## Active: Hubris MoE (efba)
**Status:** Phase 5 - Expert Expansion
```

**Pattern 2: Standalone Section**
```markdown
# NLU Enhancement Epic
## Epic: NLU Enhancement (nlu)
**Started:** 2025-12-19
**Status:** Planning
```

**Epic Entity Fields:**
- `id`: `EPIC-{short_id}` (e.g., `EPIC-nlu`)
- `title`: Epic name (e.g., "NLU Enhancement")
- `status`: active | completed | on_hold
- `phase`: Current phase number (highest in_progress/completed)
- `phases`: List of phase dictionaries with:
  - `number`: Phase number
  - `name`: Phase description
  - `status`: completed | in_progress | pending
- `properties`:
  - `short_id`: Short identifier (e.g., "nlu")
  - `started`: Start date

### Sprint Parsing

**Sprint Section Pattern:**
```markdown
## Sprint 15: Search Quality Fundamentals
**Sprint ID:** sprint-015-search-quality
**Epic:** NLU Enhancement (nlu)
**Status:** Complete ✅
**Session:** dOcbe
**Isolation:** `cortical/query/`, `cortical/code_concepts.py`

### Goals
- [x] Enable code stop word filtering
- [ ] Weight lateral expansion by TF-IDF

### Notes
- Investigation identified root causes
```

**Status Emoji Mapping:**
- 🟢 Available → `"available"`
- 🟡 In Progress → `"in_progress"`
- ✅ Complete → `"completed"`
- 🔴 Blocked → `"blocked"`

**Sprint Entity Fields:**
- `id`: `S-{sprint_id}` (e.g., `S-sprint-015-search-quality`)
- `title`: Sprint title (emoji stripped)
- `status`: available | in_progress | completed | blocked
- `epic_id`: Full Epic ID (e.g., `EPIC-nlu`)
- `number`: Sprint number (e.g., 15)
- `session_id`: Session identifier (e.g., "dOcbe")
- `isolation`: List of file paths (backticks removed)
- `goals`: List of goal dictionaries with:
  - `text`: Goal description
  - `completed`: Boolean (true if `[x]`, false if `[ ]`)
- `notes`: List of note strings
- `properties`:
  - `epic_name`: Epic name for reference
  - `sprint_id`: Original sprint ID

### Edge Creation

For each Sprint with a valid `epic_id`, a PART_OF edge is created:
- `source_id`: Sprint ID
- `target_id`: Epic ID
- `edge_type`: "PART_OF"

## Migration Results

### From CURRENT_SPRINT.md

**Epics Found:**
1. Hubris MoE (efba) - Active
2. Cortical Core (core) - Maintenance
3. NLU Enhancement (nlu) - Planning

**Sprints Found:**
- Total: 20 sprints
- Completed: 7 sprints (6, 9, 1-5, 15, 20)
- In Progress: 1 sprint (17)
- Available: 12 sprints

**Epic Linkages:**
- Hubris MoE: Sprints 1-7
- Cortical Core: Sprints 8-9
- NLU Enhancement: Sprints 15-19
- No Epic: Sprint 20 (Code Quality epic not defined)

## Dry Run Example

```bash
$ python scripts/migrate_sprints_to_got.py --dry-run

Parsing /home/user/Opus-code-test/tasks/CURRENT_SPRINT.md...

Found 3 epics and 20 sprints

======================================================================
DRY RUN MODE - No changes will be made
======================================================================

----------------------------------------------------------------------
Migrating Epics
----------------------------------------------------------------------
[DRY-RUN] Would create Epic: Hubris MoE
  ID: EPIC-efba
  Status: active
  Phase: 5/5
  Phases: 5

[DRY-RUN] Would create Epic: NLU Enhancement
  ID: EPIC-nlu
  Status: active
  Phase: 5/5
  Phases: 5

----------------------------------------------------------------------
Migrating Sprints
----------------------------------------------------------------------
[DRY-RUN] Would create Sprint: Sprint 15: Search Quality Fundamentals
  ID: S-sprint-015-search-quality
  Number: 15
  Status: completed
  Epic: EPIC-nlu
  Session: dOcbe
  Isolation: 2 paths
  Goals: 6 (6 completed)
  Notes: 4

...

======================================================================
DRY RUN COMPLETE - No changes made
======================================================================
Epics: 3
Sprints: 20
```

## After Migration

### Querying Sprints

```python
from cortical.got.api import GoTManager

manager = GoTManager(".got")

# Find all sprints
sprints = manager.find_tasks()  # Will include Sprint entities

# Find sprints by status
active_sprints = manager.find_tasks(status="in_progress")

# Get specific sprint
sprint = manager.get_task("S-sprint-015-search-quality")

# Get sprint's epic
if sprint and sprint.epic_id:
    epic = manager.read(sprint.epic_id)
```

### Querying Epics

```python
# Read epic by ID
epic = manager.read("EPIC-nlu")

# Find all sprints in an epic (via PART_OF edges)
# Note: Need to implement sprint query methods in GoT API
```

## Edge Cases Handled

1. **Missing Epics**: Sprints referencing undefined epics are created without epic_id
2. **Multiple Epic Patterns**: Both "# Epics" section and standalone sections are parsed
3. **Legacy Sprints**: Sprints without full metadata (Goals/Notes) are handled gracefully
4. **Status Variations**: Status emoji are correctly mapped to valid status strings
5. **Isolation Paths**: Backticks and comma-separated paths are properly parsed

## Validation

After migration, validate with:

```bash
# Check GoT health
python scripts/got_utils.py validate

# List all sprints
python scripts/got_utils.py list --type sprint

# View specific sprint
python scripts/got_utils.py show S-sprint-015-search-quality
```

## Rollback

To rollback the migration:

```bash
# Remove all migrated entities
rm -rf .got/entities/EPIC-*
rm -rf .got/entities/S-*
rm -rf .got/entities/E-S-*  # PART_OF edges
```

## Future Enhancements

1. **Bidirectional Sync**: Update CURRENT_SPRINT.md from GoT data
2. **Sprint Query Methods**: Add `find_sprints()` and `get_epic_sprints()` to GoT API
3. **Phase Tracking**: Link sprint completion to epic phase progression
4. **Goal Extraction**: Create Task entities for uncompleted sprint goals
5. **Dependency Tracking**: Parse and create edges for sprint dependencies

## References

- GoT Types: `/home/user/Opus-code-test/cortical/got/types.py`
- GoT API: `/home/user/Opus-code-test/cortical/got/api.py`
- Source File: `/home/user/Opus-code-test/tasks/CURRENT_SPRINT.md`
- Migration Script: `/home/user/Opus-code-test/scripts/migrate_sprints_to_got.py`
