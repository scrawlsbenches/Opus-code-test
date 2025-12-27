# Knowledge Transfer: GoT Relationships Query - Generic Entity & Edge Support

**Date:** 2025-12-27
**Session:** 7K1pS
**Branch:** `claude/sparkslm-git-training-7K1pS`
**Tags:** `got`, `bugfix`, `relationships`, `edge-types`, `entity-types`

---

## Executive Summary

Fixed the GoT `query "relationships <id>"` command to work with **all entity types** (tasks, sprints, decisions, epics, handoffs) and **all 29 edge types** (not just BLOCKS, DEPENDS_ON, CONTAINS).

---

## Problem Statement

The `relationships` query was broken for non-task entities and most edge types:

```bash
# Before fix - sprint relationships returned nothing
$ python scripts/got_utils.py query "relationships S-027"
No results found.

# Before fix - handoff with TRANSFERS edge returned nothing
$ python scripts/got_utils.py query "relationships H-20251224-070920-ab3027eb"
No results found.
```

### Root Causes

1. **`get_all_relationships()` only looked up tasks**
   - Called `get_task()` for edge targets - missed sprints, epics, handoffs
   - No converter methods for non-task entities

2. **Only 3 of 29 edge types handled**
   - Hardcoded: BLOCKS, DEPENDS_ON, CONTAINS
   - Ignored: TRANSFERS, IMPLEMENTS, REQUIRES, ENABLES, etc. (26 types)

---

## Solution

### 1. Generic Entity Lookup

Added `_get_entity_node()` helper that resolves any entity ID:

```python
def _get_entity_node(self, entity_id: str) -> Optional[ThoughtNode]:
    """Get any entity (task, sprint, decision, epic, handoff) as ThoughtNode."""

    # Fast path by prefix
    if entity_id.startswith("T-"):
        task = self._manager.get_task(entity_id)
        if task: return self._tx_task_to_node(task)

    if entity_id.startswith("S-"):
        sprint = self._manager.get_sprint(entity_id)
        if sprint: return self._tx_sprint_to_node(sprint)

    # ... similar for D-, EPIC-, H-

    # Fallback: try all types
    for getter, converter in [...]:
        entity = getter(entity_id)
        if entity: return converter(entity)
```

### 2. Converter Methods Added

New methods to convert each entity type to ThoughtNode:

| Method | Entity Type | NodeType |
|--------|-------------|----------|
| `_tx_task_to_node()` | Task | TASK |
| `_tx_sprint_to_node()` | Sprint | CONTEXT |
| `_tx_decision_to_node()` | Decision | DECISION |
| `_tx_epic_to_node()` | Epic | CONTEXT |
| `_tx_handoff_to_node()` | Handoff | TASK |

### 3. Dynamic Edge Type Handling

Edge types now processed dynamically with grammatically correct naming:

```python
def get_incoming_key(edge_type: str) -> str:
    """Convert edge type to incoming relationship key."""
    et = edge_type.lower()

    # Grammatical conjugation for common verbs
    if et == 'blocks': return 'blocked_by'
    elif et == 'contains': return 'contained_by'
    elif et == 'transfers': return 'transferred_by'
    elif et == 'triggers': return 'triggered_by'
    # ... 17 special cases total
    else:
        return f"{et}_by"  # Default fallback
```

---

## Edge Type Naming Convention

| Edge Type | Outgoing Key | Incoming Key |
|-----------|--------------|--------------|
| BLOCKS | `blocks` | `blocked_by` |
| CONTAINS | `contains` | `contained_by` |
| DEPENDS_ON | `depends_on` | `depended_by` |
| TRANSFERS | `transfers` | `transferred_by` |
| IMPLEMENTS | `implements` | `implemented_by` |
| REQUIRES | `requires` | `required_by` |
| ENABLES | `enables` | `enabled_by` |
| TRIGGERS | `triggers` | `triggered_by` |
| SUPPORTS | `supports` | `supported_by` |
| REFUTES | `refutes` | `refuted_by` |
| PRECEDES | `precedes` | `preceded_by` |
| ANSWERS | `answers` | `answered_by` |
| RAISES | `raises` | `raised_by` |
| EXPLORES | `explores` | `explored_by` |
| OBSERVES | `observes` | `observed_by` |
| SUGGESTS | `suggests` | `suggested_by` |
| TESTS | `tests` | `tested_by` |
| REFINES | `refines` | `refined_by` |
| MOTIVATES | `motivates` | `motivated_by` |
| JUSTIFIES | `justifies` | `justified_by` |
| *others* | lowercase | `{lowercase}_by` |

---

## Files Modified

| File | Changes |
|------|---------|
| `scripts/got_utils.py` | Added `_get_entity_node()`, 4 converter methods, rewrote `get_all_relationships()` |

### Key Lines

- `_get_entity_node()`: lines 1292-1346
- `_tx_sprint_to_node()`: lines 772-791
- `_tx_decision_to_node()`: lines 793-809
- `_tx_epic_to_node()`: lines 811-827
- `_tx_handoff_to_node()`: lines 829-846
- `get_all_relationships()`: lines 1348-1456

---

## Commits

| Commit | Description |
|--------|-------------|
| `a0e979c8` | Initial fix for sprints (CONTAINS edge) |
| `33fab1ca` | Add handoff support to entity lookup |
| `b45833b6` | Handle all 29 edge types dynamically |

---

## Verification Commands

```bash
# Sprint relationships (CONTAINS)
python scripts/got_utils.py query "relationships S-027"

# Task relationships (DEPENDS_ON)
python scripts/got_utils.py query "relationships T-20251227-171020-2df0d769"

# Handoff relationships (TRANSFERS)
python scripts/got_utils.py query "relationships H-20251224-070920-ab3027eb"

# Task with outgoing TRANSFERS
python scripts/got_utils.py query "relationships T-20251222-145525-445df343"

# Validate GoT health
python scripts/got_utils.py validate
```

---

## Backward Compatibility

The fix maintains backward compatibility:

1. **Existing keys preserved**: `blocks`, `blocked_by`, `depends_on`, `depended_by`, `contains`, `contained_by` always present in result dict (even if empty)

2. **New keys added dynamically**: Other edge types only appear when edges exist

3. **No API changes**: Same function signature, same return type

---

## Future-Proofing

The fix is designed to automatically support new edge types:

1. **New entity types**: Add prefix check + converter method to `_get_entity_node()`
2. **New edge types**: Automatically handled - add to grammatical mapping if verb conjugation needed

---

## Testing Checklist

- [x] Sprint relationships show contained tasks
- [x] Task relationships show dependencies and containing sprint
- [x] Handoff relationships show TRANSFERS edges
- [x] Decision relationships work
- [x] Unknown entity IDs return empty results (no crash)
- [x] GoT validate passes
- [x] Backward compatibility with existing code

---

## Related Documents

- CLAUDE.md: GoT section with query language documentation
- `.got/entities/`: Entity storage location
- `cortical/got/api.py`: `get_edges_for_task()` implementation

---

*This fix resolves the silent failure of relationships queries for non-task entities and enables the full GoT graph to be queried.*
