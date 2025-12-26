# GoT Namespace Specification

This document is the authoritative reference for entity ID formats and namespace conventions in the Graph of Thought (GoT) system.

## ID Prefix Reference

| Prefix | Entity Type | Format | Example |
|--------|-------------|--------|---------|
| `T-` | Task | `T-YYYYMMDD-HHMMSS-XXXXXXXX` | `T-20251226-141000-a1b2c3d4` |
| `D-` | Decision | `D-YYYYMMDD-HHMMSS-XXXXXXXX` | `D-20251226-141500-e5f6g7h8` |
| `S-` | Sprint | `S-NNN` or `S-sprint-NNN-name` | `S-026`, `S-sprint-017-spark` |
| `EPIC-` | Epic | `EPIC-{name}` | `EPIC-got-db`, `EPIC-woven-mind` |
| `H-` | Handoff | `H-YYYYMMDD-HHMMSS-XXXXXXXX` | `H-20251226-150000-q1r2s3t4` |
| `E-` | Edge | `E-{source}-{target}-{type}` | `E-T-xxx-T-yyy-DEPENDS_ON` |
| `DOC-` | Document | `DOC-{path-slug}` | `DOC-docs-architecture-md` |
| `CML-` | ClaudeMdLayer | `CML-{layer}-{section}` | `CML-core-architecture` |
| `CMV-` | ClaudeMdVersion | `CMV-{layer_id}-vN` | `CMV-CML-core-v1` |
| `TEAM-` | Team | `TEAM-{name}` | `TEAM-engineering-backend` |
| `PP-` | PersonaProfile | `PP-{role}-{name}` | `PP-backend-dev` |

## Timestamp Format

Timestamp-based IDs use the format: `YYYYMMDD-HHMMSS-XXXXXXXX`

- `YYYYMMDD`: Date (e.g., `20251226` for December 26, 2025)
- `HHMMSS`: Time in UTC (e.g., `141000` for 14:10:00)
- `XXXXXXXX`: 8-character hex suffix for uniqueness (e.g., `a1b2c3d4`)

## Reserved Prefixes

These prefixes are reserved and must ONLY be used for their designated entity types:

| Prefix | Reserved For | Notes |
|--------|--------------|-------|
| `E-` | Edge | **Do NOT use for Epic** (use `EPIC-`) |
| `T-` | Task | |
| `D-` | Decision | |
| `S-` | Sprint | |
| `H-` | Handoff | |

## Extended Entity Prefixes

| Prefix | Entity Type | Status |
|--------|-------------|--------|
| `G-` | Goal | Reserved (future) |
| `OP-` | OrchestrationPlan | Reserved (future) |
| `EX-` | Execution | Reserved (future) |
| `MEM-` | Memory | Reserved (future) |

## Sprint ID Format

Sprint IDs have evolved through several formats. Current practice:

### Preferred Format
- **Short form**: `S-NNN` (e.g., `S-026`)
- **Long form**: `S-sprint-NNN-name` (e.g., `S-sprint-017-spark-slm`)

### Legacy Formats (still valid, do not create new)
- `S-NNN-Name` (e.g., `S-017-Spark-SLM`)
- `S-name-hyphenated` (e.g., `S-got-db-cleanup`)

### Recommendations
1. Use short form `S-NNN` for programmatic creation
2. Use long form with descriptive name for human-created sprints
3. Do not mix casing within a single ID

## Epic ID Format

Epic IDs are human-readable identifiers:

### Format
- `EPIC-{descriptive-name}` (all lowercase, hyphen-separated)

### Examples
- `EPIC-got-db` - Graph of Thought database work
- `EPIC-woven-mind` - Woven Mind cognitive architecture
- `EPIC-nlu` - Natural Language Understanding

### Do NOT Use
- `E-{name}` - Reserved for Edge
- `EPIC_{name}` - Use hyphens, not underscores

## Edge ID Format

Edge IDs are auto-generated from source, target, and type:

### Format
- `E-{source_id}-{target_id}-{edge_type}`

### Examples
- `E-T-20251226-141000-a1b2-T-20251226-142000-b3c4-DEPENDS_ON`
- `E-S-026-T-xxx-CONTAINS`

### Valid Edge Types
```
DEPENDS_ON, BLOCKS, CONTAINS, RELATES_TO, REQUIRES,
IMPLEMENTS, SUPERSEDES, DERIVED_FROM, PARENT_OF, CHILD_OF,
PART_OF, REFERENCES, CONTRADICTS, JUSTIFIES, TRANSFERS,
PRODUCES, DOCUMENTED_BY
```

## Validation

### ID Format Validation

Use `cortical.got.types.VALID_ENTITY_TYPES` to validate entity_type fields:

```python
from cortical.got.types import VALID_ENTITY_TYPES

# Valid: 'task', 'decision', 'edge', 'sprint', 'epic', 'handoff',
#        'claudemd_layer', 'claudemd_version', 'persona_profile', 'team', 'document'
if entity_type not in VALID_ENTITY_TYPES:
    raise ValueError(f"Unknown entity type: {entity_type}")
```

### ID-to-Type Mapping

Infer entity type from ID prefix:

```python
def entity_type_from_id(entity_id: str) -> str:
    """Infer entity type from ID prefix."""
    prefix_map = {
        'T-': 'task',
        'D-': 'decision',
        'E-': 'edge',
        'S-': 'sprint',
        'EPIC-': 'epic',
        'H-': 'handoff',
        'DOC-': 'document',
        'CML-': 'claudemd_layer',
        'CMV-': 'claudemd_version',
        'TEAM-': 'team',
        'PP-': 'persona_profile',
    }
    for prefix, entity_type in prefix_map.items():
        if entity_id.startswith(prefix):
            return entity_type
    raise ValueError(f"Unknown ID prefix: {entity_id}")
```

## Canonical ID Generation

Use the canonical ID generation module for creating new IDs:

```python
from cortical.utils.id_generation import (
    generate_task_id,      # T-YYYYMMDD-HHMMSS-XXXX
    generate_decision_id,  # D-YYYYMMDD-HHMMSS-XXXX
    generate_handoff_id,   # H-YYYYMMDD-HHMMSS-XXXX
    generate_plan_id,      # OP-YYYYMMDD-HHMMSS-XXXX
)
```

## See Also

- `cortical/utils/id_generation.py` - Canonical ID generation functions
- `cortical/got/types.py` - Entity type definitions and VALID_ENTITY_TYPES
- `cortical/got/entity_schemas.py` - Schema validation for entities
