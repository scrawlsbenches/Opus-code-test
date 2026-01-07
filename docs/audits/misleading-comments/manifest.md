# Audit: Misleading Comments

*Audit ID: misleading-comments-2026-01-07*
*Created: 2026-01-07*
*Status: COMPLETE*

---

## Goal

Find and assess comments that could mislead developers or AI agents into believing things that aren't true, such as:
- Features that are "planned" but will never happen
- References to documents that don't exist
- Stale TODOs that are done or abandoned
- Aspirational statements presented as commitments

## Trigger

Agent was misled by this comment in `cortical/cdg/storage.py:342`:
```python
# FUTURE: When CDG index is implemented per the distributed graph
# specification (docs/architecture/DISTRIBUTED_GRAPH_SPECIFICATION.md),
```

The spec exists but there's no evidence of implementation. Agent presented this as a "planned fix" when it's just speculation.

---

## Scope

### In Scope
- `cortical/` directory (all Python files)
- Comment patterns: `FUTURE:`, `TODO:`, `FIXME:`, `PLANNED:`, `HACK:`, `XXX:`, `TEMPORARY:`, `WORKAROUND:`
- References to documents (check if they exist)
- Comments containing "will be", "should be", "planned to"

### Out of Scope
- `tests/` directory (test comments are less critical)
- `docs/` directory (documentation is separate concern)
- `.git/`, `__pycache__/`, virtual environments
- Third-party code

### Boundaries
- Max 50 findings per task
- If more, truncate and note for follow-up task

---

## Success Criteria

1. All comments matching patterns in scope are catalogued
2. Each finding has: file, line, content, assessment, notes
3. Referenced documents are verified (exist/don't exist)
4. Findings categorized: accurate | stale | misleading | unknown
5. Human has reviewed and made decisions

---

## Tasks

| Task ID | Directory | Pattern | Status | Claimed By | Result |
|---------|-----------|---------|--------|------------|--------|
| 20260107-120000-cdg-comments | `cortical/cdg/` | All patterns | complete | Agent (v2 re-run) | result-20260107-120000-cdg-comments-v2.md |
| 20260107-120100-got-comments | `cortical/got/` | All patterns | complete | Agent (v2 template) | result-20260107-120100-got-comments.md |
| 20260107-120200-core-comments | `cortical/core/` | All patterns | complete | Agent (v2) | result-20260107-120200-core-comments.md |
| 20260107-120300-common-comments | `cortical/common/` | All patterns | complete | Agent (v2) | result-20260107-120300-common-comments.md |
| 20260107-120400-remaining-comments | `cortical/` (remaining) | All patterns | complete | Agent (v2) | result-20260107-120400-remaining-comments.md |

---

## Progress Log

| Timestamp | Event | Agent/Human | Notes |
|-----------|-------|-------------|-------|
| 2026-01-07 | Audit created | Agent | Triggered by misleading FUTURE: comment |
| | | | |

---

## Decisions

See `decisions.md` for human decisions on findings.

---

## Conflicts

None yet.

---

## Final Status

*Audit completed 2026-01-07*

| Metric | Count |
|--------|-------|
| **Total findings** | 29 |
| **Accurate** | 16 (55%) |
| **Stale** | 0 (0%) |
| **Misleading** | 10 (34%) |
| **Unknown** | 2 (7%) |
| **Unreviewed** | 1 (3%) |

### Breakdown by Directory

| Directory | Findings | Accurate | Misleading | Unknown |
|-----------|----------|----------|------------|---------|
| cortical/cdg/ | 9 | 5 | 3 | 0 |
| cortical/got/ | 5 | 2 | 3 | 0 |
| cortical/core/ | 3 | 2 | 1 | 0 |
| cortical/common/ | 1 | 0 | 1 | 0 |
| cortical/ (remaining) | 11 | 7 | 2 | 2 |

All tasks run with v2 template (cdg/ re-run 2026-01-07).

### Next Steps

- [ ] Human review of 10 misleading comments
- [ ] Decide: fix, defer, or accept each
- [ ] Human review of 2 unknown comments (need context)
