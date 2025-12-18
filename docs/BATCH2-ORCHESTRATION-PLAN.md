# Batch 2 Orchestration Plan

**Created:** 2025-12-17
**Status:** Active
**Prerequisite:** Batch 1 Complete (18 tasks, 4591 tests passing)

## Overview

Batch 2 focuses on **MoE Foundation** and **Epic 2 Completion** - two non-overlapping domains that can execute in parallel.

## Agent Assignments

### Agent ε: MoE Foundation
**Priority:** HIGH
**Branch:** `claude/batch2-epsilon-{session_id}`

**File Claims (Exclusive):**
- `scripts/hubris/**/*.py`
- `cortical/ml_storage.py` (if needed)

**Tasks:**
| ID | Description | Priority |
|----|-------------|----------|
| T-20251217-134359-efba-001 | Design and implement MicroExpert base class with serialization | HIGH |
| T-20251217-134403-efba-002 | Implement ExpertRouter with intent classification | HIGH |
| T-20251217-134407-efba-003 | Create VotingAggregator for expert prediction fusion | MEDIUM |
| T-20251217-134422-efba-004 | Migrate FilePredictionModel to MicroExpert format | MEDIUM |

**Success Criteria:**
- [ ] MicroExpert base class with save/load methods
- [ ] ExpertRouter classifies intents (file, test, error, doc)
- [ ] VotingAggregator merges expert predictions
- [ ] Existing ml_file_prediction.py migrated to new format
- [ ] All existing tests pass
- [ ] New code has unit tests

**Reference Docs:**
- `docs/moe-thousand-brains-architecture.md`
- `docs/attention-marketplace-intelligence-exchange.md`

---

### Agent ζ: Epic 2 Completion
**Priority:** MEDIUM
**Branch:** `claude/batch2-zeta-{session_id}`

**File Claims (Exclusive):**
- `scripts/ml-session-*.sh` (read-only for hooks, can add new)
- `scripts/session_memory_generator.py` (NEW)
- `samples/memories/[DRAFT]-*.md` (auto-generated)

**Tasks:**
| ID | Description | Priority |
|----|-------------|----------|
| Sprint-2.3 | SessionEnd auto-memory generation | HIGH |
| Sprint-2.4 | Post-commit task linking (regex T-XXXXX) | MEDIUM |

**Success Criteria:**
- [ ] Session end generates draft memory from commits
- [ ] Commits with T-XXXXX update task status
- [ ] Draft memories saved to samples/memories/[DRAFT]-*.md
- [ ] All existing tests pass

---

## File Claim Matrix

```
                           ε(MoE)    ζ(Epic2)
scripts/hubris/**/*.py       ██         ░░
scripts/ml_file_prediction.py██         ░░
scripts/session_memory_*.py  ░░         ██
scripts/ml-session-*.sh      ░░         ██ (extend)
samples/memories/            ░░         ██
tasks/*.json                 ░░         ██ (update)

██ = Exclusive (can modify)
░░ = Read-only
```

---

## Coordination Rules

### Merge Order
1. Agent ε (MoE) - merged first (foundational)
2. Agent ζ (Epic 2) - merged second (uses established patterns)

### Completion Report Format
```json
{
  "agent": "epsilon|zeta",
  "status": "complete|partial|blocked",
  "files_created": ["list"],
  "files_modified": ["list"],
  "tests_added": 0,
  "tests_passed": true,
  "blockers": []
}
```

---

## Related Documents

- `docs/BATCH1-ORCHESTRATION-PLAN.md` - Previous batch
- `docs/CONTINUOUS-CONSCIOUSNESS-ROADMAP.md` - Overall roadmap
- `docs/moe-thousand-brains-architecture.md` - MoE design
