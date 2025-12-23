# Continuation Prompt

Run this first:
```bash
python scripts/claudemd_generation_demo.py --generate-layer4
```

## Session Context

Previous session (claude/investigate-layer4-diff-rCMWl) completed:
- Sprint 17 (SparkSLM) - all 4 tasks done
- Added `scripts/task_diff.py` for sub-agent diff recovery
- Fixed continuation system bugs (file location, knowledge transfer detection, handoff commands)
- Verified dog-fooding workflow works end-to-end
- Created formal handoff for next task

## Current State

- **Branch:** `claude/investigate-layer4-diff-rCMWl`
- **Sprint:** S-018 (Schema Evolution Foundation) - 33% complete, 12 pending tasks
- **Pending Handoff:** `H-20251223-200955-ed21` → Link architecture docs to tasks

## What To Do

1. Run the Layer 4 command above
2. Accept the pending handoff: `python scripts/got_utils.py handoff list`
3. Work on the handed-off task or pick another from Sprint 18
4. Or: Create PR to merge this branch to main

## Priority Tasks (Sprint 18)

| Task ID | Title | Priority | Handoff |
|---------|-------|----------|---------|
| T-20251223-153558-8edaa341 | Link architecture docs to tasks | high | ✅ H-20251223-200955-ed21 |
| T-20251223-153551-5705de6e | Design orphan detection system | high | |
| T-20251223-151749-cb683757 | Define entity schemas | medium | |

## Verification

The continuation system should show:
- Current branch = Saved branch (no alert)
- Sprint S-018 with pending tasks
- Latest knowledge transfer from 2025-12-23
- Trust protocol starting at L0

## Knowledge Transfer

`samples/memories/2025-12-23-knowledge-transfer-layer4-verification.md`
