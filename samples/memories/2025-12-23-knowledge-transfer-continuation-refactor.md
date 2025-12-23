# Knowledge Transfer: Continuation Protocol Refactor

**Date:** 2025-12-23
**Branch:** claude/verify-gitignore-LpJe4
**Sprint:** S-018 (Schema Evolution Foundation)
**Task:** T-20251223-164818-f848d31d

## Summary

Refactored `scripts/claudemd_generation_demo.py` based on sub-agent survey consensus. The script now provides **facts and investigation commands** instead of making determinations.

## Key Changes

### Philosophy Shift
- **Before:** Script made decisions like "needs_merge", "merged", "same branch"
- **After:** Script reports facts, agent investigates and decides

### New Layer 4 Structure
1. **Observable Facts** - From GoT files only (branch, sprint, tasks, handoffs)
2. **Verify State** - Commands for agent to run (git status, git log, etc.)
3. **Questions to Resolve** - Ask human before acting
4. **Trust Protocol** - L0-L3 reference table

### Functions Changed
- Removed: `analyze_branch_continuity()` (made determinations)
- Removed: `check_branch_merged()` (heuristic)
- Added: `get_branch_state()` (raw facts only)
- Updated: `generate_layer4_from_got()` (facts + commands + questions)
- Updated: `generate_continuation_context()` (no prescriptive guidance)

### CLI Commands
```bash
# Generate Layer 4 from GoT
python scripts/claudemd_generation_demo.py --generate-layer4

# Full continuation context
python scripts/claudemd_generation_demo.py --continuation

# Save branch for next session
python scripts/claudemd_generation_demo.py --save-branch

# Check branch state (facts only)
python scripts/claudemd_generation_demo.py --check-continuity
```

## Sub-Agent Survey Results

Four agents surveyed on "How do you want information?"

**Consensus:**
1. Facts, not conclusions (4/4)
2. I run the commands myself (4/4)
3. Questions to ask human, not mandates (4/4)
4. Confidence scores + fallbacks when uncertain (4/4)

**Key Quote:** "Give me the tools and context, not the mandate."

## Commits

1. `44c29e5b` - feat(continuation): Add GoT-based dynamic Layer 4 generation
2. `b8663438` - feat(continuation): Add cross-session branch continuity tracking
3. `adcbb3c3` - refactor(continuation): Facts-only Layer 4, no determinations

## What To Test

On a new branch, the system should:
1. Show current branch and saved branch differ
2. Provide investigation commands (not conclusions)
3. Ask questions about what to do (not mandate actions)
4. Let agent verify state before acting

## Files Modified

- `scripts/claudemd_generation_demo.py` - Main refactored script
- `.got/entities/S-018.json` - Sprint metadata with saved branch

## Next Steps

1. Test on new branch to verify branch difference detection
2. Verify investigation commands work correctly
3. Confirm questions are contextually appropriate
4. Consider integrating into actual CLAUDE.md generation pipeline
