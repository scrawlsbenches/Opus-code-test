# CLAUDE.md Recovery Redesign

## Problem Statement

CLAUDE.md is a cognitive prosthetic - the single artifact that allows recovery from context compaction (daydreaming). Current state fails this purpose:

1. **`recover()` method not mentioned** - The actual recovery method exists but isn't referenced
2. **`health_check()` not mentioned** - Preventative drift detection is invisible
3. **Recovery Protocol uses fragile approaches** - Points to suboptimal methods
4. **Redundancy** - "Daydreaming" concept repeated multiple times
5. **Open Questions create doubt** - Uncertainty during recovery is harmful

## Design Principles for the Fix

1. **One path, not many** - Single clear workflow for each situation
2. **Reference real methods** - Recovery Protocol must use `recover()` and `health_check()`
3. **Actionable over explanatory** - Tell me what to DO, not just what to KNOW
4. **Trust through accuracy** - Every method mentioned must exist and work

## Changes Required

### CLAUDE.md Changes

1. **Recovery Protocol** - Simplify to:
   - Step 1: `health_check()` - Am I drifting?
   - Step 2: `recover()` - Get full context summary
   - Step 3: Resume work based on recovery output

2. **Remove "Open Questions"** - Creates doubt during recovery

3. **Consolidate daydreaming references** - Explain once in Cognitive Model, reference elsewhere

4. **Update "How To Use Me"** - Already done with session() pattern

### Code Changes

None required - `recover()` and `health_check()` already exist and work correctly.
The issue is documentation not referencing them, not missing implementation.

## Recovery Flow (What CLAUDE.md Should Enable)

```
CONFUSED
    │
    ▼
Read CLAUDE.md
    │
    ▼
health_check() → Am I drifting or lost?
    │
    ▼
recover() → Get intent anchors, pending work, learnings
    │
    ▼
Resume with clear context
```

## What to Keep

- Identity section (who I am)
- Cognitive Model diagram (state awareness)
- Vulnerabilities list (self-awareness)
- Key Files table (orientation)
- Design Principles (the "why")
- How To Use Me (the pattern)
- Meta section (the purpose)

## What to Remove/Change

- Verbose Recovery Protocol steps → Replace with method calls
- Open Questions section → Remove entirely
- Redundant explanations → Consolidate

## Validation Criteria

A recovering Claude should be able to:
1. Read CLAUDE.md
2. Call `health_check()` to assess state
3. Call `recover()` to get full context
4. Know exactly what to do next

No guessing, no ambiguity, no doubt.
