# Knowledge Transfer: Director Methodology & Remaining Work

**Session:** Senior Engineering Consultation
**Date:** 2026-01-05
**Branch:** `claude/senior-engineer-consultation-T5aMm`
**Author:** Director Agent (Opus 4.5)

---

## To My Future Self

You're continuing work on the GoT Query Expression System. This document captures not just *what* remains, but *how* to work effectively with sub-agents to complete it.

---

## Part 1: The Wave Methodology

### Why Waves Work

Sub-agents are stateless. They can't see what other agents did. They can't course-correct mid-task. This means:

1. **Research before implementation** — You can't fix what you don't understand
2. **Parallel where possible** — Independent tasks run simultaneously
3. **Verify independently** — The agent that built it shouldn't verify it
4. **Document as you go** — Future you has no memory of current you

### The Three-Wave Pattern

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    THE THREE-WAVE PATTERN                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  WAVE 1: RESEARCH (Explore agents, parallel)                            │
│  ─────────────────────────────────────────                              │
│  • Audit the problem scope                                              │
│  • Find ALL affected files and line numbers                             │
│  • Identify patterns and anti-patterns                                  │
│  • Discover existing infrastructure to leverage                         │
│  • Report: "Here's what we found, here's what needs fixing"             │
│                                                                          │
│  WAVE 2: IMPLEMENT (general-purpose agents, parallel where independent) │
│  ────────────────────────────────────────────────────────────────────   │
│  • Give precise instructions: exact files, exact changes                │
│  • Include validation commands in the task                              │
│  • One focused mission per agent                                        │
│  • Report: "Here's what I changed, here's the validation output"        │
│                                                                          │
│  WAVE 3: VERIFY (Explore agents, parallel)                              │
│  ─────────────────────────────────────────                              │
│  • Different agents than Wave 2 (fresh eyes)                            │
│  • Check each fix was correctly applied                                 │
│  • Run integration tests                                                │
│  • Report: "PASS/FAIL with evidence"                                    │
│                                                                          │
│  DIRECTOR PHASE: Synthesize, Document, Commit                           │
│  ──────────────────────────────────────────                             │
│  • Review wave results                                                  │
│  • Update OUTSTANDING_ISSUES.md                                         │
│  • Commit with clear message                                            │
│  • Push to branch                                                       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### What Sub-Agents Need to Succeed

**Every sub-agent prompt MUST include:**

1. **Clear mission** — One sentence: "You are a [role] doing [task]"
2. **Context** — What problem are we solving? Why?
3. **Specific locations** — Exact file paths and line numbers
4. **Expected outcome** — What does success look like?
5. **Validation command** — How to verify the work
6. **Constraints** — What NOT to do (prevent scope creep)
7. **Report format** — What to return to the director

**Example prompt structure:**
```
You are a [role] [doing what].

**MISSION:** [One clear sentence]

**CONTEXT:**
[Why this matters, what problem we're solving]

**SPECIFIC LOCATIONS:**
- File: `path/to/file.py` line X-Y
- File: `path/to/other.py` line Z

**YOUR TASK:**
1. [Step 1]
2. [Step 2]
3. [Step 3]

**VALIDATION:**
Run: `command to verify`

**DO NOT:**
- [Thing to avoid]
- [Other thing to avoid]

**REPORT FORMAT:**
- [What to include in response]
```

---

## Part 2: Remaining Work

### Priority 1: Complete OI-001 (Status Strings)

**Status:** Partially Resolved
**Remaining:** 5 files with hardcoded status strings

| File | Lines | Entity Type | Complexity |
|------|-------|-------------|------------|
| `cortical/got/recovery.py` | 212 | Task | Simple |
| `cortical/got/cli/backlog.py` | 232, 280 | Task | Simple |
| `cortical/got/cli/orphan.py` | 305 | Task | Simple |
| `cortical/got/query_api.py` | 376-387 | Task | Medium |
| `cortical/got/types.py` | 343-524 | All | Complex |

**Fix Pattern (already proven):**
```python
from cortical.got.entity_schemas import get_valid_statuses
valid_statuses = get_valid_statuses('task')
```

**Wave Plan:**
- Wave 1: Skip (already researched)
- Wave 2: 3 parallel agents (CLI files, recovery.py, query_api.py)
- Wave 3: Verify all changes

**types.py is SPECIAL:** The dataclass `__post_init__` methods use status validation. This requires careful thought—circular import risk if entity_schemas imports types. Consider:
- Option A: Keep hardcoded in types.py (acceptable for dataclass validation)
- Option B: Create minimal constants module imported by both
- Recommend: Option A (dataclasses are foundational, schema is higher-level)

### Priority 2: Complete OI-002 (Edge Type String Literals)

**Status:** Partially Resolved
**Remaining:** 21 hardcoded string comparisons

| File | Count | Complexity |
|------|-------|------------|
| `cortical/got/expression/functions/graph.py` | 8 | Medium |
| `cortical/got/query_api.py` | 4 | Medium |
| `cortical/got/cli/decision.py` | 4 | Simple |
| `cortical/got/api.py` | 1+ | Simple |
| `cortical/got/indexer.py` | 1 | Simple |
| `cortical/got/expression/functions/filters.py` | 1 | Simple |
| `cortical/got/orphan.py` | 1 | Simple |

**Fix Pattern:**
```python
from cortical.got.types import EdgeTypes

# Instead of:
if edge.edge_type == "DEPENDS_ON":

# Use:
if edge.edge_type == EdgeTypes.DEPENDS_ON:
```

**Wave Plan:**
- Wave 1: Skip (already researched)
- Wave 2: 3 parallel agents (graph.py, query_api.py, remaining files)
- Wave 3: Verify imports work, no typos

### Priority 3: OI-003 through OI-006 (Lower Priority)

These are backlog items. Don't start until OI-001 and OI-002 are fully resolved.

---

## Part 3: Communication Style

### With the User

- **Be direct** — State what you're doing and why
- **Show progress** — Use TodoWrite to track tasks visibly
- **Report results** — Tables, not walls of text
- **Admit uncertainty** — "I'm not sure about X" is better than guessing

### With Sub-Agents

- **Be precise** — Vague instructions produce vague results
- **Include context** — They have no memory of previous agents
- **Set boundaries** — "Do NOT make changes" for research agents
- **Request structured output** — "Report in this format: ..."

### In Documentation

- **Update as you go** — OUTSTANDING_ISSUES.md is the source of truth
- **Mark progress** — "Partially Resolved" with details beats "Open"
- **Leave breadcrumbs** — Future you needs to understand past you

---

## Part 4: Quality Gates

### Before Any Wave 2 (Implementation)

```bash
# Verify system is healthy before changes
python3 scripts/got_utils.py validate
```

### After Any Wave 2 (Implementation)

```bash
# Verify system is still healthy
python3 scripts/got_utils.py validate

# Quick sanity check
python3 -c "from cortical.got import EdgeTypes, VALID_EDGE_TYPES; print('OK')"
```

### Before Commit

```bash
# Check what changed
git status
git diff --stat

# Verify GoT health
python3 scripts/got_utils.py validate
```

### After Push

- Update OUTSTANDING_ISSUES.md
- Create KT document if significant work

---

## Part 5: When Things Go Wrong

### Sub-Agent Returns Incomplete Work

1. Don't re-run the same prompt
2. Analyze what was missing from instructions
3. Create a new, more specific prompt
4. Consider breaking into smaller tasks

### Sub-Agent Made Wrong Changes

1. Check git diff to understand what changed
2. Revert if necessary: `git checkout -- path/to/file.py`
3. Create clearer instructions for next attempt
4. Consider doing it yourself if scope is small

### Circular Import or Import Error

1. This is common when adding schema introspection
2. Check import order in affected files
3. Consider lazy imports or moving code
4. The types.py dataclasses are foundational—don't make them depend on higher-level modules

### Tests Fail After Changes

1. Read the error message (obvious but often skipped)
2. Check if test expectations are outdated
3. Verify the change was correct (not just different)
4. Fix forward, don't revert unless truly broken

---

## Part 6: The Mindset

### You Are the Director

Sub-agents are your hands. You are the brain. They execute; you strategize.

- **Don't micromanage** — Give clear instructions, trust execution
- **Do verify** — Trust but verify with Wave 3
- **Stay focused** — One issue at a time, fully resolved
- **Document everything** — Your memory resets; documents persist

### The Three Promises Apply to You

```
1. I WILL NOT BREAK WHAT WORKS
   → Validate before and after every wave

2. I WILL EXPLAIN MY REASONING
   → Commit messages, OUTSTANDING_ISSUES.md, KT documents

3. I WILL LEAVE THE CODE BETTER THAN I FOUND IT
   → Partial resolution is still progress; document remaining work
```

### When to Stop

- All HIGH priority issues resolved → Commit and celebrate
- Context pressure building → Create KT, handoff cleanly
- Blocked on decision → Ask user, don't guess
- Diminishing returns → Document remaining work, move on

---

## Part 7: Quick Reference

### Commands You'll Need

```bash
# GoT validation
python3 scripts/got_utils.py validate

# Sanity check imports
python3 -c "from cortical.got.entity_schemas import get_valid_statuses; print(get_valid_statuses('task'))"
python3 -c "from cortical.got import EdgeTypes; print(EdgeTypes.DEPENDS_ON)"

# Git workflow
git status
git diff --stat
git add -A
git commit -m "message"
git push -u origin branch-name
```

### Files to Know

| File | Purpose |
|------|---------|
| `docs/OUTSTANDING_ISSUES.md` | Source of truth for remaining work |
| `docs/design/got-query-audit-and-design.md` | Design principles and architecture |
| `cortical/got/entity_schemas.py` | Schema definitions, `get_valid_statuses()` |
| `cortical/got/types.py` | `EdgeTypes` class, `VALID_EDGE_TYPES` |
| `cortical/got/validation.py` | Entity validation (uses schema now) |

### The Golden Rule

**Research → Implement → Verify → Document**

Never skip a step. Never assume. Always validate.

---

## Closing Message

Future me: You've got this. The methodology works. The infrastructure is in place. The remaining work is well-documented.

Start with OI-001 remaining files (recovery.py, CLI files, query_api.py). They're straightforward—just apply the pattern that's already proven.

Trust the waves. Trust the tests. Trust the process.

*"Every change runs through tests. No exceptions. No shortcuts."*

---

**Session:** Senior Engineering Consultation
**Commit:** cd841d25
**Branch:** `claude/senior-engineer-consultation-T5aMm`
**Next:** Complete OI-001, then OI-002, then lower priority items
