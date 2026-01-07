# Code Audit Framework

*Created: 2026-01-07*
*Purpose: Systematic discovery and resolution of code quality issues across context losses*

---

## Why This Exists

AI agents (and humans) can be misled by:
- Stale comments that describe things that never happened
- `FUTURE:` / `TODO:` / `PLANNED:` comments that became permanent
- References to documents that don't exist
- Comments written by confused previous agents
- Aspirational statements presented as commitments

**The problem compounds:** When an agent reads a misleading comment, it may repeat that misinformation confidently, creating more confusion.

**Context loss makes it worse:** If an agent's context is compacted mid-analysis, all progress is lost unless state is persisted to files.

---

## Directory Structure

```
docs/audits/
├── README.md                      # You are here
├── misleading-comments/           # Specific audit type
│   ├── inbox/                     # Tasks waiting for agents
│   │   └── task-{id}.md          # One task per file
│   ├── outbox/                    # Completed results
│   │   └── result-{id}.md        # One result per task
│   ├── parking-lot/               # Out-of-scope findings
│   │   └── finding-{id}.md       # Issues found but not in scope
│   ├── manifest.md                # State tracking (SINGLE SOURCE OF TRUTH)
│   └── decisions.md               # Human decisions on findings
└── {other-audit-type}/            # Same structure for other audits
```

---

## The Audit Process

### Phase 1: Setup (Human or Lead Agent)

1. Create audit directory with structure above
2. Write `manifest.md` with:
   - Audit goal
   - Scope boundaries
   - Success criteria
   - Patterns to search for
3. Generate task files in `inbox/`

### Phase 2: Discovery (Sub-Agents)

Each agent:
1. Claims ONE task from `inbox/` (rename to `inbox/task-{id}.claimed.md`)
2. Executes the bounded task
3. Writes results to `outbox/result-{id}.md`
4. Updates `manifest.md` to mark task complete
5. If finds out-of-scope issues, writes to `parking-lot/`

### Phase 3: Verification (Optional Second Wave)

Different agents review `outbox/` results:
1. Spot-check claims against actual code
2. Flag disagreements in result files
3. Escalate conflicts to `decisions.md`

### Phase 4: Human Review

Human reviews:
1. `manifest.md` for overall status
2. `outbox/` for findings
3. Makes decisions in `decisions.md`

### Phase 5: Execution

Agents execute approved fixes:
1. Only fix what's approved in `decisions.md`
2. Commit with reference to audit ID
3. Update `manifest.md` with completion status

---

## Critical Questions Every Agent Must Ask

### When Reading Code Comments

| Question | Why It Matters |
|----------|---------------|
| **Is this comment dated?** | Undated comments may be years old |
| **Does `git blame` show when this was written?** | Context on age and author |
| **If it references a document, does that document exist?** | Phantom references are common |
| **Is this aspirational or committed?** | "FUTURE:" often means "never" |
| **What happens if someone believes this?** | Misleading comments cause real bugs |
| **Who wrote this - human or AI agent?** | AI agents can embed confident mistakes |

### When Writing Findings

| Question | Why It Matters |
|----------|---------------|
| **Am I stating a FACT or an INFERENCE?** | Label clearly |
| **Can another agent verify this independently?** | Include file:line references |
| **What's the severity if this is wrong?** | Prioritize high-impact issues |
| **Is this in scope for my task?** | Out-of-scope goes to parking-lot |

### When Context Might Be Lost

| Question | Why It Matters |
|----------|---------------|
| **Is my current state saved to a file?** | Context can compact anytime |
| **Can a new agent continue from my files?** | Handoff must be seamless |
| **Is the manifest updated?** | Single source of truth |

---

## Guardrails

### DO

- Keep tasks small and atomic (one directory, one pattern)
- Write structured output (tables, not prose)
- Include file:line references for all findings
- Update manifest IMMEDIATELY when claiming/completing tasks
- Put out-of-scope findings in parking-lot (don't ignore them)
- Date-stamp all findings
- Separate observations from conclusions

### DO NOT

- Trust `FUTURE:` / `TODO:` / `PLANNED:` comments without verification
- Make bulk changes without human approval
- Overwrite other agents' results (append or create new)
- Assume referenced documents exist (check!)
- Hold state only in context (persist to files)
- Exceed task scope (use parking-lot for extras)
- Claim multiple tasks at once

---

## Failsafes

### Context Loss Recovery

If an agent's context is compacted:
1. New agent reads `manifest.md` to understand state
2. Claimed but incomplete tasks are visible (`.claimed.md` suffix)
3. New agent can either:
   - Complete the claimed task (if clear what remains)
   - Unclaim it (rename back, add note about partial progress)

### Conflicting Results

If two agents disagree:
1. Both results remain in `outbox/`
2. Conflict noted in `manifest.md`
3. Human resolves in `decisions.md`

### Runaway Scope

If audit finds overwhelming number of issues:
1. Stop after N findings per task (configurable in task file)
2. Note "truncated" in results
3. Human decides whether to continue or prioritize

### Agent Makes Mistake

Results are never deleted. If a result is wrong:
1. Add correction as new file: `outbox/result-{id}-correction.md`
2. Reference the original
3. Explain what was wrong and why

---

## Task File Format

```markdown
# Task: {task-id}

## Scope
- **Directory:** `cortical/cdg/`
- **Pattern:** `FUTURE:|TODO:|PLANNED:`
- **Max findings:** 50 (truncate if more)

## Instructions
1. Search for pattern in directory
2. For each match, record: file, line, content, assessment
3. Check if any referenced documents exist
4. Write results to `outbox/result-{task-id}.md`

## Success Criteria
- All matches in scope examined
- Each finding has file:line reference
- Each finding has assessment (accurate/stale/misleading/unknown)
- Manifest updated

## Claimed By
- Agent: (fill when claiming)
- Timestamp: (fill when claiming)
- Status: pending | in-progress | complete | abandoned
```

---

## Result File Format

```markdown
# Result: {task-id}

## Summary
- **Files scanned:** N
- **Matches found:** N
- **Findings:** N accurate, N stale, N misleading, N unknown
- **Truncated:** yes/no

## Findings

| File | Line | Content | Assessment | Notes |
|------|------|---------|------------|-------|
| storage.py | 342 | `FUTURE: When CDG index...` | misleading | References spec that exists but no implementation |
| ... | ... | ... | ... | ... |

## Out-of-Scope (moved to parking-lot)
- (list any findings that were out of scope)

## Completed By
- Agent: (fill when completing)
- Timestamp: (fill when completing)
```

---

## Multi-Round Communication

For large issues that can't be fixed atomically:

### Round 1: Discovery
- Find all instances of the problem
- Output: List of locations

### Round 2: Analysis
- Understand each instance
- Output: Categorized findings with recommended actions

### Round 3: Planning
- Group related fixes
- Identify dependencies between fixes
- Output: Execution plan with ordering

### Round 4: Human Approval
- Human reviews plan
- Approves/modifies/rejects
- Output: Approved plan in `decisions.md`

### Round 5: Execution
- Execute approved fixes in order
- Each fix is a separate commit
- Output: Commit references in manifest

### Round 6: Verification
- Verify fixes didn't break anything
- Run tests
- Output: Verification report

---

## Known Failure Modes

| Failure Mode | Symptom | Prevention | Recovery |
|--------------|---------|------------|----------|
| Context loss mid-task | Claimed task never completes | Small atomic tasks | Unclaim and restart |
| Agent trusts bad comment | Propagates misinformation | Verification questions | Second-wave review |
| Scope creep | Task takes forever | Strict scope + parking-lot | Truncate and note |
| Conflicting conclusions | Two results disagree | Keep both, flag conflict | Human decides |
| Overwhelming findings | Hundreds of issues | Max findings per task | Prioritize, batch |
| Stale manifest | Manifest doesn't match reality | Update immediately | Reconcile from files |

---

## Starting an Audit

1. **Define the audit** - What are we looking for? Why?
2. **Set boundaries** - Which directories? Which patterns? What's out of scope?
3. **Create manifest** - Single source of truth
4. **Generate tasks** - One per directory or logical unit
5. **Run discovery** - Agents claim and complete tasks
6. **Review findings** - Human reviews outbox
7. **Decide actions** - Human writes decisions.md
8. **Execute fixes** - Agents implement approved changes
9. **Verify** - Run tests, check nothing broke
10. **Close audit** - Update manifest with final status

---

## Questions This Framework Doesn't Answer Yet

1. **How do we prevent the same misleading comment from being re-introduced?**
   - CI lint rule? Pre-commit hook?

2. **How do we handle disagreements between human and agent?**
   - Human wins, but should we log the disagreement?

3. **What's the retention policy for old audits?**
   - Keep forever? Archive after N days?

4. **How do we track that fixes actually fixed the problem?**
   - Re-run audit? Spot-check?

5. **Should parking-lot items become new audits automatically?**
   - Or require human decision?

---

## Changelog

| Date | Change | Author |
|------|--------|--------|
| 2026-01-07 | Initial creation | Agent (claude/code-review-fixes-J4A3H) |
