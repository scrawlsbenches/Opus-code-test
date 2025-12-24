# User Feedback Survey Planning

**Status:** Draft
**Created:** 2025-12-24
**Authors:** Human + Claude (pair programming session)

---

## Purpose

Design a feedback and reward signal system that:
1. Collects data that drives verifiable improvements
2. Enables data-driven decisions on what/how/why/when we work
3. Prepares training data for statistical models (next-word prediction and beyond)
4. Strengthens human-agent collaboration through mutual understanding

---

## Current State Assessment

### What We Collect Well

| Data Type | Location | Training Value |
|-----------|----------|----------------|
| Commits with diffs | `.git-ml/commits/` | Input-output pairs for code generation |
| Chat exchanges | `.git-ml/chats/` | Query-response pairs |
| Session-commit links | `.git-ml/sessions/` | Temporal context, causality |
| Tool uses | `.git-ml/actions/` | Action selection training |
| CI results | `.git-ml/commits/` | Binary outcome signal |
| File prediction accuracy | Hubris expert system | Closed-loop reward signal |

### What We're Missing

| Signal Type | Why It Matters | Current State |
|-------------|----------------|---------------|
| **Merge/Revert outcome** | Ultimate success metric | Not tracked |
| **Coverage delta** | Quality signal for test generation | CI captures total, not delta |
| **Iteration count** | Efficiency signal | Not tracked per-task |
| **Human edit after agent edit** | Indicates incomplete/incorrect work | Not tracked |
| **Time-to-completion** | Efficiency signal | Task timestamps exist, not analyzed |
| **Explicit satisfaction** | Preference learning | Chat feedback exists, never aggregated |
| **Approach accuracy** | Did suggested approach work? | Not tracked |

### Feedback System Effectiveness

| Component | Status | Issue |
|-----------|--------|-------|
| Hubris Expert | ✅ Working | Closed loop with credit updates |
| Chat Feedback | ❌ Broken | Collected, never synthesized |
| Task Retrospectives | ❌ Broken | Optional, never prompted |
| Crisis Lessons | ⚠️ Partial | Logged, not surfaced |
| Session Memory | ⚠️ Partial | Draft generated, rarely reviewed |

**Root cause:** Most feedback lacks a closed loop. Data goes in, nothing comes out.

---

## Proposed Reward Signals

### Tier 1: Automatic (No Human Input Required)

| Signal | Source | Implementation |
|--------|--------|----------------|
| CI pass/fail | GitHub Actions | ✅ Already implemented |
| Coverage delta | CI coverage job | Add: capture before/after per commit |
| Merge success | GitHub API | Add: track PR merge within N days |
| Revert detection | Git history | Add: detect if commit reverted within N days |
| Human edit after agent edit | Git blame | Add: detect if same lines edited by human after agent |
| Iteration count | Task state transitions | Add: count in_progress → pending cycles |
| Time-to-completion | Task timestamps | Add: compute from created → completed |

### Tier 2: Low-Friction Human Input

| Signal | Collection Point | Format |
|--------|------------------|--------|
| Task satisfaction | Task completion | 1-5 scale, single click |
| Session productivity | Session end | 1-5 scale + optional one-liner |
| Handoff clarity | Handoff accept | 1-5 scale from receiving agent |
| Approach correctness | After implementation | "Did suggested approach work?" y/n |

### Tier 3: Rich Human Feedback (Conversational)

| Signal | Collection Point | Format |
|--------|------------------|--------|
| Preference learning | Ongoing conversation | Agent asks, human answers |
| Friction points | When detected | "What would have helped?" |
| Process improvements | Sprint retrospective | Open discussion, documented |

---

## Reward Signal Schema

```python
@dataclass
class RewardSignal:
    """Structured reward signal for training."""

    # Identity
    signal_id: str              # Unique identifier
    timestamp: str              # ISO format

    # Context
    commit_hash: Optional[str]  # Associated commit
    task_id: Optional[str]      # Associated task
    session_id: Optional[str]   # Associated session

    # Signal
    signal_type: str            # e.g., "ci_result", "merge_outcome", "satisfaction"
    signal_value: float         # Normalized 0.0-1.0
    raw_value: Any              # Original value before normalization

    # Metadata
    source: str                 # "automatic" | "human" | "agent"
    confidence: float           # How reliable is this signal?

    # For delayed signals
    delay_days: Optional[int]   # How long after the action was this measured?
```

### Normalization Rules

| Signal Type | Raw Value | Normalized |
|-------------|-----------|------------|
| CI result | pass/fail | 1.0 / 0.0 |
| Coverage delta | -100% to +100% | 0.0 to 1.0 (0.5 = no change) |
| Merge outcome | merged/closed/reverted | 1.0 / 0.5 / 0.0 |
| Satisfaction rating | 1-5 | 0.0 / 0.25 / 0.5 / 0.75 / 1.0 |
| Time efficiency | seconds | Inverse normalized against baseline |
| Iteration count | N iterations | 1.0 / N (fewer is better) |

---

## Implementation Plan

### Phase 1: Automatic Reward Signals (Week 1)

**Goal:** Capture signals that require no human input.

1. **Coverage delta tracking**
   - Modify CI to capture coverage before/after
   - Store delta with commit metadata
   - File: `.git-ml/commits/{hash}.json` add `coverage_delta` field

2. **Merge/revert detection**
   - Add script to check commit fate after N days
   - Run weekly via cron or manual trigger
   - File: `scripts/ml_reward_collector.py`

3. **Human-edit-after-agent detection**
   - Analyze git blame for agent commits
   - Flag if same lines edited by human within N commits
   - Indicates incomplete or incorrect agent work

### Phase 2: Low-Friction Human Signals (Week 2)

**Goal:** Add minimal-overhead feedback at natural pause points.

1. **Task completion rating**
   - Auto-prompt on `got task complete`
   - Single keystroke: 1-5 or Enter to skip
   - Store in task metadata

2. **Session end rating**
   - Add to `ml-session-capture-hook.sh`
   - "Rate this session [1-5, Enter=skip]:"
   - Store in session metadata

3. **Aggregate and surface**
   - Weekly synthesis script
   - Surface patterns in knowledge transfer docs

### Phase 3: Preference Learning (Week 3+)

**Goal:** Build mutual understanding through conversation.

1. **Agent question bank**
   - Curated questions about working preferences
   - Asked opportunistically, not intrusively
   - Answers stored in `.git-ml/preferences/`

2. **Pattern detection**
   - Analyze feedback for recurring themes
   - Surface in session start context

---

## Agent Question Bank (Getting to Know You)

Questions the agent may ask to understand human preferences:

### Working Style
- Do you prefer detailed explanations or concise answers?
- When exploring a problem, do you want me to show my reasoning or just conclusions?
- How do you feel about me asking clarifying questions vs. making reasonable assumptions?

### Code Preferences
- What's your tolerance for "clever" code vs. verbose-but-obvious?
- Do you prefer comprehensive error handling or fail-fast with clear errors?
- How much inline documentation do you like?

### Process Preferences
- Do you want me to use the todo list for small tasks or only complex ones?
- How often should I checkpoint/commit during long tasks?
- Do you prefer I ask before making architectural decisions or propose and proceed?

### Collaboration Style
- When I'm uncertain, do you prefer I say so explicitly or present options?
- How do you feel about me pushing back on requests I think are suboptimal?
- Do you want status updates during long operations or just final results?

### Meta
- What's the most frustrating thing an AI assistant has done?
- What would make our collaboration 2x more effective?
- Is there something you wish I would do that I haven't?

---

## Success Metrics

How we'll know this is working:

| Metric | Baseline | Target | Measurement |
|--------|----------|--------|-------------|
| Feedback response rate | ~5% (optional retrospectives) | >50% | Filled vs. skipped |
| Reward signal coverage | ~20% (CI only) | >80% | Commits with 3+ signals |
| Model prediction accuracy | MRR 0.43 | MRR 0.60 | File prediction eval |
| Time-to-completion trend | Unknown | Decreasing | Task duration analysis |
| Revert rate | Unknown | Decreasing | Automatic tracking |
| Human edit-after-agent rate | Unknown | Decreasing | Automatic tracking |

---

## Open Questions

1. **Storage location:** Should reward signals go in `.git-ml/rewards/` or augment existing files?
2. **Aggregation frequency:** Daily? Weekly? Per-sprint?
3. **Privacy:** What preferences/feedback should NOT be stored?
4. **Feedback fatigue:** How do we detect and respond to survey exhaustion?
5. **Delayed signals:** How long to wait for merge/revert before recording outcome?

---

## Next Steps

1. [ ] Review this document together
2. [ ] Prioritize Phase 1 automatic signals
3. [ ] Implement `ml_reward_collector.py`
4. [ ] Add coverage delta to CI
5. [ ] Begin preference learning conversation

---

## Appendix: Existing Infrastructure

Files to modify or extend:

| File | Current Purpose | Proposed Addition |
|------|-----------------|-------------------|
| `scripts/ml_data_collector.py` | Commit/chat collection | Add reward signal storage |
| `.github/workflows/ci.yml` | CI pipeline | Add coverage delta capture |
| `cortical/got/cli/task.py` | Task management | Auto-prompt satisfaction rating |
| `scripts/ml-session-capture-hook.sh` | Session end | Add productivity rating prompt |
| `scripts/got_utils.py` | GoT CLI | Add reward signal commands |

New files to create:

| File | Purpose |
|------|---------|
| `scripts/ml_reward_collector.py` | Collect automatic reward signals |
| `scripts/ml_preference_learner.py` | Store and surface preferences |
| `.git-ml/rewards/` | Reward signal storage |
| `.git-ml/preferences/` | Human preference storage |
