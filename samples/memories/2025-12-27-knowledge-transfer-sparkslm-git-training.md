# Knowledge Transfer: SparkSLM Git History Training Design

**Date:** 2025-12-27
**Session:** 7K1pS
**Branch:** `claude/sparkslm-git-training-7K1pS`
**Tags:** `sparkslm`, `ml-training`, `git-history`, `oversampling`, `n-gram`

---

## Executive Summary

This session designed an intelligent approach to training SparkSLM (our statistical n-gram language model) on git history data. The key insight is that **weighted count accumulation** is safer than naive data duplication for oversampling across branches.

---

## Problem Statement

**Goal:** Train SparkSLM on git history to learn project-specific patterns for:
- Commit message prediction
- Code change patterns
- File co-occurrence
- Developer workflow patterns

**Challenge:** Git history is non-uniform:
- Different branches represent different quality levels
- Commits can appear on multiple branches (via merges)
- Agent-generated branches (`claude/*`) need different treatment
- Temporal patterns matter (recent vs. historical)

**Question:** How do we safely oversample per-branch without introducing bias or data leakage?

---

## Solution: Weighted N-Gram Training

### Core Principle

Instead of training the same commit multiple times (which duplicates data and amplifies bias), we **weight the count contributions**:

```python
# Standard n-gram: count(context + word) += 1
# Weighted n-gram: count(context + word) += weight
```

This allows us to give more influence to high-quality commits without data duplication.

### Weight Computation

Final weight = `branch_weight × quality_multipliers × temporal_decay`

#### Branch Tier Weights

| Tier | Branch Pattern | Weight | Rationale |
|------|----------------|--------|-----------|
| 1 | `main`, `master` | 1.0 | Production code, reviewed |
| 2 | `release/*`, `hotfix/*` | 0.9-1.1 | Stable/critical fixes |
| 3 | `feature/*`, `develop` | 0.6 | Human WIP, not yet reviewed |
| 4 | `claude/*` | 0.4 | Agent-generated, needs validation |
| 5 | `experimental/*`, `wip/*` | 0.2 | Exploratory, may be abandoned |

#### Quality Signal Multipliers

| Signal | Multiplier | Detection Method |
|--------|------------|------------------|
| Merged to main | ×1.2 | `git branch --merged main` |
| Has test changes | ×1.1 | Files matching `test_*.py` |
| CI passed | ×1.1 | From `.git-ml/` CI data |
| Was reverted | ×0.0 | `git log --grep="Revert"` |
| Fixes bug | ×1.1 | Message contains "fix:", "bug" |
| Breaking change | ×0.8 | Message contains "BREAKING" |

#### Temporal Decay

```python
def temporal_weight(commit_date, half_life_months=6.0):
    age_months = (now - commit_date).days / 30.0
    return 0.5 ** (age_months / half_life_months)
```

- 0 months old → weight 1.0
- 6 months old → weight 0.5
- 12 months old → weight 0.25

---

## Safety Mechanisms

### 1. SHA-Based Deduplication

Each commit is processed **exactly once**, regardless of how many branches contain it:

```python
seen_commits: Set[str] = set()
for commit in all_commits:
    if commit.sha in seen_commits:
        continue
    seen_commits.add(commit.sha)
    # Process commit...
```

### 2. Best Branch Assignment

When a commit appears on multiple branches, we assign it to the **highest-tier** branch:
- A commit on both `feature/auth` and `main` → assigned to `main` (weight 1.0)

### 3. Reverted Commit Exclusion

Commits that were later reverted get weight=0:
```python
if was_reverted(sha):
    weight = 0.0  # Don't train on undone work
```

### 4. Minimum Weight Floor

Prevent total exclusion with a floor:
```python
weight = max(computed_weight, min_weight=0.05)
```

### 5. Stratified Sampling (Optional)

For large repos, ensure each tier is represented proportionally:
```python
samples_per_tier = {
    'main': 200,      # Full representation
    'feature': 120,   # 60% of main
    'claude': 80,     # 40% of main
}
```

---

## Implementation Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                 GIT HISTORY TRAINING PIPELINE                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  1. EXTRACT                                                          │
│     └── Get all commits across all branches                          │
│     └── Deduplicate by SHA (each commit processed once)              │
│                                                                       │
│  2. CLASSIFY                                                         │
│     └── Determine "best branch" for each commit                      │
│     └── Extract quality signals (merged, tested, reverted)           │
│                                                                       │
│  3. WEIGHT                                                           │
│     └── Branch tier weight (main=1.0, claude/*=0.4)                  │
│     └── Quality multipliers (has_tests: +0.1)                        │
│     └── Temporal decay (recent more relevant)                        │
│                                                                       │
│  4. TOKENIZE                                                         │
│     └── Commit messages → n-grams                                    │
│     └── Diffs → structured tokens (using DiffTokenizer)              │
│                                                                       │
│  5. TRAIN                                                            │
│     └── Weighted n-gram accumulation                                 │
│     └── Separate models for messages vs. diffs (optional)            │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Existing Infrastructure

### SparkSLM Components (in `cortical/spark/`)

| File | Purpose | Relevant to Training |
|------|---------|---------------------|
| `ngram.py` | N-gram model with Laplace smoothing | Core training target |
| `predictor.py` | SparkPredictor facade | High-level API |
| `diff_tokenizer.py` | Tokenize git diffs with special markers | Diff preprocessing |
| `alignment.py` | User context alignment | Can augment with git patterns |

### ML Data Collection (in `.git-ml/`)

| Directory | Contents | Usable for Training |
|-----------|----------|---------------------|
| `commits/` | Full commit data with diffs | Primary data source |
| `commits-lite/` | Lightweight commit metadata | Quick iteration |
| `sessions/` | Development sessions | Workflow patterns |
| `tracked/commits.jsonl` | Git-tracked commit log | Shared training data |

### Key Files to Modify/Create

1. **New:** `cortical/spark/git_trainer.py` - GitHistoryTrainer class
2. **Modify:** `cortical/spark/ngram.py` - Add weighted training support
3. **New:** `scripts/train_spark_from_git.py` - CLI for training

---

## Design Decisions Made

### D1: Weighted Counts vs. Data Duplication

**Decision:** Use weighted count accumulation, not data duplication.

**Rationale:**
- Duplication amplifies bias from noisy branches
- Weighted counts preserve probability semantics
- More memory-efficient (no repeated storage)
- Easier to tune (change weights, not data)

### D2: Branch Tier System

**Decision:** Use 5-tier branch weighting system.

**Rationale:**
- Simple enough to understand and debug
- Captures key quality distinctions
- Allows easy tuning per project
- Agent branches (`claude/*`) explicitly handled

### D3: Temporal Decay with Half-Life

**Decision:** Use exponential decay with 6-month half-life.

**Rationale:**
- Balances recency with institutional knowledge
- 6 months = typical project cycle
- Configurable per-project needs
- Never fully excludes old commits (asymptotic)

### D4: Revert Exclusion

**Decision:** Weight reverted commits at 0.0.

**Rationale:**
- Reverted = mistake acknowledged
- Training on mistakes teaches wrong patterns
- Still in git history for debugging
- Can be used as negative examples separately

---

## Open Questions for Future Work

1. **Cross-branch patterns:** Can we learn what makes commits "merge-worthy"?
2. **Author weighting:** Should experienced authors' commits weight more?
3. **File-type specialization:** Separate models for .py vs .md vs .json?
4. **Online learning:** Update model incrementally as new commits arrive?
5. **Negative examples:** Use reverted commits to teach what NOT to do?

---

## Data Structure Considerations

For space-efficient collection, consider:

1. **Bloom filters** for O(1) SHA deduplication with minimal memory
2. **Trie** for branch prefix matching and weight lookup
3. **Bit vectors** for compact quality signal storage
4. **Streaming updates** instead of loading all commits into memory
5. **Count-min sketch** for approximate n-gram counting at scale

(See upcoming sprint tasks for exploration of these structures)

---

## Testing Strategy

1. **Unit tests:** Weight computation, deduplication logic
2. **Integration tests:** Full pipeline on small repo
3. **Validation:** Compare perplexity on held-out commits
4. **A/B comparison:** Weighted vs. unweighted model quality

---

## Related Documents

- `CLAUDE.md` - ML Data Collection section
- `docs/ml-milestone-thresholds.md` - Training data requirements
- `docs/ml-training-best-practices.md` - General training guidance
- `cortical/spark/*.py` - SparkSLM implementation

---

## Next Steps

1. Create sprint with implementation tasks
2. Explore space-efficient data structures
3. Implement `GitHistoryTrainer` class
4. Add weighted training to `NGramModel`
5. Create training CLI script
6. Validate on this repository

---

*This knowledge transfer captures the design session for SparkSLM git training. The approach prioritizes safety (no data duplication, revert exclusion) while enabling intelligent oversampling (branch tiers, quality signals, temporal decay).*
