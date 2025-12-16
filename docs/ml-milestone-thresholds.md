# ML Milestone Thresholds

## Overview

ML milestones define the minimum amount of data required for various machine learning training scenarios. These thresholds ensure that models have sufficient examples to learn meaningful patterns without overfitting or producing unreliable predictions.

**Purpose:**
- Prevent premature training with insufficient data
- Set realistic expectations for data collection progress
- Guide when each ML capability becomes viable
- Maintain model quality standards across the project

**Location:** `scripts/ml_collector/config.py` (lines 33-37)

```python
MILESTONES = {
    "file_prediction": {"commits": 500, "sessions": 100, "chats": 200},
    "commit_messages": {"commits": 2000, "sessions": 500, "chats": 1000},
    "code_suggestions": {"commits": 5000, "sessions": 2000, "chats": 5000},
}
```

## Milestone Definitions

### File Prediction (First Milestone)

The earliest viable ML capability, requiring modest data volumes.

| Data Type | Threshold | Rationale |
|-----------|-----------|-----------|
| Commits | 500 | Generates 2,500-5,000 file pairs for co-occurrence patterns |
| Sessions | 100 | Ensures diverse coding contexts and workflows |
| Chats | 200 | Provides basic query-response patterns |

**Training Goal:** Predict which files are likely to change together based on:
- File co-occurrence patterns from commit history
- Commit type classification (feat:, fix:, docs:, etc.)
- Keyword-to-file associations from commit messages
- Module relationship patterns

**Use Cases:**
- Pre-commit hook file suggestions
- IDE integration for file recommendations
- Code review automation (missing file detection)

**Model Type:** Lightweight pattern matching with TF-IDF-style scoring (no neural networks required)

### Commit Messages (Medium Milestone)

Mid-level capability requiring more diverse examples.

| Data Type | Threshold | Rationale |
|-----------|-----------|-----------|
| Commits | 2,000 | Sufficient for commit type classification with 200+ examples per class |
| Sessions | 500 | Captures longer development cycles and patterns |
| Chats | 1,000 | Rich conversational patterns for message generation |

**Training Goal:** Generate or suggest commit messages based on:
- Code diff analysis
- Historical commit message patterns
- File change patterns
- Task/issue linkage patterns

**Use Cases:**
- Automated commit message suggestions
- Commit message quality validation
- Conventional commit format enforcement
- Semantic release automation

**Model Considerations:**
- Typical commit type distribution: 60% feat, 20% fix, 10% docs, 10% other
- Need 200+ examples per class for reliable classification
- Requires diverse examples to avoid bias toward common patterns

### Code Suggestions (Full Training)

Advanced capability requiring substantial data for robust patterns.

| Data Type | Threshold | Rationale |
|-----------|-----------|-----------|
| Commits | 5,000 | ~50,000 file pairs for robust co-occurrence and rare pattern learning |
| Sessions | 2,000 | Comprehensive coverage of development scenarios |
| Chats | 5,000 | Rich conversational corpus for context understanding |

**Training Goal:** Full code-aware assistance including:
- Code completion and generation
- Refactoring suggestions
- Bug pattern detection
- Architecture recommendations
- Test generation

**Use Cases:**
- Fine-tuning code generation models
- Training custom code assistants
- Building project-specific linters
- Automated code review systems

**Model Considerations:**
- Sufficient data for rare patterns and edge cases
- Enables transfer learning or fine-tuning approaches
- Supports multi-task learning scenarios
- Adequate for detecting project-specific idioms

## Threshold Derivation

### Statistical Foundations

The thresholds are based on statistical significance requirements and machine learning best practices:

#### File Prediction (500 commits)

**Co-occurrence Matrix Size:**
```
Average commit: 5-10 files changed
500 commits × 7.5 files/commit = 3,750 total file changes
Unique file pairs: ~2,500-5,000 (depends on project structure)
```

**Why 500 is sufficient:**
- TF-IDF requires term frequency across documents (commits)
- With 500 commits, common co-occurrence patterns emerge reliably
- Rare patterns (occurring in <5 commits) are filtered as noise
- No neural network training required (pattern matching only)

**Session/Chat Ratios:**
- Sessions: 100 ≈ 5 commits/session (realistic development pace)
- Chats: 200 ≈ 2.5 chats/session (typical user interaction rate)

#### Commit Messages (2,000 commits)

**Classification Requirements:**
```
Typical distribution:
- feat:     60% → 1,200 examples
- fix:      20% →   400 examples
- docs:     10% →   200 examples
- other:    10% →   200 examples
```

**Why 2,000 is necessary:**
- Rule of thumb: 100-200 examples per class for reliable classification
- Minority classes (docs, test, chore) need 200+ examples
- Message generation requires understanding patterns across types
- Conventional commit format has ~10 common types

**Session/Chat Ratios:**
- Sessions: 500 ≈ 4 commits/session
- Chats: 1,000 ≈ 2 chats/session

#### Code Suggestions (5,000 commits)

**Comprehensive Pattern Coverage:**
```
Unique patterns needed:
- File pairs: ~50,000 (covers rare co-occurrences)
- Code idioms: ~1,000+ (project-specific patterns)
- Rare patterns: 5-50 occurrences each (needs 5,000+ commits)
```

**Why 5,000 enables advanced features:**
- Rare patterns need 5-50 examples to be reliable
- With 500 commits, patterns occurring <1% are noise
- With 5,000 commits, patterns occurring <1% (50 examples) are learnable
- Sufficient for fine-tuning pre-trained models
- Enables transfer learning with project-specific data

**Session/Chat Ratios:**
- Sessions: 2,000 ≈ 2.5 commits/session
- Chats: 5,000 ≈ 2.5 chats/session

### Development Pace Estimates

Based on typical software development velocity:

| Scenario | Commits/Week | Time to First Milestone | Time to Full Training |
|----------|--------------|-------------------------|----------------------|
| Individual | 5-10 | 10-20 weeks (2.5-5 months) | 100-200 weeks (2-4 years) |
| Small Team (3-5) | 20-40 | 2-5 weeks | 20-50 weeks (5-12 months) |
| Active Team (10+) | 50-100 | 1-2 weeks | 10-20 weeks (2.5-5 months) |

**Assumptions:**
- Individual: 1 commit/day × 5 days/week = 5 commits/week
- Small team: 4 developers × 5 commits/week = 20 commits/week
- Active team: 15 developers × 5 commits/week = 75 commits/week

## Usage in the Codebase

### Progress Tracking

Milestones are checked automatically by `scripts/ml_collector/stats.py`:

```python
from scripts.ml_collector.stats import estimate_progress, print_stats

# Get progress toward milestones
progress = estimate_progress()
# Returns:
# {
#   "file_prediction": {
#     "commits": {"current": 403, "required": 500, "percent": 80},
#     "sessions": {"current": 25, "required": 100, "percent": 25},
#     "chats": {"current": 150, "required": 200, "percent": 75},
#     "overall": 25  # Minimum of all percentages
#   },
#   ...
# }

# Print human-readable stats
print_stats()
```

**Output format:**
```
🎯 Training Milestones:

   File Prediction: [█████░░░░░░░░░░░░░░░] 25%
      commits: 403/500
      sessions: 25/100
      chats: 150/200

   Commit Messages: [██░░░░░░░░░░░░░░░░░░] 10%
      commits: 403/2000
      sessions: 25/500
      chats: 150/1000

   Code Suggestions: [█░░░░░░░░░░░░░░░░░░░] 5%
      commits: 403/5000
      sessions: 25/2000
      chats: 150/5000
```

### Command-Line Interface

```bash
# Check milestone progress
python scripts/ml_data_collector.py stats

# Estimate when milestones will be reached
python scripts/ml_data_collector.py estimate

# Train file prediction model (only if file_prediction milestone met)
python scripts/ml_file_prediction.py train
```

### Programmatic Access

```python
from scripts.ml_collector.config import MILESTONES

# Check if enough data for file prediction
if commits_count >= MILESTONES["file_prediction"]["commits"]:
    print("Ready to train file prediction model!")

# Calculate overall progress
def get_milestone_progress(milestone_name, current_counts):
    requirements = MILESTONES[milestone_name]
    percentages = []
    for data_type, required in requirements.items():
        current = current_counts.get(data_type, 0)
        percentages.append(min(100, int(100 * current / required)))
    return min(percentages)  # Overall = minimum progress
```

### Automatic Warnings

The system warns when training with insufficient data:

**In `scripts/ml_file_prediction.py`:**
```python
WARNING_MIN_TRAINING_COMMITS = 50  # Absolute minimum (10% of milestone)

def train_model(commits):
    if len(commits) < WARNING_MIN_TRAINING_COMMITS:
        print(f"⚠️  WARNING: Only {len(commits)} commits. "
              f"Recommend {MILESTONES['file_prediction']['commits']}+ for reliability.")
    # Continue training anyway (useful for testing)
```

## Adjusting Thresholds

### When to Modify

**Increase thresholds if:**
- Model predictions are unreliable or noisy
- Project has highly diverse file patterns
- Rare co-occurrences are important (e.g., security files)
- Training data quality is low (auto-generated commits, etc.)

**Decrease thresholds if:**
- Project has predictable patterns (monorepo with clear boundaries)
- Early feedback is more valuable than accuracy
- Data collection is slow (small team, infrequent commits)
- Testing/experimentation is the goal

### How to Modify

**Step 1: Update the constant**

Edit `scripts/ml_collector/config.py`:

```python
MILESTONES = {
    "file_prediction": {"commits": 300, "sessions": 75, "chats": 150},  # Lowered for faster feedback
    "commit_messages": {"commits": 2000, "sessions": 500, "chats": 1000},  # Unchanged
    "code_suggestions": {"commits": 5000, "sessions": 2000, "chats": 5000},  # Unchanged
}
```

**Step 2: Document the change**

Update this file with:
- Why the threshold was changed
- Expected impact on model quality
- Date and context of the change

**Step 3: Test the model**

```bash
# Verify the model still trains correctly
python scripts/ml_file_prediction.py train

# Evaluate performance with new threshold
python scripts/ml_file_prediction.py evaluate --split 0.2

# Check predictions
python scripts/ml_file_prediction.py predict "Add authentication feature"
```

**Step 4: Monitor quality**

Track model performance metrics:
- MRR (Mean Reciprocal Rank): Average position of first correct prediction
- Recall@10: Percentage of actual files in top 10 predictions
- Precision@1: Percentage of top predictions that are correct

### Relationship Between Data Types

The thresholds maintain ratios based on typical development patterns:

| Ratio | Typical Value | Meaning |
|-------|---------------|---------|
| Commits per Session | 2-5 | Development sessions involve multiple commits |
| Chats per Session | 2-3 | Users ask multiple questions per session |
| Chats per Commit | 0.4-2.5 | Not all commits involve AI assistance |

**Balancing the ratios:**

If you increase commit threshold, consider increasing session/chat thresholds proportionally:

```python
# Example: 2x increase across the board
"file_prediction": {
    "commits": 1000,  # Was 500 (2x)
    "sessions": 200,  # Was 100 (2x)
    "chats": 400,     # Was 200 (2x)
}
```

### Project-Specific Adjustments

**Monorepo with clear boundaries:**
```python
# Files rarely co-occur across boundaries
# → Increase threshold to capture cross-boundary patterns
"file_prediction": {"commits": 800, "sessions": 150, "chats": 300}
```

**Microservices with shared patterns:**
```python
# Similar patterns across services
# → Decrease threshold (patterns emerge faster)
"file_prediction": {"commits": 300, "sessions": 75, "chats": 150}
```

**High-frequency commits (CI/automation):**
```python
# Many small commits (linting, formatting, etc.)
# → Increase threshold to get meaningful changes
"commit_messages": {"commits": 3000, "sessions": 500, "chats": 1000}
```

## Related Documentation

- **ML Data Collection Overview:** See `CLAUDE.md` section "ML Data Collection"
- **File Prediction Model:** `scripts/ml_file_prediction.py` header comments
- **Pre-Commit Suggestions:** `docs/ml-precommit-suggestions.md`
- **Collection Statistics:** Run `python scripts/ml_data_collector.py stats`
- **Training Guide:** `docs/ml-data-collection-knowledge-transfer.md`

## Change Log

| Date | Change | Rationale |
|------|--------|-----------|
| 2025-12-13 | Initial thresholds set | Based on ML best practices and project velocity estimates |

---

**Next Steps:**

1. **Check your progress:** `python scripts/ml_data_collector.py stats`
2. **Estimate completion:** `python scripts/ml_data_collector.py estimate`
3. **Start training:** When file_prediction milestone is reached, run `python scripts/ml_file_prediction.py train`
