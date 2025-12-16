# ML Data Collection System - Knowledge Transfer

**Date:** 2025-12-16
**Session:** Review of model creation files and chunked storage implementation

---

## Executive Summary

This project implements an automated ML data collection pipeline to train a **project-specific micro-model**. The system captures development context (commits, chat sessions, tool usage) that can be exported for fine-tuning a language model specialized for this codebase.

Key achievements in this session:
1. Reviewed and documented the model creation system
2. Fixed infinite commit loop in ML hooks
3. Implemented chunked storage for git-friendly large file storage

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA COLLECTION LAYER                        │
├──────────────────┬──────────────────┬──────────────────────────────┤
│   SessionStart   │   Post-Commit    │         Stop Hook            │
│   Hook           │   Hook           │                              │
│   ┌──────────┐   │   ┌──────────┐   │   ┌────────────────────┐    │
│   │ Start    │   │   │ Capture  │   │   │ Process transcript │    │
│   │ session  │   │   │ commit   │   │   │ Extract chats      │    │
│   │ Install  │   │   │ metadata │   │   │ Link to commits    │    │
│   │ hooks    │   │   │ + diffs  │   │   └────────────────────┘    │
│   └──────────┘   │   └──────────┘   │                              │
└──────────────────┴──────────────────┴──────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         STORAGE LAYER                               │
├─────────────────────────────────────────────────────────────────────┤
│  .git-ml/                                                           │
│  ├── tracked/              # GIT-TRACKED (small, shareable)         │
│  │   ├── commits.jsonl     # Lightweight commit metadata            │
│  │   ├── sessions.jsonl    # Session summaries                      │
│  │   ├── orchestration.jsonl                                        │
│  │   └── chunked/          # NEW: Compressed chat/commit storage    │
│  │       └── chats-*.jsonl # Chunked, compressed records            │
│  │                                                                  │
│  ├── chats/                # Individual chat JSON files             │
│  │   └── YYYY-MM-DD/       # Date-organized                         │
│  │                                                                  │
│  ├── commits/              # LOCAL ONLY (gitignored - too large)    │
│  │                         # Full commit data with diffs            │
│  │                                                                  │
│  └── shared/               # Aggregated patterns (safe to commit)   │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         EXPORT LAYER                                │
├─────────────────────────────────────────────────────────────────────┤
│  Formats:  JSONL  │  CSV  │  HuggingFace Dataset                   │
│                                                                     │
│  Training Record Format:                                            │
│  {                                                                  │
│    "type": "chat" | "commit",                                       │
│    "timestamp": "2025-12-16T10:00:00Z",                            │
│    "input": "query or commit message",                              │
│    "output": "response or diff summary",                            │
│    "context": { "files": [...], "tools_used": [...] }              │
│  }                                                                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Files

| File | Purpose | Lines |
|------|---------|-------|
| `scripts/ml_data_collector.py` | Main CLI and orchestrator | ~4300 |
| `scripts/ml_collector/` | Modular package (14 modules) | ~95KB |
| `scripts/ml_collector/chunked_storage.py` | Git-friendly storage | ~400 |
| `scripts/ml_file_prediction.py` | **NEW:** File prediction model | ~620 |
| `scripts/ml-session-start-hook.sh` | SessionStart hook | 64 |
| `scripts/ml-session-capture-hook.sh` | Stop hook (transcript) | 68 |
| `.git/hooks/post-commit` | Commit data capture | 17 |
| `tests/unit/test_ml_file_prediction.py` | File prediction tests | ~480 |

---

## Training Milestones

The system tracks progress toward three training tiers:

| Milestone | Commits | Sessions | Chats | Use Case |
|-----------|---------|----------|-------|----------|
| **File Prediction** | 500 | 100 | 200 | Predict which files to modify |
| **Commit Messages** | 2,000 | 500 | 1,000 | Generate commit messages |
| **Code Suggestions** | 5,000 | 2,000 | 5,000 | Full code assistance |

**Current Progress (as of session):**
- Commits (lite): 509
- Chats: 22
- Sessions: 0
- Progress: ~0% toward first milestone

---

## Changes Made This Session

### 1. Fixed Infinite Commit Loop

**Problem:** Each commit triggered the post-commit hook → saved ML data → needed another commit → infinite loop.

**Solution:** Modified post-commit hook to skip ML-only commits:

```bash
# .git/hooks/post-commit
COMMIT_MSG=$(git log -1 --format=%s HEAD 2>/dev/null)
if [[ "$COMMIT_MSG" == "data: ML tracking data"* ]] || [[ "$COMMIT_MSG" == "data: ML"* ]]; then
    exit 0
fi
```

**Files changed:**
- `.git/hooks/post-commit` - Installed hook
- `scripts/ml_data_collector.py` - Template for new installations (line ~3168)

### 2. Implemented Chunked Storage

**Problem:** Large chat files and commit diffs can exceed GitHub's size limits.

**Solution:** New chunked storage system with compression:

```python
# scripts/ml_collector/chunked_storage.py

Features:
- Compresses content >5KB with zlib (~60-90% size reduction)
- Stores as JSONL (one record per line) for git-friendly diffs
- Deduplicates via content hashing
- Supports full reconstruction from chunks
- Includes compaction utility (like git gc)
```

**New CLI commands:**
```bash
python scripts/ml_data_collector.py chunked migrate     # Migrate existing data
python scripts/ml_data_collector.py chunked compact     # Consolidate old chunks
python scripts/ml_data_collector.py chunked stats       # Show statistics
python scripts/ml_data_collector.py chunked reconstruct # Rebuild from chunks
```

### 3. Fixed .gitignore Pattern

**Problem:** `.git-ml/commits/` pattern with inline comment wasn't being recognized.

**Solution:** Moved comment to separate line:
```gitignore
# Full commit data with diffs is too large for GitHub
.git-ml/commits/
```

---

## Common Operations

### Check Collection Status
```bash
python scripts/ml_data_collector.py stats
```

### Estimate Time to Training
```bash
python scripts/ml_data_collector.py estimate
```

### Export for Training
```bash
# JSONL format (recommended)
python scripts/ml_data_collector.py export --format jsonl --output training.jsonl

# HuggingFace format
python scripts/ml_data_collector.py export --format huggingface --output dataset.json
```

### Add Feedback to Chats
```bash
# List chats needing feedback
python scripts/ml_data_collector.py feedback --list

# Add feedback
python scripts/ml_data_collector.py feedback --chat-id <id> --rating good
```

### Data Quality Report
```bash
python scripts/ml_data_collector.py quality-report
```

### Migrate to Chunked Storage
```bash
# Migrate all existing chats and commits
python scripts/ml_data_collector.py chunked migrate

# Verify migration
python scripts/ml_data_collector.py chunked stats
```

---

## File Prediction Model

The first ML model built from collected data: **predict which files to modify** based on a task description.

### Training
```bash
# Train on commit history (creates .git-ml/models/file_prediction.json)
python scripts/ml_file_prediction.py train
```

### Prediction
```bash
# Predict files for a task
python scripts/ml_file_prediction.py predict "Add authentication feature"

# With seed files for co-occurrence boosting
python scripts/ml_file_prediction.py predict "Fix related bug" --seed auth.py login.py --top 10
```

### Evaluation
```bash
# Evaluate on 20% holdout set
python scripts/ml_file_prediction.py evaluate --split 0.2

# View model statistics
python scripts/ml_file_prediction.py stats
```

### How It Works

1. **Feature Extraction:**
   - Commit type patterns (feat:, fix:, docs:, refactor:, etc.)
   - Task references (Task #42, etc.)
   - Action verbs (add, fix, update, implement)
   - Module keywords (test, api, config, etc.)

2. **Model Components:**
   - **File co-occurrence matrix**: Files changed together in commits
   - **Type-to-files mapping**: Which files change for each commit type
   - **Keyword-to-files mapping**: Keywords → associated files
   - **File frequency**: How often each file is changed

3. **Scoring:**
   - TF-IDF style scoring for commit type and keyword matches
   - Jaccard similarity for co-occurrence boosting
   - Frequency penalty to avoid always suggesting high-frequency files

### Current Metrics (403 commits, 20% test split)

| Metric | Value | Description |
|--------|-------|-------------|
| MRR | 0.43 | Mean Reciprocal Rank - first correct ~position 2-3 |
| Recall@10 | 0.48 | 48% of actual files appear in top 10 |
| Precision@1 | 0.31 | 31% of top predictions are correct |
| Recall@5 | 0.38 | 38% of actual files appear in top 5 |

### Model File

Stored at: `.git-ml/models/file_prediction.json`

Contains:
- `file_cooccurrence`: File → {co-occurring file → count}
- `type_to_files`: Commit type → {file → count}
- `keyword_to_files`: Keyword → {file → count}
- `file_frequency`: File → total change count
- `total_commits`: Training commit count
- `trained_at`: ISO timestamp
- `version`: Model version

---

## Data Schemas

### Commit Schema
```python
{
    "hash": str,           # Git commit hash
    "message": str,        # Commit message
    "author": str,         # Author name
    "timestamp": str,      # ISO timestamp
    "branch": str,         # Branch name
    "files_changed": list, # List of file paths
    "insertions": int,     # Lines added
    "deletions": int,      # Lines removed
    "hunks": list,         # Diff hunks (full commits only)
    "hour_of_day": int,    # 0-23
    "day_of_week": str,    # Monday-Sunday
    "session_id": str,     # Linked session (optional)
    "related_chats": list  # Linked chat IDs
}
```

### Chat Schema
```python
{
    "id": str,              # Unique chat ID
    "timestamp": str,       # ISO timestamp
    "session_id": str,      # Parent session
    "query": str,           # User query
    "response": str,        # Assistant response
    "files_referenced": list,  # Files read
    "files_modified": list,    # Files changed
    "tools_used": list,        # Tools invoked
    "query_tokens": int,       # Token count
    "response_tokens": int,    # Token count
    "user_feedback": dict      # Optional feedback
}
```

---

## Privacy & Security

### Automatic Redaction
The system automatically redacts sensitive data before storage:
- API keys and tokens
- Passwords and secrets
- AWS credentials
- Private keys
- Database connection strings
- GitHub/Slack tokens
- JWTs

### Data Retention
- Default: 730 days (2 years)
- Configurable via `cleanup` command:
  ```bash
  python scripts/ml_data_collector.py cleanup --days 365
  ```

### Contribution Consent
Optional opt-in to share anonymized patterns:
```bash
python scripts/ml_data_collector.py contribute status
python scripts/ml_data_collector.py contribute enable
```

---

## Troubleshooting

### Hook Not Running
```bash
# Reinstall hooks
python scripts/ml_data_collector.py install-hooks

# Verify hook is executable
ls -la .git/hooks/post-commit
```

### Excessive ML Commits
If you see repeated "data: ML tracking data" commits, verify the post-commit hook has the skip logic (lines 7-12).

### Large Files Warning
```bash
# Migrate to chunked storage
python scripts/ml_data_collector.py chunked migrate

# Compact old chunks
python scripts/ml_data_collector.py chunked compact
```

### Disable Collection Temporarily
```bash
export ML_COLLECTION_ENABLED=0
# ... do work ...
unset ML_COLLECTION_ENABLED
```

---

## Future Considerations

1. **Automated Training Pipeline**: When milestones are reached, trigger model fine-tuning
2. **Feedback Integration**: Use chat feedback ratings to weight training examples
3. **Cross-Project Learning**: Share anonymized patterns across repositories
4. **Incremental Training**: Update model as new data is collected

---

## Test Coverage

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `tests/unit/test_ml_export.py` | 20 | Export formats |
| `tests/unit/test_ml_feedback.py` | 25 | Feedback system |
| `tests/unit/test_ml_quality.py` | ~25 | Quality analysis |
| `tests/unit/test_chunked_storage.py` | 25 | Chunked storage |
| `tests/unit/test_ml_file_prediction.py` | 33 | File prediction model |

Run all ML tests:
```bash
python -m unittest tests.unit.test_ml_export tests.unit.test_ml_feedback \
    tests.unit.test_ml_quality tests.unit.test_chunked_storage \
    tests.unit.test_ml_file_prediction -v
```

---

## References

- Main documentation: `CLAUDE.md` (section: ML Data Collection)
- Test files: `tests/unit/test_ml_*.py`
- Hook scripts: `scripts/ml-session-*.sh`
- Config: `scripts/ml_collector/config.py`
