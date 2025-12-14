# Knowledge Transfer: Security Hardening & Semantic Diff

**Date:** 2025-12-14
**Branch:** `claude/plan-next-priorities-zwRV6`
**Focus:** Security improvements, semantic diff feature, CI fixes

---

## Executive Summary

This session completed 8 tasks focused on:
1. Security hardening (CI scanning, documentation, deprecation warnings)
2. New "What Changed?" semantic diff feature
3. Critical CI fix for test failures

### Commits Made

| Commit | Description |
|--------|-------------|
| `90b989f` | Security CI scanning and pickle deprecation warnings |
| `091167a` | MCP server security documentation |
| `a31d1c7` | Semantic diff feature (LEGACY-075) |
| `a5abfb9` | Task-manager skill security documentation |
| `0da2739` | Fix CI test failures from deprecation warnings |

---

## Part 1: Security Improvements

### SEC-001: Pickle Security Warning in README

**Location:** `README.md` → "Security Considerations" section

Added comprehensive warning about pickle deserialization risks:
- Explains RCE risk from malicious pickle files
- Recommends JSON StateLoader for untrusted sources
- Links to Python documentation

### SEC-002, SEC-004, SEC-005: CI Security Scanning

**Location:** `.github/workflows/ci.yml` → `security-scan` job

Added new parallel CI job with three security tools:

```yaml
security-scan:
  name: "🔐 Security Scan"
  steps:
    - name: Run Bandit (SAST)      # Static analysis
    - name: Run pip-audit          # Dependency vulnerabilities
    - name: Run detect-secrets     # Secret scanning
```

**Configuration Details:**
- Bandit: `-ll -s B101` (medium+ severity, skip assert warnings)
- pip-audit: Full dependency scan with descriptions
- detect-secrets: Excludes `.pkl` and `.json` files, reports non-test findings

### SEC-006: MCP Server Security Model

**Location:** `docs/mcp-security.md`

Comprehensive security documentation covering:
- Tool capabilities and risk levels (search, passages, expand_query, corpus_stats, add_document)
- Input validation performed by each endpoint
- Trust model (stdio transport, single-tenant, no authentication)
- Security boundary diagram
- Rate limiting recommendations
- Deployment checklist

### SEC-007: Task-Manager Skill Permissions

**Location:** `.claude/skills/task-manager/SKILL.md` → "Security Model" section

Documented why the skill needs `Read, Bash, Write` permissions:
- Write needed for JSON task file creation
- Alternatives considered (Bash heredocs) and rejected
- Recommendations for strict environments

### SEC-008: Pickle Deprecation Warnings

**Location:** `cortical/persistence.py` → `save_processor()` and `load_processor()`

Added `DeprecationWarning` when using pickle format:
```python
warnings.warn(
    "Pickle format is deprecated due to security concerns (arbitrary code execution). "
    "Consider using format='protobuf' or the StateLoader JSON format instead. "
    "See README.md 'Security Considerations' for details.",
    DeprecationWarning,
    stacklevel=2
)
```

---

## Part 2: Semantic Diff Feature (LEGACY-075)

### New Module: `cortical/diff.py`

Implements "What Changed?" functionality for comparing processor states.

**Key Classes:**

| Class | Purpose |
|-------|---------|
| `SemanticDiff` | Complete diff report with all changes |
| `TermChange` | Tracks PageRank/TF-IDF changes per term |
| `RelationChange` | Tracks typed connection changes |
| `ClusterChange` | Tracks concept cluster reorganization |

**Key Functions:**

| Function | Purpose |
|----------|---------|
| `compare_processors(old, new)` | Full semantic diff between processor states |
| `compare_documents(proc, doc1, doc2)` | Compare two documents in same corpus |
| `what_changed(proc, old_text, new_text)` | Quick text comparison without corpus changes |

### New CorticalTextProcessor Methods

```python
# Compare two processor states
diff = new_processor.compare_with(old_processor)
print(diff.summary())

# Compare documents within corpus
comparison = processor.compare_documents("doc1", "doc2")
print(f"Similarity: {comparison['jaccard_similarity']:.2%}")

# Quick text comparison
result = processor.what_changed(old_code, new_code)
print(f"Tokens added: {result['tokens']['added']}")
```

### Example Output

```
# Semantic Diff Summary

## Documents
- Added: 1 documents
  + doc3

## Terms
- New terms: 2
  + subset
  + deep

## Importance Shifts (PageRank)
### Rising Terms
  + learning: +0.058332
  + machine: +0.012112
### Falling Terms
  - recognition: -0.051109
  - pattern: -0.051109

## Relations
- New relations: 11
  + learning --co_occurrence--> L0_deep

## Statistics
- Total term changes: 11
- Total relation changes: 11
- Total cluster changes: 2
```

---

## Part 3: CI Fix

### Problem

After adding pickle deprecation warnings (SEC-008), CI tests started failing because:
1. `pyproject.toml` has `filterwarnings = ["error"]` (treat warnings as errors)
2. Smoke tests call `save()` and `load()` which emit `DeprecationWarning`
3. The warning became a test failure

### Solution

**Location:** `pyproject.toml` → `[tool.pytest.ini_options]`

Added filter to ignore our intentional deprecation warning:
```toml
filterwarnings = [
    "error",
    # Allow our intentional pickle deprecation warnings (SEC-008)
    "ignore:Pickle format is deprecated:DeprecationWarning",
]
```

### Lesson Learned

When adding deprecation warnings to code that's exercised by tests, remember to update `filterwarnings` in `pyproject.toml` to prevent false CI failures.

---

## Part 4: Remaining Work

### Security Tasks (Still Pending)

| Task | Priority | Effort | Description |
|------|----------|--------|-------------|
| SEC-003 | Medium | 4h | HMAC verification for pickle files |
| SEC-009 | Low | 4h | Security-focused test suite |
| SEC-010 | Low | 8h | Input fuzzing with Hypothesis |

### Feature Tasks (Still Pending)

| Task | Priority | Description |
|------|----------|-------------|
| LEGACY-078 | Medium | Code pattern detection |
| LEGACY-095 | Medium | Split processor.py (2,301 lines) |
| LEGACY-187 | Medium | Async API support |
| LEGACY-190 | Medium | REST API wrapper (FastAPI) |

---

## Files Changed in This Session

### New Files
- `cortical/diff.py` - Semantic diff module (610 lines)
- `docs/mcp-security.md` - MCP security documentation

### Modified Files
- `.github/workflows/ci.yml` - Added security-scan job
- `README.md` - Added Security Considerations section
- `cortical/__init__.py` - Exported diff module classes
- `cortical/persistence.py` - Added deprecation warnings
- `cortical/processor.py` - Added semantic diff methods
- `pyproject.toml` - Fixed pytest filterwarnings
- `.claude/skills/task-manager/SKILL.md` - Added security model section
- `docs/README.md` - Added Security section to docs index

### Task Files Updated
- `tasks/2025-12-14_11-15-01_41d5.json` - Marked SEC tasks completed
- `tasks/legacy_migration.json` - Marked LEGACY-075 completed

---

## How to Use New Features

### Security Scanning (Automatic)

Security scans run automatically on every push:
- Check CI for "🔐 Security Scan" job
- Review any Bandit, pip-audit, or detect-secrets findings

### Semantic Diff

```python
from cortical import CorticalTextProcessor

# Load two versions of your corpus
old = CorticalTextProcessor.load("corpus_v1.pkl")
new = CorticalTextProcessor.load("corpus_v2.pkl")

# See what changed
diff = new.compare_with(old)
print(diff.summary())

# Or export as dict for programmatic use
data = diff.to_dict()
```

---

## Testing Changes

To verify all changes work:

```bash
# Run smoke tests
python -m pytest tests/smoke/ -v

# Test semantic diff
python -c "
from cortical import CorticalTextProcessor
p1 = CorticalTextProcessor()
p1.process_document('d1', 'Hello world')
p1.compute_all()

p2 = CorticalTextProcessor()
p2.process_document('d1', 'Hello world')
p2.process_document('d2', 'Goodbye world')
p2.compute_all()

diff = p2.compare_with(p1)
print(diff.summary())
"
```

---

*Document created: 2025-12-14*
