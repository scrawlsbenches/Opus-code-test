# Knowledge Transfer: Security Review & Test Coverage

**Date:** 2025-12-14
**Branch:** `claude/review-security-features-OtjPE`
**Focus:** Review security features, add missing tests, complete SEC-003
**Status:** COMPLETED

---

## Executive Summary

This session completed three major tasks:
1. **Added 70 tests for diff.py** - Coverage improved from 17% to 87%
2. **Implemented SEC-003: HMAC verification** - Pickle file integrity protection
3. **Added 23 tests for SEC-003** - Comprehensive security test coverage

### Commits Made

| Commit | Description |
|--------|-------------|
| (pending) | feat: Add HMAC signature verification for pickle files (SEC-003) |
| (pending) | test: Add comprehensive tests for diff.py and SEC-003 |

---

## Current State Assessment

### What Was Done (Previous Session)
| Task | Status | Notes |
|------|--------|-------|
| SEC-001: README security warning | Done | Pickle risks documented |
| SEC-002: Bandit SAST in CI | Done | Added to security-scan job |
| SEC-004: pip-audit in CI | Done | Dependency scanning |
| SEC-005: detect-secrets in CI | Done | Secret scanning |
| SEC-006: MCP security docs | Done | docs/mcp-security.md |
| SEC-007: task-manager permissions | Done | SKILL.md updated |
| SEC-008: Pickle deprecation warning | Done | persistence.py |
| LEGACY-075: Semantic diff | Done | cortical/diff.py |

### Test Status
```
Smoke tests:  18 passed (0.49s)
Quick tests:  2166 passed (13s)
diff.py:      17% coverage (279 statements, 210 missed)
```

### Key Gap: diff.py Has No Tests
The new 625-line `cortical/diff.py` module has:
- 4 dataclasses: `TermChange`, `RelationChange`, `ClusterChange`, `SemanticDiff`
- 3 public functions: `compare_processors()`, `compare_documents()`, `what_changed()`
- 2 private functions: `_compare_relations()`, `_compare_clusters()`
- 0 dedicated test files

The module works (verified manually), but untested code is risky.

---

## Work Plan

### Phase 1: Add Tests for diff.py (Priority: HIGH)

**Why first:** "Did I do it right?" - We can't ship untested code.

**Deliverable:** `tests/unit/test_diff.py`

**Test cases needed:**
1. `TestTermChange` - Dataclass property tests
   - `pagerank_delta` calculation
   - `tfidf_delta` calculation
   - `documents_added` / `documents_removed`
2. `TestRelationChange` - Basic dataclass
3. `TestClusterChange` - Basic dataclass
4. `TestSemanticDiff` - Summary and to_dict methods
5. `TestCompareProcessors` - Main comparison function
   - Empty processors
   - Single document change
   - Multiple document changes
   - PageRank importance shifts
6. `TestCompareDocuments` - Within-corpus comparison
7. `TestWhatChanged` - Quick text comparison

**Estimated effort:** 2 hours
**Sub-agent friendly:** Yes - bounded, well-defined task

### Phase 2: Implement SEC-003 (Priority: MEDIUM)

**Why second:** Completes the pickle security story.

**Deliverable:** HMAC verification in `cortical/persistence.py`

**Implementation:**
```python
import hmac
import hashlib

def _compute_signature(data: bytes, key: bytes) -> bytes:
    return hmac.new(key, data, hashlib.sha256).digest()

def save_processor(..., signing_key: Optional[bytes] = None):
    # If key provided, write signature to .pkl.sig

def load_processor(..., verify_key: Optional[bytes] = None):
    # If key provided, verify signature before loading
```

**Design decisions:**
- Signature file: `{filename}.sig` (separate file, not embedded)
- Algorithm: HMAC-SHA256 (standard, no external deps)
- Key management: User-provided (not auto-generated)
- Behavior on mismatch: Raise `SecurityError` or similar

**Estimated effort:** 4 hours
**Sub-agent friendly:** No - security code needs careful review

### Phase 3: Update Tests for SEC-003 (Priority: LOW)

**Deliverable:** Tests in `tests/unit/test_persistence.py` or `tests/security/`

**Test cases:**
1. Save with signing key creates .sig file
2. Load with verify key and valid signature succeeds
3. Load with verify key and invalid signature fails
4. Load with verify key and missing signature fails
5. Backward compatibility: Load without key still works (with warning)

---

## Sub-Agent Guidance

### For diff.py Test Agent

**Context:** You're writing tests for a new semantic diff module.

**Files to read first:**
- `cortical/diff.py` - The module under test
- `tests/unit/test_*.py` - Existing test patterns
- `tests/conftest.py` - Available fixtures

**Fixtures available:**
- `small_processor` - Pre-loaded 25-doc corpus
- `fresh_processor` - Empty processor
- `small_corpus_docs` - Raw document dict

**Test file location:** `tests/unit/test_diff.py`

**Style requirements:**
- Use pytest, not unittest
- Group tests by class being tested
- Include docstrings explaining what each test verifies
- Test edge cases: empty processors, single doc, identical inputs

**Do NOT:**
- Modify any existing files
- Add tests to existing test files
- Change the diff.py implementation

### For SEC-003 Implementation

**Context:** Adding HMAC signature verification to pickle save/load.

**Files to modify:**
- `cortical/persistence.py` - Add signing logic

**Design constraints:**
- No external dependencies (use stdlib hmac/hashlib)
- Signature in separate file (.sig extension)
- Backward compatible (signing optional)
- Clear error messages on verification failure

**Security considerations:**
- Use constant-time comparison (hmac.compare_digest)
- Don't leak timing information
- Clear documentation of threat model

---

## Decision Log

| Decision | Rationale |
|----------|-----------|
| Tests before SEC-003 | Can't verify security code without tests |
| diff.py tests first | Lower risk, enables sub-agent delegation |
| HMAC not embedded in pickle | Separate .sig file is cleaner, easier to audit |
| User-provided keys | No key management complexity in library |

---

## Files Expected to Change

### New Files
- `tests/unit/test_diff.py` - Tests for diff module

### Modified Files
- `cortical/persistence.py` - SEC-003 HMAC verification
- Possibly `tests/unit/test_persistence.py` - SEC-003 tests

---

## Verification Checklist

Before marking complete:
- [x] All tests pass: `python -m pytest tests/ -v` - 2259 passed
- [x] Coverage for diff.py > 80%: 87% achieved
- [x] SEC-003 implementation reviewed for security issues
- [x] Documentation updated if API changed
- [x] Commit messages reference task IDs

---

## What Was Implemented

### SEC-003: HMAC Signature Verification

**Files Modified:**
- `cortical/persistence.py` - Added signature functions and verification logic
- `cortical/processor.py` - Added `signing_key` and `verify_key` parameters
- `cortical/__init__.py` - Exported `SignatureVerificationError`

**New Functions:**
```python
# Helper functions in persistence.py
_get_signature_path(filepath)      # Returns filepath + ".sig"
_compute_signature(data, key)      # HMAC-SHA256 computation
_save_signature(filepath, sig)     # Save signature to .sig file
_load_signature(filepath)          # Load signature from .sig file
_verify_signature(data, sig, key)  # Constant-time signature verification
```

**Usage:**
```python
from cortical import CorticalTextProcessor, SignatureVerificationError

# Save with signature
key = b'my-secret-key'
processor.save("corpus.pkl", signing_key=key)
# Creates corpus.pkl and corpus.pkl.sig

# Load with verification
loaded = CorticalTextProcessor.load("corpus.pkl", verify_key=key)
# Raises SignatureVerificationError if tampered

# Backward compatible - no key required
loaded = CorticalTextProcessor.load("corpus.pkl")
```

**Security Features:**
- HMAC-SHA256 for signing (32-byte signature)
- Constant-time comparison (`hmac.compare_digest`) prevents timing attacks
- Separate .sig file (cleaner, easier to audit)
- Clear error messages on verification failure

### Tests Added

**tests/unit/test_diff.py** - 70 tests covering:
- TermChange, RelationChange, ClusterChange dataclasses
- SemanticDiff summary and to_dict methods
- compare_processors() function
- compare_documents() function
- what_changed() function

**tests/unit/test_persistence.py** - 23 tests covering:
- Signature helper functions
- SignatureVerificationError exception
- Save/load with signing and verification
- Tamper detection
- Backward compatibility

---

## Files Changed

### New Files
- `tests/unit/test_diff.py` - 70 tests for diff module

### Modified Files
- `cortical/persistence.py` - SEC-003 HMAC implementation
- `cortical/processor.py` - signing_key/verify_key parameters
- `cortical/__init__.py` - Export SignatureVerificationError
- `tests/unit/test_persistence.py` - SEC-003 tests
- `docs/session-2025-12-14-review-and-testing.md` - This document

---

## Remaining Tasks

### Security Tasks (Still Pending)
| Task | Priority | Effort | Description |
|------|----------|--------|-------------|
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

*Document created: 2025-12-14*
*Document completed: 2025-12-14*
