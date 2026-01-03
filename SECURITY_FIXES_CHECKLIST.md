# Security Fixes Checklist - Prioritized Action Plan

**Created:** 2026-01-03
**Status:** PENDING IMPLEMENTATION

---

## Critical Priority (Fix Immediately)

### 1. Path Traversal Protection - ALL FILES
**Affected Files:** All 6 files
**Effort:** 2-3 hours
**Risk:** CRITICAL - Arbitrary file write/read

**Implementation:**
```python
# Add to cortical/got/security_utils.py (NEW FILE)

from pathlib import Path
from typing import Union

class SecurityValidator:
    """Security validation utilities for GoT system."""

    @staticmethod
    def validate_got_dir(got_dir: Union[str, Path], repo_root: Path) -> Path:
        """Validate GoT directory path is within repository."""
        got_dir = Path(got_dir).resolve()

        try:
            got_dir.relative_to(repo_root.resolve())
        except ValueError:
            raise ValueError(f"got_dir must be within repository: {repo_root}")

        return got_dir

    @staticmethod
    def validate_subdirectory(parent: Path, subdir: Path) -> Path:
        """Validate subdirectory is within parent."""
        subdir = subdir.resolve()
        parent = parent.resolve()

        if not subdir.is_relative_to(parent):
            raise ValueError(f"Directory traversal detected: {subdir} not in {parent}")

        return subdir

    @staticmethod
    def validate_task_id(task_id: str) -> str:
        """Validate task ID format."""
        import re
        TASK_ID_PATTERN = re.compile(r'^T-\d{8}-\d{6}-[0-9a-f]{8}$')

        if not TASK_ID_PATTERN.match(task_id):
            raise ValueError(f"Invalid task_id format: {task_id[:50]}")

        return task_id
```

**Files to Update:**
- [ ] `cortical/got/learning_integration.py` - Line 69
- [ ] `cortical/got/cli/failure.py` - Lines 28, 76
- [ ] `scripts/commit_task_linker.py` - Lines 37, 40
- [ ] `scripts/training_data_exporter.py` - Line 261
- [ ] `scripts/auto_learning_capture.py` - Line 58
- [ ] `.claude/hooks/session_learning_capture.py` - Lines 55, 218

---

### 2. JSON Size Limits - 4 FILES
**Affected Files:** failure.py, commit_task_linker.py, training_data_exporter.py, session_learning_capture.py
**Effort:** 1-2 hours
**Risk:** CRITICAL - Memory exhaustion DoS

**Implementation:**
```python
# Add to security_utils.py

import json
from typing import Any, Dict

MAX_JSON_SIZE = 10 * 1024 * 1024  # 10MB

class SafeJSONLoader:
    """Safe JSON loading with size limits."""

    @staticmethod
    def load_file(file_path: Path, max_size: int = MAX_JSON_SIZE) -> Dict[str, Any]:
        """Load JSON file with size check."""
        file_size = file_path.stat().st_size
        if file_size > max_size:
            raise ValueError(f"JSON file too large: {file_size} bytes (max: {max_size})")

        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
```

**Files to Update:**
- [ ] `cortical/got/cli/failure.py` - Lines 91-92, 103-104
- [ ] `scripts/commit_task_linker.py` - Lines 164, 221
- [ ] `scripts/training_data_exporter.py` - Line 276
- [ ] `.claude/hooks/session_learning_capture.py` - Line 219

---

### 3. O(n²) Resource Exhaustion - commit_task_linker.py
**Affected Files:** scripts/commit_task_linker.py
**Effort:** 3-4 hours
**Risk:** CRITICAL - CPU exhaustion DoS

**Implementation:** See detailed fix in SECURITY_REVIEW_REPORT.md, File 3, CRITICAL-3

**Tasks:**
- [ ] Add MAX_COMMITS = 10,000 limit
- [ ] Add MAX_TASKS = 5,000 limit
- [ ] Add MAX_COMPARISONS = 100,000 limit
- [ ] Add progress logging every 10,000 comparisons
- [ ] Truncate commit/task text to 1,000 chars before similarity computation

---

## High Priority (Fix This Week)

### 4. Input Validation Framework
**Effort:** 4-5 hours
**Risk:** HIGH - Multiple injection vectors

**Implementation:**
```python
# Add to security_utils.py

import re
from typing import Optional

class InputValidator:
    """Input validation utilities."""

    MAX_TEXT_LENGTH = 10000
    MAX_SHORT_TEXT = 1000
    MAX_ID_LENGTH = 100

    @staticmethod
    def validate_text(text: str, max_length: Optional[int] = None,
                     field_name: str = "input") -> str:
        """Validate and sanitize text input."""
        max_length = max_length or InputValidator.MAX_TEXT_LENGTH

        if not text:
            raise ValueError(f"{field_name} cannot be empty")

        if len(text) > max_length:
            raise ValueError(
                f"{field_name} exceeds maximum length of {max_length} "
                f"(got {len(text)})"
            )

        # Remove null bytes and control characters (except \n, \t)
        sanitized = ''.join(
            c for c in text
            if c == '\n' or c == '\t' or c >= ' '
        )

        return sanitized

    @staticmethod
    def validate_failure_id(failure_id: str) -> str:
        """Validate failure ID format."""
        if not re.match(r'^F-\d{8}-\d{6}-[0-9a-f]{8}$', failure_id):
            raise ValueError(f"Invalid failure_id format: {failure_id}")
        return failure_id

    @staticmethod
    def validate_commit_hash(commit_hash: str, min_length: int = 6) -> str:
        """Validate commit hash format."""
        if not re.match(r'^[0-9a-f]{6,40}$', commit_hash.lower()):
            raise ValueError(f"Invalid commit hash format: {commit_hash}")

        if len(commit_hash) < min_length:
            raise ValueError(
                f"Commit hash must be at least {min_length} characters"
            )

        return commit_hash.lower()
```

**Files to Update:**
- [ ] `cortical/got/learning_integration.py` - Add validation to capture methods
- [ ] `cortical/got/cli/failure.py` - Lines 127-131, validate all inputs
- [ ] `scripts/commit_task_linker.py` - Line 519, validate commit hashes
- [ ] All files - Validate task_id format

---

### 5. Atomic File Operations
**Effort:** 2-3 hours
**Risk:** HIGH - Data corruption

**Implementation:**
```python
# Add to security_utils.py

import os
import tempfile
import shutil
from pathlib import Path
from typing import Any, Callable

class AtomicFileWriter:
    """Atomic file write operations."""

    @staticmethod
    def write_json(file_path: Path, data: Any, indent: int = 2) -> None:
        """Write JSON atomically using temp file."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Create temp file in same directory
        temp_fd, temp_path = tempfile.mkstemp(
            dir=file_path.parent,
            prefix=f'.tmp_{file_path.stem}_',
            suffix='.json'
        )

        try:
            with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent)

            # Atomic move (platform-dependent behavior)
            shutil.move(temp_path, file_path)

        except Exception:
            # Clean up temp file on error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise
```

**Files to Update:**
- [ ] `cortical/got/cli/failure.py` - Line 77
- [ ] `scripts/commit_task_linker.py` - Lines 245-246

---

### 6. Sanitize Exported Training Data
**Effort:** 3-4 hours
**Risk:** HIGH - Sensitive data exposure

**Implementation:**
```python
# Add to security_utils.py

import re
from pathlib import Path
from typing import Any, Dict

class DataSanitizer:
    """Sanitize data for export."""

    @staticmethod
    def sanitize_text(text: str, max_length: int = 500) -> str:
        """Sanitize text for export, removing sensitive patterns."""
        if not text:
            return ""

        # Truncate
        text = text[:max_length]

        # Remove email addresses
        text = re.sub(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            '[EMAIL]',
            text
        )

        # Remove potential API keys (32+ alphanumeric strings)
        text = re.sub(r'\b[A-Za-z0-9]{32,}\b', '[KEY]', text)

        # Remove absolute file paths
        text = re.sub(r'(?:^|[\s])(/[^\s]+)', r' [PATH]', text)

        # Remove IP addresses
        text = re.sub(
            r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
            '[IP]',
            text
        )

        # Remove URLs
        text = re.sub(
            r'https?://[^\s]+',
            '[URL]',
            text
        )

        return text

    @staticmethod
    def sanitize_file_paths(paths: list) -> list:
        """Keep only filenames, remove directory structure."""
        return [Path(p).name for p in paths]

    @staticmethod
    def sanitize_training_example(example: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize a complete training example."""
        sanitized = example.copy()

        # Text fields to sanitize
        text_fields = [
            'commit_message', 'task_title', 'task_description',
            'rationale', 'retrospective', 'instructions', 'summary'
        ]

        for field in text_fields:
            if field in sanitized:
                sanitized[field] = DataSanitizer.sanitize_text(
                    sanitized[field]
                )

        # File paths
        if 'files_changed' in sanitized:
            sanitized['files_changed'] = DataSanitizer.sanitize_file_paths(
                sanitized['files_changed']
            )

        # Truncate hashes
        if 'commit_hash' in sanitized:
            sanitized['commit_hash'] = sanitized['commit_hash'][:12]

        return sanitized
```

**Files to Update:**
- [ ] `scripts/commit_task_linker.py` - Lines 606-627
- [ ] `scripts/training_data_exporter.py` - Lines 621-627, 378-379, 441-442

---

### 7. Redact Sensitive IDs in Logs
**Effort:** 2 hours
**Risk:** HIGH - Information disclosure

**Implementation:**
```python
# Add to security_utils.py

class LogSanitizer:
    """Sanitize sensitive data in logs."""

    @staticmethod
    def redact_id(entity_id: str, show_chars: int = 4) -> str:
        """Redact middle portion of ID for logging."""
        if not entity_id or len(entity_id) < show_chars * 2:
            return "***"

        return f"{entity_id[:show_chars]}...{entity_id[-show_chars:]}"

    @staticmethod
    def redact_path(path: str) -> str:
        """Redact directory structure, keep filename."""
        return Path(path).name
```

**Files to Update:**
- [ ] `cortical/got/learning_integration.py` - Lines 176, 276, 319
- [ ] All files - Replace direct ID logging with redacted versions

---

## Medium Priority (Fix This Month)

### 8. Resource Limits on Caching
**Effort:** 2-3 hours

```python
# Add to security_utils.py

class ResourceLimits:
    MAX_CACHE_SIZE = 100_000  # 100k entities
    MAX_ENTITY_SIZE = 1_000_000  # 1MB per entity
    MAX_LINKS = 1_000_000  # 1M links
    MAX_FILES_TO_SCAN = 100_000  # 100k files
    MAX_DAYS_LOOKBACK = 365  # 1 year
```

**Files to Update:**
- [ ] `scripts/training_data_exporter.py` - Line 266, add cache size limit
- [ ] `scripts/commit_task_linker.py` - Lines 136, 223, add link limit
- [ ] `scripts/auto_learning_capture.py` - Line 94, add file scan limit

---

### 9. Anti-Gaming Quality Score
**Effort:** 2 hours

**Files to Update:**
- [ ] `scripts/training_data_exporter.py` - Lines 303-334

---

### 10. Timestamp Validation
**Effort:** 1 hour

**Files to Update:**
- [ ] `scripts/auto_learning_capture.py` - Lines 115-118
- [ ] All files parsing timestamps

---

### 11. Tag Validation
**Effort:** 1 hour

**Files to Update:**
- [ ] `cortical/got/learning_integration.py` - Lines 167-171

---

### 12. Replace Reflection with Explicit API Checks
**Effort:** 1 hour

**Files to Update:**
- [ ] `cortical/got/cli/failure.py` - Lines 166-175

---

## Low Priority (Fix When Convenient)

### 13. Markdown Escaping
**Files:** training_data_exporter.py

### 14. CLI Rate Limiting
**Files:** All CLI scripts

### 15. Improve Error Messages
**Files:** All files

---

## Implementation Plan

### Week 1: Critical Fixes
- Day 1: Create security_utils.py with all utility classes
- Day 2: Fix path traversal in all 6 files
- Day 3: Add JSON size limits to all 4 affected files
- Day 4: Fix O(n²) resource exhaustion
- Day 5: Testing and validation

### Week 2: High Priority Fixes
- Day 1-2: Implement input validation framework
- Day 3: Atomic file operations
- Day 4: Sanitize exported data
- Day 5: Redact sensitive logs

### Week 3: Medium Priority Fixes
- Day 1-2: Resource limits on caching
- Day 3: Anti-gaming quality score
- Day 4-5: Timestamp and tag validation

### Week 4: Testing & Documentation
- Day 1-2: Security testing (fuzzing, load tests)
- Day 3: Update documentation
- Day 4-5: Code review and final fixes

---

## Testing Checklist

### Path Traversal Tests
- [ ] Test `got_dir` with `../../../etc`
- [ ] Test `got_dir` with absolute paths like `/tmp`
- [ ] Test task_id with path traversal: `../../../etc/passwd`
- [ ] Test failure_id with traversal sequences

### Resource Exhaustion Tests
- [ ] Load 1GB JSON file
- [ ] Create 1 million links
- [ ] Run semantic similarity with 100k commits
- [ ] Cache 200k entities

### Input Validation Tests
- [ ] Submit 100MB retrospective text
- [ ] Use control characters in task descriptions
- [ ] Use null bytes in commit messages
- [ ] Test with Unicode edge cases

### Data Sanitization Tests
- [ ] Export with email addresses in descriptions
- [ ] Export with API keys in commit messages
- [ ] Export with file paths
- [ ] Verify no PII in exported training data

---

## Sign-off

Once all critical and high priority fixes are implemented:

- [ ] Code review by security engineer
- [ ] Penetration testing
- [ ] Update threat model
- [ ] Document security controls
- [ ] Train developers on secure coding practices

---

**Estimated Total Effort:** 30-35 hours
**Recommended Timeline:** 4 weeks
**Priority:** HIGH - Multiple critical vulnerabilities
