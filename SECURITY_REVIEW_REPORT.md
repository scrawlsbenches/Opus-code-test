# Security Review Report: Learning Integration Files

**Review Date:** 2026-01-03
**Reviewer:** Security Analysis Bot
**Scope:** 6 new files created for LLM learning integration

---

## Executive Summary

This security review identified **3 CRITICAL**, **8 HIGH**, **12 MEDIUM**, and **9 LOW** severity vulnerabilities across 6 files. The most severe issues involve path traversal, deserialization attacks, resource exhaustion, and information disclosure.

**Critical Issues Requiring Immediate Attention:**
1. Path traversal in file operations (all files)
2. Unbounded JSON deserialization (4 files)
3. O(n²) computational complexity without rate limiting (commit_task_linker.py)

---

## Vulnerability Summary by File

| File | Critical | High | Medium | Low |
|------|----------|------|--------|-----|
| learning_integration.py | 1 | 2 | 3 | 2 |
| cli/failure.py | 1 | 2 | 2 | 1 |
| commit_task_linker.py | 1 | 2 | 3 | 2 |
| training_data_exporter.py | 0 | 1 | 2 | 2 |
| auto_learning_capture.py | 0 | 1 | 1 | 1 |
| session_learning_capture.py | 0 | 0 | 1 | 1 |

---

## Detailed Vulnerability Analysis

### File 1: `/home/user/Opus-code-test/cortical/got/learning_integration.py`

#### CRITICAL-1: Path Traversal in Directory Creation
**Location:** Lines 69-71
**Severity:** CRITICAL
**OWASP Category:** Path Traversal (A01:2021)

```python
# VULNERABLE CODE:
def __init__(self, got_dir: Path):
    self.got_dir = Path(got_dir)
    self.learning_dir = self.got_dir / "learning"
    self.learning_dir.mkdir(parents=True, exist_ok=True)
```

**Issue:** Accepts arbitrary `got_dir` path without validation. An attacker could pass `../../etc` or absolute paths to write to system directories.

**Fix:**
```python
def __init__(self, got_dir: Path):
    # Validate and resolve path
    self.got_dir = Path(got_dir).resolve()

    # Ensure path is within expected repository
    expected_root = Path(__file__).parent.parent.parent.resolve()
    try:
        self.got_dir.relative_to(expected_root)
    except ValueError:
        raise ValueError(f"got_dir must be within repository: {expected_root}")

    # Prevent directory traversal in subdirectory
    self.learning_dir = self.got_dir / "learning"
    if not self.learning_dir.resolve().is_relative_to(self.got_dir.resolve()):
        raise ValueError("Invalid learning directory path")

    self.learning_dir.mkdir(parents=True, exist_ok=True)
```

---

#### HIGH-1: Unbounded File Path Processing
**Location:** Lines 434-450
**Severity:** HIGH
**OWASP Category:** Injection (A03:2021)

```python
# VULNERABLE CODE:
def _infer_domain_from_files(self, files: List[str]) -> str:
    if not files:
        return "general"

    dirs = set()
    for file_path in files:
        parts = Path(file_path).parts  # No validation!
```

**Issue:** Processes arbitrary file paths without validation. Could include path traversal sequences (`../../../etc/passwd`).

**Fix:**
```python
def _infer_domain_from_files(self, files: List[str]) -> str:
    if not files:
        return "general"

    # Limit number of files processed
    MAX_FILES = 1000
    if len(files) > MAX_FILES:
        files = files[:MAX_FILES]

    dirs = set()
    for file_path in files:
        # Validate file path
        if not file_path or len(file_path) > 4096:  # PATH_MAX
            continue
        if ".." in file_path or file_path.startswith("/"):
            continue  # Skip potential traversal attempts

        try:
            parts = Path(file_path).parts
            if len(parts) > 1:
                dirs.add(parts[0])
        except (ValueError, OSError):
            continue  # Skip invalid paths
```

---

#### HIGH-2: Information Disclosure in Logs
**Location:** Lines 176, 276, 319
**Severity:** HIGH
**OWASP Category:** Security Logging and Monitoring Failures (A09:2021)

```python
# VULNERABLE CODE:
logger.info(f"Captured task completion: {task_id} -> {experience.id}")
```

**Issue:** Logs potentially sensitive task IDs and experience data. In production, logs may be accessible to unauthorized users.

**Fix:**
```python
# Redact sensitive IDs in logs
def _redact_id(entity_id: str) -> str:
    """Redact middle portion of ID for logging."""
    if len(entity_id) < 10:
        return "***"
    return f"{entity_id[:4]}...{entity_id[-4:]}"

logger.info(
    f"Captured task completion: {self._redact_id(task_id)} -> "
    f"{self._redact_id(experience.id)}"
)
```

---

#### MEDIUM-1: Unbounded Text Processing
**Location:** Lines 520-567
**Severity:** MEDIUM
**OWASP Category:** Resource Exhaustion

```python
# VULNERABLE CODE:
def _parse_retrospective(self, retrospective: str) -> Dict[str, List[str]]:
    # No length limit on input!
    sentences = [s.strip() for s in retrospective.split('.') if s.strip()]
```

**Issue:** No size limits on retrospective text. A malicious user could provide gigabytes of text causing memory exhaustion.

**Fix:**
```python
def _parse_retrospective(self, retrospective: str) -> Dict[str, List[str]]:
    MAX_RETRO_LENGTH = 10000  # 10KB limit

    if not retrospective:
        return {'worked': [], 'didnt_work': [], 'different': []}

    # Truncate if too long
    if len(retrospective) > MAX_RETRO_LENGTH:
        retrospective = retrospective[:MAX_RETRO_LENGTH]
        logger.warning(f"Retrospective truncated to {MAX_RETRO_LENGTH} chars")

    # Continue with parsing...
```

---

#### MEDIUM-2: No Input Validation on task_id
**Location:** Lines 78-88
**Severity:** MEDIUM
**OWASP Category:** Input Validation

**Issue:** `task_id` parameter not validated. Could contain malicious strings, control characters, or extremely long values.

**Fix:**
```python
import re

TASK_ID_PATTERN = re.compile(r'^T-\d{8}-\d{6}-[0-9a-f]{8}$')

def capture_task_completion(self, task_id: str, ...):
    # Validate task ID format
    if not TASK_ID_PATTERN.match(task_id):
        raise ValueError(f"Invalid task_id format: {task_id[:50]}")

    # Continue with capture...
```

---

#### MEDIUM-3: Unsafe Store Operations
**Location:** Lines 174, 273
**Severity:** MEDIUM
**OWASP Category:** Broken Access Control

```python
# VULNERABLE CODE:
self.cycle.store.save(experience)
```

**Issue:** No verification of what's being saved or access control. Could potentially overwrite existing data.

**Fix:**
```python
# Add validation before save
if not experience.id or not experience.context:
    raise ValueError("Invalid experience object")

# Check for existing experience
existing = self.cycle.store.get(experience.id)
if existing and existing.timestamp != experience.timestamp:
    logger.warning(f"Overwriting existing experience: {experience.id}")

self.cycle.store.save(experience)
```

---

#### LOW-1: Hardcoded Efficiency Thresholds
**Location:** Lines 487-518
**Severity:** LOW
**OWASP Category:** Security Misconfiguration

**Issue:** Hardcoded thresholds not configurable, could be abused to game metrics.

**Fix:** Make thresholds configurable via environment or config file.

---

#### LOW-2: Weak Tag Validation
**Location:** Lines 167-171
**Severity:** LOW
**OWASP Category:** Input Validation

**Issue:** Tags not validated, could inject malicious values.

**Fix:**
```python
def _validate_tag(tag: str) -> bool:
    """Validate tag format."""
    if not tag or len(tag) > 200:
        return False
    # Only allow alphanumeric, colon, hyphen, underscore
    return bool(re.match(r'^[a-zA-Z0-9:_-]+$', tag))

# Before adding tags:
for tag in [f"task:{task_id}", f"category:{task_category}", ...]:
    if self._validate_tag(tag):
        experience.tags.add(tag)
```

---

### File 2: `/home/user/Opus-code-test/cortical/got/cli/failure.py`

#### CRITICAL-2: Path Traversal in Failure File Creation
**Location:** Lines 76-78
**Severity:** CRITICAL
**OWASP Category:** Path Traversal

```python
# VULNERABLE CODE:
failure_file = failures_dir / f"{failure_id}.json"
with open(failure_file, 'w') as f:
    json.dump(failure_data, f, indent=2)
```

**Issue:** While `failure_id` is generated by the function, there's no validation on the `got_dir` parameter passed to `_save_failure`.

**Fix:**
```python
def _save_failure(got_dir: Path, ...):
    # Validate got_dir
    got_dir = Path(got_dir).resolve()
    failures_dir = _get_failures_dir(got_dir)

    # Validate failure_id format
    if not re.match(r'^F-\d{8}-\d{6}-[0-9a-f]{8}$', failure_id):
        raise ValueError("Invalid failure_id format")

    failure_file = failures_dir / f"{failure_id}.json"

    # Ensure file is within failures directory
    if not failure_file.resolve().is_relative_to(failures_dir.resolve()):
        raise ValueError("Invalid failure file path")

    with open(failure_file, 'w') as f:
        json.dump(failure_data, f, indent=2)
```

---

#### HIGH-3: Unsafe JSON Deserialization
**Location:** Lines 91-92, 103-104
**Severity:** HIGH
**OWASP Category:** Deserialization of Untrusted Data (A08:2021)

```python
# VULNERABLE CODE:
with open(failure_file, 'r') as f:
    return json.load(f)  # No size limit!
```

**Issue:** Loads JSON files without size limits. An attacker could create a multi-GB JSON file causing memory exhaustion.

**Fix:**
```python
MAX_JSON_SIZE = 10 * 1024 * 1024  # 10MB

def _load_failure(got_dir: Path, failure_id: str) -> Optional[Dict[str, Any]]:
    failures_dir = _get_failures_dir(got_dir)
    failure_file = failures_dir / f"{failure_id}.json"

    if not failure_file.exists():
        return None

    # Check file size before loading
    file_size = failure_file.stat().st_size
    if file_size > MAX_JSON_SIZE:
        logger.error(f"Failure file too large: {file_size} bytes")
        return None

    try:
        with open(failure_file, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in {failure_file}")
        return None
```

---

#### HIGH-4: Information Disclosure in Error Messages
**Location:** Lines 106, 153-160
**Severity:** HIGH
**OWASP Category:** Security Misconfiguration

```python
# VULNERABLE CODE:
print(f"Warning: Could not load {failure_file}: {e}")
```

**Issue:** Reveals internal file paths and error details to users.

**Fix:**
```python
logger.warning(f"Could not load failure file: {e}")
print(f"Warning: Could not load failure record")
```

---

#### MEDIUM-4: No Input Validation on User-Provided Strings
**Location:** Lines 127-131
**Severity:** MEDIUM
**OWASP Category:** Input Validation

```python
# VULNERABLE CODE:
task_id = args.task_id
attempt = args.attempt
error = args.error
```

**Issue:** No validation on length or content of user inputs. Could be extremely long or contain malicious content.

**Fix:**
```python
MAX_TEXT_LENGTH = 10000

def _validate_text_input(text: str, max_length: int = MAX_TEXT_LENGTH) -> str:
    """Validate and sanitize text input."""
    if not text:
        raise ValueError("Input cannot be empty")
    if len(text) > max_length:
        raise ValueError(f"Input exceeds maximum length of {max_length}")
    # Remove null bytes and control characters except newline/tab
    sanitized = ''.join(c for c in text if c == '\n' or c == '\t' or c >= ' ')
    return sanitized

task_id = _validate_text_input(args.task_id, 100)
attempt = _validate_text_input(args.attempt, 5000)
error = _validate_text_input(args.error, 5000)
```

---

#### MEDIUM-5: Reflection Code Execution Risk
**Location:** Lines 166-175
**Severity:** MEDIUM
**OWASP Category:** Code Injection

```python
# VULNERABLE CODE:
import inspect
sig = inspect.signature(manager.add_edge)
```

**Issue:** Uses reflection to detect API capabilities. If `manager` object is compromised, could execute arbitrary code.

**Fix:**
```python
# Use explicit API version check instead of reflection
try:
    # Try calling with validate_refs parameter
    manager.add_edge(
        source_id=failure_id,
        target_id=task_id,
        edge_type="FAILED_ATTEMPT",
        reason=f"Failed attempt: {attempt[:50]}...",
        validate_refs=False,
    )
except TypeError:
    # Fallback to older API
    manager.add_edge(
        source_id=failure_id,
        target_id=task_id,
        edge_type="FAILED_ATTEMPT",
        reason=f"Failed attempt: {attempt[:50]}...",
    )
```

---

#### LOW-3: Timestamp Parsing Without Validation
**Location:** Lines 236-240
**Severity:** LOW
**OWASP Category:** Input Validation

**Issue:** Parses timestamps without validation, could raise unhandled exceptions.

**Fix:** Already has try/except, but should validate format first.

---

### File 3: `/home/user/Opus-code-test/scripts/commit_task_linker.py`

#### CRITICAL-3: Resource Exhaustion via O(n²) Computation
**Location:** Lines 338-363
**Severity:** CRITICAL
**OWASP Category:** Denial of Service

```python
# VULNERABLE CODE:
for commit_hash, commit in self.commits.items():
    for task_id, task in self.tasks.items():
        similarity = compute_semantic_similarity(
            commit.message,
            task.get_text_for_similarity()
        )
```

**Issue:** Nested loop with expensive similarity computation. With 10,000 commits and 1,000 tasks, this performs 10 million operations. No rate limiting or resource constraints.

**Fix:**
```python
def link_semantic_similarity(self, threshold: float = None) -> int:
    threshold = threshold or SEMANTIC_SIMILARITY_THRESHOLD
    logger.info(f"Finding semantic similarities (threshold={threshold})...")

    # Add resource limits
    MAX_COMMITS = 10000
    MAX_TASKS = 5000
    MAX_COMPARISONS = 100000

    if len(self.commits) > MAX_COMMITS:
        logger.warning(f"Too many commits ({len(self.commits)}), limiting to {MAX_COMMITS}")
        commits_to_process = dict(list(self.commits.items())[:MAX_COMMITS])
    else:
        commits_to_process = self.commits

    if len(self.tasks) > MAX_TASKS:
        logger.warning(f"Too many tasks ({len(self.tasks)}), limiting to {MAX_TASKS}")
        tasks_to_process = dict(list(self.tasks.items())[:MAX_TASKS])
    else:
        tasks_to_process = self.tasks

    links_found = 0
    comparisons = 0

    for commit_hash, commit in commits_to_process.items():
        for task_id, task in tasks_to_process.items():
            comparisons += 1
            if comparisons > MAX_COMPARISONS:
                logger.warning(f"Hit comparison limit ({MAX_COMPARISONS}), stopping")
                return links_found

            # Skip if already linked
            if task_id in self._commit_to_tasks[commit_hash]:
                continue

            # Compute similarity
            similarity = compute_semantic_similarity(
                commit.message[:1000],  # Limit message length
                task.get_text_for_similarity()[:1000]
            )

            if similarity >= threshold:
                self._add_link(...)
                links_found += 1

    return links_found
```

---

#### HIGH-5: Unsafe JSON Deserialization
**Location:** Lines 164, 221
**Severity:** HIGH
**OWASP Category:** Deserialization of Untrusted Data

```python
# VULNERABLE CODE:
with open(task_file, 'r', encoding='utf-8') as f:
    task_data = json.load(f)
```

**Issue:** Same as previous files - no size limits on JSON loading.

**Fix:** (Same pattern as File 2, HIGH-3)

---

#### HIGH-6: Information Disclosure in Exported Data
**Location:** Lines 606-627
**Severity:** HIGH
**OWASP Category:** Sensitive Data Exposure

```python
# VULNERABLE CODE:
example = {
    'commit_hash': commit.hash,
    'commit_message': commit.message,  # May contain sensitive info
    'task_id': task.id,
    'task_title': task.title,
    'task_description': task.description,  # May contain sensitive info
    'files_changed': commit.files_changed,  # Reveals internal structure
}
```

**Issue:** Exports potentially sensitive data (commit messages, task descriptions, file paths) without sanitization.

**Fix:**
```python
def _sanitize_for_export(text: str, max_length: int = 500) -> str:
    """Sanitize text for export."""
    if not text:
        return ""

    # Truncate
    text = text[:max_length]

    # Remove potentially sensitive patterns
    # Email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    # API keys (common patterns)
    text = re.sub(r'\b[A-Za-z0-9]{32,}\b', '[KEY]', text)
    # File paths
    text = re.sub(r'/[^\s]+', '[PATH]', text)

    return text

example = {
    'commit_hash': commit.hash[:12],  # Truncate hash
    'commit_message': self._sanitize_for_export(commit.message),
    'task_id': task.id,
    'task_title': self._sanitize_for_export(task.title),
    'task_description': self._sanitize_for_export(task.description),
    'files_changed': [Path(f).name for f in commit.files_changed],  # Only filenames
    'metadata': link.metadata
}
```

---

#### MEDIUM-6: Race Condition in File Operations
**Location:** Lines 234-248
**Severity:** MEDIUM
**OWASP Category:** Race Condition

```python
# VULNERABLE CODE:
self.links_file.parent.mkdir(parents=True, exist_ok=True)
with open(self.links_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2)
```

**Issue:** Non-atomic file write. If process crashes between mkdir and write, file could be corrupted or partially written.

**Fix:**
```python
import tempfile
import shutil

def _save_links(self) -> None:
    """Save links to file atomically."""
    self.links_file.parent.mkdir(parents=True, exist_ok=True)

    data = {
        'version': '1.0.0',
        'generated_at': datetime.now().isoformat(),
        'total_links': len(self.links),
        'links': [link.to_dict() for link in self.links]
    }

    # Write to temporary file first
    temp_fd, temp_path = tempfile.mkstemp(
        dir=self.links_file.parent,
        prefix='.tmp_links_',
        suffix='.json'
    )

    try:
        with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        # Atomic move
        shutil.move(temp_path, self.links_file)
        logger.info(f"Saved {len(self.links)} links to {self.links_file}")
    except Exception:
        # Clean up temp file on error
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise
```

---

#### MEDIUM-7: No Validation on Partial Commit Hashes
**Location:** Lines 519-526
**Severity:** MEDIUM
**OWASP Category:** Input Validation

```python
# VULNERABLE CODE:
matching_hashes = [h for h in self.commits.keys() if h.startswith(commit_hash)]
```

**Issue:** Accepts arbitrary partial hashes without validation. Could match unintended commits.

**Fix:**
```python
def get_links_for_commit(self, commit_hash: str) -> List[Tuple[TaskInfo, CommitTaskLink]]:
    # Validate commit hash format
    if not re.match(r'^[0-9a-f]{6,40}$', commit_hash.lower()):
        raise ValueError(f"Invalid commit hash format: {commit_hash}")

    # Minimum 6 characters for partial hash
    if len(commit_hash) < 6:
        raise ValueError("Commit hash must be at least 6 characters")

    # Support partial commit hashes
    matching_hashes = [
        h for h in self.commits.keys()
        if h.lower().startswith(commit_hash.lower())
    ]

    if len(matching_hashes) > 1:
        raise ValueError(f"Ambiguous commit hash: {commit_hash} matches {len(matching_hashes)} commits")

    if not matching_hashes:
        return []

    # Continue with processing...
```

---

#### MEDIUM-8: Memory Exhaustion via Large Link Lists
**Location:** Lines 136, 223
**Severity:** MEDIUM
**OWASP Category:** Resource Exhaustion

**Issue:** No limits on number of links stored in memory.

**Fix:**
```python
MAX_LINKS = 1000000  # 1 million links

def _add_link(self, ...):
    if len(self.links) >= MAX_LINKS:
        raise RuntimeError(f"Maximum link count ({MAX_LINKS}) exceeded")
    # Continue...
```

---

#### LOW-4: Insecure Regex Pattern
**Location:** Line 47
**Severity:** LOW
**OWASP Category:** Regular Expression Denial of Service (ReDoS)

**Issue:** Current pattern is safe, but could be DoS vector if made more complex.

**Fix:** Already safe, document why.

---

#### LOW-5: No Rate Limiting on CLI Operations
**Location:** Lines 634-793
**Severity:** LOW
**OWASP Category:** Missing Rate Limiting

**Issue:** CLI can be called repeatedly without limits.

**Fix:** Add cooldown file or rate limiting mechanism.

---

### File 4: `/home/user/Opus-code-test/scripts/training_data_exporter.py`

#### HIGH-7: Sensitive Data in Training Export
**Location:** Lines 621-627
**Severity:** HIGH
**OWASP Category:** Sensitive Data Exposure

```python
# VULNERABLE CODE:
with open(output_file, 'w', encoding='utf-8') as f:
    for example in training_examples:
        f.write(json.dumps(example) + '\n')
```

**Issue:** Exports all task/decision data to JSONL without sanitization. Could include passwords, API keys, or PII in task descriptions.

**Fix:**
```python
def _sanitize_training_example(example: Dict[str, Any]) -> Dict[str, Any]:
    """Remove sensitive data from training examples."""
    sanitized = example.copy()

    # Sanitize text fields
    text_fields = ['commit_message', 'task_title', 'task_description', 'rationale']
    for field in text_fields:
        if field in sanitized:
            sanitized[field] = self._sanitize_for_export(sanitized[field])

    # Remove file paths
    if 'files_changed' in sanitized:
        sanitized['files_changed'] = [Path(f).name for f in sanitized['files_changed']]

    return sanitized

# In export method:
with open(output_file, 'w', encoding='utf-8') as f:
    for example in training_examples:
        sanitized = self._sanitize_training_example(example)
        f.write(json.dumps(sanitized) + '\n')
```

---

#### MEDIUM-9: Unbounded Entity Cache
**Location:** Line 266
**Severity:** MEDIUM
**OWASP Category:** Resource Exhaustion

```python
# VULNERABLE CODE:
self._entity_cache: Dict[str, Dict[str, Any]] = {}
```

**Issue:** Loads all entities into memory without limits. With thousands of tasks, could exhaust memory.

**Fix:**
```python
MAX_CACHE_SIZE = 100000  # 100k entities
MAX_ENTITY_SIZE = 1_000_000  # 1MB per entity

def _load_entities(self):
    logger.info(f"Loading entities from {self.got_dir}")

    entity_count = 0
    for entity_file in self.got_dir.glob("*.json"):
        # Check cache size limit
        if len(self._entity_cache) >= MAX_CACHE_SIZE:
            logger.warning(f"Hit cache size limit ({MAX_CACHE_SIZE}), stopping load")
            break

        # Check file size
        file_size = entity_file.stat().st_size
        if file_size > MAX_ENTITY_SIZE:
            logger.warning(f"Skipping large entity file: {entity_file.name} ({file_size} bytes)")
            continue

        try:
            with open(entity_file, 'r', encoding='utf-8') as f:
                entity = json.load(f)

            if isinstance(entity, dict) and 'data' in entity:
                entity_data = entity['data']
                entity_id = entity_data.get('id', entity_file.stem)
                self._entity_cache[entity_id] = entity_data
                entity_count += 1

        except (json.JSONDecodeError, KeyError, IOError) as e:
            logger.warning(f"Failed to load {entity_file.name}: {e}")

    logger.info(f"Loaded {entity_count} entities (cache size: {len(self._entity_cache)})")
```

---

#### MEDIUM-10: Quality Score Algorithm Can Be Gamed
**Location:** Lines 303-334
**Severity:** MEDIUM
**OWASP Category:** Business Logic Bypass

```python
# VULNERABLE CODE:
def _calculate_quality_score(self, text: str, min_length: int = 50) -> float:
    # Simple heuristics that can be gamed
    sentences = text.count('.') + text.count('!') + text.count('?')
```

**Issue:** Quality score can be artificially inflated by adding periods or repeated words.

**Fix:**
```python
def _calculate_quality_score(self, text: str, min_length: int = 50) -> float:
    """Calculate quality score with anti-gaming measures."""
    if not text or not text.strip():
        return 0.0

    text = text.strip()
    length = len(text)

    # Detect gaming attempts
    # Check for repeated characters
    if len(set(text)) < len(text) * 0.1:  # Less than 10% unique chars
        return 0.0

    # Check for excessive punctuation
    punct_ratio = sum(text.count(c) for c in '.!?') / max(len(text), 1)
    if punct_ratio > 0.1:  # More than 10% punctuation
        return max(0.3, min(0.7, 1.0 - punct_ratio))

    # Length score (with diminishing returns)
    length_score = min(1.0, length / (min_length * 3))

    # Structure score
    sentences = min(20, text.count('.') + text.count('!') + text.count('?'))
    structure_score = min(1.0, sentences / 5)

    # Information density (unique words / total words)
    words = text.lower().split()
    if not words:
        return 0.0

    unique_words = len(set(words))
    total_words = len(words)

    # Penalize repetitive text
    if unique_words < total_words * 0.3:  # Less than 30% unique
        return 0.2

    density_score = unique_words / total_words

    # Weighted average
    quality = (length_score * 0.3 + structure_score * 0.3 + density_score * 0.4)

    return round(quality, 2)
```

---

#### LOW-6: Markdown Injection in Summary
**Location:** Lines 676-679
**Severity:** LOW
**OWASP Category:** Injection

**Issue:** Writes user data to markdown without escaping. Could inject malicious markdown.

**Fix:**
```python
def _escape_markdown(text: str) -> str:
    """Escape markdown special characters."""
    special_chars = ['*', '_', '[', ']', '(', ')', '#', '`']
    for char in special_chars:
        text = text.replace(char, '\\' + char)
    return text

# In _write_markdown_summary:
f.write(f"- **{self._escape_markdown(d.title)}** (Q={d.quality_score:.2f})\n")
```

---

#### LOW-7: Verbose Error Messages
**Location:** Line 285
**Severity:** LOW
**OWASP Category:** Information Disclosure

**Fix:** Same pattern as other files - sanitize error messages.

---

### File 5: `/home/user/Opus-code-test/scripts/auto_learning_capture.py`

#### HIGH-8: Unbounded File Globbing
**Location:** Line 94
**Severity:** HIGH
**OWASP Category:** Resource Exhaustion

```python
# VULNERABLE CODE:
for task_file in self.entities_dir.glob("T-*.json"):
```

**Issue:** Globs all matching files without limit. With millions of tasks, could exhaust resources.

**Fix:**
```python
def get_recent_tasks(self, days: int = 1, status_filter: Optional[List[str]] = None):
    if not self.entities_dir.exists():
        logger.warning(f"Entities directory not found: {self.entities_dir}")
        return []

    cutoff_time = datetime.now(timezone.utc) - timedelta(days=days)
    recent_tasks = []

    # Limit number of files scanned
    MAX_FILES_TO_SCAN = 100000
    files_scanned = 0

    for task_file in self.entities_dir.glob("T-*.json"):
        files_scanned += 1
        if files_scanned > MAX_FILES_TO_SCAN:
            logger.warning(f"Hit scan limit ({MAX_FILES_TO_SCAN} files), stopping")
            break

        # Check file size before opening
        if task_file.stat().st_size > 1_000_000:  # 1MB
            logger.warning(f"Skipping large task file: {task_file.name}")
            continue

        # Continue with processing...
```

---

#### MEDIUM-11: Timestamp Parsing Without Validation
**Location:** Lines 115-118
**Severity:** MEDIUM
**OWASP Category:** Input Validation

**Issue:** Parses timestamps from user data without validation.

**Fix:**
```python
def _safe_parse_timestamp(ts_str: Optional[str]) -> Optional[datetime]:
    """Safely parse ISO timestamp."""
    if not ts_str:
        return None

    # Validate format first
    if not re.match(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', ts_str):
        return None

    try:
        return datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
    except (ValueError, AttributeError):
        return None

# Usage:
updated_at = self._safe_parse_timestamp(updated_at_str)
if not updated_at:
    continue
```

---

#### LOW-8: Unconstrained Days Parameter
**Location:** Line 90
**Severity:** LOW
**OWASP Category:** Business Logic

**Issue:** Accepts arbitrary `days` parameter. Could scan years of data.

**Fix:**
```python
MAX_DAYS_LOOKBACK = 365  # 1 year maximum

def get_recent_tasks(self, days: int = 1, ...):
    if days < 1:
        raise ValueError("days must be positive")
    if days > MAX_DAYS_LOOKBACK:
        raise ValueError(f"days cannot exceed {MAX_DAYS_LOOKBACK}")
    # Continue...
```

---

### File 6: `/home/user/Opus-code-test/.claude/hooks/session_learning_capture.py`

#### MEDIUM-12: Unsafe Task Metadata Loading
**Location:** Lines 218-219
**Severity:** MEDIUM
**OWASP Category:** Path Traversal

```python
# VULNERABLE CODE:
task_file = entities_dir / f"{task_id}.json"
```

**Issue:** Constructs file path from user input without validation.

**Fix:**
```python
TASK_ID_PATTERN = re.compile(r'^T-\d{8}-\d{6}-[0-9a-f]{8}$')

def load_task_metadata(task_id: str) -> Optional[Dict[str, Any]]:
    # Validate task ID format
    if not TASK_ID_PATTERN.match(task_id):
        logger.error(f"Invalid task_id format: {task_id}")
        return None

    try:
        entities_dir = GOT_DIR / "entities"
        task_file = entities_dir / f"{task_id}.json"

        # Verify path is within entities directory
        if not task_file.resolve().is_relative_to(entities_dir.resolve()):
            logger.error(f"Invalid task file path: {task_file}")
            return None

        # Continue with loading...
```

---

#### LOW-9: Verbose Exception Logging
**Location:** Line 247
**Severity:** LOW
**OWASP Category:** Information Disclosure

**Fix:** Sanitize exception messages before logging.

---

## Summary of Recommendations

### Immediate Actions (Critical/High)

1. **Add Path Validation:** Implement strict path validation in all file operations
2. **Limit JSON Size:** Add MAX_JSON_SIZE checks before all `json.load()` calls
3. **Rate Limit Computations:** Add MAX_COMPARISONS limit to O(n²) operations
4. **Sanitize Exports:** Remove sensitive data from all exported training data
5. **Validate IDs:** Use regex patterns to validate all task/failure/commit IDs

### Short-Term Actions (Medium)

1. **Atomic File Writes:** Use temp files + atomic moves for all file writes
2. **Input Length Limits:** Add MAX_LENGTH validation to all text inputs
3. **Resource Limits:** Implement MAX_CACHE_SIZE, MAX_FILES limits
4. **Quality Score Hardening:** Add anti-gaming measures to quality calculations
5. **Tag Validation:** Validate tag format before adding to experiences

### Long-Term Actions (Low)

1. **Centralized Sanitization:** Create shared sanitization utilities
2. **Security Logging:** Implement redaction for all sensitive IDs in logs
3. **Rate Limiting:** Add CLI rate limiting to prevent abuse
4. **Markdown Escaping:** Escape user content in markdown generation
5. **Error Message Sanitization:** Standardize error message handling

---

## Security Testing Recommendations

1. **Fuzzing:** Test all file path inputs with path traversal sequences
2. **Load Testing:** Send large JSON files to test size limits
3. **Memory Profiling:** Verify cache limits work under load
4. **Injection Testing:** Test SQL-like injection in task descriptions
5. **Export Analysis:** Scan exported training data for leaked secrets

---

## Compliance Notes

These vulnerabilities map to:
- **OWASP Top 10 2021:** A01 (Broken Access Control), A03 (Injection), A08 (Deserialization)
- **CWE:** CWE-22 (Path Traversal), CWE-502 (Deserialization), CWE-400 (Resource Exhaustion)
- **NIST:** SC-5 (Denial of Service Protection), SI-10 (Information Input Validation)

---

**End of Report**
