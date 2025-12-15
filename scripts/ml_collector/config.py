"""
Configuration module for ML Data Collector

Contains paths, schemas, milestones, and redaction patterns.
"""

import re
from pathlib import Path
from typing import Dict, List


# ============================================================================
# PATHS
# ============================================================================

ML_DATA_DIR = Path(".git-ml")
COMMITS_DIR = ML_DATA_DIR / "commits"           # Full commit data with diffs (large, local only)
COMMITS_LITE_DIR = ML_DATA_DIR / "commits-lite" # Legacy: individual JSON files (being phased out)
SESSIONS_DIR = ML_DATA_DIR / "sessions"         # Full session data (local only)
CHATS_DIR = ML_DATA_DIR / "chats"
ACTIONS_DIR = ML_DATA_DIR / "actions"
SHARED_DIR = ML_DATA_DIR / "shared"  # Aggregated patterns - safe to commit
GITHUB_DIR = ML_DATA_DIR / "github"  # PR/Issue data - local (contains discussions)
CURRENT_SESSION_FILE = ML_DATA_DIR / "current_session.json"

# Git-tracked JSONL files (single file, append-only, git-friendly)
TRACKED_DIR = ML_DATA_DIR / "tracked"           # Directory for git-tracked data
COMMITS_LITE_FILE = TRACKED_DIR / "commits.jsonl"   # Commit metadata (one per line)
SESSIONS_LITE_FILE = TRACKED_DIR / "sessions.jsonl" # Session summaries (one per line)

# Training milestones
MILESTONES = {
    "file_prediction": {"commits": 500, "sessions": 100, "chats": 200},
    "commit_messages": {"commits": 2000, "sessions": 500, "chats": 1000},
    "code_suggestions": {"commits": 5000, "sessions": 2000, "chats": 5000},
}

# Data retention configuration (days)
# 2 years - enough time to hit training milestones at typical dev pace
# (~66 commits/active day observed, need 5000 for code suggestions)
DEFAULT_RETENTION_DAYS = 730
CONSENT_FILE = ML_DATA_DIR / "contribution_consent.json"


# ============================================================================
# SCHEMAS
# ============================================================================

COMMIT_SCHEMA = {
    "required": ["hash", "message", "author", "timestamp", "branch", "files_changed",
                 "insertions", "deletions", "hunks", "hour_of_day", "day_of_week"],
    "types": {
        "hash": str, "message": str, "author": str, "timestamp": str, "branch": str,
        "files_changed": list, "insertions": int, "deletions": int, "hunks": list,
        "hour_of_day": int, "day_of_week": str, "is_merge": bool, "is_initial": bool,
        "parent_count": int, "session_id": (str, type(None)), "related_chats": list,
    }
}

CHAT_SCHEMA = {
    "required": ["id", "timestamp", "session_id", "query", "response",
                 "files_referenced", "files_modified", "tools_used"],
    "types": {
        "id": str, "timestamp": str, "session_id": str, "query": str, "response": str,
        "files_referenced": list, "files_modified": list, "tools_used": list,
        "query_tokens": int, "response_tokens": int,
        "user_feedback": (dict, str, type(None)),  # Can be dict, legacy string, or None
    }
}

ACTION_SCHEMA = {
    "required": ["id", "timestamp", "session_id", "action_type", "target"],
    "types": {
        "id": str, "timestamp": str, "session_id": str,
        "action_type": str, "target": str, "success": bool,
    }
}


# ============================================================================
# REDACTION PATTERNS
# ============================================================================

# Sensitive data patterns to redact before storage
# These patterns are applied to query/response text before saving
REDACTION_PATTERNS = [
    # API keys and tokens
    (r'(?i)(api[_-]?key|apikey|api_secret|secret[_-]?key)\s*[=:]\s*["\']?[\w\-]{20,}["\']?', r'\1=<REDACTED>'),
    (r'(?i)(token|bearer|authorization)\s*[=:]\s*["\']?[\w\-\.]{20,}["\']?', r'\1=<REDACTED>'),
    # Passwords and secrets
    (r'(?i)(password|passwd|pwd|secret)\s*[=:]\s*["\']?[^\s"\']{8,}["\']?', r'\1=<REDACTED>'),
    # AWS credentials
    (r'(?i)(aws[_-]?access[_-]?key[_-]?id|aws[_-]?secret[_-]?access[_-]?key)\s*[=:]\s*["\']?[\w\+/]{16,}["\']?', r'\1=<REDACTED>'),
    (r'AKIA[0-9A-Z]{16}', '<AWS_KEY_REDACTED>'),
    # Private keys
    (r'-----BEGIN\s+(RSA\s+)?PRIVATE KEY-----[\s\S]*?-----END\s+(RSA\s+)?PRIVATE KEY-----', '<PRIVATE_KEY_REDACTED>'),
    (r'-----BEGIN\s+OPENSSH\s+PRIVATE KEY-----[\s\S]*?-----END\s+OPENSSH\s+PRIVATE KEY-----', '<SSH_KEY_REDACTED>'),
    # Database connection strings
    (r'(?i)(mongodb|postgresql|mysql|redis)://[^\s"\']+', r'\1://<CONNECTION_REDACTED>'),
    # Generic credentials in URLs
    (r'://[^:]+:[^@]+@', '://<CREDENTIALS>@'),
    # GitHub tokens
    (r'ghp_[a-zA-Z0-9]{36}', '<GITHUB_TOKEN_REDACTED>'),
    (r'gho_[a-zA-Z0-9]{36}', '<GITHUB_OAUTH_REDACTED>'),
    # Slack tokens
    (r'xox[baprs]-[0-9a-zA-Z\-]+', '<SLACK_TOKEN_REDACTED>'),
    # JWT tokens (basic pattern)
    (r'eyJ[a-zA-Z0-9_-]*\.eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*', '<JWT_REDACTED>'),
]


# ============================================================================
# VALIDATION
# ============================================================================

def validate_schema(data: dict, schema: dict, data_type: str) -> List[str]:
    """Validate data against a schema. Returns list of errors (empty if valid)."""
    errors = []
    for field in schema["required"]:
        if field not in data:
            errors.append(f"{data_type}: missing required field '{field}'")
    for field, expected in schema["types"].items():
        if field in data and data[field] is not None:
            if isinstance(expected, tuple):
                if not isinstance(data[field], expected):
                    errors.append(f"{data_type}: field '{field}' has wrong type")
            elif not isinstance(data[field], expected):
                errors.append(f"{data_type}: field '{field}' has type {type(data[field]).__name__}, expected {expected.__name__}")
    return errors


# ============================================================================
# REDACTION
# ============================================================================

def redact_sensitive_data(text: str) -> str:
    """Redact sensitive data patterns from text before storage.

    Args:
        text: The text to redact

    Returns:
        Text with sensitive patterns replaced with redaction markers
    """
    if not text:
        return text

    redacted = text
    for pattern, replacement in REDACTION_PATTERNS:
        redacted = re.sub(pattern, replacement, redacted)
    return redacted


def count_redactions(original: str, redacted: str) -> int:
    """Count how many redactions were made."""
    if not original or not redacted:
        return 0
    # Count occurrences of redaction markers
    markers = ['<REDACTED>', '<AWS_KEY_REDACTED>', '<PRIVATE_KEY_REDACTED>',
               '<SSH_KEY_REDACTED>', '<CONNECTION_REDACTED>', '<CREDENTIALS>',
               '<GITHUB_TOKEN_REDACTED>', '<GITHUB_OAUTH_REDACTED>',
               '<SLACK_TOKEN_REDACTED>', '<JWT_REDACTED>']
    return sum(redacted.count(m) for m in markers)
