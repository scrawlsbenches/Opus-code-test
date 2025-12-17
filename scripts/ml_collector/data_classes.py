"""
Data classes for ML Data Collector

Contains all dataclass definitions for structured data.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class TranscriptExchange:
    """A single query/response exchange extracted from a transcript."""
    query: str
    response: str
    tools_used: List[str]
    tool_inputs: List[Dict]
    tool_outputs: List[Dict]  # Tool results with output, success status
    timestamp: str
    thinking: Optional[str] = None


@dataclass
class DiffHunk:
    """A single diff hunk from a commit."""
    file: str
    function: Optional[str]  # Function/class containing the change
    change_type: str  # add, modify, delete, rename
    start_line: int
    lines_added: List[str]
    lines_removed: List[str]
    context_before: List[str]
    context_after: List[str]


@dataclass
class CommitContext:
    """Rich context captured at commit time."""
    # Git metadata
    hash: str
    message: str
    author: str
    timestamp: str
    branch: str

    # Files changed
    files_changed: List[str]
    insertions: int
    deletions: int

    # Diff structure
    hunks: List[Dict]

    # Temporal context
    hour_of_day: int
    day_of_week: str
    seconds_since_last_commit: Optional[int]

    # Commit type detection
    is_merge: bool = False
    is_initial: bool = False
    parent_count: int = 1

    # Session context (if available)
    session_id: Optional[str] = None
    related_chats: List[str] = field(default_factory=list)

    # Outcome tracking (filled in later)
    ci_result: Optional[str] = None
    reverted: bool = False
    amended: bool = False


@dataclass
class ChatEntry:
    """A query/response pair from a chat session."""
    id: str
    timestamp: str
    session_id: str

    # The conversation
    query: str
    response: str

    # Context
    files_referenced: List[str]
    files_modified: List[str]
    tools_used: List[str]
    tool_outputs: List[Dict] = field(default_factory=list)  # Tool results with output, success status

    # Outcome
    user_feedback: Optional[str] = None  # positive, negative, neutral
    resulted_in_commit: bool = False
    related_commit: Optional[str] = None

    # Metadata
    query_tokens: int = 0
    response_tokens: int = 0
    duration_seconds: Optional[float] = None


@dataclass
class ActionEntry:
    """A discrete action taken during development."""
    id: str
    timestamp: str
    session_id: str

    action_type: str  # search, read, edit, test, commit, etc.
    target: str  # file path, query string, etc.

    # Context
    context: Dict[str, Any] = field(default_factory=dict)

    # Outcome
    success: bool = True
    result_summary: Optional[str] = None
