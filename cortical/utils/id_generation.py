"""
Canonical ID generation utilities for all GoT and task systems.

All IDs use format: {PREFIX}-YYYYMMDD-HHMMSS-XXXXXXXX
- PREFIX: T (task), D (decision), E (edge), S (sprint), G (goal)
- Timestamp: UTC timezone for consistency
- Suffix: 8 hex characters (4 bytes = ~4 billion unique values)

This is the single source of truth for ID generation across:
- cortical/got/api.py (GoT transaction system)
- scripts/got_utils.py (CLI utilities)
- scripts/task_utils.py (task management)

Examples:
    >>> generate_task_id()
    'T-20251222-143052-a1b2c3d4'

    >>> generate_decision_id()
    'D-20251222-143052-e5f6g7h8'

    >>> generate_sprint_id(number=5)
    'S-005'

    >>> normalize_id('task:T-20251222-143052-a1b2c3d4')
    'T-20251222-143052-a1b2c3d4'
"""

import secrets
from datetime import datetime, timezone
from typing import Optional


def generate_task_id() -> str:
    """
    Generate unique task ID.

    Format: T-YYYYMMDD-HHMMSS-XXXXXXXX where XXXXXXXX is random hex.

    Returns:
        Task ID string (e.g., 'T-20251222-143052-a1b2c3d4')

    Note:
        - Uses UTC timezone for consistency
        - Random suffix provides ~4 billion unique values
        - No 'task:' prefix (that's legacy format)
    """
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    suffix = secrets.token_hex(4)  # 8 hex chars
    return f"T-{timestamp}-{suffix}"


def generate_decision_id() -> str:
    """
    Generate unique decision ID.

    Format: D-YYYYMMDD-HHMMSS-XXXXXXXX where XXXXXXXX is random hex.

    Returns:
        Decision ID string (e.g., 'D-20251222-143052-e5f6g7h8')

    Note:
        - Uses UTC timezone for consistency
        - Random suffix provides ~4 billion unique values
        - No 'decision:' prefix (that's legacy format)
    """
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    suffix = secrets.token_hex(4)  # 8 hex chars
    return f"D-{timestamp}-{suffix}"


def generate_edge_id() -> str:
    """
    Generate unique edge ID.

    Format: E-YYYYMMDD-HHMMSS-XXXXXXXX where XXXXXXXX is random hex.

    Returns:
        Edge ID string (e.g., 'E-20251222-143052-i9j0k1l2')

    Note:
        - Uses UTC timezone for consistency
        - Random suffix provides ~4 billion unique values
    """
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    suffix = secrets.token_hex(4)  # 8 hex chars
    return f"E-{timestamp}-{suffix}"


def generate_sprint_id(number: Optional[int] = None) -> str:
    """
    Generate sprint ID.

    Args:
        number: Optional sprint number. If provided, uses format S-NNN.
                If None, uses current year-month format S-YYYY-MM.

    Returns:
        Sprint ID string (e.g., 'S-005' or 'S-2025-12')

    Examples:
        >>> generate_sprint_id(number=5)
        'S-005'

        >>> generate_sprint_id()  # Current month
        'S-2025-12'
    """
    if number is not None:
        return f"S-{number:03d}"
    return f"S-{datetime.now(timezone.utc).strftime('%Y-%m')}"


def generate_goal_id() -> str:
    """
    Generate goal ID.

    Format: G-YYYYMMDD-XXXXXXXX where XXXXXXXX is random hex.

    Returns:
        Goal ID string (e.g., 'G-20251222-m3n4o5p6')

    Note:
        - Uses UTC timezone for consistency
        - No hour/minute/second (goals are day-level granularity)
    """
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%d")
    suffix = secrets.token_hex(4)  # 8 hex chars
    return f"G-{timestamp}-{suffix}"


def normalize_id(id_str: str) -> str:
    """
    Remove legacy prefixes from IDs.

    Legacy format used prefixes like 'task:', 'decision:', etc.
    This function strips them for backward compatibility.

    Args:
        id_str: ID string (may have legacy prefix)

    Returns:
        Normalized ID without prefix

    Examples:
        >>> normalize_id('task:T-20251222-143052-a1b2c3d4')
        'T-20251222-143052-a1b2c3d4'

        >>> normalize_id('T-20251222-143052-a1b2c3d4')
        'T-20251222-143052-a1b2c3d4'
    """
    for prefix in ('task:', 'decision:', 'edge:', 'sprint:', 'goal:'):
        if id_str.startswith(prefix):
            return id_str[len(prefix):]
    return id_str
