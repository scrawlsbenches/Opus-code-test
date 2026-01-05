"""
Natural Language Query Translator for GoT Expression DSL.

Translates legacy natural language query patterns to expression DSL syntax,
enabling backward compatibility while leveraging the new expression system.

Design Reference: docs/design/got-query-audit-and-design.md Section 1.5
"""

import re
from typing import List


def translate(query: str) -> str:
    """
    Translate a natural language query to expression DSL syntax.

    Args:
        query: Natural language query string

    Returns:
        Expression DSL string that can be parsed and executed

    Examples:
        >>> translate("blocked tasks")
        'blocked()'

        >>> translate("high priority pending")
        "priority = 'high' AND status = 'pending'"

        >>> translate("what blocks T-001")
        "blockers('T-001')"

        >>> translate("orphan tasks")
        'orphan_nodes()'

    Pattern Matching:
        - Case-insensitive matching
        - Whitespace normalization
        - Unknown patterns pass through unchanged
    """
    # Normalize input: strip and collapse whitespace
    query = query.strip()
    query = re.sub(r'\s+', ' ', query)

    # Preserve original for ID extraction (case-sensitive)
    original_query = query
    query_lower = query.lower()

    # =========================================================================
    # Parameterized Patterns (must be checked before static patterns)
    # =========================================================================

    # "what blocks T-XXX" → blockers('T-XXX')
    if query_lower.startswith("what blocks "):
        task_id = original_query[12:].strip()
        return f"blockers('{task_id}')"

    # "what depends on T-XXX" → dependents('T-XXX')
    if query_lower.startswith("what depends on "):
        task_id = original_query[16:].strip()
        return f"dependents('{task_id}')"

    # "tasks in sprint S-XXX" → in_sprint('S-XXX')
    if query_lower.startswith("tasks in sprint "):
        sprint_id = original_query[16:].strip()
        return f"in_sprint('{sprint_id}')"

    # "tasks in S-XXX" (short form) → in_sprint('S-XXX')
    if query_lower.startswith("tasks in "):
        # Check if it looks like a sprint ID
        potential_id = original_query[9:].strip()
        if potential_id.startswith("S-"):
            return f"in_sprint('{potential_id}')"

    # "relationships T-XXX" → connected_to('T-XXX')
    if query_lower.startswith("relationships "):
        entity_id = original_query[14:].strip()
        return f"connected_to('{entity_id}')"

    # =========================================================================
    # Static Patterns
    # =========================================================================

    # Status Queries
    # -------------------------------------------------------------------------
    if query_lower == "blocked tasks":
        return "blocked()"

    if query_lower == "pending tasks":
        return "status = 'pending'"

    if query_lower == "active tasks":
        return "status = 'in_progress'"

    if query_lower == "completed tasks":
        return "status = 'completed'"

    if query_lower in ("in_progress tasks", "in progress tasks"):
        return "status = 'in_progress'"

    if query_lower == "all tasks":
        # Return empty string - this means "no filter"
        return ""

    # Priority Queries
    # -------------------------------------------------------------------------
    if query_lower == "high priority tasks":
        return "priority = 'high'"

    if query_lower == "critical tasks":
        return "priority = 'critical'"

    # Compound Priority + Status
    if query_lower == "high priority pending":
        return "priority = 'high' AND status = 'pending'"

    # Graph Queries
    # -------------------------------------------------------------------------
    if query_lower in ("orphan tasks", "orphan nodes", "orphans"):
        return "orphan_nodes()"

    # Time Queries
    # -------------------------------------------------------------------------
    if query_lower in ("recent tasks", "tasks today"):
        return "recent(1)"

    if query_lower == "stale tasks":
        return "stale(7)"

    # Entity Type Queries
    # -------------------------------------------------------------------------
    if query_lower in ("decisions", "all decisions"):
        return "entity_type('decision')"

    if query_lower in ("sprints", "all sprints"):
        return "entity_type('sprint')"

    if query_lower in ("current sprint", "active sprint"):
        return "entity_type('sprint') AND status = 'in_progress'"

    # =========================================================================
    # Unknown Pattern - Pass Through
    # =========================================================================
    # If no pattern matches, return the original query unchanged
    # This allows direct DSL expressions to work
    return original_query


def get_supported_patterns() -> List[str]:
    """
    Return a list of supported natural language query patterns.

    Returns:
        List of pattern strings describing supported queries

    Example:
        >>> patterns = get_supported_patterns()
        >>> "blocked tasks" in patterns
        True
    """
    return [
        # Status queries
        "blocked tasks",
        "pending tasks",
        "active tasks",
        "completed tasks",
        "in_progress tasks",
        "all tasks",

        # Priority queries
        "high priority tasks",
        "critical tasks",
        "high priority pending",

        # Graph queries
        "orphan tasks",
        "orphan nodes",
        "orphans",

        # Time queries
        "recent tasks",
        "tasks today",
        "stale tasks",

        # Entity queries
        "decisions",
        "all decisions",
        "sprints",
        "all sprints",
        "current sprint",
        "active sprint",

        # Parameterized patterns (with placeholders)
        "what blocks <task_id>",
        "what depends on <task_id>",
        "tasks in sprint <sprint_id>",
        "tasks in <sprint_id>",
        "relationships <entity_id>",
    ]
