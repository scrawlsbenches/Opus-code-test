"""
Pattern definitions for comment classification.

This module contains all pattern constants used for classifying comments
as misleading, accurate, or noise. Patterns are tuples of (regex, category).

Categories help with:
- Debugging classification decisions
- Generating statistics about pattern matches
- Fine-tuning classification rules
"""

from typing import List, Tuple

# ==============================================================================
# MISLEADING PATTERNS
# ==============================================================================
# Comments that are speculative, outdated, or potentially incorrect

MISLEADING_PATTERNS: List[Tuple[str, str]] = [
    # Speculative "will be" patterns - promises that may never be fulfilled
    (r'will be implemented', 'speculative'),
    (r'will be added', 'speculative'),
    (r'will be handled', 'speculative'),
    (r'will be replaced', 'speculative'),
    (r'will be done', 'speculative'),
    (r'will be fixed', 'speculative'),

    # FUTURE markers - often become stale
    (r'^FUTURE:', 'future_marker'),
    (r'FUTURE\s*when', 'future_marker'),

    # "When X is implemented" patterns - conditional promises
    (r'when .* is implemented', 'speculative'),
    (r'when .* is ready', 'speculative'),
    (r'when .* is done', 'speculative'),
    (r'when feature is', 'speculative'),

    # Placeholder/stub markers - incomplete implementations
    (r'placeholder', 'placeholder'),
    (r'\bstub\b', 'placeholder'),
    (r'not implemented yet', 'placeholder'),

    # Vague future references - undefined timeline
    (r'eventually', 'vague'),
    (r'someday', 'vague'),
    (r'in the future', 'vague'),
    (r'later we', 'vague'),
    (r'planned to', 'vague'),

    # Potentially stale documentation references
    (r'See:.*\.md', 'doc_reference'),
    (r'see docs/', 'doc_reference'),
]


# ==============================================================================
# ACCURATE PATTERNS
# ==============================================================================
# Comments that are factual, documented behavior

ACCURATE_PATTERNS: List[Tuple[str, str]] = [
    # Return value documentation - factual statements about behavior
    (r'^Returns?\s+', 'returns'),
    (r'^Returns:\s+', 'returns'),
    (r'returns\s+(True|False|None|the|a|an)\b', 'returns'),

    # Parameter documentation
    (r'^Args?:\s*', 'args'),
    (r'^Parameters?:\s*', 'args'),
    (r'^Params?:\s*', 'args'),

    # Exception documentation
    (r'^Raises?\s+', 'raises'),
    (r'^Raises:\s+', 'raises'),
    (r'raises\s+(ValueError|TypeError|KeyError|RuntimeError)', 'raises'),

    # Implementation facts - statements about what code does
    (r'^This (is|uses|implements|creates|computes|validates)', 'implementation'),
    (r'^Implements\s+', 'implementation'),
    (r'^Uses\s+', 'implementation'),
    (r'^Creates\s+', 'implementation'),

    # Complexity/performance notes - measurable facts
    (r'O\([nN1]\)', 'complexity'),
    (r'O\(n\s*(log\s*n)?\)', 'complexity'),
    (r'runs in O\(', 'complexity'),
    (r'time complexity', 'complexity'),

    # Type annotations
    (r'^type:\s*', 'type_hint'),

    # Factual notes
    (r'^NOTE:\s+\w', 'note'),
    (r'^IMPORTANT:\s+', 'note'),

    # Valid TODOs with specific actions (capitalized = actionable)
    (r'^TODO:\s+[A-Z]', 'todo'),
    (r'^FIXME:\s+[A-Z]', 'todo'),
]


# ==============================================================================
# EXCLUDE PATTERNS
# ==============================================================================
# Comments that should be excluded from training (noise)

EXCLUDE_PATTERNS: List[str] = [
    r'^-+$',            # Separator lines: ----
    r'^=+$',            # Separator lines: ====
    r'^\s*$',           # Empty or whitespace only
    r'^#\s*$',          # Just hash
    r'^\d+$',           # Just numbers
    r'^[a-z]$',         # Single letter
    r'^type:\s*ignore', # Type ignore comments
    r'^noqa',           # Linter ignores
    r'^pylint',         # Linter directives
    r'^pragma',         # Pragma directives
]


# ==============================================================================
# SUSPICIOUS PATTERNS (for Bloom filter pre-screening)
# ==============================================================================
# Lower-cased phrases that suggest a comment might be misleading

SUSPICIOUS_PATTERNS: List[str] = [
    "will be implemented",
    "will be done",
    "will be replaced",
    "will be handled",
    "when cdg index is implemented",
    "when feature is ready",
    "placeholder",
    "stub",
    "not implemented yet",
    "coming soon",
    "tbd",
    "fixme later",
    "hack",
    "temporary fix",
    "workaround",
]


# ==============================================================================
# COMMENT MARKERS (for Trie indexing)
# ==============================================================================
# Standard comment markers to index and track

COMMENT_MARKERS: List[str] = [
    "FUTURE:",
    "TODO:",
    "FIXME:",
    "HACK:",
    "XXX:",
    "NOTE:",
    "WARNING:",
    "BUG:",
    "OPTIMIZE:",
    "REFACTOR:",
]
