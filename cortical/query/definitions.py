"""
Definition Search Module
========================

Functions for finding and boosting code definitions (classes, functions, methods).

This module handles:
- Detection of definition-seeking queries ("class Foo", "def bar")
- Pattern-based search for definitions in source code
- Boosting mechanisms for definition passages
- Test file detection and penalty application
"""

from typing import Dict, List, Tuple, Optional, TypedDict, Any
import re

from .utils import is_test_file


# Patterns for detecting definition queries
DEFINITION_QUERY_PATTERNS = [
    # "class Foo" or "class foo"
    (r'\bclass\s+(\w+)', 'class'),
    # "def bar" or "function bar"
    (r'\bdef\s+(\w+)', 'function'),
    (r'\bfunction\s+(\w+)', 'function'),
    # "method baz"
    (r'\bmethod\s+(\w+)', 'method'),
]

# Regex patterns for finding definitions in source code
# Format: (pattern_template, definition_type)
# pattern_template uses {name} placeholder for the identifier
DEFINITION_SOURCE_PATTERNS = {
    'python_class': r'^class\s+{name}\b[^:]*:',
    'python_function': r'^def\s+{name}\s*\(',
    'python_method': r'^\s+def\s+{name}\s*\(',
    'javascript_function': r'^function\s+{name}\s*\(',
    'javascript_class': r'^class\s+{name}\b',
    'javascript_const_fn': r'^const\s+{name}\s*=\s*(?:async\s*)?\(',
}

# Default boost for definition matches
DEFINITION_BOOST = 5.0


class DefinitionQuery(TypedDict):
    """Info about a definition-seeking query."""
    is_definition_query: bool
    definition_type: Optional[str]  # 'class', 'function', 'method', 'variable'
    identifier: Optional[str]       # The identifier being searched for
    pattern: Optional[str]          # Regex pattern to find the definition


def is_definition_query(query_text: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Detect if a query is looking for a code definition.

    Recognizes patterns like:
    - "class Minicolumn"
    - "def compute_pagerank"
    - "function tokenize"
    - "method process_document"

    Args:
        query_text: The search query

    Returns:
        Tuple of (is_definition, definition_type, identifier_name)
        If not a definition query, returns (False, None, None)
    """
    query_lower = query_text.strip()

    for pattern, def_type in DEFINITION_QUERY_PATTERNS:
        match = re.search(pattern, query_lower, re.IGNORECASE)
        if match:
            identifier = match.group(1)
            return (True, def_type, identifier)

    return (False, None, None)


def find_definition_in_text(
    text: str,
    identifier: str,
    def_type: str,
    context_chars: int = 500
) -> Optional[Tuple[str, int, int]]:
    """
    Find a definition in source text and extract surrounding context.

    Args:
        text: Source code text to search
        identifier: Name of the class/function/method to find
        def_type: Type of definition ('class', 'function', 'method')
        context_chars: Number of characters of context to include after the definition

    Returns:
        Tuple of (passage_text, start_char, end_char) or None if not found
    """
    # Build patterns to try based on definition type
    patterns_to_try = []

    if def_type == 'class':
        patterns_to_try = [
            DEFINITION_SOURCE_PATTERNS['python_class'],
            DEFINITION_SOURCE_PATTERNS['javascript_class'],
        ]
    elif def_type in ('function', 'method'):
        patterns_to_try = [
            DEFINITION_SOURCE_PATTERNS['python_function'],
            DEFINITION_SOURCE_PATTERNS['python_method'],
            DEFINITION_SOURCE_PATTERNS['javascript_function'],
            DEFINITION_SOURCE_PATTERNS['javascript_const_fn'],
        ]

    # Try each pattern
    for pattern_template in patterns_to_try:
        pattern = pattern_template.format(name=re.escape(identifier))
        match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
        if match:
            # Find the start of the line containing the definition
            # This ensures the passage starts with the actual definition line
            line_start = text.rfind('\n', 0, match.start())
            if line_start == -1:
                # Match is on the first line of the text
                start = 0
            else:
                # Start from the character after the newline
                start = line_start + 1

            # Extract context after the definition
            end = min(len(text), match.end() + context_chars)

            # Try to extend to next blank line or class/function boundary
            remaining = text[match.end():end]
            # Look for a good boundary (blank line followed by non-indented text)
            boundary_match = re.search(r'\n\n(?=[^\s])', remaining)
            if boundary_match:
                end = match.end() + boundary_match.end()

            passage = text[start:end]
            return (passage, start, end)

    return None


def find_definition_passages(
    query_text: str,
    documents: Dict[str, str],
    context_chars: int = 500,
    boost: float = DEFINITION_BOOST
) -> List[Tuple[str, str, int, int, float]]:
    """
    Find definition passages for a definition query.

    If the query is looking for a class/function/method definition,
    directly search source files for the definition and return
    high-scoring passages.

    Args:
        query_text: Search query (e.g., "class Minicolumn", "def compute_pagerank")
        documents: Dict mapping doc_id to document text
        context_chars: Characters of context to include after definition
        boost: Score boost for definition matches

    Returns:
        List of (passage_text, doc_id, start_char, end_char, score) tuples.
        Returns empty list if query is not a definition query.
    """
    is_def, def_type, identifier = is_definition_query(query_text)

    if not is_def or not identifier:
        return []

    results = []

    # Search all documents for the definition
    for doc_id, text in documents.items():
        # Prefer source files over test files for definitions
        is_test = is_test_file(doc_id)

        result = find_definition_in_text(text, identifier, def_type, context_chars)
        if result:
            passage, start, end = result
            # Apply boost, with penalty for test files
            score = boost * (0.6 if is_test else 1.0)
            results.append((passage, doc_id, start, end, score))

    # Sort by score (highest first)
    results.sort(key=lambda x: -x[4])
    return results


def detect_definition_query(query_text: str) -> DefinitionQuery:
    """
    Detect if a query is searching for a code definition.

    Recognizes patterns like:
    - "class Minicolumn" -> looking for class definition
    - "def compute_tfidf" -> looking for function definition
    - "function handleClick" -> looking for function definition

    Args:
        query_text: The search query

    Returns:
        DefinitionQuery with detection results and pattern to search for
    """
    query_lower = query_text.lower().strip()

    # Patterns for definition searches
    patterns = [
        # "class ClassName" or "class ClassName definition"
        (r'\bclass\s+([A-Za-z_][A-Za-z0-9_]*)', 'class',
         lambda name: rf'\bclass\s+{re.escape(name)}\s*[:\(]'),
        # "def function_name" or "function function_name"
        (r'\b(?:def|function)\s+([A-Za-z_][A-Za-z0-9_]*)', 'function',
         lambda name: rf'\bdef\s+{re.escape(name)}\s*\('),
        # "method methodName"
        (r'\bmethod\s+([A-Za-z_][A-Za-z0-9_]*)', 'method',
         lambda name: rf'\bdef\s+{re.escape(name)}\s*\('),
    ]

    for regex, def_type, pattern_fn in patterns:
        match = re.search(regex, query_text, re.IGNORECASE)
        if match:
            identifier = match.group(1)
            return DefinitionQuery(
                is_definition_query=True,
                definition_type=def_type,
                identifier=identifier,
                pattern=pattern_fn(identifier)
            )

    return DefinitionQuery(
        is_definition_query=False,
        definition_type=None,
        identifier=None,
        pattern=None
    )


def apply_definition_boost(
    passages: List[Tuple[str, str, int, int, float]],
    query_text: str,
    boost_factor: float = 3.0
) -> List[Tuple[str, str, int, int, float]]:
    """
    Boost passages that contain actual code definitions matching the query.

    When searching for "class Minicolumn", passages containing the actual
    class definition (`class Minicolumn:`) get boosted over passages that
    merely reference or use the class.

    Args:
        passages: List of (text, doc_id, start, end, score) tuples
        query_text: The original search query
        boost_factor: Multiplier for definition-containing passages (default 3.0)

    Returns:
        Re-scored passages with definition boost applied, sorted by new score
    """
    definition_info = detect_definition_query(query_text)

    if not definition_info['is_definition_query'] or not definition_info['pattern']:
        return passages

    pattern = re.compile(definition_info['pattern'], re.IGNORECASE)
    boosted_passages = []

    for text, doc_id, start, end, score in passages:
        if pattern.search(text):
            # This passage contains the actual definition
            boosted_passages.append((text, doc_id, start, end, score * boost_factor))
        else:
            boosted_passages.append((text, doc_id, start, end, score))

    # Re-sort by boosted scores
    boosted_passages.sort(key=lambda x: x[4], reverse=True)
    return boosted_passages


def boost_definition_documents(
    doc_results: List[Tuple[str, float]],
    query_text: str,
    documents: Dict[str, str],
    boost_factor: float = 2.0,
    test_with_definition_penalty: float = 0.5,
    test_without_definition_penalty: float = 0.7
) -> List[Tuple[str, float]]:
    """
    Boost documents that contain the actual definition being searched for.

    This helps ensure the source file containing a class/function definition
    is included in the document candidates, even if test files mention the
    identifier more frequently.

    For definition queries:
    - Source files with the definition pattern get boost_factor (default 2.0x)
    - Test files with the definition pattern get test_with_definition_penalty (default 0.5x)
    - All other test files get test_without_definition_penalty (default 0.7x)

    Args:
        doc_results: List of (doc_id, score) tuples
        query_text: The original search query
        documents: Dict mapping doc_id to document text
        boost_factor: Multiplier for definition-containing source docs (default 2.0)
        test_with_definition_penalty: Multiplier for test files that contain the definition
            (default 0.5). Even test files with definitions are penalized vs source files.
        test_without_definition_penalty: Multiplier for test files without the definition
            (default 0.7). Set to 1.0 to disable test file penalty.

    Returns:
        Re-scored document results with definition boost applied
    """
    definition_info = detect_definition_query(query_text)

    if not definition_info['is_definition_query'] or not definition_info['pattern']:
        return doc_results

    pattern = re.compile(definition_info['pattern'], re.IGNORECASE)
    boosted_docs = []

    for doc_id, score in doc_results:
        doc_text = documents.get(doc_id, '')
        has_definition = pattern.search(doc_text)
        is_test = is_test_file(doc_id)

        if has_definition:
            if is_test:
                # Test file with definition: still penalized vs source files
                boosted_docs.append((doc_id, score * test_with_definition_penalty))
            else:
                # Source file with definition: apply full boost
                boosted_docs.append((doc_id, score * boost_factor))
        elif is_test:
            # Test file without definition: apply penalty to deprioritize
            boosted_docs.append((doc_id, score * test_without_definition_penalty))
        else:
            # Source file without definition: keep original score
            boosted_docs.append((doc_id, score))

    # Re-sort by boosted scores
    boosted_docs.sort(key=lambda x: x[1], reverse=True)
    return boosted_docs
