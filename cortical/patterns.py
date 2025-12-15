"""
Code Pattern Detection Module
==============================

Detects common programming patterns in indexed code.

Identifies design patterns, idioms, and code structures including:
- Singleton pattern
- Factory pattern
- Decorator usage
- Context managers
- Error handling patterns
- Generator patterns
- Async patterns
- Property patterns
- Class patterns
"""

import re
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict


# =============================================================================
# PATTERN DEFINITIONS
# =============================================================================

# Pattern name -> (regex pattern, description, category)
PATTERN_DEFINITIONS: Dict[str, Tuple[str, str, str]] = {
    # Creational Patterns
    'singleton': (
        r'(_instance\s*=\s*None|__new__.*cls\._instance|if\s+not\s+hasattr\(cls,\s*["\']_instance|'
        r'def\s+__new__\s*\(\s*cls.*\).*return.*cls\._instance)',
        'Singleton pattern (single instance control)',
        'creational'
    ),
    'factory': (
        r'(def\s+(create|make|build|get)_\w+|class\s+\w*Factory\w*|'
        r'@staticmethod\s+def\s+(create|make|build))',
        'Factory pattern (object creation)',
        'creational'
    ),
    'builder': (
        r'(def\s+with_\w+\(self|def\s+set_\w+\(self.*return\s+self|'
        r'class\s+\w*Builder\w*)',
        'Builder pattern (fluent construction)',
        'creational'
    ),

    # Structural Patterns
    'decorator': (
        r'(@\w+\s*(\(.*\))?\s*\n\s*def\s+\w+|'
        r'def\s+\w+\s*\([^)]*\)\s*:\s*def\s+wrapper|'
        r'@(property|staticmethod|classmethod|wraps))',
        'Decorator pattern (wrapping behavior)',
        'structural'
    ),
    'adapter': (
        r'(class\s+\w*Adapter\w*|def\s+adapt\w*\()',
        'Adapter pattern (interface conversion)',
        'structural'
    ),
    'proxy': (
        r'(class\s+\w*Proxy\w*|def\s+__getattr__)',
        'Proxy pattern (access control)',
        'structural'
    ),

    # Behavioral Patterns
    'context_manager': (
        r'(def\s+__enter__|def\s+__exit__|@contextmanager|'
        r'with\s+\w+.*as\s+\w+:)',
        'Context manager (resource management)',
        'behavioral'
    ),
    'generator': (
        r'(yield\s+\w|yield\s+from\s+|yield\s*$|'
        r'def\s+\w+.*\):.*yield)',
        'Generator pattern (lazy iteration)',
        'behavioral'
    ),
    'iterator': (
        r'(def\s+__iter__|def\s+__next__|class\s+\w*Iterator\w*)',
        'Iterator pattern (custom iteration)',
        'behavioral'
    ),
    'observer': (
        r'(def\s+(notify|subscribe|unsubscribe|attach|detach)|(on|handle)_\w+_event|'
        r'class\s+\w*(Observer|Subject|Publisher)\w*)',
        'Observer pattern (event notification)',
        'behavioral'
    ),
    'strategy': (
        r'(class\s+\w*Strategy\w*|def\s+set_strategy)',
        'Strategy pattern (algorithm selection)',
        'behavioral'
    ),

    # Concurrency Patterns
    'async_await': (
        r'(async\s+def\s+\w+|await\s+\w+|async\s+with|async\s+for)',
        'Async/await pattern (asynchronous code)',
        'concurrency'
    ),
    'thread_safety': (
        r'(threading\.Lock|threading\.RLock|threading\.Semaphore|'
        r'with\s+\w*lock\w*:|@synchronized)',
        'Thread safety (locks and synchronization)',
        'concurrency'
    ),
    'concurrent_futures': (
        r'(concurrent\.futures|ThreadPoolExecutor|ProcessPoolExecutor|'
        r'executor\.submit|executor\.map)',
        'Concurrent futures (thread/process pools)',
        'concurrency'
    ),

    # Error Handling
    'error_handling': (
        r'(try\s*:|except\s+\w+:|except\s*:|finally\s*:|raise\s+\w+)',
        'Error handling (try/except blocks)',
        'error_handling'
    ),
    'custom_exception': (
        r'(class\s+\w*(Error|Exception)\w*\s*\(.*Exception|raise\s+\w+Error\()',
        'Custom exception classes',
        'error_handling'
    ),
    'assertion': (
        r'(assert\s+\w+|AssertionError)',
        'Assertions (runtime checks)',
        'error_handling'
    ),

    # Python-Specific Idioms
    'property_decorator': (
        r'(@property|@\w+\.setter|@\w+\.deleter)',
        'Property decorator (computed attributes)',
        'idiom'
    ),
    'dataclass': (
        r'(@dataclass|@dataclasses\.dataclass)',
        'Dataclass (structured data)',
        'idiom'
    ),
    'slots': (
        r'(__slots__\s*=)',
        'Slots (memory optimization)',
        'idiom'
    ),
    'magic_methods': (
        r'(def\s+__(repr|str|eq|lt|le|gt|ge|hash|bool|len|getitem|setitem|delitem|call)__)',
        'Magic methods (operator overloading)',
        'idiom'
    ),
    'comprehension': (
        r'(\[.+for\s+\w+\s+in\s+.+\]|\{.+for\s+\w+\s+in\s+.+\}|'
        r'\(.+for\s+\w+\s+in\s+.+\))',
        'List/dict/set comprehension',
        'idiom'
    ),
    'unpacking': (
        r'(\*args|\*\*kwargs|\*\w+,|def\s+\w+\([^)]*\*|'
        r'\w+,\s*\*\w+\s*=)',
        'Argument unpacking (*args, **kwargs)',
        'idiom'
    ),

    # Testing Patterns
    'unittest_class': (
        r'(class\s+Test\w+\(.*TestCase|def\s+test_\w+\(self|'
        r'def\s+setUp\(self|def\s+tearDown\(self)',
        'Unittest test class',
        'testing'
    ),
    'pytest_test': (
        r'(def\s+test_\w+\(|@pytest\.\w+|assert\s+\w+\s*(==|!=|is|in))',
        'Pytest test function',
        'testing'
    ),
    'mock_usage': (
        r'(@mock\.|Mock\(|MagicMock\(|patch\(|@patch)',
        'Mocking (test doubles)',
        'testing'
    ),
    'fixture': (
        r'(@pytest\.fixture|@fixture)',
        'Pytest fixture (test setup)',
        'testing'
    ),

    # Functional Programming
    'lambda': (
        r'(lambda\s+\w+:|lambda\s*:)',
        'Lambda functions (anonymous functions)',
        'functional'
    ),
    'map_filter_reduce': (
        r'(map\(|filter\(|reduce\(|functools\.reduce)',
        'Map/filter/reduce (functional operations)',
        'functional'
    ),
    'partial_application': (
        r'(functools\.partial|partial\()',
        'Partial application (currying)',
        'functional'
    ),

    # Type Annotations
    'type_hints': (
        r'(def\s+\w+\([^)]*:\s*\w+|def\s+\w+.*->\s*\w+:|'
        r'from\s+typing\s+import|List\[|Dict\[|Optional\[|Union\[)',
        'Type hints (static typing)',
        'typing'
    ),
    'type_checking': (
        r'(if\s+TYPE_CHECKING:|from\s+typing\s+import\s+TYPE_CHECKING)',
        'TYPE_CHECKING guard (import-time types)',
        'typing'
    ),
}


# Pattern categories for grouping
PATTERN_CATEGORIES: Dict[str, List[str]] = defaultdict(list)
for pattern_name, (_, _, category) in PATTERN_DEFINITIONS.items():
    PATTERN_CATEGORIES[category].append(pattern_name)


# =============================================================================
# PATTERN DETECTION FUNCTIONS
# =============================================================================


def detect_patterns_in_text(text: str, patterns: Optional[List[str]] = None) -> Dict[str, List[int]]:
    """
    Detect programming patterns in a text string.

    Args:
        text: Source code text to analyze
        patterns: Specific pattern names to search for (None = all patterns)

    Returns:
        Dict mapping pattern names to list of line numbers where found

    Example:
        >>> code = "async def fetch():\\n    await get_data()"
        >>> patterns = detect_patterns_in_text(code)
        >>> 'async_await' in patterns
        True
    """
    if patterns is None:
        patterns = list(PATTERN_DEFINITIONS.keys())

    results: Dict[str, List[int]] = {}
    lines = text.split('\n')

    for pattern_name in patterns:
        if pattern_name not in PATTERN_DEFINITIONS:
            continue

        regex_pattern, _, _ = PATTERN_DEFINITIONS[pattern_name]
        pattern = re.compile(regex_pattern, re.MULTILINE | re.DOTALL)

        # Track which lines match this pattern
        matching_lines: Set[int] = set()

        # Search line by line for better line number tracking
        for line_num, line in enumerate(lines, start=1):
            if pattern.search(line):
                matching_lines.add(line_num)

        # Also search the full text for multi-line patterns
        for match in pattern.finditer(text):
            # Find which line this match starts on
            start_pos = match.start()
            line_num = text[:start_pos].count('\n') + 1
            matching_lines.add(line_num)

        if matching_lines:
            results[pattern_name] = sorted(matching_lines)

    return results


def detect_patterns_in_documents(
    documents: Dict[str, str],
    patterns: Optional[List[str]] = None
) -> Dict[str, Dict[str, List[int]]]:
    """
    Detect patterns across multiple documents.

    Args:
        documents: Dict mapping doc_id to content
        patterns: Specific pattern names to search for (None = all patterns)

    Returns:
        Dict mapping doc_id to pattern detection results

    Example:
        >>> docs = {'file1.py': 'async def foo(): pass'}
        >>> results = detect_patterns_in_documents(docs)
        >>> 'async_await' in results['file1.py']
        True
    """
    results = {}
    for doc_id, content in documents.items():
        patterns_found = detect_patterns_in_text(content, patterns)
        if patterns_found:
            results[doc_id] = patterns_found

    return results


def get_pattern_summary(
    pattern_results: Dict[str, List[int]]
) -> Dict[str, int]:
    """
    Summarize pattern detection results by counting occurrences.

    Args:
        pattern_results: Output from detect_patterns_in_text()

    Returns:
        Dict mapping pattern names to occurrence counts

    Example:
        >>> results = {'async_await': [1, 5, 10], 'generator': [3]}
        >>> summary = get_pattern_summary(results)
        >>> summary['async_await']
        3
    """
    return {
        pattern_name: len(line_numbers)
        for pattern_name, line_numbers in pattern_results.items()
    }


def get_patterns_by_category(
    pattern_results: Dict[str, List[int]]
) -> Dict[str, Dict[str, int]]:
    """
    Group pattern results by category.

    Args:
        pattern_results: Output from detect_patterns_in_text()

    Returns:
        Dict mapping category to {pattern_name: count}

    Example:
        >>> results = {'async_await': [1, 2], 'generator': [3]}
        >>> by_category = get_patterns_by_category(results)
        >>> 'concurrency' in by_category
        True
    """
    categorized: Dict[str, Dict[str, int]] = defaultdict(dict)

    for pattern_name, line_numbers in pattern_results.items():
        if pattern_name in PATTERN_DEFINITIONS:
            _, _, category = PATTERN_DEFINITIONS[pattern_name]
            count = len(line_numbers)
            categorized[category][pattern_name] = count

    return dict(categorized)


def get_pattern_description(pattern_name: str) -> Optional[str]:
    """
    Get the description for a pattern.

    Args:
        pattern_name: Name of the pattern

    Returns:
        Description string, or None if pattern not found

    Example:
        >>> get_pattern_description('singleton')
        'Singleton pattern (single instance control)'
    """
    if pattern_name in PATTERN_DEFINITIONS:
        _, description, _ = PATTERN_DEFINITIONS[pattern_name]
        return description
    return None


def get_pattern_category(pattern_name: str) -> Optional[str]:
    """
    Get the category for a pattern.

    Args:
        pattern_name: Name of the pattern

    Returns:
        Category string, or None if pattern not found

    Example:
        >>> get_pattern_category('singleton')
        'creational'
    """
    if pattern_name in PATTERN_DEFINITIONS:
        _, _, category = PATTERN_DEFINITIONS[pattern_name]
        return category
    return None


def list_all_patterns() -> List[str]:
    """
    List all available pattern names.

    Returns:
        Sorted list of pattern names

    Example:
        >>> patterns = list_all_patterns()
        >>> 'singleton' in patterns
        True
    """
    return sorted(PATTERN_DEFINITIONS.keys())


def list_patterns_by_category(category: str) -> List[str]:
    """
    List all patterns in a specific category.

    Args:
        category: Category name

    Returns:
        Sorted list of pattern names in that category

    Example:
        >>> patterns = list_patterns_by_category('creational')
        >>> 'singleton' in patterns
        True
    """
    return sorted(PATTERN_CATEGORIES.get(category, []))


def list_all_categories() -> List[str]:
    """
    List all pattern categories.

    Returns:
        Sorted list of category names

    Example:
        >>> categories = list_all_categories()
        >>> 'creational' in categories
        True
    """
    return sorted(PATTERN_CATEGORIES.keys())


def format_pattern_report(
    pattern_results: Dict[str, List[int]],
    show_lines: bool = False
) -> str:
    """
    Format pattern detection results as a human-readable report.

    Args:
        pattern_results: Output from detect_patterns_in_text()
        show_lines: Whether to show line numbers

    Returns:
        Formatted report string

    Example:
        >>> results = {'async_await': [1, 5], 'generator': [10]}
        >>> report = format_pattern_report(results)
        >>> 'async_await' in report
        True
    """
    if not pattern_results:
        return "No patterns detected."

    lines = []
    lines.append(f"Detected {len(pattern_results)} pattern types:")
    lines.append("")

    # Group by category
    by_category = get_patterns_by_category(pattern_results)

    for category in sorted(by_category.keys()):
        lines.append(f"{category.upper()}:")
        patterns_in_cat = by_category[category]

        for pattern_name in sorted(patterns_in_cat.keys()):
            count = patterns_in_cat[pattern_name]
            description = get_pattern_description(pattern_name)

            if show_lines and pattern_name in pattern_results:
                line_nums = pattern_results[pattern_name]
                lines.append(f"  - {pattern_name}: {count} occurrences")
                lines.append(f"    {description}")
                lines.append(f"    Lines: {', '.join(map(str, line_nums[:10]))}")
                if len(line_nums) > 10:
                    lines.append(f"    ... and {len(line_nums) - 10} more")
            else:
                lines.append(f"  - {pattern_name}: {count} occurrences - {description}")

        lines.append("")

    return '\n'.join(lines)


def get_corpus_pattern_statistics(
    doc_patterns: Dict[str, Dict[str, List[int]]]
) -> Dict[str, any]:
    """
    Compute statistics across all documents.

    Args:
        doc_patterns: Output from detect_patterns_in_documents()

    Returns:
        Dict with corpus-wide statistics

    Example:
        >>> docs = {'f1.py': {'async_await': [1]}, 'f2.py': {'async_await': [2]}}
        >>> stats = get_corpus_pattern_statistics(docs)
        >>> stats['total_documents']
        2
    """
    pattern_doc_counts: Dict[str, int] = defaultdict(int)
    pattern_total_occurrences: Dict[str, int] = defaultdict(int)

    for doc_id, doc_patterns_result in doc_patterns.items():
        for pattern_name, line_numbers in doc_patterns_result.items():
            pattern_doc_counts[pattern_name] += 1
            pattern_total_occurrences[pattern_name] += len(line_numbers)

    return {
        'total_documents': len(doc_patterns),
        'patterns_found': len(pattern_doc_counts),
        'pattern_document_counts': dict(pattern_doc_counts),
        'pattern_occurrences': dict(pattern_total_occurrences),
        'most_common_pattern': max(pattern_total_occurrences.items(), key=lambda x: x[1])[0]
            if pattern_total_occurrences else None,
    }
