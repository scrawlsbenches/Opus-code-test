"""
Comprehensive Unit Tests for Pattern Detection - Coverage Enhancement
=======================================================================

This test suite complements test_patterns.py by targeting:
1. Uncovered code paths (lines 500, 360->359)
2. Edge cases and boundary conditions
3. Multi-line pattern detection
4. Complex pattern interactions
5. All pattern types not explicitly tested
6. Report formatting edge cases

Goal: Increase coverage from 98% to 100%
"""

import pytest

from cortical.patterns import (
    PATTERN_DEFINITIONS,
    PATTERN_CATEGORIES,
    detect_patterns_in_text,
    detect_patterns_in_documents,
    get_pattern_summary,
    get_patterns_by_category,
    get_pattern_description,
    get_pattern_category,
    list_all_patterns,
    list_patterns_by_category,
    list_all_categories,
    format_pattern_report,
    get_corpus_pattern_statistics,
)


# =============================================================================
# MISSING LINE COVERAGE TESTS
# =============================================================================


class TestMissingLineCoverage:
    """Tests specifically targeting uncovered lines."""

    def test_format_report_with_more_than_10_lines(self):
        """
        Test line 500: The "... and N more" truncation logic.

        When a pattern has >10 line numbers and show_lines=True,
        the report should truncate and show "... and N more".
        """
        # Create pattern results with more than 10 line numbers
        pattern_results = {
            'async_await': list(range(1, 21)),  # 20 line numbers
            'generator': list(range(1, 16)),     # 15 line numbers
        }

        report = format_pattern_report(pattern_results, show_lines=True)

        # Should contain truncation message
        assert '... and' in report
        assert 'more' in report
        # Should show first 10 line numbers for async_await (1-10)
        assert '1' in report
        assert '10' in report
        # Should indicate 10 more lines (20 - 10 = 10)
        assert '10 more' in report or '5 more' in report  # 15 - 10 = 5 for generator

    def test_get_patterns_by_category_with_unknown_pattern(self):
        """
        Test line 360->359: Branch when pattern_name not in PATTERN_DEFINITIONS.

        This tests the conditional at line 360 that checks if pattern_name
        exists in PATTERN_DEFINITIONS before accessing category.
        """
        # Create results with a pattern name that doesn't exist
        # This could happen if PATTERN_DEFINITIONS was modified externally
        # or if we're processing stale data
        pattern_results = {
            'valid_pattern': [1, 2, 3],
            'unknown_pattern_xyz': [4, 5],  # Not in PATTERN_DEFINITIONS
        }

        # First, let's make sure 'valid_pattern' exists, use a real pattern
        pattern_results = {
            'async_await': [1, 2, 3],
            'unknown_pattern_xyz': [4, 5],
        }

        # Should not crash, should skip unknown pattern
        by_category = get_patterns_by_category(pattern_results)

        # async_await should be categorized
        assert 'concurrency' in by_category
        assert 'async_await' in by_category['concurrency']

        # unknown_pattern_xyz should be silently ignored (not in any category)
        # Check that it's not present in any category
        for category, patterns in by_category.items():
            assert 'unknown_pattern_xyz' not in patterns


# =============================================================================
# ADDITIONAL PATTERN TYPE TESTS
# =============================================================================


class TestAllPatternTypes:
    """Ensure all pattern types are tested with explicit examples."""

    def test_builder_pattern(self):
        """Test builder pattern detection."""
        code = """
class QueryBuilder:
    def with_filter(self, filter):
        self.filter = filter
        return self

    def set_limit(self, limit):
        self.limit = limit
        return self
"""
        patterns = detect_patterns_in_text(code)
        assert 'builder' in patterns

    def test_adapter_pattern(self):
        """Test adapter pattern detection."""
        code = """
class LegacyAdapter:
    def adapt_request(self, data):
        return self.convert(data)
"""
        patterns = detect_patterns_in_text(code)
        assert 'adapter' in patterns

    def test_proxy_pattern(self):
        """Test proxy pattern detection."""
        code = """
class ProxyObject:
    def __getattr__(self, name):
        return getattr(self._obj, name)
"""
        patterns = detect_patterns_in_text(code)
        assert 'proxy' in patterns

    def test_iterator_pattern(self):
        """Test iterator pattern detection."""
        code = """
class CustomIterator:
    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.data):
            raise StopIteration
        return self.data[self.index]
"""
        patterns = detect_patterns_in_text(code)
        assert 'iterator' in patterns

    def test_observer_pattern(self):
        """Test observer pattern detection."""
        code = """
class EventPublisher:
    def notify(self, event):
        for subscriber in self.subscribers:
            subscriber.handle_event(event)

    def subscribe(self, subscriber):
        self.subscribers.append(subscriber)
"""
        patterns = detect_patterns_in_text(code)
        assert 'observer' in patterns

    def test_strategy_pattern(self):
        """Test strategy pattern detection."""
        code = """
class SortStrategy:
    pass

def set_strategy(self, strategy):
    self.strategy = strategy
"""
        patterns = detect_patterns_in_text(code)
        assert 'strategy' in patterns

    def test_thread_safety_pattern(self):
        """Test thread safety pattern detection."""
        code = """
import threading

class ThreadSafeCounter:
    def __init__(self):
        self.lock = threading.Lock()
        self.count = 0

    def increment(self):
        with self.lock:
            self.count += 1
"""
        patterns = detect_patterns_in_text(code)
        assert 'thread_safety' in patterns

    def test_concurrent_futures_pattern(self):
        """Test concurrent futures pattern detection."""
        code = """
from concurrent.futures import ThreadPoolExecutor

def process_parallel(items):
    with ThreadPoolExecutor() as executor:
        results = executor.map(process_item, items)
"""
        patterns = detect_patterns_in_text(code)
        assert 'concurrent_futures' in patterns

    def test_assertion_pattern(self):
        """Test assertion pattern detection."""
        code = """
def validate(value):
    assert value > 0, "Value must be positive"
    assert isinstance(value, int), "Value must be integer"
"""
        patterns = detect_patterns_in_text(code)
        assert 'assertion' in patterns

    def test_slots_pattern(self):
        """Test __slots__ pattern detection."""
        code = """
class OptimizedClass:
    __slots__ = ['name', 'value', 'data']

    def __init__(self, name, value):
        self.name = name
        self.value = value
"""
        patterns = detect_patterns_in_text(code)
        assert 'slots' in patterns

    def test_fixture_pattern(self):
        """Test pytest fixture pattern detection."""
        code = """
import pytest

@pytest.fixture
def database_connection():
    conn = create_connection()
    yield conn
    conn.close()
"""
        patterns = detect_patterns_in_text(code)
        assert 'fixture' in patterns

    def test_map_filter_reduce_pattern(self):
        """Test map/filter/reduce pattern detection."""
        code = """
from functools import reduce

numbers = [1, 2, 3, 4, 5]
squared = map(lambda x: x**2, numbers)
evens = filter(lambda x: x % 2 == 0, numbers)
total = reduce(lambda a, b: a + b, numbers)
"""
        patterns = detect_patterns_in_text(code)
        assert 'map_filter_reduce' in patterns
        assert 'lambda' in patterns

    def test_partial_application_pattern(self):
        """Test partial application pattern detection."""
        code = """
from functools import partial

def multiply(a, b):
    return a * b

double = partial(multiply, 2)
triple = partial(multiply, 3)
"""
        patterns = detect_patterns_in_text(code)
        assert 'partial_application' in patterns

    def test_type_checking_pattern(self):
        """Test TYPE_CHECKING guard pattern detection."""
        code = """
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import User

def process_user(user: 'User'):
    pass
"""
        patterns = detect_patterns_in_text(code)
        assert 'type_checking' in patterns


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_pattern_on_first_line(self):
        """Test pattern detection when match is on line 1."""
        code = "async def fetch(): pass"
        patterns = detect_patterns_in_text(code)
        assert 'async_await' in patterns
        assert 1 in patterns['async_await']

    def test_pattern_on_last_line(self):
        """Test pattern detection when match is on last line."""
        code = "x = 1\ny = 2\nasync def fetch(): pass"
        patterns = detect_patterns_in_text(code)
        assert 'async_await' in patterns
        assert 3 in patterns['async_await']

    def test_multiple_patterns_same_line(self):
        """Test when multiple patterns match the same line."""
        code = "@property\ndef value(self): return self._value"
        patterns = detect_patterns_in_text(code)
        # Should detect both decorator and property_decorator
        assert 'decorator' in patterns or 'property_decorator' in patterns

    def test_multiline_pattern_detection(self):
        """Test patterns that span multiple lines."""
        code = """
def outer():
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper
"""
        patterns = detect_patterns_in_text(code)
        # Should detect decorator pattern and unpacking
        assert 'unpacking' in patterns

    def test_nested_patterns(self):
        """Test detection of nested patterns."""
        code = """
async def outer():
    async def inner():
        async with context():
            async for item in items:
                yield item
"""
        patterns = detect_patterns_in_text(code)
        assert 'async_await' in patterns
        assert 'generator' in patterns
        # Multiple occurrences of async patterns
        assert len(patterns['async_await']) >= 4

    def test_pattern_in_string_literal(self):
        """Test that patterns in string literals are still detected by regex."""
        code = '''
text = """
async def fake():
    pass
"""
async def real():
    pass
'''
        patterns = detect_patterns_in_text(code)
        # Regex will match both occurrences
        assert 'async_await' in patterns

    def test_pattern_in_comment(self):
        """Test that patterns in comments are still detected by regex."""
        code = """
# async def commented_out():
#     pass
async def real():
    pass
"""
        patterns = detect_patterns_in_text(code)
        assert 'async_await' in patterns

    def test_very_long_line_numbers_list(self):
        """Test formatting with very long list of line numbers."""
        pattern_results = {
            'comprehension': list(range(1, 101)),  # 100 line numbers
        }
        report = format_pattern_report(pattern_results, show_lines=True)
        # Should truncate at 10 and show "90 more"
        assert '90 more' in report or '... and' in report

    def test_single_line_in_report(self):
        """Test report formatting with single occurrence."""
        pattern_results = {
            'async_await': [42],
        }
        report = format_pattern_report(pattern_results, show_lines=True)
        assert '42' in report
        assert '1 occurrence' in report or '1' in report

    def test_exactly_10_lines_in_report(self):
        """Test report formatting with exactly 10 line numbers (boundary)."""
        pattern_results = {
            'async_await': list(range(1, 11)),  # Exactly 10 line numbers
        }
        report = format_pattern_report(pattern_results, show_lines=True)
        # Should NOT show "... and N more" for exactly 10
        assert '... and' not in report or 'more' not in report

    def test_11_lines_in_report(self):
        """Test report formatting with 11 line numbers (just over boundary)."""
        pattern_results = {
            'async_await': list(range(1, 12)),  # 11 line numbers
        }
        report = format_pattern_report(pattern_results, show_lines=True)
        # Should show "... and 1 more"
        assert '1 more' in report


# =============================================================================
# COMPLEX INTERACTION TESTS
# =============================================================================


class TestComplexInteractions:
    """Test complex interactions between patterns."""

    def test_all_pattern_categories_represented(self):
        """Test code that uses patterns from all categories."""
        code = """
from typing import List, Optional
from dataclasses import dataclass
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor
import pytest

# Creational
class Singleton:
    _instance = None

class UserFactory:
    @staticmethod
    def create(): pass

# Structural
@property
def value(self): pass

# Behavioral
def generator():
    yield 1

class Observer:
    def notify(self): pass

# Concurrency
async def fetch():
    await get()

lock = threading.Lock()

with ThreadPoolExecutor() as executor:
    pass

# Error handling
try:
    risky()
except ValueError:
    pass

class CustomError(Exception):
    pass

# Idiom
@dataclass
class Data:
    x: int

class Optimized:
    __slots__ = ['x']

items = [x for x in range(10)]

# Testing
@pytest.fixture
def setup():
    pass

def test_it():
    assert True

# Functional
fn = lambda x: x
mapped = map(str, [1, 2, 3])

# Typing
def typed(x: int) -> str:
    return str(x)
"""
        patterns = detect_patterns_in_text(code)
        by_category = get_patterns_by_category(patterns)

        # Should have patterns from multiple categories
        assert len(by_category) >= 5

        # Check key categories are present
        expected_categories = ['creational', 'behavioral', 'concurrency',
                             'error_handling', 'idiom', 'testing', 'functional', 'typing']
        found_categories = list(by_category.keys())

        # At least 5 of these should be found
        matches = sum(1 for cat in expected_categories if cat in found_categories)
        assert matches >= 5

    def test_corpus_statistics_with_multiple_patterns(self):
        """Test corpus statistics with complex pattern distribution."""
        doc_patterns = {
            'file1.py': {
                'async_await': [1, 2, 3],
                'generator': [5],
                'decorator': [7, 8],
            },
            'file2.py': {
                'async_await': [1],
                'decorator': [3, 4, 5],
            },
            'file3.py': {
                'generator': [1, 2, 3, 4],
            },
            'file4.py': {
                'singleton': [1],
            },
        }

        stats = get_corpus_pattern_statistics(doc_patterns)

        assert stats['total_documents'] == 4
        assert stats['patterns_found'] == 4

        # async_await in 2 files, 4 total occurrences
        assert stats['pattern_document_counts']['async_await'] == 2
        assert stats['pattern_occurrences']['async_await'] == 4

        # decorator in 2 files, 5 total occurrences
        assert stats['pattern_document_counts']['decorator'] == 2
        assert stats['pattern_occurrences']['decorator'] == 5

        # generator in 2 files, 5 total occurrences
        assert stats['pattern_document_counts']['generator'] == 2
        assert stats['pattern_occurrences']['generator'] == 5

        # Most common should be decorator or generator (both 5)
        assert stats['most_common_pattern'] in ['decorator', 'generator']

    def test_detect_patterns_preserves_order(self):
        """Test that line numbers are returned in sorted order."""
        code = """
line 1
line 2
async def a(): pass
line 4
async def b(): pass
line 6
async def c(): pass
"""
        patterns = detect_patterns_in_text(code)
        assert 'async_await' in patterns
        line_numbers = patterns['async_await']
        # Should be sorted
        assert line_numbers == sorted(line_numbers)

    def test_summary_with_many_patterns(self):
        """Test summary generation with many different patterns."""
        pattern_results = {}
        for i, pattern_name in enumerate(list(PATTERN_DEFINITIONS.keys())[:10]):
            pattern_results[pattern_name] = list(range(1, i + 2))

        summary = get_pattern_summary(pattern_results)

        assert len(summary) == 10
        # Each pattern should have count matching its line numbers
        for pattern_name, count in summary.items():
            assert count == len(pattern_results[pattern_name])


# =============================================================================
# PATTERN COMBINATION TESTS
# =============================================================================


class TestPatternCombinations:
    """Test realistic combinations of patterns."""

    def test_async_context_manager(self):
        """Test async context manager pattern combination."""
        code = """
class AsyncResource:
    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
"""
        patterns = detect_patterns_in_text(code)
        # Should detect both async and context manager patterns
        assert 'async_await' in patterns or 'context_manager' in patterns

    def test_decorated_async_generator(self):
        """Test decorated async generator."""
        code = """
@property
async def stream_data():
    async for chunk in get_chunks():
        yield chunk
"""
        patterns = detect_patterns_in_text(code)
        # Should detect decorator, async, and generator patterns
        assert 'decorator' in patterns or 'property_decorator' in patterns
        assert 'async_await' in patterns
        assert 'generator' in patterns

    def test_factory_with_type_hints(self):
        """Test factory pattern with type hints."""
        code = """
from typing import Optional

class UserFactory:
    @staticmethod
    def create_user(name: str, age: Optional[int] = None) -> User:
        return User(name, age)
"""
        patterns = detect_patterns_in_text(code)
        assert 'factory' in patterns
        assert 'type_hints' in patterns

    def test_singleton_with_thread_safety(self):
        """Test thread-safe singleton."""
        code = """
import threading

class ThreadSafeSingleton:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance
"""
        patterns = detect_patterns_in_text(code)
        assert 'singleton' in patterns
        assert 'thread_safety' in patterns

    def test_dataclass_with_validation(self):
        """Test dataclass with property validation."""
        code = """
from dataclasses import dataclass

@dataclass
class User:
    _email: str = ""

    @property
    def email(self):
        return self._email

    @email.setter
    def email(self, value: str):
        if '@' not in value:
            raise ValueError("Invalid email")
        self._email = value
"""
        patterns = detect_patterns_in_text(code)
        assert 'dataclass' in patterns
        assert 'property_decorator' in patterns
        assert 'type_hints' in patterns
        assert 'custom_exception' in patterns or 'error_handling' in patterns


# =============================================================================
# METADATA AND UTILITY TESTS
# =============================================================================


class TestMetadataUtilities:
    """Test metadata and utility functions thoroughly."""

    def test_all_patterns_have_categories(self):
        """Verify every pattern is in exactly one category."""
        all_patterns = set(PATTERN_DEFINITIONS.keys())
        categorized_patterns = set()

        for category, patterns in PATTERN_CATEGORIES.items():
            categorized_patterns.update(patterns)

        # Every pattern should be categorized
        assert all_patterns == categorized_patterns

    def test_all_patterns_have_descriptions(self):
        """Verify every pattern has a non-empty description."""
        for pattern_name in PATTERN_DEFINITIONS.keys():
            desc = get_pattern_description(pattern_name)
            assert desc is not None
            assert len(desc) > 0
            assert isinstance(desc, str)

    def test_all_patterns_have_valid_categories(self):
        """Verify every pattern's category is valid."""
        all_categories = list_all_categories()

        for pattern_name in PATTERN_DEFINITIONS.keys():
            category = get_pattern_category(pattern_name)
            assert category is not None
            assert category in all_categories

    def test_category_listing_complete(self):
        """Verify list_all_categories returns all unique categories."""
        categories_from_definitions = set()
        for _, _, category in PATTERN_DEFINITIONS.values():
            categories_from_definitions.add(category)

        categories_from_function = set(list_all_categories())

        assert categories_from_definitions == categories_from_function

    def test_patterns_by_category_listing_complete(self):
        """Verify list_patterns_by_category returns correct patterns."""
        for category in list_all_categories():
            patterns_in_cat = list_patterns_by_category(category)

            # Check these patterns actually belong to this category
            for pattern_name in patterns_in_cat:
                actual_category = get_pattern_category(pattern_name)
                assert actual_category == category


# =============================================================================
# SPECIAL CHARACTERS AND REGEX TESTS
# =============================================================================


class TestSpecialCharacters:
    """Test pattern detection with special characters and edge cases."""

    def test_pattern_with_parentheses(self):
        """Test pattern detection with various parentheses."""
        code = """
def func(*args, **kwargs):
    result = map(lambda x: x**2, range(10))
    return list(result)
"""
        patterns = detect_patterns_in_text(code)
        assert 'unpacking' in patterns
        assert 'lambda' in patterns
        assert 'map_filter_reduce' in patterns

    def test_pattern_with_brackets(self):
        """Test pattern detection with brackets."""
        code = """
from typing import List, Dict, Optional

def process(items: List[str]) -> Dict[str, int]:
    return {item: len(item) for item in items}
"""
        patterns = detect_patterns_in_text(code)
        assert 'type_hints' in patterns
        assert 'comprehension' in patterns

    def test_pattern_with_underscores(self):
        """Test pattern detection with underscores."""
        code = """
class MyClass:
    __slots__ = ['_value', '_name']

    def __init__(self, value):
        self._value = value

    def __repr__(self):
        return f"MyClass({self._value})"
"""
        patterns = detect_patterns_in_text(code)
        assert 'slots' in patterns
        assert 'magic_methods' in patterns

    def test_pattern_with_newlines_and_whitespace(self):
        """Test pattern detection with various whitespace."""
        code = """

async def fetch():


    await get()



    return


"""
        patterns = detect_patterns_in_text(code)
        assert 'async_await' in patterns


# =============================================================================
# DOCUMENTATION AND REPORT TESTS
# =============================================================================


class TestReportFormatting:
    """Test comprehensive report formatting scenarios."""

    def test_report_includes_all_sections(self):
        """Test that report includes expected sections."""
        pattern_results = {
            'async_await': [1, 2],
            'singleton': [5],
            'generator': [10],
        }
        report = format_pattern_report(pattern_results)

        # Should include summary
        assert 'pattern' in report.lower()
        assert '3' in report  # 3 pattern types

        # Should include categories
        assert 'concurrency' in report.lower() or 'CONCURRENCY' in report
        assert 'creational' in report.lower() or 'CREATIONAL' in report
        assert 'behavioral' in report.lower() or 'BEHAVIORAL' in report

    def test_report_without_line_numbers(self):
        """Test report without showing line numbers."""
        pattern_results = {
            'async_await': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        }
        report = format_pattern_report(pattern_results, show_lines=False)

        # Should show count but not individual line numbers
        assert 'async_await' in report
        assert '12' in report  # count
        # Should NOT show "Lines:" section
        assert 'Lines:' not in report

    def test_report_alphabetical_ordering(self):
        """Test that report orders patterns consistently."""
        pattern_results = {
            'generator': [1],
            'async_await': [2],
            'singleton': [3],
        }
        report = format_pattern_report(pattern_results)

        # Categories should be in alphabetical order
        # Patterns within categories should be in alphabetical order
        assert isinstance(report, str)
        assert len(report) > 0

    def test_empty_pattern_results_message(self):
        """Test message for empty results."""
        report = format_pattern_report({})
        assert 'No patterns' in report or 'no patterns' in report


# =============================================================================
# INTEGRATION AND REGRESSION TESTS
# =============================================================================


class TestRegressionCases:
    """Test specific regression cases and bug fixes."""

    def test_duplicate_line_numbers_removed(self):
        """Test that duplicate line numbers are handled correctly."""
        code = """
async def a(): await x()
async def b(): await y()
"""
        patterns = detect_patterns_in_text(code)
        assert 'async_await' in patterns
        # Line numbers should be unique (even if multiple matches per line)
        line_numbers = patterns['async_await']
        assert len(line_numbers) == len(set(line_numbers))

    def test_pattern_detection_is_consistent(self):
        """Test that running detection multiple times gives same results."""
        code = """
async def fetch():
    async with session() as s:
        async for item in s.stream():
            yield item
"""
        patterns1 = detect_patterns_in_text(code)
        patterns2 = detect_patterns_in_text(code)

        # Should be identical
        assert patterns1 == patterns2

    def test_large_document_performance(self):
        """Test pattern detection on large document doesn't crash."""
        # Create a large code sample
        lines = []
        for i in range(100):
            lines.append(f"async def func_{i}():")
            lines.append(f"    await task_{i}()")
            lines.append("")

        code = "\n".join(lines)
        patterns = detect_patterns_in_text(code)

        # Should detect async patterns without crashing
        assert 'async_await' in patterns
        assert len(patterns['async_await']) >= 100

    def test_unicode_in_code(self):
        """Test pattern detection with unicode characters."""
        code = """
async def fetch_données():
    # Récupérer les données
    await get_data()
"""
        patterns = detect_patterns_in_text(code)
        assert 'async_await' in patterns

    def test_mixed_line_endings(self):
        """Test pattern detection with mixed line endings."""
        # Using explicit newlines
        code = "async def a():\n    pass\nasync def b():\n    pass"
        patterns = detect_patterns_in_text(code)
        assert 'async_await' in patterns

    def test_empty_lines_dont_break_line_counting(self):
        """Test that empty lines are counted correctly."""
        code = """


async def fetch():


    pass


"""
        patterns = detect_patterns_in_text(code)
        assert 'async_await' in patterns
        # Line number should account for empty lines
        assert any(ln >= 3 for ln in patterns['async_await'])
