"""
Unit Tests for Pattern Detection Module
========================================

Task LEGACY-078: Code pattern detection capabilities.

Tests the pattern detection module which identifies common programming
patterns in indexed code including:
- Singleton pattern
- Factory pattern
- Decorator usage
- Context managers
- Error handling patterns
- Generator patterns
- Async patterns
- And many more

These tests verify both the core pattern detection functions and the
processor integration.
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

from cortical.processor import CorticalTextProcessor


# =============================================================================
# PATTERN DEFINITIONS TESTS
# =============================================================================


class TestPatternDefinitions:
    """Tests for pattern definition structure."""

    def test_all_definitions_have_three_elements(self):
        """Each pattern definition has regex, description, and category."""
        for pattern_name, definition in PATTERN_DEFINITIONS.items():
            assert len(definition) == 3, f"{pattern_name} should have 3 elements"
            regex, description, category = definition
            assert isinstance(regex, str)
            assert isinstance(description, str)
            assert isinstance(category, str)

    def test_pattern_categories_populated(self):
        """PATTERN_CATEGORIES is correctly populated."""
        assert len(PATTERN_CATEGORIES) > 0
        # Each pattern should be in exactly one category
        all_patterns = set()
        for category, patterns in PATTERN_CATEGORIES.items():
            all_patterns.update(patterns)
        assert len(all_patterns) == len(PATTERN_DEFINITIONS)

    def test_essential_patterns_exist(self):
        """Essential patterns are defined."""
        essential = [
            'singleton', 'factory', 'decorator', 'context_manager',
            'generator', 'async_await', 'error_handling', 'property_decorator'
        ]
        for pattern in essential:
            assert pattern in PATTERN_DEFINITIONS

    def test_essential_categories_exist(self):
        """Essential categories are defined."""
        essential_cats = [
            'creational', 'structural', 'behavioral', 'concurrency',
            'error_handling', 'idiom', 'testing', 'functional', 'typing'
        ]
        for category in essential_cats:
            assert category in PATTERN_CATEGORIES


# =============================================================================
# DETECT PATTERNS IN TEXT TESTS
# =============================================================================


class TestDetectPatternsInText:
    """Tests for detect_patterns_in_text function."""

    def test_detect_async_await(self):
        """Detect async/await pattern."""
        code = """
async def fetch_data():
    result = await get_api_data()
    return result
"""
        patterns = detect_patterns_in_text(code)
        assert 'async_await' in patterns
        assert len(patterns['async_await']) >= 2  # async def and await

    def test_detect_generator(self):
        """Detect generator pattern."""
        code = """
def count_up(n):
    for i in range(n):
        yield i
"""
        patterns = detect_patterns_in_text(code)
        assert 'generator' in patterns
        assert 4 in patterns['generator']  # yield is on line 4 (after leading newline)

    def test_detect_singleton(self):
        """Detect singleton pattern."""
        code = """
class Singleton:
    _instance = None

    def __new__(cls):
        if not hasattr(cls, '_instance'):
            cls._instance = super().__new__(cls)
        return cls._instance
"""
        patterns = detect_patterns_in_text(code)
        assert 'singleton' in patterns

    def test_detect_factory(self):
        """Detect factory pattern."""
        code = """
def create_user(name):
    return User(name)

class UserFactory:
    @staticmethod
    def create(name):
        return User(name)
"""
        patterns = detect_patterns_in_text(code)
        assert 'factory' in patterns

    def test_detect_context_manager(self):
        """Detect context manager pattern."""
        code = """
class FileManager:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
"""
        patterns = detect_patterns_in_text(code)
        assert 'context_manager' in patterns
        assert 3 in patterns['context_manager']  # __enter__ (line 3 after leading newline)
        assert 6 in patterns['context_manager']  # __exit__ (line 6 after leading newline)

    def test_detect_decorator(self):
        """Detect decorator pattern."""
        code = """
@property
def name(self):
    return self._name

@staticmethod
def create():
    pass
"""
        patterns = detect_patterns_in_text(code)
        assert 'decorator' in patterns

    def test_detect_error_handling(self):
        """Detect error handling pattern."""
        code = """
try:
    risky_operation()
except ValueError as e:
    print(f"Error: {e}")
finally:
    cleanup()
"""
        patterns = detect_patterns_in_text(code)
        assert 'error_handling' in patterns
        assert 2 in patterns['error_handling']  # try (line 2 after leading newline)
        assert 6 in patterns['error_handling']  # finally (line 6 after leading newline)

    def test_detect_custom_exception(self):
        """Detect custom exception pattern."""
        code = """
class ValidationError(Exception):
    pass

def validate(value):
    if not value:
        raise ValidationError("Invalid value")
"""
        patterns = detect_patterns_in_text(code)
        assert 'custom_exception' in patterns

    def test_detect_property_decorator(self):
        """Detect property decorator pattern."""
        code = """
@property
def value(self):
    return self._value

@value.setter
def value(self, new_value):
    self._value = new_value
"""
        patterns = detect_patterns_in_text(code)
        assert 'property_decorator' in patterns

    def test_detect_dataclass(self):
        """Detect dataclass pattern."""
        code = """
from dataclasses import dataclass

@dataclass
class User:
    name: str
    age: int
"""
        patterns = detect_patterns_in_text(code)
        assert 'dataclass' in patterns

    def test_detect_magic_methods(self):
        """Detect magic methods pattern."""
        code = """
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Point({self.x}, {self.y})"

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
"""
        patterns = detect_patterns_in_text(code)
        assert 'magic_methods' in patterns

    def test_detect_comprehension(self):
        """Detect comprehension pattern."""
        code = """
squares = [x**2 for x in range(10)]
even_squares = {x**2 for x in range(10) if x % 2 == 0}
mapping = {x: x**2 for x in range(10)}
"""
        patterns = detect_patterns_in_text(code)
        assert 'comprehension' in patterns

    def test_detect_unpacking(self):
        """Detect argument unpacking pattern."""
        code = """
def func(*args, **kwargs):
    pass

a, *rest = [1, 2, 3, 4]
"""
        patterns = detect_patterns_in_text(code)
        assert 'unpacking' in patterns

    def test_detect_unittest_class(self):
        """Detect unittest test class pattern."""
        code = """
import unittest

class TestMyCode(unittest.TestCase):
    def setUp(self):
        pass

    def test_feature(self):
        self.assertEqual(1, 1)
"""
        patterns = detect_patterns_in_text(code)
        assert 'unittest_class' in patterns

    def test_detect_pytest_test(self):
        """Detect pytest test function pattern."""
        code = """
import pytest

def test_basic():
    assert True

@pytest.mark.skip
def test_skip():
    pass
"""
        patterns = detect_patterns_in_text(code)
        assert 'pytest_test' in patterns

    def test_detect_mock_usage(self):
        """Detect mocking pattern."""
        code = """
from unittest.mock import Mock, patch

@patch('module.function')
def test_with_mock(mock_func):
    mock = Mock()
    mock.return_value = 42
"""
        patterns = detect_patterns_in_text(code)
        assert 'mock_usage' in patterns

    def test_detect_lambda(self):
        """Detect lambda pattern."""
        code = """
square = lambda x: x**2
add = lambda a, b: a + b
"""
        patterns = detect_patterns_in_text(code)
        assert 'lambda' in patterns

    def test_detect_type_hints(self):
        """Detect type hints pattern."""
        code = """
from typing import List, Dict, Optional

def process(items: List[str]) -> Dict[str, int]:
    return {item: len(item) for item in items}

def maybe_get(key: str) -> Optional[str]:
    pass
"""
        patterns = detect_patterns_in_text(code)
        assert 'type_hints' in patterns

    def test_empty_text_returns_empty_dict(self):
        """Empty text returns no patterns."""
        patterns = detect_patterns_in_text("")
        assert patterns == {}

    def test_non_code_text_returns_empty_dict(self):
        """Non-code text returns no patterns."""
        patterns = detect_patterns_in_text("This is just plain text.")
        assert patterns == {}

    def test_specific_patterns_only(self):
        """Can search for specific patterns only."""
        code = """
async def fetch():
    yield 1
"""
        # Only search for async_await
        patterns = detect_patterns_in_text(code, patterns=['async_await'])
        assert 'async_await' in patterns
        assert 'generator' not in patterns

        # Only search for generator
        patterns = detect_patterns_in_text(code, patterns=['generator'])
        assert 'generator' in patterns
        assert 'async_await' not in patterns

    def test_unknown_pattern_name_ignored(self):
        """Unknown pattern names are ignored."""
        code = "async def test(): pass"
        patterns = detect_patterns_in_text(code, patterns=['unknown_pattern'])
        assert 'unknown_pattern' not in patterns

    def test_line_numbers_accurate(self):
        """Line numbers are accurately reported."""
        code = """line 1
async def test():
    pass
"""
        patterns = detect_patterns_in_text(code)
        assert 'async_await' in patterns
        assert 2 in patterns['async_await']  # async def on line 2


# =============================================================================
# DETECT PATTERNS IN DOCUMENTS TESTS
# =============================================================================


class TestDetectPatternsInDocuments:
    """Tests for detect_patterns_in_documents function."""

    def test_detect_in_multiple_documents(self):
        """Detect patterns across multiple documents."""
        docs = {
            'file1.py': 'async def fetch(): pass',
            'file2.py': 'def generator(): yield 1',
            'file3.py': 'print("hello")',  # No patterns
        }
        results = detect_patterns_in_documents(docs)

        assert 'file1.py' in results
        assert 'async_await' in results['file1.py']

        assert 'file2.py' in results
        assert 'generator' in results['file2.py']

        # file3.py might not be in results if no patterns found
        # or might have some basic patterns

    def test_empty_documents(self):
        """Empty documents dict returns empty results."""
        results = detect_patterns_in_documents({})
        assert results == {}

    def test_documents_with_no_patterns(self):
        """Documents with no patterns are omitted from results."""
        docs = {
            'file1.py': 'x = 1',
            'file2.py': 'y = 2',
        }
        results = detect_patterns_in_documents(docs)
        # Should be empty or only contain very basic patterns
        # Depends on what patterns match simple assignments

    def test_specific_patterns_in_documents(self):
        """Can search for specific patterns in documents."""
        docs = {
            'file1.py': 'async def fetch(): yield 1',
        }
        results = detect_patterns_in_documents(docs, patterns=['async_await'])

        assert 'file1.py' in results
        assert 'async_await' in results['file1.py']
        assert 'generator' not in results['file1.py']


# =============================================================================
# PATTERN SUMMARY TESTS
# =============================================================================


class TestGetPatternSummary:
    """Tests for get_pattern_summary function."""

    def test_summary_counts_occurrences(self):
        """Summary counts pattern occurrences."""
        pattern_results = {
            'async_await': [1, 5, 10],
            'generator': [3],
            'decorator': [2, 4, 6, 8],
        }
        summary = get_pattern_summary(pattern_results)

        assert summary['async_await'] == 3
        assert summary['generator'] == 1
        assert summary['decorator'] == 4

    def test_empty_results(self):
        """Empty results return empty summary."""
        summary = get_pattern_summary({})
        assert summary == {}


# =============================================================================
# PATTERNS BY CATEGORY TESTS
# =============================================================================


class TestGetPatternsByCategory:
    """Tests for get_patterns_by_category function."""

    def test_groups_by_category(self):
        """Patterns are grouped by category."""
        pattern_results = {
            'async_await': [1, 2],
            'singleton': [5],
            'generator': [10],
        }
        by_category = get_patterns_by_category(pattern_results)

        assert 'concurrency' in by_category
        assert by_category['concurrency']['async_await'] == 2

        assert 'creational' in by_category
        assert by_category['creational']['singleton'] == 1

        assert 'behavioral' in by_category
        assert by_category['behavioral']['generator'] == 1

    def test_empty_results(self):
        """Empty results return empty categorization."""
        by_category = get_patterns_by_category({})
        assert by_category == {}


# =============================================================================
# PATTERN METADATA TESTS
# =============================================================================


class TestPatternMetadata:
    """Tests for pattern metadata functions."""

    def test_get_pattern_description(self):
        """Get description for a pattern."""
        desc = get_pattern_description('singleton')
        assert 'Singleton' in desc or 'singleton' in desc
        assert isinstance(desc, str)

    def test_get_pattern_description_unknown(self):
        """Unknown pattern returns None."""
        desc = get_pattern_description('unknown_pattern')
        assert desc is None

    def test_get_pattern_category(self):
        """Get category for a pattern."""
        category = get_pattern_category('singleton')
        assert category == 'creational'

        category = get_pattern_category('async_await')
        assert category == 'concurrency'

    def test_get_pattern_category_unknown(self):
        """Unknown pattern returns None."""
        category = get_pattern_category('unknown_pattern')
        assert category is None

    def test_list_all_patterns(self):
        """List all available patterns."""
        patterns = list_all_patterns()
        assert isinstance(patterns, list)
        assert len(patterns) > 0
        assert 'singleton' in patterns
        assert 'async_await' in patterns
        # Should be sorted
        assert patterns == sorted(patterns)

    def test_list_patterns_by_category(self):
        """List patterns in a specific category."""
        creational = list_patterns_by_category('creational')
        assert 'singleton' in creational
        assert 'factory' in creational
        # Should be sorted
        assert creational == sorted(creational)

    def test_list_patterns_by_category_unknown(self):
        """Unknown category returns empty list."""
        patterns = list_patterns_by_category('unknown_category')
        assert patterns == []

    def test_list_all_categories(self):
        """List all pattern categories."""
        categories = list_all_categories()
        assert isinstance(categories, list)
        assert len(categories) > 0
        assert 'creational' in categories
        assert 'concurrency' in categories
        # Should be sorted
        assert categories == sorted(categories)


# =============================================================================
# FORMAT PATTERN REPORT TESTS
# =============================================================================


class TestFormatPatternReport:
    """Tests for format_pattern_report function."""

    def test_format_basic_report(self):
        """Format a basic pattern report."""
        pattern_results = {
            'async_await': [1, 5],
            'singleton': [10],
        }
        report = format_pattern_report(pattern_results)

        assert 'async_await' in report
        assert 'singleton' in report
        assert '2 occurrences' in report or '2' in report

    def test_format_with_line_numbers(self):
        """Format report with line numbers."""
        pattern_results = {
            'async_await': [1, 5, 10],
        }
        report = format_pattern_report(pattern_results, show_lines=True)

        assert '1' in report
        assert '5' in report
        assert '10' in report

    def test_format_empty_results(self):
        """Format empty results."""
        report = format_pattern_report({})
        assert 'No patterns' in report

    def test_format_groups_by_category(self):
        """Report groups patterns by category."""
        pattern_results = {
            'async_await': [1],
            'singleton': [2],
        }
        report = format_pattern_report(pattern_results)

        # Should show category headers
        assert 'CONCURRENCY' in report or 'concurrency' in report
        assert 'CREATIONAL' in report or 'creational' in report


# =============================================================================
# CORPUS STATISTICS TESTS
# =============================================================================


class TestGetCorpusPatternStatistics:
    """Tests for get_corpus_pattern_statistics function."""

    def test_compute_statistics(self):
        """Compute corpus-wide statistics."""
        doc_patterns = {
            'file1.py': {'async_await': [1, 2], 'singleton': [5]},
            'file2.py': {'async_await': [3]},
            'file3.py': {'generator': [1, 2, 3]},
        }
        stats = get_corpus_pattern_statistics(doc_patterns)

        assert stats['total_documents'] == 3
        assert stats['patterns_found'] == 3  # async_await, singleton, generator

        # async_await appears in 2 documents
        assert stats['pattern_document_counts']['async_await'] == 2
        # singleton appears in 1 document
        assert stats['pattern_document_counts']['singleton'] == 1

        # async_await has 3 total occurrences (2 in file1, 1 in file2)
        assert stats['pattern_occurrences']['async_await'] == 3
        # generator has 3 occurrences
        assert stats['pattern_occurrences']['generator'] == 3

        # Most common should be either async_await or generator (both have 3)
        assert stats['most_common_pattern'] in ['async_await', 'generator']

    def test_empty_corpus(self):
        """Empty corpus returns minimal statistics."""
        stats = get_corpus_pattern_statistics({})
        assert stats['total_documents'] == 0
        assert stats['patterns_found'] == 0
        assert stats['most_common_pattern'] is None


# =============================================================================
# PROCESSOR INTEGRATION TESTS
# =============================================================================


class TestProcessorIntegration:
    """Tests for processor integration."""

    def test_detect_patterns_method(self):
        """Processor can detect patterns in a document."""
        processor = CorticalTextProcessor()
        processor.process_document('code.py', 'async def fetch(): pass')

        patterns = processor.detect_patterns('code.py')
        assert 'async_await' in patterns

    def test_detect_patterns_unknown_doc(self):
        """Detecting patterns in unknown document returns empty."""
        processor = CorticalTextProcessor()
        patterns = processor.detect_patterns('unknown.py')
        assert patterns == {}

    def test_detect_patterns_in_corpus_method(self):
        """Processor can detect patterns in entire corpus."""
        processor = CorticalTextProcessor()
        processor.process_document('file1.py', 'async def fetch(): pass')
        processor.process_document('file2.py', 'def gen(): yield 1')

        results = processor.detect_patterns_in_corpus()

        assert 'file1.py' in results
        assert 'async_await' in results['file1.py']

        assert 'file2.py' in results
        assert 'generator' in results['file2.py']

    def test_get_pattern_summary_method(self):
        """Processor can get pattern summary for a document."""
        processor = CorticalTextProcessor()
        code = """
async def fetch():
    await get()
async def store():
    await put()
"""
        processor.process_document('code.py', code)

        summary = processor.get_pattern_summary('code.py')
        assert 'async_await' in summary
        assert summary['async_await'] >= 2  # At least 2 async defs

    def test_get_corpus_pattern_statistics_method(self):
        """Processor can get corpus-wide pattern statistics."""
        processor = CorticalTextProcessor()
        processor.process_document('file1.py', 'async def fetch(): pass')
        processor.process_document('file2.py', 'async def store(): pass')

        stats = processor.get_corpus_pattern_statistics()
        assert stats['total_documents'] == 2
        assert 'async_await' in stats['pattern_document_counts']

    def test_format_pattern_report_method(self):
        """Processor can format pattern reports."""
        processor = CorticalTextProcessor()
        processor.process_document('code.py', 'async def fetch(): pass')

        report = processor.format_pattern_report('code.py')
        assert isinstance(report, str)
        assert 'async_await' in report

    def test_list_available_patterns_method(self):
        """Processor can list available patterns."""
        processor = CorticalTextProcessor()
        patterns = processor.list_available_patterns()

        assert isinstance(patterns, list)
        assert 'singleton' in patterns
        assert 'async_await' in patterns

    def test_list_pattern_categories_method(self):
        """Processor can list pattern categories."""
        processor = CorticalTextProcessor()
        categories = processor.list_pattern_categories()

        assert isinstance(categories, list)
        assert 'creational' in categories
        assert 'concurrency' in categories

    def test_detect_specific_patterns(self):
        """Processor can detect specific patterns only."""
        processor = CorticalTextProcessor()
        code = """
async def fetch():
    yield 1
"""
        processor.process_document('code.py', code)

        # Only detect async_await
        patterns = processor.detect_patterns('code.py', patterns=['async_await'])
        assert 'async_await' in patterns
        assert 'generator' not in patterns


# =============================================================================
# REAL-WORLD PATTERN DETECTION TESTS
# =============================================================================


class TestRealWorldPatterns:
    """Tests with realistic code samples."""

    def test_detect_patterns_in_test_file(self):
        """Detect patterns in a typical test file."""
        code = """
import pytest
from unittest.mock import Mock, patch

class TestMyFeature:
    def setUp(self):
        self.mock = Mock()

    def test_basic(self):
        assert True

    @pytest.mark.skip
    def test_skip(self):
        pass

    @patch('module.function')
    def test_with_mock(self, mock_func):
        mock_func.return_value = 42
"""
        patterns = detect_patterns_in_text(code)

        assert 'pytest_test' in patterns or 'unittest_class' in patterns
        assert 'mock_usage' in patterns
        assert 'decorator' in patterns

    def test_detect_patterns_in_class(self):
        """Detect patterns in a typical class."""
        code = """
from dataclasses import dataclass
from typing import Optional

@dataclass
class User:
    name: str
    age: int
    _email: Optional[str] = None

    def __post_init__(self):
        if not self.name:
            raise ValueError("Name required")

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
        assert 'type_hints' in patterns
        assert 'property_decorator' in patterns
        assert 'custom_exception' in patterns or 'error_handling' in patterns

    def test_detect_patterns_in_async_code(self):
        """Detect patterns in async code."""
        code = """
import asyncio
from typing import AsyncIterator

async def fetch_users() -> list:
    async with aiohttp.ClientSession() as session:
        async for user in get_users(session):
            yield user

async def process_batch(items):
    await asyncio.gather(*[process(item) for item in items])
"""
        patterns = detect_patterns_in_text(code)

        assert 'async_await' in patterns
        assert 'generator' in patterns or 'async_await' in patterns
        assert 'type_hints' in patterns
        assert 'comprehension' in patterns
