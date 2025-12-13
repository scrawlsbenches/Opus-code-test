"""
Unit Tests for Query Definitions Module
========================================

Task #173: Unit tests for cortical/query/definitions.py.

Tests definition detection and extraction functions:
- is_definition_query: Query classification
- find_definition_in_text: Definition extraction from source
- find_definition_passages: Main definition search
- detect_definition_query: Structured detection
- apply_definition_boost: Passage boosting
- is_test_file: Test file detection
- boost_definition_documents: Document boosting

All tests use mock data and run in <2 seconds.
"""

import re
import pytest

from cortical.query.definitions import (
    is_definition_query,
    find_definition_in_text,
    find_definition_passages,
    detect_definition_query,
    apply_definition_boost,
    is_test_file,
    boost_definition_documents,
    DEFINITION_BOOST,
    DEFINITION_QUERY_PATTERNS,
    DEFINITION_SOURCE_PATTERNS,
)


# =============================================================================
# QUERY CLASSIFICATION TESTS
# =============================================================================


class TestIsDefinitionQuery:
    """Tests for is_definition_query() - query classification."""

    def test_empty_query(self):
        """Empty query is not a definition query."""
        result = is_definition_query("")
        assert result == (False, None, None)

    def test_plain_text_query(self):
        """Plain text query is not a definition query."""
        result = is_definition_query("how does this work")
        assert result == (False, None, None)

    def test_class_query_lowercase(self):
        """Recognizes 'class Minicolumn' as class definition query."""
        is_def, def_type, identifier = is_definition_query("class Minicolumn")
        assert is_def is True
        assert def_type == 'class'
        assert identifier == 'Minicolumn'

    def test_class_query_uppercase(self):
        """Recognizes 'CLASS Processor' (case insensitive)."""
        is_def, def_type, identifier = is_definition_query("CLASS Processor")
        assert is_def is True
        assert def_type == 'class'
        assert identifier == 'Processor'

    def test_def_query(self):
        """Recognizes 'def compute_pagerank' as function definition query."""
        is_def, def_type, identifier = is_definition_query("def compute_pagerank")
        assert is_def is True
        assert def_type == 'function'
        assert identifier == 'compute_pagerank'

    def test_function_keyword(self):
        """Recognizes 'function tokenize' as function definition query."""
        is_def, def_type, identifier = is_definition_query("function tokenize")
        assert is_def is True
        assert def_type == 'function'
        assert identifier == 'tokenize'

    def test_method_query(self):
        """Recognizes 'method process_document' as method definition query."""
        is_def, def_type, identifier = is_definition_query("method process_document")
        assert is_def is True
        assert def_type == 'method'
        assert identifier == 'process_document'

    def test_query_with_extra_words(self):
        """Handles query with extra words after identifier."""
        is_def, def_type, identifier = is_definition_query("class Minicolumn definition")
        assert is_def is True
        assert def_type == 'class'
        assert identifier == 'Minicolumn'

    def test_query_with_leading_words(self):
        """Handles query with words before the pattern."""
        is_def, def_type, identifier = is_definition_query("find class Minicolumn")
        assert is_def is True
        assert def_type == 'class'
        assert identifier == 'Minicolumn'

    def test_snake_case_identifier(self):
        """Handles snake_case identifiers."""
        is_def, def_type, identifier = is_definition_query("def compute_all")
        assert is_def is True
        assert def_type == 'function'
        assert identifier == 'compute_all'

    def test_camel_case_identifier(self):
        """Handles CamelCase identifiers."""
        is_def, def_type, identifier = is_definition_query("class CorticalTextProcessor")
        assert is_def is True
        assert def_type == 'class'
        assert identifier == 'CorticalTextProcessor'

    def test_pattern_not_at_start(self):
        """Pattern can appear anywhere in query."""
        is_def, def_type, identifier = is_definition_query("where is class Foo defined")
        assert is_def is True
        assert def_type == 'class'
        assert identifier == 'Foo'

    def test_first_pattern_wins(self):
        """If multiple patterns match, first one wins."""
        is_def, def_type, identifier = is_definition_query("class Foo def bar")
        assert is_def is True
        assert def_type == 'class'
        assert identifier == 'Foo'


# =============================================================================
# DEFINITION EXTRACTION TESTS
# =============================================================================


class TestFindDefinitionInText:
    """Tests for find_definition_in_text() - extract definition from source."""

    def test_empty_text(self):
        """Empty text returns None."""
        result = find_definition_in_text("", "Foo", "class")
        assert result is None

    def test_class_not_found(self):
        """Class not in text returns None."""
        text = "def some_function():\n    pass"
        result = find_definition_in_text(text, "Minicolumn", "class")
        assert result is None

    def test_python_class_definition(self):
        """Finds Python class definition."""
        text = """
import sys

class Minicolumn:
    '''A test class.'''
    def __init__(self):
        pass
"""
        result = find_definition_in_text(text, "Minicolumn", "class")
        assert result is not None
        passage, start, end = result
        assert "class Minicolumn:" in passage
        assert start >= 0
        assert end > start

    def test_python_function_definition(self):
        """Finds Python function definition."""
        text = """
def compute_pagerank(graph, damping=0.85):
    '''Compute PageRank scores.'''
    return {}
"""
        result = find_definition_in_text(text, "compute_pagerank", "function")
        assert result is not None
        passage, start, end = result
        assert "def compute_pagerank" in passage

    def test_python_method_definition(self):
        """Finds Python method definition inside a class."""
        text = """
class Processor:
    def process_document(self, doc_id, text):
        '''Process a document.'''
        pass
"""
        result = find_definition_in_text(text, "process_document", "method")
        assert result is not None
        passage, start, end = result
        assert "def process_document" in passage

    def test_javascript_class_definition(self):
        """Finds JavaScript class definition."""
        text = """
class UserManager {
  constructor() {
    this.users = [];
  }
}
"""
        result = find_definition_in_text(text, "UserManager", "class")
        assert result is not None
        passage, start, end = result
        assert "class UserManager" in passage

    def test_javascript_function_definition(self):
        """Finds JavaScript function definition."""
        text = """
function handleClick(event) {
  console.log(event);
}
"""
        result = find_definition_in_text(text, "handleClick", "function")
        assert result is not None
        passage, start, end = result
        assert "function handleClick" in passage

    def test_javascript_const_function(self):
        """Finds JavaScript const arrow function."""
        text = """
const fetchData = async (url) => {
  return await fetch(url);
};
"""
        result = find_definition_in_text(text, "fetchData", "function")
        assert result is not None
        passage, start, end = result
        assert "const fetchData" in passage

    def test_case_insensitive_match(self):
        """Definition matching is case insensitive."""
        text = "class minicolumn:\n    pass"
        result = find_definition_in_text(text, "Minicolumn", "class")
        assert result is not None

    def test_context_chars_respected(self):
        """Context characters parameter controls passage length."""
        text = "def foo():\n" + "    pass\n" * 100  # Long function

        short_result = find_definition_in_text(text, "foo", "function", context_chars=50)
        long_result = find_definition_in_text(text, "foo", "function", context_chars=500)

        assert short_result is not None
        assert long_result is not None
        short_passage, _, _ = short_result
        long_passage, _, _ = long_result
        assert len(long_passage) >= len(short_passage)

    def test_boundary_detection(self):
        """Extracts up to next blank line boundary."""
        text = """
def compute_all(self):
    '''Compute everything.'''
    self.compute_tfidf()
    self.compute_importance()

def other_function():
    pass
"""
        result = find_definition_in_text(text, "compute_all", "function")
        assert result is not None
        passage, _, _ = result
        # Should stop at blank line before other_function
        assert "compute_all" in passage
        assert "compute_importance" in passage
        # Should not include other_function
        assert "other_function" not in passage

    def test_multiline_definition(self):
        """Handles multiline function signatures."""
        text = """
def complex_function(
    arg1,
    arg2,
    arg3
):
    return arg1 + arg2 + arg3
"""
        result = find_definition_in_text(text, "complex_function", "function")
        assert result is not None
        passage, _, _ = result
        assert "complex_function" in passage

    def test_identifier_with_special_chars(self):
        """Handles identifiers that need escaping in regex."""
        text = "def __init__(self):\n    pass"
        result = find_definition_in_text(text, "__init__", "function")
        assert result is not None
        passage, _, _ = result
        assert "__init__" in passage

    def test_invalid_def_type(self):
        """Invalid def_type returns None."""
        text = "class Foo:\n    pass"
        result = find_definition_in_text(text, "Foo", "invalid_type")
        assert result is None

    def test_passage_starts_with_definition_line(self):
        """
        Regression test for Task #179: Passage must start with definition line.

        Bug: Previously, find_definition_in_text used `start = match.start() - 50`,
        which could place start in the middle of an earlier line. When showcase.py
        extracted the first line, it showed truncated/wrong content.

        Fix: Now finds the start of the line containing the match, ensuring the
        passage always starts with the actual definition line.
        """
        # Simulate a realistic file structure with content before the definition
        text = """
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class DataRecord:
    id: str
    content: str

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DataProcessor:
    '''Main processor for handling data records.'''

    def __init__(self):
        self._records = {}

    def clear(self):
        '''Remove all records.'''
        self._records.clear()


def calculate_statistics(records: List[DataRecord]) -> Dict:
    '''Calculate statistics for records.'''
    if not records:
        return {}
    return {'count': len(records)}
"""

        # Test class definition
        result = find_definition_in_text(text, "DataProcessor", "class")
        assert result is not None
        passage, start, end = result

        # The passage should start with the actual definition line
        first_line = passage.strip().split('\n')[0]
        assert first_line.startswith("class DataProcessor"), (
            f"Expected first line to start with 'class DataProcessor', "
            f"but got: {first_line!r}"
        )
        # Should NOT start with truncated content like "etadata is None"
        assert "metadata" not in first_line.lower() or "dataprocessor" in first_line.lower()

        # Test function definition
        result = find_definition_in_text(text, "calculate_statistics", "function")
        assert result is not None
        passage, start, end = result

        # The passage should start with the function definition
        first_line = passage.strip().split('\n')[0]
        assert first_line.startswith("def calculate_statistics"), (
            f"Expected first line to start with 'def calculate_statistics', "
            f"but got: {first_line!r}"
        )
        # Should NOT start with truncated content like "records.clear()"
        assert "calculate_statistics" in first_line

    def test_definition_at_file_start(self):
        """Definition at the very start of file works correctly."""
        text = "class FirstClass:\n    pass"
        result = find_definition_in_text(text, "FirstClass", "class")
        assert result is not None
        passage, start, end = result

        # Start should be 0 (beginning of file)
        assert start == 0
        # First line should be the definition
        first_line = passage.strip().split('\n')[0]
        assert first_line.startswith("class FirstClass")


# =============================================================================
# DEFINITION PASSAGES SEARCH TESTS
# =============================================================================


class TestFindDefinitionPassages:
    """Tests for find_definition_passages() - main search function."""

    def test_non_definition_query(self):
        """Non-definition query returns empty list."""
        documents = {"doc1": "class Foo:\n    pass"}
        result = find_definition_passages("how does this work", documents)
        assert result == []

    def test_no_documents(self):
        """Empty documents dict returns empty list."""
        result = find_definition_passages("class Foo", {})
        assert result == []

    def test_definition_not_found(self):
        """Definition not in any document returns empty list."""
        documents = {"doc1": "def bar():\n    pass"}
        result = find_definition_passages("class Foo", documents)
        assert result == []

    def test_find_class_definition(self):
        """Finds class definition and returns boosted passage."""
        documents = {
            "minicolumn.py": """
class Minicolumn:
    '''Core data structure.'''
    def __init__(self):
        pass
"""
        }
        results = find_definition_passages("class Minicolumn", documents)

        assert len(results) > 0
        passage, doc_id, start, end, score = results[0]
        assert "class Minicolumn:" in passage
        assert doc_id == "minicolumn.py"
        assert score == DEFINITION_BOOST  # Default boost

    def test_find_function_definition(self):
        """Finds function definition."""
        documents = {
            "analysis.py": """
def compute_pagerank(graph):
    return {}
"""
        }
        results = find_definition_passages("def compute_pagerank", documents)

        assert len(results) > 0
        passage, doc_id, _, _, score = results[0]
        assert "compute_pagerank" in passage
        assert doc_id == "analysis.py"

    def test_multiple_documents(self):
        """Searches across multiple documents."""
        documents = {
            "file1.py": "class Foo:\n    pass",
            "file2.py": "def bar():\n    pass",
            "file3.py": "class Foo:\n    # Another definition\n    pass"
        }
        results = find_definition_passages("class Foo", documents)

        # Should find Foo in both file1 and file3
        assert len(results) == 2
        doc_ids = {r[1] for r in results}
        assert "file1.py" in doc_ids
        assert "file3.py" in doc_ids

    def test_test_file_penalty(self):
        """Test files get score penalty."""
        documents = {
            "minicolumn.py": "class Minicolumn:\n    pass",
            "test_minicolumn.py": "class Minicolumn:\n    pass"  # Test file
        }
        results = find_definition_passages("class Minicolumn", documents)

        assert len(results) == 2
        # Results are sorted by score, so source file should be first
        assert results[0][1] == "minicolumn.py"
        assert results[1][1] == "test_minicolumn.py"
        # Test file should have lower score
        assert results[0][4] > results[1][4]

    def test_custom_boost(self):
        """Custom boost factor is applied."""
        documents = {"file.py": "class Foo:\n    pass"}
        results = find_definition_passages("class Foo", documents, boost=10.0)

        assert len(results) > 0
        score = results[0][4]
        assert score == 10.0  # Custom boost applied

    def test_custom_context_chars(self):
        """Custom context_chars parameter is passed through."""
        documents = {
            "file.py": "def foo():\n" + "    pass\n" * 50
        }
        short_results = find_definition_passages(
            "def foo", documents, context_chars=50
        )
        long_results = find_definition_passages(
            "def foo", documents, context_chars=500
        )

        short_passage = short_results[0][0]
        long_passage = long_results[0][0]
        assert len(long_passage) >= len(short_passage)

    def test_results_sorted_by_score(self):
        """Results are sorted by score (highest first)."""
        documents = {
            "source.py": "class Foo:\n    pass",
            "tests/test_foo.py": "class Foo:\n    pass",
            "tests/unit/test_foo.py": "class Foo:\n    pass"
        }
        results = find_definition_passages("class Foo", documents)

        # Scores should be descending
        scores = [r[4] for r in results]
        assert scores == sorted(scores, reverse=True)


# =============================================================================
# STRUCTURED DETECTION TESTS
# =============================================================================


class TestDetectDefinitionQuery:
    """Tests for detect_definition_query() - structured detection."""

    def test_non_definition_query(self):
        """Non-definition query returns all None/False."""
        result = detect_definition_query("how does this work")
        assert result['is_definition_query'] is False
        assert result['definition_type'] is None
        assert result['identifier'] is None
        assert result['pattern'] is None

    def test_class_query_detection(self):
        """Detects class query with pattern."""
        result = detect_definition_query("class Minicolumn")
        assert result['is_definition_query'] is True
        assert result['definition_type'] == 'class'
        assert result['identifier'] == 'Minicolumn'
        assert result['pattern'] is not None
        # Pattern should match the actual definition
        pattern = re.compile(result['pattern'], re.IGNORECASE)
        assert pattern.search("class Minicolumn:")
        assert pattern.search("class Minicolumn(object):")

    def test_def_query_detection(self):
        """Detects def query with pattern."""
        result = detect_definition_query("def compute_pagerank")
        assert result['is_definition_query'] is True
        assert result['definition_type'] == 'function'
        assert result['identifier'] == 'compute_pagerank'
        assert result['pattern'] is not None
        # Pattern should match actual definition
        pattern = re.compile(result['pattern'], re.IGNORECASE)
        assert pattern.search("def compute_pagerank(")

    def test_function_keyword_detection(self):
        """Detects 'function' keyword queries."""
        result = detect_definition_query("function handleClick")
        assert result['is_definition_query'] is True
        assert result['definition_type'] == 'function'
        assert result['identifier'] == 'handleClick'

    def test_method_query_detection(self):
        """Detects method query."""
        result = detect_definition_query("method process_document")
        assert result['is_definition_query'] is True
        assert result['definition_type'] == 'method'
        assert result['identifier'] == 'process_document'

    def test_pattern_matches_actual_code(self):
        """Generated pattern matches actual code definitions."""
        result = detect_definition_query("class Processor")
        pattern = re.compile(result['pattern'], re.IGNORECASE)

        # Should match various class definition styles
        assert pattern.search("class Processor:")
        assert pattern.search("class Processor(Base):")
        assert pattern.search("class Processor ( Base ) :")

        # Should not match non-definitions
        assert not pattern.search("# class Processor is great")
        assert not pattern.search("processor = Processor()")

    def test_identifier_with_underscores(self):
        """Handles identifiers with underscores."""
        result = detect_definition_query("def __init__")
        assert result['identifier'] == "__init__"
        # Pattern should escape special regex chars
        pattern = re.compile(result['pattern'], re.IGNORECASE)
        assert pattern.search("def __init__(")

    def test_case_insensitive_keyword(self):
        """Keywords are case insensitive."""
        result_lower = detect_definition_query("class Foo")
        result_upper = detect_definition_query("CLASS Foo")

        assert result_lower['is_definition_query'] is True
        assert result_upper['is_definition_query'] is True
        assert result_lower['identifier'] == result_upper['identifier']


# =============================================================================
# PASSAGE BOOSTING TESTS
# =============================================================================


class TestApplyDefinitionBoost:
    """Tests for apply_definition_boost() - boost definition passages."""

    def test_non_definition_query(self):
        """Non-definition query returns passages unchanged."""
        passages = [
            ("some text", "doc1", 0, 100, 1.0),
            ("other text", "doc2", 0, 100, 0.5)
        ]
        result = apply_definition_boost(passages, "how does this work")
        assert result == passages

    def test_empty_passages(self):
        """Empty passages returns empty."""
        result = apply_definition_boost([], "class Foo")
        assert result == []

    def test_passage_with_definition_gets_boost(self):
        """Passage containing actual definition gets boosted."""
        passages = [
            ("class Minicolumn:\n    pass", "minicolumn.py", 0, 100, 1.0),
            ("using Minicolumn in code", "usage.py", 0, 100, 1.0)
        ]
        result = apply_definition_boost(passages, "class Minicolumn", boost_factor=3.0)

        # First passage has definition, should be boosted
        assert result[0][4] == 3.0  # 1.0 * 3.0
        # Second passage is just usage, unchanged
        assert result[1][4] == 1.0

    def test_results_sorted_by_boosted_score(self):
        """Results are re-sorted after boosting."""
        passages = [
            ("using Foo", "usage.py", 0, 100, 5.0),  # High score, no definition
            ("class Foo:\n    pass", "foo.py", 0, 100, 1.0)  # Low score, has definition
        ]
        result = apply_definition_boost(passages, "class Foo", boost_factor=10.0)

        # After boosting, definition passage should be first
        # foo.py: 1.0 * 10.0 = 10.0 (now highest)
        # usage.py: 5.0 (unchanged)
        assert result[0][1] == "foo.py"  # Definition file now first
        assert result[0][4] == 10.0
        assert result[1][1] == "usage.py"
        assert result[1][4] == 5.0

    def test_custom_boost_factor(self):
        """Custom boost factor is applied."""
        passages = [("class Foo:\n    pass", "foo.py", 0, 100, 2.0)]
        result = apply_definition_boost(passages, "class Foo", boost_factor=5.0)
        assert result[0][4] == 10.0  # 2.0 * 5.0

    def test_multiple_definitions_all_boosted(self):
        """Multiple passages with definitions all get boosted."""
        passages = [
            ("class Foo:\n    # Version 1", "foo_v1.py", 0, 100, 1.0),
            ("class Foo:\n    # Version 2", "foo_v2.py", 0, 100, 1.0),
            ("using Foo", "usage.py", 0, 100, 1.0)
        ]
        result = apply_definition_boost(passages, "class Foo", boost_factor=3.0)

        # Both definition passages boosted
        assert result[0][4] == 3.0 or result[1][4] == 3.0
        # Usage passage not boosted (will be last after sorting)
        assert result[2][4] == 1.0

    def test_function_definition_boost(self):
        """Function definitions are boosted correctly."""
        passages = [
            ("def compute():\n    pass", "analysis.py", 0, 100, 1.0),
            ("result = compute()", "main.py", 0, 100, 1.0)
        ]
        result = apply_definition_boost(passages, "def compute", boost_factor=4.0)

        assert result[0][4] == 4.0  # Definition boosted
        assert result[1][4] == 1.0  # Usage not boosted


# =============================================================================
# TEST FILE DETECTION TESTS
# =============================================================================


class TestIsTestFile:
    """Tests for is_test_file() - detect test files."""

    def test_source_file(self):
        """Regular source file is not a test file."""
        assert is_test_file("cortical/processor.py") is False
        assert is_test_file("analysis.py") is False
        assert is_test_file("src/main.py") is False

    def test_test_directory_path(self):
        """Files in tests/ directory are test files."""
        assert is_test_file("tests/test_processor.py") is True
        assert is_test_file("tests/unit/test_analysis.py") is True
        assert is_test_file("/path/to/tests/test_file.py") is True

    def test_test_prefix(self):
        """Files starting with test_ are test files."""
        assert is_test_file("test_processor.py") is True
        assert is_test_file("test_integration.py") is True
        assert is_test_file("path/test_something.py") is True

    def test_test_suffix(self):
        """Files ending with _test.py are test files."""
        assert is_test_file("processor_test.py") is True
        assert is_test_file("integration_test.py") is True
        assert is_test_file("path/module_test.py") is True

    def test_mock_file(self):
        """Files with 'mock' in name are test files."""
        assert is_test_file("mocks.py") is True
        assert is_test_file("test_mocks.py") is True
        assert is_test_file("mock_data.py") is True

    def test_fixture_file(self):
        """Files with 'fixture' in name are test files."""
        assert is_test_file("fixtures.py") is True
        assert is_test_file("test_fixtures.py") is True
        assert is_test_file("fixture_data.py") is True

    def test_case_insensitive(self):
        """Detection is case insensitive."""
        assert is_test_file("Tests/TEST_PROCESSOR.PY") is True
        assert is_test_file("MOCK_DATA.PY") is True

    def test_test_in_middle_of_path(self):
        """'test' in middle of path component is detected."""
        assert is_test_file("myproject/test/data.py") is True
        assert is_test_file("src/tests/unit.py") is True

    def test_similar_but_not_test(self):
        """Files with similar names but not test files."""
        assert is_test_file("latest_version.py") is False
        assert is_test_file("contest.py") is False
        assert is_test_file("attest.py") is False

    def test_without_extension(self):
        """Works with paths without .py extension."""
        assert is_test_file("tests/test_something") is True
        assert is_test_file("test_file") is True
        assert is_test_file("src/module") is False


# =============================================================================
# DOCUMENT BOOSTING TESTS
# =============================================================================


class TestBoostDefinitionDocuments:
    """Tests for boost_definition_documents() - boost source files."""

    def test_non_definition_query(self):
        """Non-definition query returns documents unchanged."""
        doc_results = [("doc1", 1.0), ("doc2", 0.5)]
        documents = {"doc1": "text", "doc2": "text"}

        result = boost_definition_documents(doc_results, "how does this work", documents)
        assert result == doc_results

    def test_empty_documents(self):
        """Empty document results returns empty."""
        result = boost_definition_documents([], "class Foo", {})
        assert result == []

    def test_source_file_with_definition_boosted(self):
        """Source file containing definition gets boosted."""
        doc_results = [
            ("minicolumn.py", 1.0),
            ("usage.py", 1.0)
        ]
        documents = {
            "minicolumn.py": "class Minicolumn:\n    pass",
            "usage.py": "mc = Minicolumn()"
        }

        result = boost_definition_documents(
            doc_results, "class Minicolumn", documents, boost_factor=2.0
        )

        # minicolumn.py has definition, should be boosted
        minicolumn_score = next(s for d, s in result if d == "minicolumn.py")
        usage_score = next(s for d, s in result if d == "usage.py")

        assert minicolumn_score == 2.0  # 1.0 * 2.0
        assert usage_score == 1.0  # Unchanged

    def test_test_file_with_definition_penalized(self):
        """Test file with definition gets penalty instead of boost."""
        doc_results = [
            ("minicolumn.py", 1.0),
            ("test_minicolumn.py", 1.0)
        ]
        documents = {
            "minicolumn.py": "class Minicolumn:\n    pass",
            "test_minicolumn.py": "class Minicolumn:\n    pass"
        }

        result = boost_definition_documents(
            doc_results, "class Minicolumn", documents,
            boost_factor=2.0,
            test_with_definition_penalty=0.5
        )

        source_score = next(s for d, s in result if d == "minicolumn.py")
        test_score = next(s for d, s in result if d == "test_minicolumn.py")

        assert source_score == 2.0  # Boosted
        assert test_score == 0.5  # Penalized

    def test_test_file_without_definition_penalized(self):
        """Test file without definition gets different penalty."""
        doc_results = [
            ("minicolumn.py", 1.0),
            ("test_usage.py", 1.0)
        ]
        documents = {
            "minicolumn.py": "class Minicolumn:\n    pass",
            "test_usage.py": "mc = Minicolumn()"
        }

        result = boost_definition_documents(
            doc_results, "class Minicolumn", documents,
            test_without_definition_penalty=0.7
        )

        test_score = next(s for d, s in result if d == "test_usage.py")
        assert test_score == 0.7  # 1.0 * 0.7

    def test_results_sorted_by_boosted_score(self):
        """Results are re-sorted after boosting."""
        doc_results = [
            ("test_foo.py", 10.0),  # High score, test file with definition
            ("foo.py", 1.0),  # Low score, source with definition
            ("usage.py", 5.0)  # Medium score, source without definition
        ]
        documents = {
            "test_foo.py": "class Foo:\n    pass",
            "foo.py": "class Foo:\n    pass",
            "usage.py": "f = Foo()"
        }

        result = boost_definition_documents(
            doc_results, "class Foo", documents,
            boost_factor=2.0,
            test_with_definition_penalty=0.5,
            test_without_definition_penalty=0.7
        )

        # Expected scores:
        # test_foo.py: 10.0 * 0.5 = 5.0 (test with def)
        # foo.py: 1.0 * 2.0 = 2.0 (source with def)
        # usage.py: 5.0 * 1.0 = 5.0 (source without def)

        # Results should be sorted descending
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

        # foo.py should move up despite lower initial score
        assert result[0][0] in ["test_foo.py", "usage.py"]  # Tied at 5.0
        assert result[2][0] == "foo.py"  # Boosted but still lowest

    def test_custom_boost_factor(self):
        """Custom boost factor is applied."""
        doc_results = [("foo.py", 2.0)]
        documents = {"foo.py": "class Foo:\n    pass"}

        result = boost_definition_documents(
            doc_results, "class Foo", documents, boost_factor=5.0
        )

        assert result[0][1] == 10.0  # 2.0 * 5.0

    def test_custom_penalties(self):
        """Custom penalty factors are applied."""
        doc_results = [
            ("test_with_def.py", 1.0),
            ("test_without_def.py", 1.0)
        ]
        documents = {
            "test_with_def.py": "class Foo:\n    pass",
            "test_without_def.py": "f = Foo()"
        }

        result = boost_definition_documents(
            doc_results, "class Foo", documents,
            test_with_definition_penalty=0.3,
            test_without_definition_penalty=0.6
        )

        with_def_score = next(s for d, s in result if d == "test_with_def.py")
        without_def_score = next(s for d, s in result if d == "test_without_def.py")

        assert with_def_score == 0.3
        assert without_def_score == 0.6

    def test_no_test_penalty(self):
        """Can disable test penalty by setting to 1.0."""
        doc_results = [
            ("foo.py", 1.0),
            ("test_foo.py", 1.0)
        ]
        documents = {
            "foo.py": "using Foo",
            "test_foo.py": "using Foo"
        }

        result = boost_definition_documents(
            doc_results, "class Foo", documents,
            test_without_definition_penalty=1.0  # No penalty
        )

        # Both should have same score
        assert result[0][1] == 1.0
        assert result[1][1] == 1.0

    def test_multiple_source_files_with_definition(self):
        """Multiple source files with definitions all get boosted."""
        doc_results = [
            ("foo_v1.py", 1.0),
            ("foo_v2.py", 1.0),
            ("usage.py", 1.0)
        ]
        documents = {
            "foo_v1.py": "class Foo:\n    # Version 1",
            "foo_v2.py": "class Foo:\n    # Version 2",
            "usage.py": "f = Foo()"
        }

        result = boost_definition_documents(
            doc_results, "class Foo", documents, boost_factor=3.0
        )

        v1_score = next(s for d, s in result if d == "foo_v1.py")
        v2_score = next(s for d, s in result if d == "foo_v2.py")
        usage_score = next(s for d, s in result if d == "usage.py")

        assert v1_score == 3.0
        assert v2_score == 3.0
        assert usage_score == 1.0
