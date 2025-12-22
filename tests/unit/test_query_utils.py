"""
Tests for query utility functions.
"""

import pytest
from cortical.query.utils import get_tfidf_score, normalize_scores, is_test_file
from cortical.minicolumn import Minicolumn


class TestGetTfidfScore:
    """Tests for get_tfidf_score helper."""

    def test_global_tfidf_when_no_doc_id(self):
        """Returns global TF-IDF when doc_id is not provided."""
        col = Minicolumn("L0_neural", "neural", 0)
        col.tfidf = 2.5
        col.tfidf_per_doc = {"doc1": 3.0, "doc2": 1.5}

        assert get_tfidf_score(col) == 2.5

    def test_global_tfidf_when_doc_id_is_none(self):
        """Returns global TF-IDF when doc_id is None."""
        col = Minicolumn("L0_neural", "neural", 0)
        col.tfidf = 2.5
        col.tfidf_per_doc = {"doc1": 3.0}

        assert get_tfidf_score(col, None) == 2.5

    def test_per_doc_tfidf_when_doc_id_exists(self):
        """Returns per-document TF-IDF when doc_id exists in tfidf_per_doc."""
        col = Minicolumn("L0_neural", "neural", 0)
        col.tfidf = 2.5
        col.tfidf_per_doc = {"doc1": 3.0, "doc2": 1.5}

        assert get_tfidf_score(col, "doc1") == 3.0
        assert get_tfidf_score(col, "doc2") == 1.5

    def test_fallback_to_global_when_doc_id_not_in_dict(self):
        """Falls back to global TF-IDF when doc_id not in tfidf_per_doc."""
        col = Minicolumn("L0_neural", "neural", 0)
        col.tfidf = 2.5
        col.tfidf_per_doc = {"doc1": 3.0}

        assert get_tfidf_score(col, "doc_unknown") == 2.5

    def test_fallback_when_no_tfidf_per_doc_attribute(self):
        """Falls back to global TF-IDF when tfidf_per_doc doesn't exist."""
        col = Minicolumn("L0_neural", "neural", 0)
        col.tfidf = 2.5
        # Don't set tfidf_per_doc

        assert get_tfidf_score(col, "doc1") == 2.5


class TestNormalizeScores:
    """Tests for normalize_scores helper."""

    def test_empty_dict(self):
        """Returns empty dict when input is empty."""
        assert normalize_scores({}) == {}

    def test_single_score(self):
        """Normalizes single score to 1.0."""
        result = normalize_scores({"doc1": 5.0})
        assert result == {"doc1": 1.0}

    def test_multiple_scores(self):
        """Normalizes multiple scores by dividing by max."""
        result = normalize_scores({
            "doc1": 10.0,
            "doc2": 5.0,
            "doc3": 2.5
        })
        assert result == {
            "doc1": 1.0,
            "doc2": 0.5,
            "doc3": 0.25
        }

    def test_all_zero_scores(self):
        """Returns original scores when all are zero."""
        scores = {"doc1": 0.0, "doc2": 0.0}
        result = normalize_scores(scores)
        assert result == scores

    def test_negative_scores(self):
        """Handles negative scores correctly."""
        result = normalize_scores({
            "doc1": 10.0,
            "doc2": -5.0,
            "doc3": 0.0
        })
        assert result == {
            "doc1": 1.0,
            "doc2": -0.5,
            "doc3": 0.0
        }

    def test_preserves_original_dict(self):
        """Returns new dict, doesn't modify original."""
        original = {"doc1": 10.0, "doc2": 5.0}
        result = normalize_scores(original)

        # Original unchanged
        assert original == {"doc1": 10.0, "doc2": 5.0}
        # Result is normalized
        assert result == {"doc1": 1.0, "doc2": 0.5}


class TestIsTestFile:
    """Tests for is_test_file helper."""

    def test_source_file(self):
        """Returns False for source files."""
        assert is_test_file("cortical/processor.py") is False
        assert is_test_file("src/main.py") is False

    def test_tests_directory_with_slash(self):
        """Returns True for files in /tests/ directory."""
        assert is_test_file("tests/test_processor.py") is True
        assert is_test_file("project/tests/unit.py") is True

    def test_test_directory_singular(self):
        """Returns True for files in /test/ directory (needs slashes)."""
        assert is_test_file("project/test/integration.py") is True
        assert is_test_file("src/test/unit.py") is True

    def test_test_prefix(self):
        """Returns True for files starting with test_."""
        assert is_test_file("test_processor.py") is True
        assert is_test_file("test_utils.py") is True

    def test_test_suffix(self):
        """Returns True for files ending with _test.py."""
        assert is_test_file("processor_test.py") is True
        assert is_test_file("utils_test.py") is True

    def test_mock_file(self):
        """Returns True for files with 'mock' in filename."""
        assert is_test_file("mock_data.py") is True
        assert is_test_file("user_mock.py") is True

    def test_fixture_file(self):
        """Returns True for files with 'fixture' in filename."""
        assert is_test_file("fixture_data.py") is True
        assert is_test_file("data_fixture.py") is True

    def test_case_insensitive(self):
        """Detection is case-insensitive."""
        assert is_test_file("project/Tests/Test_Processor.py") is True
        assert is_test_file("src/Test/unit.py") is True

    def test_test_in_middle_of_path(self):
        """Returns True if /tests/ or /test/ anywhere in path."""
        assert is_test_file("project/tests/unit/test_processor.py") is True
        assert is_test_file("src/test/integration/suite.py") is True

    def test_similar_but_not_test(self):
        """Returns False for files with 'test' in non-test contexts."""
        assert is_test_file("contest_handler.py") is False
        assert is_test_file("latest_version.py") is False

    def test_without_extension(self):
        """Works with files without extensions."""
        assert is_test_file("project/tests/data") is True
        assert is_test_file("test_script") is True
        assert is_test_file("src/main") is False
