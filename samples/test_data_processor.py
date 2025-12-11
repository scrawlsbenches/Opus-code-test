"""
Unit tests for the DataProcessor module.

This test file demonstrates various testing patterns for the data processing
functionality including fixtures, edge cases, and integration tests.
"""

import unittest
from typing import List

# Note: In a real project, this would import from the actual module
# from data_processor import DataProcessor, DataRecord, calculate_statistics


class MockDataRecord:
    """Mock DataRecord for testing purposes."""
    def __init__(self, id: str, content: str, metadata=None, tags=None):
        self.id = id
        self.content = content
        self.metadata = metadata or {}
        self.tags = tags or []


class TestDataProcessorBasics(unittest.TestCase):
    """Test basic DataProcessor functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.processor = MockDataProcessor()
        self.sample_record = MockDataRecord(
            id="test1",
            content="Hello world test content",
            tags=["test", "sample"]
        )

    def test_add_record(self):
        """Test adding a single record."""
        self.processor.add_record(self.sample_record)
        result = self.processor.get_record("test1")
        self.assertIsNotNone(result)
        self.assertEqual(result.content, "Hello world test content")

    def test_add_multiple_records(self):
        """Test adding multiple records."""
        records = [
            MockDataRecord("r1", "First record"),
            MockDataRecord("r2", "Second record"),
            MockDataRecord("r3", "Third record"),
        ]
        for record in records:
            self.processor.add_record(record)

        self.assertEqual(len(self.processor._records), 3)

    def test_get_nonexistent_record(self):
        """Test retrieving a record that doesn't exist."""
        result = self.processor.get_record("nonexistent")
        self.assertIsNone(result)


class TestDataProcessorFiltering(unittest.TestCase):
    """Test filtering functionality."""

    def setUp(self):
        """Set up test data with various records."""
        self.processor = MockDataProcessor()
        self.processor.add_record(MockDataRecord("1", "Python programming"))
        self.processor.add_record(MockDataRecord("2", "JavaScript development"))
        self.processor.add_record(MockDataRecord("3", "Python web development"))
        self.processor.add_record(MockDataRecord("4", "Database design", tags=["db"]))

    def test_filter_by_content_single_match(self):
        """Test content filter with single match."""
        results = self.processor.filter_by_content("JavaScript")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id, "2")

    def test_filter_by_content_multiple_matches(self):
        """Test content filter with multiple matches."""
        results = self.processor.filter_by_content("Python")
        self.assertEqual(len(results), 2)

    def test_filter_by_content_case_insensitive(self):
        """Test that content filtering is case insensitive."""
        results = self.processor.filter_by_content("python")
        self.assertEqual(len(results), 2)

    def test_filter_by_content_no_match(self):
        """Test content filter with no matches."""
        results = self.processor.filter_by_content("Ruby")
        self.assertEqual(len(results), 0)

    def test_filter_by_tag(self):
        """Test filtering by tag."""
        results = self.processor.filter_by_tag("db")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id, "4")


class TestDataProcessorTransformation(unittest.TestCase):
    """Test transformation functionality."""

    def setUp(self):
        """Set up test processor."""
        self.processor = MockDataProcessor()
        self.processor.add_record(MockDataRecord("1", "hello world"))
        self.processor.add_record(MockDataRecord("2", "test content"))

    def test_transform_uppercase(self):
        """Test transforming content to uppercase."""
        results = self.processor.transform_content(str.upper)
        self.assertEqual(results[0].content, "HELLO WORLD")

    def test_transform_preserves_metadata(self):
        """Test that transformation preserves record metadata."""
        self.processor._records["1"].metadata = {"key": "value"}
        results = self.processor.transform_content(str.upper)
        self.assertEqual(results[0].metadata, {"key": "value"})


class TestStatisticsCalculation(unittest.TestCase):
    """Test statistics calculation functions."""

    def test_empty_records(self):
        """Test statistics with empty record list."""
        stats = mock_calculate_statistics([])
        self.assertEqual(stats['count'], 0)
        self.assertEqual(stats['avg_content_length'], 0)

    def test_single_record(self):
        """Test statistics with single record."""
        records = [MockDataRecord("1", "Hello")]
        stats = mock_calculate_statistics(records)
        self.assertEqual(stats['count'], 1)
        self.assertEqual(stats['avg_content_length'], 5)

    def test_multiple_records(self):
        """Test statistics with multiple records."""
        records = [
            MockDataRecord("1", "Hi", tags=["a"]),
            MockDataRecord("2", "Hello", tags=["a", "b"]),
            MockDataRecord("3", "Goodbye", tags=["c"]),
        ]
        stats = mock_calculate_statistics(records)
        self.assertEqual(stats['count'], 3)
        self.assertEqual(stats['unique_tags'], {"a", "b", "c"})


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def test_empty_content(self):
        """Test record with empty content."""
        processor = MockDataProcessor()
        record = MockDataRecord("empty", "")
        processor.add_record(record)
        self.assertEqual(processor.get_record("empty").content, "")

    def test_special_characters(self):
        """Test content with special characters."""
        processor = MockDataProcessor()
        record = MockDataRecord("special", "Hello! @#$% World?")
        processor.add_record(record)
        results = processor.filter_by_content("@#$%")
        self.assertEqual(len(results), 1)

    def test_unicode_content(self):
        """Test content with unicode characters."""
        processor = MockDataProcessor()
        record = MockDataRecord("unicode", "Hello 世界 🌍")
        processor.add_record(record)
        results = processor.filter_by_content("世界")
        self.assertEqual(len(results), 1)


# Mock implementations for testing
class MockDataProcessor:
    """Mock implementation of DataProcessor for testing."""

    def __init__(self):
        self._records = {}
        self._tag_index = {}

    def add_record(self, record):
        self._records[record.id] = record
        for tag in record.tags:
            if tag not in self._tag_index:
                self._tag_index[tag] = []
            self._tag_index[tag].append(record.id)

    def get_record(self, record_id):
        return self._records.get(record_id)

    def filter_by_content(self, query):
        query_lower = query.lower()
        return [r for r in self._records.values() if query_lower in r.content.lower()]

    def filter_by_tag(self, tag):
        ids = self._tag_index.get(tag, [])
        return [self._records[i] for i in ids if i in self._records]

    def transform_content(self, func):
        results = []
        for r in self._records.values():
            new_record = MockDataRecord(r.id, func(r.content), r.metadata.copy(), r.tags.copy())
            results.append(new_record)
        return results


def mock_calculate_statistics(records: List) -> dict:
    """Mock implementation of calculate_statistics."""
    if not records:
        return {'count': 0, 'avg_content_length': 0, 'unique_tags': set(), 'records_with_metadata': 0}

    total_length = sum(len(r.content) for r in records)
    all_tags = set()
    for r in records:
        all_tags.update(r.tags)

    return {
        'count': len(records),
        'avg_content_length': total_length / len(records),
        'unique_tags': all_tags,
        'records_with_metadata': sum(1 for r in records if r.metadata)
    }


if __name__ == '__main__':
    unittest.main()
