"""
Pattern Detection Demo
======================

Demonstrates the code pattern detection capabilities of the
Cortical Text Processor.
"""

from cortical.processor import CorticalTextProcessor
from cortical.patterns import (
    detect_patterns_in_text,
    format_pattern_report,
    list_all_patterns,
    list_all_categories,
)


def main():
    print("=" * 70)
    print("Code Pattern Detection Demo")
    print("=" * 70)

    # Sample code files
    sample_files = {
        'singleton.py': """
class DatabaseConnection:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def connect(self):
        print("Connected to database")
""",
        'async_handler.py': """
import asyncio
from typing import List

async def fetch_users() -> List[dict]:
    try:
        async with aiohttp.ClientSession() as session:
            async for user in get_users(session):
                yield user
    except Exception as e:
        raise FetchError(f"Failed to fetch users: {e}")
""",
        'factory.py': """
from dataclasses import dataclass

@dataclass
class User:
    name: str
    email: str

    @property
    def display_name(self):
        return f"{self.name} <{self.email}>"

class UserFactory:
    @staticmethod
    def create_user(name, email):
        return User(name=name, email=email)

    @staticmethod
    def create_admin(name, email):
        user = User(name=name, email=email)
        user.is_admin = True
        return user
""",
        'test_features.py': """
import pytest
from unittest.mock import Mock, patch

class TestUserFeatures:
    def setUp(self):
        self.user = User("test")

    def test_login(self):
        assert self.user.login("password")

    @pytest.mark.skip
    def test_logout(self):
        assert self.user.logout()

    @patch('module.authenticate')
    def test_with_mock(self, mock_auth):
        mock_auth.return_value = True
        assert self.user.verify()
"""
    }

    # Create processor and add documents
    processor = CorticalTextProcessor()
    print("\n1. Adding sample code files...")
    for filename, code in sample_files.items():
        processor.process_document(filename, code)
        print(f"   ✓ {filename}")

    # List available patterns
    print(f"\n2. Available pattern types ({len(list_all_patterns())} patterns):")
    categories = list_all_categories()
    for category in categories[:5]:  # Show first 5 categories
        print(f"   - {category}")
    print(f"   ... and {len(categories) - 5} more categories")

    # Detect patterns in each file
    print("\n3. Detecting patterns in each file...")
    for filename in sample_files.keys():
        patterns = processor.detect_patterns(filename)
        if patterns:
            print(f"\n   {filename}:")
            for pattern_name in sorted(patterns.keys()):
                lines = patterns[pattern_name]
                print(f"     - {pattern_name}: {len(lines)} occurrence(s)")

    # Detailed report for one file
    print("\n" + "=" * 70)
    print("4. Detailed pattern report for 'async_handler.py':")
    print("=" * 70)
    report = processor.format_pattern_report('async_handler.py', show_lines=True)
    print(report)

    # Corpus-wide statistics
    print("=" * 70)
    print("5. Corpus-wide pattern statistics:")
    print("=" * 70)
    stats = processor.get_corpus_pattern_statistics()
    print(f"Total documents analyzed: {stats['total_documents']}")
    print(f"Unique patterns found: {stats['patterns_found']}")
    print(f"Most common pattern: {stats['most_common_pattern']}")

    print("\nTop patterns by occurrence:")
    sorted_patterns = sorted(
        stats['pattern_occurrences'].items(),
        key=lambda x: -x[1]
    )
    for pattern, count in sorted_patterns[:10]:
        doc_count = stats['pattern_document_counts'][pattern]
        print(f"  {pattern}: {count} occurrences in {doc_count} file(s)")

    # Search for specific pattern types
    print("\n" + "=" * 70)
    print("6. Finding all files with specific patterns:")
    print("=" * 70)

    patterns_to_find = ['singleton', 'factory', 'async_await', 'dataclass']
    for pattern_name in patterns_to_find:
        corpus_patterns = processor.detect_patterns_in_corpus(patterns=[pattern_name])
        files_with_pattern = [
            doc_id for doc_id, patterns in corpus_patterns.items()
            if pattern_name in patterns
        ]
        if files_with_pattern:
            print(f"{pattern_name}:")
            for filename in files_with_pattern:
                print(f"  ✓ {filename}")
        else:
            print(f"{pattern_name}: Not found")

    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
