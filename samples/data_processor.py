"""
Data Processor Module - Sample code for demonstrating code search features.

This module provides utilities for processing and transforming data records.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class DataRecord:
    """Represents a single data record with metadata.

    Attributes:
        id: Unique identifier for the record
        content: The main content of the record
        metadata: Optional dictionary of metadata fields
        tags: List of tags associated with the record
    """
    id: str
    content: str
    metadata: Optional[Dict[str, Any]] = None
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}


class DataProcessor:
    """Main processor for handling data records.

    The DataProcessor provides methods for filtering, transforming,
    and aggregating data records efficiently.

    Example:
        processor = DataProcessor()
        processor.add_record(DataRecord("1", "Hello world"))
        results = processor.filter_by_content("hello")
    """

    def __init__(self):
        """Initialize the data processor with empty storage."""
        self._records: Dict[str, DataRecord] = {}
        self._tag_index: Dict[str, List[str]] = defaultdict(list)

    def add_record(self, record: DataRecord) -> None:
        """Add a record to the processor.

        Args:
            record: The DataRecord to add
        """
        self._records[record.id] = record
        for tag in record.tags:
            self._tag_index[tag].append(record.id)

    def get_record(self, record_id: str) -> Optional[DataRecord]:
        """Retrieve a record by its ID.

        Args:
            record_id: The unique identifier of the record

        Returns:
            The DataRecord if found, None otherwise
        """
        return self._records.get(record_id)

    def filter_by_content(self, query: str) -> List[DataRecord]:
        """Filter records by content matching.

        Args:
            query: Search string to match against content

        Returns:
            List of matching DataRecord objects
        """
        query_lower = query.lower()
        return [
            record for record in self._records.values()
            if query_lower in record.content.lower()
        ]

    def filter_by_tag(self, tag: str) -> List[DataRecord]:
        """Filter records by tag.

        Args:
            tag: The tag to filter by

        Returns:
            List of DataRecord objects with the specified tag
        """
        record_ids = self._tag_index.get(tag, [])
        return [self._records[rid] for rid in record_ids if rid in self._records]

    def transform_content(self, transformer_func) -> List[DataRecord]:
        """Apply a transformation function to all record contents.

        Args:
            transformer_func: Callable that takes content string and returns transformed string

        Returns:
            List of new DataRecord objects with transformed content
        """
        results = []
        for record in self._records.values():
            new_content = transformer_func(record.content)
            new_record = DataRecord(
                id=record.id,
                content=new_content,
                metadata=record.metadata.copy(),
                tags=record.tags.copy()
            )
            results.append(new_record)
        return results

    def aggregate_by_tag(self) -> Dict[str, int]:
        """Count records per tag.

        Returns:
            Dictionary mapping tag names to record counts
        """
        return {tag: len(ids) for tag, ids in self._tag_index.items()}

    def clear(self) -> None:
        """Remove all records from the processor."""
        self._records.clear()
        self._tag_index.clear()


def calculate_statistics(records: List[DataRecord]) -> Dict[str, Any]:
    """Calculate statistics for a list of records.

    Args:
        records: List of DataRecord objects to analyze

    Returns:
        Dictionary containing:
            - count: Number of records
            - avg_content_length: Average content length
            - unique_tags: Set of all unique tags
            - records_with_metadata: Count of records with non-empty metadata
    """
    if not records:
        return {
            'count': 0,
            'avg_content_length': 0,
            'unique_tags': set(),
            'records_with_metadata': 0
        }

    total_length = sum(len(r.content) for r in records)
    all_tags = set()
    metadata_count = 0

    for record in records:
        all_tags.update(record.tags)
        if record.metadata:
            metadata_count += 1

    return {
        'count': len(records),
        'avg_content_length': total_length / len(records),
        'unique_tags': all_tags,
        'records_with_metadata': metadata_count
    }


def merge_records(records: List[DataRecord], separator: str = '\n') -> DataRecord:
    """Merge multiple records into a single record.

    Args:
        records: List of DataRecord objects to merge
        separator: String to use between merged contents

    Returns:
        A new DataRecord with combined content, merged metadata, and all tags
    """
    if not records:
        return DataRecord(id='merged', content='')

    merged_content = separator.join(r.content for r in records)
    merged_metadata = {}
    merged_tags = []

    for record in records:
        merged_metadata.update(record.metadata)
        merged_tags.extend(record.tags)

    return DataRecord(
        id='merged_' + '_'.join(r.id for r in records[:3]),
        content=merged_content,
        metadata=merged_metadata,
        tags=list(set(merged_tags))
    )
