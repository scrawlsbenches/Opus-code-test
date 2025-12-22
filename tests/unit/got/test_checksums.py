"""
Tests for checksum utilities.

Note: Tests updated to import from cortical.utils.checksums (the new location)
instead of cortical.got.checksums (deprecated).
"""

import json
import pytest
from pathlib import Path
from cortical.utils.checksums import (
    compute_checksum,
    verify_checksum,
    compute_file_checksum,
    verify_file_checksum,
)


class TestComputeChecksum:
    """Test checksum computation."""

    def test_compute_checksum_deterministic(self):
        """Same input always gives same output."""
        data = {"key": "value", "number": 42}
        checksum1 = compute_checksum(data)
        checksum2 = compute_checksum(data)
        assert checksum1 == checksum2

    def test_compute_checksum_changes_with_data(self):
        """Different data gives different checksum."""
        data1 = {"key": "value1"}
        data2 = {"key": "value2"}
        checksum1 = compute_checksum(data1)
        checksum2 = compute_checksum(data2)
        assert checksum1 != checksum2

    def test_checksum_is_hex_string(self):
        """Checksum is 16-character hex string."""
        data = {"test": "data"}
        checksum = compute_checksum(data)
        assert len(checksum) == 16
        # Verify all characters are valid hex
        assert all(c in '0123456789abcdef' for c in checksum)

    def test_checksum_ignores_key_order(self):
        """Key order doesn't affect checksum."""
        data1 = {"a": 1, "b": 2, "c": 3}
        data2 = {"c": 3, "b": 2, "a": 1}
        data3 = {"b": 2, "a": 1, "c": 3}

        checksum1 = compute_checksum(data1)
        checksum2 = compute_checksum(data2)
        checksum3 = compute_checksum(data3)

        assert checksum1 == checksum2 == checksum3


class TestVerifyChecksum:
    """Test checksum verification."""

    def test_verify_checksum_valid(self):
        """Returns True for correct checksum."""
        data = {"key": "value", "number": 42}
        checksum = compute_checksum(data)
        assert verify_checksum(data, checksum) is True

    def test_verify_checksum_invalid(self):
        """Returns False for wrong checksum."""
        data = {"key": "value"}
        wrong_checksum = "0" * 16
        assert verify_checksum(data, wrong_checksum) is False

    def test_verify_checksum_after_data_change(self):
        """Checksum fails after data modification."""
        data = {"key": "value"}
        checksum = compute_checksum(data)

        # Modify data
        data["key"] = "modified"

        assert verify_checksum(data, checksum) is False


class TestFileChecksum:
    """Test file checksum functions."""

    def test_file_checksum_matches_data_checksum(self, tmp_path):
        """File checksum equals data checksum."""
        data = {"key": "value", "number": 42, "nested": {"a": 1}}

        # Write data to file
        test_file = tmp_path / "test.json"
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(data, f)

        # Compute checksums
        data_checksum = compute_checksum(data)
        file_checksum = compute_file_checksum(test_file)

        assert file_checksum == data_checksum

    def test_verify_file_checksum_valid(self, tmp_path):
        """verify_file_checksum returns True for correct checksum."""
        data = {"test": "data"}
        test_file = tmp_path / "test.json"

        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(data, f)

        expected = compute_checksum(data)
        assert verify_file_checksum(test_file, expected) is True

    def test_verify_file_checksum_invalid(self, tmp_path):
        """verify_file_checksum returns False for wrong checksum."""
        data = {"test": "data"}
        test_file = tmp_path / "test.json"

        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(data, f)

        wrong_checksum = "0" * 16
        assert verify_file_checksum(test_file, wrong_checksum) is False

    def test_file_checksum_ignores_formatting(self, tmp_path):
        """File checksum ignores JSON formatting differences."""
        data = {"a": 1, "b": 2}

        # Write with different formatting
        file1 = tmp_path / "compact.json"
        with open(file1, 'w', encoding='utf-8') as f:
            json.dump(data, f, separators=(',', ':'))

        file2 = tmp_path / "pretty.json"
        with open(file2, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        # Checksums should match
        checksum1 = compute_file_checksum(file1)
        checksum2 = compute_file_checksum(file2)

        assert checksum1 == checksum2
