"""
Unit tests for cortical/utils/persistence.py

Tests atomic file writing utilities.
"""

import json
import pytest
from pathlib import Path

from cortical.utils.persistence import atomic_write, atomic_write_json


class TestAtomicWrite:
    """Tests for atomic_write()."""

    def test_basic_write(self, tmp_path):
        """Write content to file."""
        path = tmp_path / "test.txt"
        atomic_write(path, "Hello, world!")

        assert path.exists()
        assert path.read_text() == "Hello, world!"

    def test_overwrite(self, tmp_path):
        """Overwrite existing file."""
        path = tmp_path / "test.txt"
        path.write_text("old content")

        atomic_write(path, "new content")

        assert path.read_text() == "new content"

    def test_custom_encoding(self, tmp_path):
        """Write with custom encoding."""
        path = tmp_path / "test.txt"
        atomic_write(path, "Héllo wörld", encoding='utf-8')

        assert path.read_text(encoding='utf-8') == "Héllo wörld"

    def test_no_temp_file_left_on_success(self, tmp_path):
        """Temp file is removed after successful write."""
        path = tmp_path / "test.txt"
        atomic_write(path, "content")

        temp_path = path.with_suffix('.txt.tmp')
        assert not temp_path.exists()

    def test_creates_parent_directories_not_needed(self, tmp_path):
        """Write works when parent directory exists."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        path = subdir / "test.txt"

        atomic_write(path, "content")
        assert path.read_text() == "content"

    def test_multiline_content(self, tmp_path):
        """Write multiline content."""
        path = tmp_path / "test.txt"
        content = "line1\nline2\nline3"
        atomic_write(path, content)

        assert path.read_text() == content

    def test_empty_content(self, tmp_path):
        """Write empty content."""
        path = tmp_path / "test.txt"
        atomic_write(path, "")

        assert path.exists()
        assert path.read_text() == ""


class TestAtomicWriteJson:
    """Tests for atomic_write_json()."""

    def test_basic_write(self, tmp_path):
        """Write JSON data to file."""
        path = tmp_path / "test.json"
        data = {"key": "value"}
        atomic_write_json(path, data)

        assert path.exists()
        assert json.loads(path.read_text()) == data

    def test_nested_data(self, tmp_path):
        """Write nested JSON data."""
        path = tmp_path / "test.json"
        data = {
            "level1": {
                "level2": {
                    "items": [1, 2, 3]
                }
            }
        }
        atomic_write_json(path, data)

        assert json.loads(path.read_text()) == data

    def test_list_data(self, tmp_path):
        """Write JSON list."""
        path = tmp_path / "test.json"
        data = [1, 2, 3, {"key": "value"}]
        atomic_write_json(path, data)

        assert json.loads(path.read_text()) == data

    def test_custom_indent(self, tmp_path):
        """Write with custom indent."""
        path = tmp_path / "test.json"
        data = {"key": "value"}
        atomic_write_json(path, data, indent=4)

        content = path.read_text()
        assert '    "key"' in content  # 4 spaces indent

    def test_no_indent(self, tmp_path):
        """Write with no indent."""
        path = tmp_path / "test.json"
        data = {"key": "value"}
        atomic_write_json(path, data, indent=None)

        content = path.read_text()
        # No newlines in compact format
        assert '\n' not in content or content.count('\n') <= 1

    def test_overwrite(self, tmp_path):
        """Overwrite existing JSON file."""
        path = tmp_path / "test.json"
        atomic_write_json(path, {"old": "data"})
        atomic_write_json(path, {"new": "data"})

        assert json.loads(path.read_text()) == {"new": "data"}

    def test_unicode_data(self, tmp_path):
        """Write JSON with unicode characters."""
        path = tmp_path / "test.json"
        data = {"message": "Héllo wörld 你好"}
        atomic_write_json(path, data)

        assert json.loads(path.read_text()) == data

    def test_empty_dict(self, tmp_path):
        """Write empty dictionary."""
        path = tmp_path / "test.json"
        atomic_write_json(path, {})

        assert json.loads(path.read_text()) == {}

    def test_empty_list(self, tmp_path):
        """Write empty list."""
        path = tmp_path / "test.json"
        atomic_write_json(path, [])

        assert json.loads(path.read_text()) == []

    def test_non_serializable_raises(self, tmp_path):
        """Non-serializable data raises error."""
        path = tmp_path / "test.json"

        class NotSerializable:
            pass

        with pytest.raises(TypeError):
            atomic_write_json(path, {"obj": NotSerializable()})

        # File should not exist after failure
        assert not path.exists()
