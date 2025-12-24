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

    def test_cleanup_temp_file_on_error(self, tmp_path, monkeypatch):
        """Temp file is cleaned up if rename fails."""
        path = tmp_path / "test.json"

        # Mock rename to fail after temp file is created
        original_rename = Path.rename

        def failing_rename(self, target):
            if str(self).endswith('.tmp'):
                raise OSError("Simulated rename failure")
            return original_rename(self, target)

        monkeypatch.setattr(Path, 'rename', failing_rename)

        with pytest.raises(OSError, match="Simulated rename failure"):
            atomic_write_json(path, {"key": "value"})

        # Temp file should be cleaned up
        temp_path = path.with_suffix('.json.tmp')
        assert not temp_path.exists()

    def test_error_when_temp_file_doesnt_exist_json(self, tmp_path, monkeypatch):
        """Handle error case when temp file was already deleted (JSON)."""
        path = tmp_path / "test.json"

        # Mock exists() to return False, simulating temp file already gone
        original_exists = Path.exists

        def mock_exists(self):
            if str(self).endswith('.tmp'):
                return False  # Simulate temp file already deleted
            return original_exists(self)

        # Also mock rename to fail
        original_rename = Path.rename

        def failing_rename(self, target):
            if str(self).endswith('.tmp'):
                raise OSError("Simulated rename failure")
            return original_rename(self, target)

        monkeypatch.setattr(Path, 'exists', mock_exists)
        monkeypatch.setattr(Path, 'rename', failing_rename)

        with pytest.raises(OSError, match="Simulated rename failure"):
            atomic_write_json(path, {"key": "value"})

        # Main file shouldn't exist since write failed
        assert not path.exists()


class TestAtomicWriteErrorHandling:
    """Tests for error handling in atomic_write()."""

    def test_cleanup_temp_file_on_write_error(self, tmp_path, monkeypatch):
        """Temp file is cleaned up if write fails."""
        path = tmp_path / "test.txt"

        # Mock write to fail
        import builtins
        original_open = builtins.open

        def failing_open(file, mode='r', *args, **kwargs):
            if str(file).endswith('.tmp') and 'w' in mode:
                raise OSError("Simulated write failure")
            return original_open(file, mode, *args, **kwargs)

        monkeypatch.setattr(builtins, 'open', failing_open)

        with pytest.raises(OSError, match="Simulated write failure"):
            atomic_write(path, "content")

        # Temp file should be cleaned up
        temp_path = path.with_suffix('.txt.tmp')
        assert not temp_path.exists()

    def test_cleanup_temp_file_on_rename_error(self, tmp_path, monkeypatch):
        """Temp file is cleaned up if rename fails."""
        path = tmp_path / "test.txt"

        # Mock rename to fail after temp file is created
        original_rename = Path.rename

        def failing_rename(self, target):
            if str(self).endswith('.tmp'):
                raise OSError("Simulated rename failure")
            return original_rename(self, target)

        monkeypatch.setattr(Path, 'rename', failing_rename)

        with pytest.raises(OSError, match="Simulated rename failure"):
            atomic_write(path, "content")

        # Temp file should be cleaned up
        temp_path = path.with_suffix('.txt.tmp')
        assert not temp_path.exists()

    def test_error_when_temp_file_doesnt_exist(self, tmp_path, monkeypatch):
        """Handle error case when temp file was already deleted."""
        path = tmp_path / "test.txt"

        # Mock exists() to return False, simulating temp file already gone
        original_exists = Path.exists

        def mock_exists(self):
            if str(self).endswith('.tmp'):
                return False  # Simulate temp file already deleted
            return original_exists(self)

        # Also mock rename to fail
        original_rename = Path.rename

        def failing_rename(self, target):
            if str(self).endswith('.tmp'):
                raise OSError("Simulated rename failure")
            return original_rename(self, target)

        monkeypatch.setattr(Path, 'exists', mock_exists)
        monkeypatch.setattr(Path, 'rename', failing_rename)

        with pytest.raises(OSError, match="Simulated rename failure"):
            atomic_write(path, "content")

        # Main file shouldn't exist since write failed
        assert not path.exists()
