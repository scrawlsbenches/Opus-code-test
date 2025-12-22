"""
Atomic file persistence utilities.

This module provides utilities for safely writing files using the
write-to-temp-then-rename pattern, which ensures atomicity on POSIX systems.
"""

import json
import os
from pathlib import Path
from typing import Any


def atomic_write(path: Path, content: str, encoding: str = 'utf-8') -> None:
    """
    Write content to a file atomically.

    Uses write-to-temp-then-rename pattern to prevent data loss
    if the process crashes during write.

    Args:
        path: Target file path
        content: String content to write
        encoding: Text encoding (default: utf-8)

    Raises:
        OSError: If write or rename fails

    Example:
        >>> from pathlib import Path
        >>> atomic_write(Path("data.txt"), "Hello, world!")
    """
    path = Path(path)
    temp_path = path.with_suffix(path.suffix + '.tmp')

    try:
        # Write to temp file first
        with open(temp_path, 'w', encoding=encoding) as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())  # Ensure data is on disk

        # Atomic rename (on POSIX systems)
        temp_path.rename(path)
    except Exception:
        # Clean up temp file on failure
        if temp_path.exists():
            temp_path.unlink()
        raise


def atomic_write_json(path: Path, data: Any, indent: int = 2, encoding: str = 'utf-8') -> None:
    """
    Write JSON data to a file atomically.

    Uses write-to-temp-then-rename pattern to prevent data loss
    if the process crashes during write.

    Args:
        path: Target file path
        data: JSON-serializable data to write
        indent: JSON indentation level (default: 2)
        encoding: Text encoding (default: utf-8)

    Raises:
        OSError: If write or rename fails
        TypeError: If data is not JSON-serializable

    Example:
        >>> from pathlib import Path
        >>> atomic_write_json(Path("data.json"), {"key": "value"})
    """
    path = Path(path)
    temp_path = path.with_suffix(path.suffix + '.tmp')

    try:
        # Write to temp file first
        with open(temp_path, 'w', encoding=encoding) as f:
            json.dump(data, f, indent=indent)
            f.flush()
            os.fsync(f.fileno())  # Ensure data is on disk

        # Atomic rename (on POSIX systems)
        temp_path.rename(path)
    except Exception:
        # Clean up temp file on failure
        if temp_path.exists():
            temp_path.unlink()
        raise
