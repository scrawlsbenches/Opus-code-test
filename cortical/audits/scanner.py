"""
Codebase scanning utilities.

This module provides utilities for scanning codebases and extracting comments.
"""

import os
import re
from pathlib import Path
from typing import List, Tuple, Iterator, Optional
from dataclasses import dataclass


@dataclass
class Comment:
    """Represents an extracted comment."""
    file_path: str
    line_number: int
    text: str
    # Optional metadata
    context: Optional[str] = None  # Surrounding code context


@dataclass
class ScanResult:
    """Result of scanning a directory."""
    files_scanned: int
    comments_found: int
    comments: List[Comment]


def find_python_files(directory: str, exclude_dirs: Optional[List[str]] = None) -> List[str]:
    """
    Recursively find all Python files in directory.

    Args:
        directory: Root directory to scan
        exclude_dirs: Directory names to exclude (default: hidden, __pycache__, node_modules)

    Returns:
        List of absolute file paths
    """
    if exclude_dirs is None:
        exclude_dirs = ['.', '__pycache__', 'node_modules', '.git', '.venv', 'venv']

    python_files = []

    for root, dirs, files in os.walk(directory):
        # Filter out excluded directories
        dirs[:] = [d for d in dirs if not any(
            d.startswith(exc) if exc == '.' else d == exc
            for exc in exclude_dirs
        )]

        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))

    return python_files


def extract_comments_from_file(file_path: str) -> List[Tuple[int, str]]:
    """
    Extract comments from a Python file.

    Args:
        file_path: Path to Python file

    Returns:
        List of (line_number, comment_text) tuples
    """
    comments = []

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                # Find Python comments (# ...)
                match = re.search(r'#\s*(.+)$', line)
                if match:
                    comment_text = match.group(1).strip()
                    if comment_text:  # Skip empty comments
                        comments.append((line_num, comment_text))
    except Exception:
        # Silently skip files that can't be read
        pass

    return comments


def extract_comments(file_path: str) -> List[Comment]:
    """
    Extract comments from a Python file as Comment objects.

    Args:
        file_path: Path to Python file

    Returns:
        List of Comment objects
    """
    raw_comments = extract_comments_from_file(file_path)
    return [
        Comment(file_path=file_path, line_number=line_num, text=text)
        for line_num, text in raw_comments
    ]


def scan_directory(
    directory: str,
    exclude_dirs: Optional[List[str]] = None,
    min_comment_length: int = 1,
) -> ScanResult:
    """
    Scan a directory for Python files and extract all comments.

    Args:
        directory: Directory to scan
        exclude_dirs: Directories to exclude
        min_comment_length: Minimum comment length to include

    Returns:
        ScanResult with all found comments
    """
    python_files = find_python_files(directory, exclude_dirs)
    all_comments = []

    for file_path in python_files:
        comments = extract_comments(file_path)
        for comment in comments:
            if len(comment.text) >= min_comment_length:
                all_comments.append(comment)

    return ScanResult(
        files_scanned=len(python_files),
        comments_found=len(all_comments),
        comments=all_comments,
    )


def iter_comments(directory: str) -> Iterator[Comment]:
    """
    Iterate over comments in a directory (memory-efficient).

    Args:
        directory: Directory to scan

    Yields:
        Comment objects
    """
    for file_path in find_python_files(directory):
        for comment in extract_comments(file_path):
            yield comment
