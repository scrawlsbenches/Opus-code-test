"""
FileSystem Abstraction Layer

Provides a protocol-based abstraction over filesystem operations, enabling:
- Real disk I/O for production
- In-memory storage for fast testing
- Future: remote storage, encrypted storage, etc.

This follows the Dependency Inversion Principle - high-level modules (CDGStore)
depend on abstractions (FileSystem), not concrete implementations.

Example:
    # Production
    fs = RealFileSystem()
    store = CDGStore(filesystem=fs)

    # Testing (10x faster, no disk I/O)
    fs = InMemoryFileSystem()
    store = CDGStore(filesystem=fs)

Design Principles:
    1. Protocol-based for duck typing compatibility
    2. Minimal interface - only what CDGStore needs
    3. Path objects throughout (no string paths)
    4. Atomic operations where possible
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Protocol, Set, Union


class FileSystem(Protocol):
    """
    Protocol defining filesystem operations needed by CDGStore.

    Implementations must provide all methods. The protocol is designed
    to be minimal - only operations actually used by storage code.
    """

    # =========================================================================
    # Directory Operations
    # =========================================================================

    def mkdir(self, path: Path, parents: bool = False, exist_ok: bool = False) -> None:
        """
        Create a directory.

        Args:
            path: Directory path to create
            parents: If True, create parent directories as needed
            exist_ok: If True, don't raise if directory exists
        """
        ...

    def exists(self, path: Path) -> bool:
        """Check if path exists (file or directory)."""
        ...

    def is_dir(self, path: Path) -> bool:
        """Check if path is a directory."""
        ...

    def glob(self, path: Path, pattern: str) -> List[Path]:
        """
        Find files matching a glob pattern.

        Args:
            path: Base directory to search in
            pattern: Glob pattern (e.g., "*.json", "**/*.py")

        Returns:
            List of matching Path objects
        """
        ...

    def iterdir(self, path: Path) -> Iterator[Path]:
        """
        Iterate over directory contents.

        Args:
            path: Directory path to iterate

        Returns:
            Iterator of Path objects for directory contents
        """
        ...

    # =========================================================================
    # File Operations
    # =========================================================================

    def read_text(self, path: Path) -> str:
        """
        Read entire file as text.

        Args:
            path: File path to read

        Returns:
            File contents as string

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        ...

    def write_text(self, path: Path, content: str) -> None:
        """
        Write text to file (overwrites existing).

        Args:
            path: File path to write
            content: Text content to write
        """
        ...

    def append_text(self, path: Path, content: str) -> None:
        """
        Append text to file (creates if doesn't exist).

        Args:
            path: File path to append to
            content: Text content to append
        """
        ...

    def unlink(self, path: Path, missing_ok: bool = False) -> None:
        """
        Delete a file.

        Args:
            path: File path to delete
            missing_ok: If True, don't raise if file doesn't exist
        """
        ...

    def rename(self, src: Path, dst: Path) -> None:
        """
        Rename/move a file atomically.

        Args:
            src: Source path
            dst: Destination path
        """
        ...

    # =========================================================================
    # Durability Operations
    # =========================================================================

    def fsync(self, path: Path) -> None:
        """
        Force flush file to disk.

        For in-memory implementations, this is a no-op.

        Args:
            path: File path to sync
        """
        ...

    def fsync_dir(self, path: Path) -> None:
        """
        Force flush directory metadata to disk.

        For in-memory implementations, this is a no-op.

        Args:
            path: Directory path to sync
        """
        ...


class RealFileSystem:
    """
    Real filesystem implementation using actual disk I/O.

    This is the production implementation. All operations go to disk.
    """

    def mkdir(self, path: Path, parents: bool = False, exist_ok: bool = False) -> None:
        path.mkdir(parents=parents, exist_ok=exist_ok)

    def exists(self, path: Path) -> bool:
        return path.exists()

    def is_dir(self, path: Path) -> bool:
        return path.is_dir()

    def glob(self, path: Path, pattern: str) -> List[Path]:
        return list(path.glob(pattern))

    def iterdir(self, path: Path) -> Iterator[Path]:
        return path.iterdir()

    def read_text(self, path: Path) -> str:
        return path.read_text(encoding='utf-8')

    def write_text(self, path: Path, content: str) -> None:
        path.write_text(content, encoding='utf-8')

    def append_text(self, path: Path, content: str) -> None:
        with open(path, 'a', encoding='utf-8') as f:
            f.write(content)

    def unlink(self, path: Path, missing_ok: bool = False) -> None:
        try:
            path.unlink()
        except FileNotFoundError:
            if not missing_ok:
                raise

    def rename(self, src: Path, dst: Path) -> None:
        src.rename(dst)

    def fsync(self, path: Path) -> None:
        """Fsync file to disk."""
        with open(path, 'r+b') as f:
            os.fsync(f.fileno())

    def fsync_dir(self, path: Path) -> None:
        """Fsync directory to ensure rename durability on POSIX."""
        fd = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


class InMemoryFileSystem:
    """
    In-memory filesystem implementation for testing.

    All operations work on an in-memory dict structure. No disk I/O.
    ~10x faster than RealFileSystem for tests.

    Features:
    - Full API compatibility with RealFileSystem
    - Simulates directory structure
    - Supports glob patterns
    - fsync operations are no-ops (instant)
    - Operation tracking for behavioral test assertions

    Example:
        fs = InMemoryFileSystem()
        fs.mkdir(Path("/data"), parents=True)
        fs.write_text(Path("/data/test.json"), '{"key": "value"}')
        content = fs.read_text(Path("/data/test.json"))

        # Assert operations occurred
        fs.assert_file_was_read(Path("/data/test.json"))
        fs.assert_directory_was_created(Path("/data"))
    """

    def __init__(self):
        # Files: path_str -> content
        self._files: Dict[str, str] = {}
        # Directories: set of path_str
        self._dirs: Set[str] = set()
        # Root always exists
        self._dirs.add("/")

        # Operation tracking for assertions
        self._operations: List[Dict[str, any]] = []
        self._files_read: Set[str] = set()
        self._files_written: Set[str] = set()
        self._directories_created: Set[str] = set()
        self._directories_scanned: Set[str] = set()

    def _normalize(self, path: Path) -> str:
        """Normalize path to string for dict keys."""
        return str(path.resolve())

    def _record_operation(self, op_type: str, path: Path, **kwargs) -> None:
        """Record an operation for later assertion."""
        self._operations.append({
            "type": op_type,
            "path": str(path),
            **kwargs
        })

    def mkdir(self, path: Path, parents: bool = False, exist_ok: bool = False) -> None:
        path_str = self._normalize(path)

        if path_str in self._dirs:
            if not exist_ok:
                raise FileExistsError(f"Directory exists: {path}")
            return

        if path_str in self._files:
            raise FileExistsError(f"File exists at path: {path}")

        # Check parent exists
        parent_str = self._normalize(path.parent)
        if parent_str not in self._dirs and parent_str != path_str:
            if parents:
                self.mkdir(path.parent, parents=True, exist_ok=True)
            else:
                raise FileNotFoundError(f"Parent directory doesn't exist: {path.parent}")

        self._dirs.add(path_str)
        self._directories_created.add(path_str)
        self._record_operation("mkdir", path, parents=parents, exist_ok=exist_ok)

    def exists(self, path: Path) -> bool:
        path_str = self._normalize(path)
        return path_str in self._files or path_str in self._dirs

    def is_dir(self, path: Path) -> bool:
        return self._normalize(path) in self._dirs

    def glob(self, path: Path, pattern: str) -> List[Path]:
        """
        Simple glob implementation for in-memory filesystem.

        Supports: *, ?, **
        """
        import fnmatch

        base_str = self._normalize(path)
        self._directories_scanned.add(base_str)
        self._record_operation("glob", path, pattern=pattern)

        results = []

        # Combine normalized base path with pattern (use / to join, not Path)
        full_pattern = base_str.rstrip("/") + "/" + pattern

        # Check all files
        for file_path in self._files.keys():
            if fnmatch.fnmatch(file_path, full_pattern):
                results.append(Path(file_path))

        # For ** patterns, also need to check recursively
        if "**" in pattern:
            # Simplified: just check if path starts with base and matches pattern
            for file_path in self._files.keys():
                if file_path.startswith(base_str):
                    rel_path = file_path[len(base_str):].lstrip("/")
                    if fnmatch.fnmatch(rel_path, pattern.replace("**/", "")):
                        results.append(Path(file_path))

        return sorted(set(results))

    def iterdir(self, path: Path) -> Iterator[Path]:
        """Iterate over directory contents."""
        path_str = self._normalize(path)
        if path_str not in self._dirs:
            raise FileNotFoundError(f"Directory not found: {path}")

        self._directories_scanned.add(path_str)
        self._record_operation("iterdir", path)

        # Find all files and dirs that are direct children of this directory
        children = set()
        prefix = path_str.rstrip("/") + "/"

        # Check files
        for file_path in self._files.keys():
            if file_path.startswith(prefix):
                # Get the immediate child name
                remainder = file_path[len(prefix):]
                if "/" not in remainder:
                    children.add(file_path)

        # Check directories
        for dir_path in self._dirs:
            if dir_path.startswith(prefix) and dir_path != path_str:
                remainder = dir_path[len(prefix):]
                if "/" not in remainder:
                    children.add(dir_path)

        return iter(sorted(Path(p) for p in children))

    def read_text(self, path: Path) -> str:
        path_str = self._normalize(path)
        if path_str not in self._files:
            raise FileNotFoundError(f"File not found: {path}")
        self._files_read.add(path_str)
        self._record_operation("read", path)
        return self._files[path_str]

    def write_text(self, path: Path, content: str) -> None:
        path_str = self._normalize(path)

        # Ensure parent directory exists
        parent_str = self._normalize(path.parent)
        if parent_str not in self._dirs:
            raise FileNotFoundError(f"Parent directory doesn't exist: {path.parent}")

        self._files[path_str] = content
        self._files_written.add(path_str)
        self._record_operation("write", path, content_length=len(content))

    def append_text(self, path: Path, content: str) -> None:
        path_str = self._normalize(path)

        # Ensure parent directory exists
        parent_str = self._normalize(path.parent)
        if parent_str not in self._dirs:
            raise FileNotFoundError(f"Parent directory doesn't exist: {path.parent}")

        if path_str in self._files:
            self._files[path_str] += content
        else:
            self._files[path_str] = content

    def unlink(self, path: Path, missing_ok: bool = False) -> None:
        path_str = self._normalize(path)
        if path_str not in self._files:
            if not missing_ok:
                raise FileNotFoundError(f"File not found: {path}")
            return
        del self._files[path_str]

    def rename(self, src: Path, dst: Path) -> None:
        src_str = self._normalize(src)
        dst_str = self._normalize(dst)

        if src_str not in self._files:
            raise FileNotFoundError(f"Source file not found: {src}")

        # Ensure destination parent exists
        dst_parent_str = self._normalize(dst.parent)
        if dst_parent_str not in self._dirs:
            raise FileNotFoundError(f"Destination parent doesn't exist: {dst.parent}")

        # Move content
        self._files[dst_str] = self._files[src_str]
        del self._files[src_str]

    def fsync(self, path: Path) -> None:
        """No-op for in-memory filesystem."""
        pass

    def fsync_dir(self, path: Path) -> None:
        """No-op for in-memory filesystem."""
        pass

    # =========================================================================
    # Testing Utilities
    # =========================================================================

    def clear(self) -> None:
        """Clear all files and directories (except root)."""
        self._files.clear()
        self._dirs.clear()
        self._dirs.add("/")
        self.reset_tracking()

    def reset_tracking(self) -> None:
        """Reset operation tracking (keeps files/dirs intact)."""
        self._operations.clear()
        self._files_read.clear()
        self._files_written.clear()
        self._directories_created.clear()
        self._directories_scanned.clear()

    def list_all_files(self) -> List[str]:
        """List all files (for debugging)."""
        return sorted(self._files.keys())

    def list_all_dirs(self) -> List[str]:
        """List all directories (for debugging)."""
        return sorted(self._dirs)

    def dump(self) -> Dict[str, any]:
        """Dump entire filesystem state (for debugging/assertions)."""
        return {
            "files": dict(self._files),
            "dirs": list(self._dirs),
        }

    # =========================================================================
    # Operation Tracking - Properties
    # =========================================================================

    @property
    def operations(self) -> List[Dict[str, any]]:
        """Get list of all operations performed."""
        return list(self._operations)

    @property
    def files_read(self) -> Set[str]:
        """Get set of files that were read."""
        return set(self._files_read)

    @property
    def files_written(self) -> Set[str]:
        """Get set of files that were written."""
        return set(self._files_written)

    @property
    def directories_created(self) -> Set[str]:
        """Get set of directories that were created."""
        return set(self._directories_created)

    @property
    def directories_scanned(self) -> Set[str]:
        """Get set of directories that were scanned (via glob or iterdir)."""
        return set(self._directories_scanned)

    @property
    def read_count(self) -> int:
        """Count of read operations."""
        return len(self._files_read)

    @property
    def write_count(self) -> int:
        """Count of write operations."""
        return len(self._files_written)

    # =========================================================================
    # Assertions - For Behavioral Tests
    # =========================================================================

    def assert_file_was_read(self, path: Path) -> None:
        """
        Assert that a specific file was read.

        Args:
            path: The file path that should have been read

        Raises:
            AssertionError: If the file was not read
        """
        path_str = self._normalize(path)
        if path_str not in self._files_read:
            read_list = ", ".join(sorted(self._files_read)[:5])
            if len(self._files_read) > 5:
                read_list += f"... ({len(self._files_read)} total)"
            raise AssertionError(
                f"Expected file to be read: {path}\n"
                f"Files actually read: [{read_list}]"
            )

    def assert_file_was_written(self, path: Path) -> None:
        """
        Assert that a specific file was written.

        Args:
            path: The file path that should have been written

        Raises:
            AssertionError: If the file was not written
        """
        path_str = self._normalize(path)
        if path_str not in self._files_written:
            written_list = ", ".join(sorted(self._files_written)[:5])
            if len(self._files_written) > 5:
                written_list += f"... ({len(self._files_written)} total)"
            raise AssertionError(
                f"Expected file to be written: {path}\n"
                f"Files actually written: [{written_list}]"
            )

    def assert_directory_was_created(self, path: Path) -> None:
        """
        Assert that a specific directory was created.

        Args:
            path: The directory path that should have been created

        Raises:
            AssertionError: If the directory was not created
        """
        path_str = self._normalize(path)
        if path_str not in self._directories_created:
            raise AssertionError(
                f"Expected directory to be created: {path}\n"
                f"Directories created: {sorted(self._directories_created)}"
            )

    def assert_directory_was_scanned(self, path: Path) -> None:
        """
        Assert that a specific directory was scanned (glob or iterdir).

        Args:
            path: The directory path that should have been scanned

        Raises:
            AssertionError: If the directory was not scanned
        """
        path_str = self._normalize(path)
        if path_str not in self._directories_scanned:
            raise AssertionError(
                f"Expected directory to be scanned: {path}\n"
                f"Directories scanned: {sorted(self._directories_scanned)}"
            )

    def assert_no_writes(self) -> None:
        """
        Assert that no files were written.

        Raises:
            AssertionError: If any files were written
        """
        if self._files_written:
            raise AssertionError(
                f"Expected no writes, but {len(self._files_written)} files were written: "
                f"{sorted(self._files_written)}"
            )

    def assert_file_count_read(self, expected: int) -> None:
        """
        Assert that exactly N files were read.

        Args:
            expected: The expected number of files read

        Raises:
            AssertionError: If the count doesn't match
        """
        actual = len(self._files_read)
        if actual != expected:
            raise AssertionError(
                f"Expected {expected} files to be read, but {actual} were read"
            )

    def assert_operation_sequence(self, expected_types: List[str]) -> None:
        """
        Assert that operations occurred in a specific sequence.

        Args:
            expected_types: List of operation types in expected order

        Raises:
            AssertionError: If the sequence doesn't match
        """
        actual_types = [op["type"] for op in self._operations]
        if actual_types != expected_types:
            raise AssertionError(
                f"Operation sequence mismatch:\n"
                f"  Expected: {expected_types}\n"
                f"  Actual:   {actual_types}"
            )
