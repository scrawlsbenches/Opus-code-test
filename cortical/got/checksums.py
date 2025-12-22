"""
Checksum utilities for GoT transactional system.

DEPRECATED: This module is deprecated. Import from cortical.utils.checksums instead.

Provides functions for computing and verifying SHA256 checksums of JSON data
to ensure data integrity in the event log and snapshots.

This module now re-exports from cortical.utils.checksums for backward compatibility.
"""

import warnings
from pathlib import Path
from typing import Dict, Any

# Re-export from centralized location
from cortical.utils.checksums import (
    compute_checksum as _compute_checksum,
    verify_checksum as _verify_checksum,
    compute_file_checksum as _compute_file_checksum,
    verify_file_checksum as _verify_file_checksum,
)


def compute_checksum(data: Dict[str, Any]) -> str:
    """
    Compute SHA256 checksum of JSON data.

    DEPRECATED: Use cortical.utils.checksums.compute_checksum instead.

    Args:
        data: Dictionary to compute checksum for

    Returns:
        First 16 characters of hex digest (64 bits of entropy)
    """
    warnings.warn(
        "cortical.got.checksums is deprecated. Use cortical.utils.checksums instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return _compute_checksum(data, truncate=16)


def verify_checksum(data: Dict[str, Any], expected: str) -> bool:
    """
    Verify that data matches expected checksum.

    DEPRECATED: Use cortical.utils.checksums.verify_checksum instead.

    Args:
        data: Dictionary to verify
        expected: Expected checksum hex string

    Returns:
        True if checksum matches, False otherwise
    """
    warnings.warn(
        "cortical.got.checksums is deprecated. Use cortical.utils.checksums instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return _verify_checksum(data, expected, truncate=16)


def compute_file_checksum(path: Path) -> str:
    """
    Compute checksum of JSON file contents.

    DEPRECATED: Use cortical.utils.checksums.compute_file_checksum instead.

    Args:
        path: Path to JSON file

    Returns:
        First 16 characters of hex digest
    """
    warnings.warn(
        "cortical.got.checksums is deprecated. Use cortical.utils.checksums instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return _compute_file_checksum(path, truncate=16)


def verify_file_checksum(path: Path, expected: str) -> bool:
    """
    Verify that file contents match expected checksum.

    DEPRECATED: Use cortical.utils.checksums.verify_file_checksum instead.

    Args:
        path: Path to JSON file
        expected: Expected checksum hex string

    Returns:
        True if checksum matches, False otherwise
    """
    warnings.warn(
        "cortical.got.checksums is deprecated. Use cortical.utils.checksums instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return _verify_file_checksum(path, expected, truncate=16)
