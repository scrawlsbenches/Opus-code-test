"""
Checksum utilities for GoT transactional system.

Provides functions for computing and verifying SHA256 checksums of JSON data
to ensure data integrity in the event log and snapshots.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, Any


def compute_checksum(data: Dict[str, Any]) -> str:
    """
    Compute SHA256 checksum of JSON data.

    Args:
        data: Dictionary to compute checksum for

    Returns:
        First 16 characters of hex digest (64 bits of entropy)
    """
    json_str = json.dumps(data, sort_keys=True, separators=(',', ':'))
    hash_obj = hashlib.sha256(json_str.encode('utf-8'))
    return hash_obj.hexdigest()[:16]


def verify_checksum(data: Dict[str, Any], expected: str) -> bool:
    """
    Verify that data matches expected checksum.

    Args:
        data: Dictionary to verify
        expected: Expected checksum hex string

    Returns:
        True if checksum matches, False otherwise
    """
    actual = compute_checksum(data)
    return actual == expected


def compute_file_checksum(path: Path) -> str:
    """
    Compute checksum of JSON file contents.

    Args:
        path: Path to JSON file

    Returns:
        First 16 characters of hex digest
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return compute_checksum(data)


def verify_file_checksum(path: Path, expected: str) -> bool:
    """
    Verify that file contents match expected checksum.

    Args:
        path: Path to JSON file
        expected: Expected checksum hex string

    Returns:
        True if checksum matches, False otherwise
    """
    actual = compute_file_checksum(path)
    return actual == expected
