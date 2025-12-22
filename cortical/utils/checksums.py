"""
Checksum utilities for data integrity verification.

Provides functions for computing and verifying SHA256 checksums of various data types
to ensure data integrity in the event log, snapshots, and storage systems.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, Any, Union


def compute_checksum(data: Union[bytes, Dict[str, Any]], truncate: int = 16) -> str:
    """
    Compute SHA256 checksum of data.

    Args:
        data: Either raw bytes or dictionary to compute checksum for.
              If dict, it will be JSON-serialized with sorted keys.
        truncate: Number of hex characters to return (default: 16 = 64 bits of entropy).
                  Use 0 or None for full hash (64 characters).

    Returns:
        Hex digest string, truncated to specified length
    """
    if isinstance(data, dict):
        # JSON-serialize dictionary with sorted keys for deterministic output
        json_str = json.dumps(data, sort_keys=True, separators=(',', ':'))
        bytes_data = json_str.encode('utf-8')
    elif isinstance(data, bytes):
        bytes_data = data
    else:
        raise TypeError(f"data must be bytes or dict, got {type(data).__name__}")

    hash_obj = hashlib.sha256(bytes_data)
    hex_digest = hash_obj.hexdigest()

    if truncate and truncate > 0:
        return hex_digest[:truncate]
    return hex_digest


def verify_checksum(data: Union[bytes, Dict[str, Any]], expected: str, truncate: int = 16) -> bool:
    """
    Verify that data matches expected checksum.

    Args:
        data: Either raw bytes or dictionary to verify
        expected: Expected checksum hex string
        truncate: Number of hex characters used in checksum (must match compute_checksum)

    Returns:
        True if checksum matches, False otherwise
    """
    actual = compute_checksum(data, truncate=truncate)
    return actual == expected


def compute_file_checksum(path: Path, truncate: int = 16) -> str:
    """
    Compute checksum of JSON file contents.

    Args:
        path: Path to JSON file
        truncate: Number of hex characters to return

    Returns:
        Hex digest string, truncated to specified length
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return compute_checksum(data, truncate=truncate)


def verify_file_checksum(path: Path, expected: str, truncate: int = 16) -> bool:
    """
    Verify that file contents match expected checksum.

    Args:
        path: Path to JSON file
        expected: Expected checksum hex string
        truncate: Number of hex characters used in checksum

    Returns:
        True if checksum matches, False otherwise
    """
    actual = compute_file_checksum(path, truncate=truncate)
    return actual == expected


def compute_dict_checksum(data: Dict[str, Any], truncate: int = 16) -> str:
    """
    Compute checksum of a dictionary (convenience wrapper).

    Args:
        data: Dictionary to compute checksum for
        truncate: Number of hex characters to return

    Returns:
        Hex digest string, truncated to specified length
    """
    return compute_checksum(data, truncate=truncate)


def compute_bytes_checksum(data: bytes, truncate: int = 16) -> str:
    """
    Compute checksum of raw bytes (convenience wrapper).

    Args:
        data: Raw bytes to compute checksum for
        truncate: Number of hex characters to return

    Returns:
        Hex digest string, truncated to specified length
    """
    return compute_checksum(data, truncate=truncate)
