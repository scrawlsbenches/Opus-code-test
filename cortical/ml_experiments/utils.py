"""
Utility functions for ML experiment tracking.

Provides reproducibility helpers, hashing, ID generation, and git integration.
Zero external dependencies - uses only Python standard library.
"""

import hashlib
import json
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def compute_file_hash(path: Path, algorithm: str = 'sha256') -> str:
    """
    Compute hash of a file for integrity checking.

    Args:
        path: Path to the file
        algorithm: Hash algorithm ('sha256', 'md5')

    Returns:
        First 12 characters of the hex digest
    """
    hasher = hashlib.new(algorithm)
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()[:12]


def compute_content_hash(content: str, algorithm: str = 'sha256') -> str:
    """
    Compute hash of string content.

    Args:
        content: String to hash
        algorithm: Hash algorithm

    Returns:
        First 12 characters of the hex digest
    """
    hasher = hashlib.new(algorithm)
    hasher.update(content.encode('utf-8'))
    return hasher.hexdigest()[:12]


def compute_dict_hash(data: Dict[str, Any]) -> str:
    """
    Compute hash of a dictionary (for config fingerprinting).

    Sorts keys for deterministic hashing.

    Args:
        data: Dictionary to hash

    Returns:
        First 12 characters of the hex digest
    """
    # Sort keys recursively for deterministic serialization
    serialized = json.dumps(data, sort_keys=True, default=str)
    return compute_content_hash(serialized)


def generate_experiment_id() -> str:
    """
    Generate unique experiment ID.

    Format: exp-YYYYMMDD-HHMMSS-XXXX
    where XXXX is a random hex suffix for uniqueness.

    Returns:
        Unique experiment identifier
    """
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = ''.join(random.choices('0123456789abcdef', k=4))
    return f"exp-{ts}-{suffix}"


def generate_dataset_id(name: str) -> str:
    """
    Generate dataset version ID.

    Format: ds-{name}-YYYYMMDD-XXXX

    Args:
        name: Human-readable dataset name

    Returns:
        Unique dataset identifier
    """
    ts = datetime.now().strftime("%Y%m%d")
    suffix = ''.join(random.choices('0123456789abcdef', k=4))
    # Sanitize name (lowercase, replace spaces with underscores)
    safe_name = name.lower().replace(' ', '_').replace('-', '_')[:20]
    return f"ds-{safe_name}-{ts}-{suffix}"


def now_iso() -> str:
    """
    Get current UTC timestamp in ISO format.

    Returns:
        ISO 8601 formatted timestamp
    """
    return datetime.now(timezone.utc).isoformat()


def get_git_hash() -> Optional[str]:
    """
    Get current git commit hash.

    Returns:
        First 12 characters of HEAD commit hash, or None if not a git repo
    """
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def get_git_status() -> str:
    """
    Check if git working directory is clean.

    Returns:
        'clean' if no uncommitted changes, 'dirty' otherwise
    """
    try:
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return 'clean' if not result.stdout.strip() else 'dirty'
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return 'unknown'


def set_random_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Integer seed value
    """
    random.seed(seed)


def reproducible_shuffle(items: List[Any], seed: int) -> List[Any]:
    """
    Shuffle a list reproducibly with a given seed.

    Args:
        items: List to shuffle
        seed: Random seed

    Returns:
        New shuffled list (original unchanged)
    """
    rng = random.Random(seed)
    shuffled = items.copy()
    rng.shuffle(shuffled)
    return shuffled


def split_indices(
    total: int,
    ratios: Dict[str, float],
    seed: int,
    strategy: str = 'random'
) -> Dict[str, List[int]]:
    """
    Split indices into train/val/test sets.

    Args:
        total: Total number of items
        ratios: Dict like {'train': 0.7, 'val': 0.15, 'test': 0.15}
        seed: Random seed for reproducibility
        strategy: 'random' or 'temporal' (temporal preserves order)

    Returns:
        Dict mapping split names to lists of indices

    Raises:
        ValueError: If ratios don't sum to ~1.0
    """
    # Validate ratios
    ratio_sum = sum(ratios.values())
    if not (0.99 <= ratio_sum <= 1.01):
        raise ValueError(f"Ratios must sum to 1.0, got {ratio_sum}")

    indices = list(range(total))

    if strategy == 'random':
        indices = reproducible_shuffle(indices, seed)
    # For 'temporal', keep original order

    splits = {}
    start = 0

    # Sort splits by name for deterministic ordering
    for split_name in sorted(ratios.keys()):
        ratio = ratios[split_name]
        count = int(total * ratio)

        # Ensure last split gets remaining items (handles rounding)
        if split_name == sorted(ratios.keys())[-1]:
            splits[split_name] = indices[start:]
        else:
            splits[split_name] = indices[start:start + count]
            start += count

    return splits


def ensure_directory(path: Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        The path (for chaining)
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    """
    Append a record to a JSONL file (append-only, merge-friendly).

    Args:
        path: Path to JSONL file
        record: Dictionary to append
    """
    ensure_directory(path.parent)
    with open(path, 'a') as f:
        f.write(json.dumps(record, default=str) + '\n')


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """
    Read all records from a JSONL file.

    Args:
        path: Path to JSONL file

    Returns:
        List of dictionaries (empty list if file doesn't exist)
    """
    if not path.exists():
        return []

    records = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # Skip malformed lines
    return records


def filter_jsonl(
    path: Path,
    filters: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Read and filter records from a JSONL file.

    Args:
        path: Path to JSONL file
        filters: Dict of field -> value to match

    Returns:
        List of matching records
    """
    records = read_jsonl(path)

    if not filters:
        return records

    def matches(record: Dict[str, Any]) -> bool:
        for key, value in filters.items():
            # Support nested keys with dot notation
            parts = key.split('.')
            current = record
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return False
            if current != value:
                return False
        return True

    return [r for r in records if matches(r)]


class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for ML experiment data types."""

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, Path):
            return str(obj)
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        if hasattr(obj, '__dataclass_fields__'):
            from dataclasses import asdict
            return asdict(obj)
        return super().default(obj)


def save_json(path: Path, data: Any, indent: int = 2) -> None:
    """
    Save data to a JSON file with custom encoding.

    Args:
        path: Output path
        data: Data to save
        indent: Indentation level (None for compact)
    """
    ensure_directory(path.parent)
    with open(path, 'w') as f:
        json.dump(data, f, cls=JSONEncoder, indent=indent)


def load_json(path: Path) -> Any:
    """
    Load data from a JSON file.

    Args:
        path: Input path

    Returns:
        Loaded data, or None if file doesn't exist
    """
    if not path.exists():
        return None
    with open(path, 'r') as f:
        return json.load(f)
