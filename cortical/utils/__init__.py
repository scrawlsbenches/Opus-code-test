"""
Utility modules for the Cortical Text Processor.

Provides shared utilities for:
- Process-safe locking (ProcessLock)
- ID generation (generate_task_id, generate_decision_id, etc.)
- Text manipulation (slugify)
- Atomic file persistence (atomic_write, atomic_write_json)
- Checksum computation and verification (compute_checksum, verify_checksum)
"""

from .locking import ProcessLock
from .id_generation import (
    generate_task_id,
    generate_decision_id,
    generate_edge_id,
    generate_sprint_id,
    generate_goal_id,
    normalize_id,
)
from .text import slugify
from .persistence import atomic_write, atomic_write_json
from .checksums import (
    compute_checksum,
    verify_checksum,
    compute_file_checksum,
    verify_file_checksum,
    compute_dict_checksum,
    compute_bytes_checksum,
)

__all__ = [
    'ProcessLock',
    'generate_task_id',
    'generate_decision_id',
    'generate_edge_id',
    'generate_sprint_id',
    'generate_goal_id',
    'normalize_id',
    'slugify',
    'atomic_write',
    'atomic_write_json',
    'compute_checksum',
    'verify_checksum',
    'compute_file_checksum',
    'verify_file_checksum',
    'compute_dict_checksum',
    'compute_bytes_checksum',
]
