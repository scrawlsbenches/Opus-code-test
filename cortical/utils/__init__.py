"""
Utility modules for the Cortical Text Processor.

Provides shared utilities for:
- Process-safe locking (ProcessLock)
- ID generation (generate_task_id, generate_decision_id, etc.)
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

__all__ = [
    'ProcessLock',
    'generate_task_id',
    'generate_decision_id',
    'generate_edge_id',
    'generate_sprint_id',
    'generate_goal_id',
    'normalize_id',
]
