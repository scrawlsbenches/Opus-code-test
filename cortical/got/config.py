"""
GoT configuration module.

Provides configuration options for the Graph of Thought transactional system,
including durability modes that control fsync behavior.
"""

from enum import Enum
from dataclasses import dataclass


class DurabilityMode(Enum):
    """
    Durability mode controls when data is synced to disk.

    PARANOID: fsync on every operation (slowest, safest)
        - Every WAL entry fsynced immediately
        - Every entity file fsynced immediately
        - Guarantees zero data loss even on power failure mid-operation
        - Performance: ~36 ops/s (single), ~114 ops/s (bulk)

    BALANCED: fsync on transaction commit only (recommended)
        - WAL entries buffered, fsynced only on commit
        - Entity files fsynced in batch at commit
        - Guarantees committed transactions survive power loss
        - Performance: Expected ~150-200 ops/s

    RELAXED: no fsync, rely on OS buffer cache (fastest, least safe)
        - No explicit fsync calls
        - Relies on OS to flush buffers (~5-30s window)
        - Guarantees data survives process crash (data in kernel buffer)
        - Risk: Power loss within OS flush window loses uncommitted data
        - Performance: Expected ~500+ ops/s
    """

    PARANOID = "paranoid"
    BALANCED = "balanced"
    RELAXED = "relaxed"


@dataclass
class GoTConfig:
    """
    Configuration for GoT transactional system.

    Attributes:
        durability: Durability mode (default: BALANCED)
    """

    durability: DurabilityMode = DurabilityMode.BALANCED
