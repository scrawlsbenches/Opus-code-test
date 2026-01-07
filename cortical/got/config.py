"""
GoT configuration module.

Provides configuration options for the Graph of Thought transactional system.

Note: GoT delegates all storage to CDG. DurabilityMode is re-exported from
CDG for backward compatibility. GoT itself should not know about storage details.
"""

from dataclasses import dataclass

# Re-export DurabilityMode from CDG (the canonical source)
from cortical.cdg.config import DurabilityMode


@dataclass
class GoTConfig:
    """
    Configuration for GoT transactional system.

    Attributes:
        durability: Durability mode (default: BALANCED)

    Note: This config is a thin wrapper. CDGConfig holds the full storage
    configuration. GoT delegates to CDG for all storage operations.
    """

    durability: DurabilityMode = DurabilityMode.BALANCED
