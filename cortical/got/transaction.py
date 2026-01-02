"""
Transaction management for Graph of Thought ACID-compliant storage.

This module now re-exports transaction infrastructure from CDG (Cortical
Distributed Graph) to ensure compatibility when GoT delegates to CDG.

Migration Note (2025-12-31):
    GoT now uses CDG's Transaction, TransactionState, and generate_transaction_id.
    This ensures that transaction objects returned by TransactionManager are
    compatible with GoT's existing code and tests.

    The CDG transaction module is nearly identical to GoT's original implementation,
    with minor extensions (touched_partitions, metadata) that don't affect GoT's usage.
"""

from __future__ import annotations

# Re-export CDG transaction infrastructure for GoT compatibility
from cortical.cdg.transaction import (
    Transaction,
    TransactionState,
    generate_transaction_id,
)

__all__ = [
    "Transaction",
    "TransactionState",
    "generate_transaction_id",
]
