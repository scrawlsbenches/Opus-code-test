"""
GoT Backend Factory.

Creates instances of the transactional GoT backend.
"""

import os
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from cortical.got.adapter import TransactionalGoTAdapter

# Project root for default paths
_PROJECT_ROOT = Path(__file__).parent.parent.parent

# Allow GOT_DIR to be overridden via environment variable (for testing)
GOT_DIR = Path(os.environ.get("GOT_DIR", _PROJECT_ROOT / ".got"))


class GoTBackendFactory:
    """Factory for creating GoT backend instances (transactional only)."""

    @staticmethod
    def create(
        backend: Optional[str] = None,
        got_dir: Optional[Path] = None,
    ) -> "TransactionalGoTAdapter":
        """
        Create transactional GoT backend.

        Args:
            backend: Ignored (kept for compatibility), always uses transactional
            got_dir: Override default directory

        Returns:
            TransactionalGoTAdapter instance

        Raises:
            RuntimeError: If transactional backend not available
        """
        # Import here to avoid circular imports
        from cortical.got.adapter import TransactionalGoTAdapter

        return TransactionalGoTAdapter(got_dir or GOT_DIR)

    @staticmethod
    def get_available_backends() -> List[str]:
        """Get list of available backends (transactional only)."""
        return ["transactional"]
