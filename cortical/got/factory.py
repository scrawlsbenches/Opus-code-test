"""
GoT Backend Factory.

Creates instances of the GoT backend via DI container.
"""

import os
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from cortical.got.api import GoTManager

# Project root for default paths
_PROJECT_ROOT = Path(__file__).parent.parent.parent

# Allow GOT_DIR to be overridden via environment variable (for testing)
GOT_DIR = Path(os.environ.get("GOT_DIR", _PROJECT_ROOT / ".got"))


class GoTBackendFactory:
    """Factory for creating GoT backend instances."""

    @staticmethod
    def create(
        backend: Optional[str] = None,
        got_dir: Optional[Path] = None,
    ) -> "GoTManager":
        """
        Create GoT backend via DI container.

        Args:
            backend: Ignored (kept for compatibility)
            got_dir: Override default directory

        Returns:
            GoTManager instance
        """
        from cortical.got.api import GoTManager
        from cortical.core.bootstrap import create_container

        container = create_container(got_dir=got_dir or GOT_DIR)
        return container.resolve(GoTManager)

    @staticmethod
    def get_available_backends() -> List[str]:
        """Get list of available backends."""
        return ["transactional"]
