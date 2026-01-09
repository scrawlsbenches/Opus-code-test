"""
Test Bootstrap - Container wiring for tests.

This is the test equivalent of cortical/core/bootstrap.py.
Use this for behavioral and integration tests instead of the app bootstrap.

Key differences from app bootstrap:
    - Uses InMemoryFileSystem by default (no disk I/O)
    - Registers only the services needed for tests
    - No global state

Usage:
    from tests.fixtures.test_bootstrap import create_test_container
    from cortical.cdg.index_manager import CDGIndexManager

    container = create_test_container()
    manager = container.resolve(CDGIndexManager)

For pytest fixtures:
    @pytest.fixture
    def container():
        return create_test_container()

    def test_something(container):
        manager = container.resolve(CDGIndexManager)
"""

from pathlib import Path
from typing import Optional

from cortical.common import Container, Lifecycle
from cortical.common.filesystem import FileSystem, InMemoryFileSystem


def create_test_container(
    base_dir: Optional[Path] = None,
    filesystem: Optional[FileSystem] = None,
) -> Container:
    """
    Create a container configured for testing.

    Args:
        base_dir: Base directory for storage (default: /test/store)
        filesystem: FileSystem to use (default: InMemoryFileSystem)

    Returns:
        Container with test-appropriate service registrations

    Example:
        container = create_test_container()
        index_manager = container.resolve(CDGIndexManager)
    """
    container = Container()

    # Defaults
    effective_dir = base_dir or Path("/test/store")
    effective_fs = filesystem or InMemoryFileSystem(effective_dir)

    # Register filesystem first - other services depend on it
    container.register_instance(FileSystem, effective_fs)
    container.register_instance(Path, effective_dir)

    # Ensure base directory exists in filesystem
    effective_fs.mkdir(effective_dir, parents=True, exist_ok=True)

    # Register CDG services
    _register_cdg_services(container, effective_dir, effective_fs)

    return container


def _register_cdg_services(
    container: Container,
    base_dir: Path,
    filesystem: FileSystem
) -> None:
    """Register CDG layer services."""
    from cortical.cdg.index_manager import CDGIndexManager

    # CDGIndexManager factory (schema-driven indexing)
    def create_index_manager() -> CDGIndexManager:
        return CDGIndexManager(
            store_dir=base_dir,
            filesystem=filesystem,
        )

    container.register(
        CDGIndexManager,
        create_index_manager,
        lifecycle=Lifecycle.SINGLETON,
    )

    # Add other CDG services as needed:
    # - CDGStore
    # - CDGTransactionManager
    # - etc.
