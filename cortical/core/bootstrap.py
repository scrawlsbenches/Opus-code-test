"""
Cortical Bootstrap - Application Container Configuration.

This module provides the central configuration point for the DI container.
It is the FIRST PLACE to look when understanding how cortical is wired.

Container is a FIRST-CLASS CITIZEN:
    All cortical components receive dependencies through the container.
    Direct instantiation with hardcoded dependencies is prohibited.

Architecture:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                         Bootstrap                                     │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                   │
    │  │ CDG Module  │  │ GoT Module  │  │ CEL Module  │  ... more         │
    │  │ - Storage   │  │ - TxManager │  │ - Lattice   │                   │
    │  │ - WAL       │  │ - Indexer   │  │ - Events    │                   │
    │  └─────────────┘  └─────────────┘  └─────────────┘                   │
    │                           │                                           │
    │                           ▼                                           │
    │                    ┌─────────────┐                                   │
    │                    │  Container  │                                   │
    │                    │  (Central)  │                                   │
    │                    └─────────────┘                                   │
    └─────────────────────────────────────────────────────────────────────┘

Usage:
    # Application startup
    from cortical.core.bootstrap import create_container

    container = create_container()
    tx_manager = container.resolve(TransactionManager)

    # For testing
    test_container = container.create_child()
    test_container.register(StorageBackend, MockStorage)

Adding New Modules:
    1. Create a module class in cortical/core/modules/
    2. Import and apply in create_container()
    3. Document the services provided

Breaking Changes:
    This module intentionally DOES NOT provide backward-compatible defaults.
    Components must be wired through the container.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from cortical.common import Container


# Global container instance (lazy-initialized)
_default_container: Optional[Container] = None


def create_container(
    got_dir: Optional[Path] = None,
    config: Optional[dict] = None,
) -> Container:
    """
    Create and configure the application container.

    This is the main entry point for bootstrapping cortical.
    All service registrations happen here or in applied modules.

    Args:
        got_dir: Optional GoT directory path (defaults to .got in cwd)
        config: Optional configuration dictionary

    Returns:
        Fully configured container

    Example:
        container = create_container()
        tx_manager = container.resolve(TransactionManager)

        # With custom paths
        container = create_container(got_dir=Path("/data/.got"))
    """
    container = Container()

    # Store configuration
    if config is not None:
        container.register_instance(dict, config)

    # Store paths
    if got_dir is not None:
        container.register_instance(Path, got_dir)

    # TODO: Apply subsystem modules as they are created
    # container.apply_module(CDGModule(config))
    # container.apply_module(GoTModule(config))
    # container.apply_module(CELModule(config))

    return container


def get_container() -> Container:
    """
    Get the global container instance.

    Creates the container on first access. Use this for application code
    that needs access to the container but isn't receiving it via injection.

    For testing, use create_container() to get an isolated instance,
    or create a child container with create_child().

    Returns:
        The global container instance

    Example:
        container = get_container()
        logger = container.resolve(Logger)
    """
    global _default_container
    if _default_container is None:
        _default_container = create_container()
    return _default_container


def reset_container() -> None:
    """
    Reset the global container.

    Use only in tests to ensure isolation between test cases.
    Production code should never call this.
    """
    global _default_container
    _default_container = None
