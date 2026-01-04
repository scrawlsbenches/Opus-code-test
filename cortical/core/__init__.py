"""
Cortical Core - Application Bootstrap and Container Configuration.

This package provides the central entry point for configuring and
bootstrapping the cortical application with dependency injection.

Container is a FIRST-CLASS CITIZEN:
    All cortical components receive dependencies through the container.
    Direct instantiation with hardcoded dependencies is prohibited.

Usage:
    from cortical.core.bootstrap import create_container, get_container

    # Application startup
    container = create_container()

    # Resolve services
    tx_manager = container.resolve(TransactionManager)
    storage = container.resolve(StorageBackend)

    # For testing - create isolated child
    test_container = container.create_child()
    test_container.register(StorageBackend, MockStorage)

Modules:
    - bootstrap: Container creation and configuration
    - modules/: Service registration modules for each subsystem
"""

from .bootstrap import create_container, get_container

__all__ = [
    'create_container',
    'get_container',
]
