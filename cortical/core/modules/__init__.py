"""
Container Modules for Cortical Subsystems.

Each module bundles related service registrations for a specific subsystem.
Modules are applied to the container in bootstrap.py.

Available Modules:
    - SchemaModule: Entity schema registry (foundation)
    - CDGModule: Storage, WAL, transactions (foundation layer)
    - IndexInitializationModule: Schema-driven index creation
    - GoTModule: Graph of Thought (tasks, decisions, edges)
    - CELModule: Cognitive Event Lattice (event sourcing)

Module Application Order:
    1. SchemaModule (registers SchemaRegistry)
    2. CDGModule (registers IndexManager, CDGStore, etc.)
    3. IndexInitializationModule (creates indexes from schemas)
    4. GoTModule, CELModule, etc. (domain layers)
"""

from .cdg_module import CDGModule
from .got_module import GoTModule
from .index_init_module import IndexInitializationModule
from .schema_module import SchemaModule

__all__ = [
    'SchemaModule',
    'CDGModule',
    'IndexInitializationModule',
    'GoTModule',
]
