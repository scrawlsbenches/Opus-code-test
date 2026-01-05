"""
Container Modules for Cortical Subsystems.

Each module bundles related service registrations for a specific subsystem.
Modules are applied to the container in bootstrap.py.

Available Modules:
    - CDGModule: Storage, WAL, transactions (foundation layer)
    - GoTModule: Graph of Thought (tasks, decisions, edges)
    - CELModule: Cognitive Event Lattice (event sourcing)
"""

from .cdg_module import CDGModule
from .got_module import GoTModule

__all__ = [
    'CDGModule',
    'GoTModule',
]
