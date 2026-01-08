"""
Container Modules for Cortical Subsystems.

Each module bundles related service registrations for a specific subsystem.
Modules are applied to the container in bootstrap.py.

Available Modules:
    - SchemaModule: Entity schema registry (foundation)
    - CDGModule: Storage, WAL, transactions (foundation layer)
    - GoTModule: Graph of Thought (tasks, decisions, edges)
    - CELModule: Cognitive Event Lattice (event sourcing)
    - AuditModule: Codebase quality analysis (Bloom filter, Naive Bayes, LSH)
"""

from .cdg_module import CDGModule
from .got_module import GoTModule
from .schema_module import SchemaModule
from .audit_module import AuditModule, AuditConfig

__all__ = [
    'SchemaModule',
    'CDGModule',
    'GoTModule',
    'AuditModule',
    'AuditConfig',
]
