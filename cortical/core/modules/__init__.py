"""
Container Modules for Cortical Subsystems.

Each module bundles related service registrations for a specific subsystem.
Modules are applied to the container in bootstrap.py.

Available Modules:
    - SchemaModule: Entity schema registry (foundation)
    - CDGModule: Storage, WAL, transactions, indexing (foundation layer)
    - GoTModule: Graph of Thought (tasks, decisions, edges)
    - CELModule: Cognitive Event Lattice (event sourcing)
    - AuditModule: Codebase quality analysis (Bloom filter, Naive Bayes, LSH)

Module Application Order:
    1. SchemaModule (registers SchemaRegistry)
    2. CDGModule (registers CDGIndexManager, CDGStore, etc.)
    3. GoTModule, CELModule, AuditModule (domain layers)

Note: CDGIndexManager handles schema-driven index creation automatically.
      Indexes are created based on Field(indexed=True) annotations in schemas.
"""

from .cdg_module import CDGModule
from .got_module import GoTModule
from .schema_module import SchemaModule
from .audit_module import AuditModule, AuditConfig
from .cognitive_module import CognitiveModule

__all__ = [
    'SchemaModule',
    'CDGModule',
    'GoTModule',
    'AuditModule',
    'AuditConfig',
    'CognitiveModule',
]
