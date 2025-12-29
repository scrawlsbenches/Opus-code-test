"""
Sanity strand for the Cognitive Event Lattice.

The Sanity strand complements Wisdom by providing:
- Health monitoring and self-diagnosis
- Schema migration and evolution
- Semantic compaction for storage efficiency

Together, Wisdom (knowledge) and Sanity (health) form the
double helix that allows the system to reason about itself
while maintaining structural integrity.

Design Philosophy:
    "A system that can't monitor itself can't heal itself."

    The Sanity strand enables meta-cognition - the system
    observing and reasoning about its own health. This is
    the foundation for self-maintenance and evolution.

Contents:
    health.py - Health monitoring and anomaly detection
    migration.py - Schema evolution and data migration
    compaction.py - Semantic compression strategies
"""

from .health import (
    HealthMetric,
    HealthStatus,
    HealthReport,
    EventStoreHealthMonitor,
)
from .migration import (
    MigrationStep,
    MigrationPlan,
    SchemaMigrationEngine,
)
from .compaction import (
    CompactionResult,
    SemanticCompactor,
    TimeWindowCompactor,
)

__all__ = [
    # Health
    'HealthMetric',
    'HealthStatus',
    'HealthReport',
    'EventStoreHealthMonitor',
    # Migration
    'MigrationStep',
    'MigrationPlan',
    'SchemaMigrationEngine',
    # Compaction
    'CompactionResult',
    'SemanticCompactor',
    'TimeWindowCompactor',
]
