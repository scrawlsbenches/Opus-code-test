"""
CDG query functions.

This module provides:
1. Core functions (count, exists, type_of, fields, entity_types) - always available
2. GoT functions (graph traversal, filters, aggregates) - require got_manager extension

Extension modules can register additional functions using FunctionRegistry.

See: docs/design/cdg-query-language.md
"""

# Core functions - always available
from .core import (
    CountFunction,
    ExistsFunction,
    TypeOfFunction,
    FieldsFunction,
    EntityTypesFunction,
)

# GoT functions - require got_manager extension in QueryContext
from .got import (
    # Graph traversal
    ConnectedToFunction,
    PathFunction,
    ChildrenFunction,
    ParentsFunction,
    DescendantsFunction,
    AncestorsFunction,
    OrphanNodesFunction,
    BlockersFunction,
    DependentsFunction,
    AllDependenciesFunction,
    CycleDetectFunction,
    # Filters
    RecentFunction,
    StaleFunction,
    HasEdgeFunction,
    BlockedFunction,
    BlockingFunction,
    InSprintFunction,
    UnassignedFunction,
    OverdueFunction,
    # Aggregates
    AggregateFunction,
)

__all__ = [
    # Core
    'CountFunction',
    'ExistsFunction',
    'TypeOfFunction',
    'FieldsFunction',
    'EntityTypesFunction',
    # Graph traversal
    'ConnectedToFunction',
    'PathFunction',
    'ChildrenFunction',
    'ParentsFunction',
    'DescendantsFunction',
    'AncestorsFunction',
    'OrphanNodesFunction',
    'BlockersFunction',
    'DependentsFunction',
    'AllDependenciesFunction',
    'CycleDetectFunction',
    # Filters
    'RecentFunction',
    'StaleFunction',
    'HasEdgeFunction',
    'BlockedFunction',
    'BlockingFunction',
    'InSprintFunction',
    'UnassignedFunction',
    'OverdueFunction',
    # Aggregates
    'AggregateFunction',
]
