"""
Graph of Thought (GoT) reasoning framework for the Cortical Text Processor.

This module provides a network-based approach to complex reasoning tasks, where
thoughts are represented as nodes in a graph and relationships between thoughts
are edges. This enables:

- Multi-step reasoning with explicit dependency tracking
- Parallel exploration of reasoning paths
- Aggregation of insights from multiple perspectives
- Validation and refinement of conclusions

The framework is inspired by Graph of Thoughts (GoT) research but adapted for
text analysis and information retrieval domains.

Core Components:
    - ThoughtNode: Individual reasoning units with content and metadata
    - ThoughtEdge: Typed relationships between thoughts
    - ThoughtCluster: Grouped thoughts for hierarchical reasoning
    - ThoughtGraph: Network structure managing nodes and edges
    - Pattern factories: Pre-built reasoning patterns (chain, tree, etc.)

Example:
    >>> from cortical.reasoning import ThoughtGraph, NodeType
    >>> graph = ThoughtGraph()
    >>> node = graph.add_thought("Initial observation", NodeType.OBSERVATION)
    >>> reasoning = graph.add_thought("Analysis", NodeType.REASONING, parents=[node.id])
"""

from .graph_of_thought import (
    NodeType,
    EdgeType,
    ThoughtNode,
    ThoughtEdge,
    ThoughtCluster,
)

from .thought_graph import ThoughtGraph

from .thought_patterns import (
    create_investigation_graph,
    create_decision_graph,
    create_debug_graph,
    create_feature_graph,
    create_requirements_graph,
    create_analysis_graph,
    create_pattern_graph,
)

__all__ = [
    # Core types
    'NodeType',
    'EdgeType',
    'ThoughtNode',
    'ThoughtEdge',
    'ThoughtCluster',
    # Graph structure
    'ThoughtGraph',
    # Pattern factories
    'create_investigation_graph',
    'create_decision_graph',
    'create_debug_graph',
    'create_feature_graph',
    'create_requirements_graph',
    'create_analysis_graph',
    'create_pattern_graph',
]
