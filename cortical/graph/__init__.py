"""
Graph Package: Unified Graph Architecture for Cortical.

This package provides both the SemanticKnowledgeGraph (domain-specific)
and a composable BaseGraph architecture for building custom graphs.

BaseGraph Architecture (NEW):
    - BaseGraph: Abstract base class for custom graph implementations
    - NodeBase, EdgeBase: Protocol-based node/edge contracts
    - InMemoryGraphStorage: High-performance in-memory storage
    - Algorithm mixins: PageRank, Clustering, SpreadingActivation

    Example:
        >>> from cortical.graph import SimpleGraph
        >>> graph = SimpleGraph()
        >>> graph.add_node("A", content="Concept A")
        >>> graph.add_node("B", content="Concept B")
        >>> graph.add_edge("A", "B", edge_type="related")
        >>> pagerank = graph.compute_pagerank()

SemanticKnowledgeGraph (Existing):
    - SemanticKnowledgeGraph: Unified orchestrator for cognitive architecture
    - GraphNode: Node in the knowledge graph
    - GraphEdge: Edge with semantic relation
    - Integration with CEL, GoT, WovenMind, PRISM, SparkSLM

    Example:
        >>> from cortical.graph import SemanticKnowledgeGraph
        >>> skg = SemanticKnowledgeGraph()
        >>> skg.add_document("intro", "Machine learning enables AI.")
        >>> skg.build()
        >>> results = skg.search("artificial intelligence")

See docs/base-graph-design.md for the BaseGraph architecture.
"""

# =============================================================================
# BaseGraph Architecture (NEW)
# =============================================================================

from .protocols import (
    NodeBase,
    EdgeBase,
    NodeProtocol,
    EdgeProtocol,
)

from .storage import (
    GraphStorage,
    InMemoryGraphStorage,
)

from .base import BaseGraph

from .algorithms import (
    PageRankMixin,
    ClusteringMixin,
    SpreadingActivationMixin,
    CentralityMixin,
)

from .implementations import (
    SimpleNode,
    SimpleEdge,
    SimpleGraph,
    DAGGraph,
    WeightedEdge,
    WeightedGraph,
)

# =============================================================================
# SemanticKnowledgeGraph (Existing)
# =============================================================================

from .knowledge_graph import (
    SemanticKnowledgeGraph,
    GraphNode,
    GraphEdge,
    ConnectionType,
    RelationType,
    SearchResult,
    LayerStatistics,
)

# Integration adapters (optional imports)
from .integrations import (
    CELAdapter,
    CELEvent,
    GoTAdapter,
    LinkedTask,
    LinkedDecision,
    WovenMindAdapter,
    WovenMindResult,
    ConsolidationResult,
    ThinkingMode,
    PRISMAdapter,
    AttentionResult,
    SparkSLMAdapter,
    PrimeResult,
    AnomalyResult,
)

__all__ = [
    # BaseGraph Architecture (NEW)
    'NodeBase',
    'EdgeBase',
    'NodeProtocol',
    'EdgeProtocol',
    'GraphStorage',
    'InMemoryGraphStorage',
    'BaseGraph',
    'PageRankMixin',
    'ClusteringMixin',
    'SpreadingActivationMixin',
    'CentralityMixin',
    'SimpleNode',
    'SimpleEdge',
    'SimpleGraph',
    'DAGGraph',
    'WeightedEdge',
    'WeightedGraph',
    # SemanticKnowledgeGraph (Existing)
    'SemanticKnowledgeGraph',
    'GraphNode',
    'GraphEdge',
    'ConnectionType',
    'RelationType',
    'SearchResult',
    'LayerStatistics',
    # Integration adapters
    'CELAdapter',
    'CELEvent',
    'GoTAdapter',
    'LinkedTask',
    'LinkedDecision',
    'WovenMindAdapter',
    'WovenMindResult',
    'ConsolidationResult',
    'ThinkingMode',
    'PRISMAdapter',
    'AttentionResult',
    'SparkSLMAdapter',
    'PrimeResult',
    'AnomalyResult',
]
