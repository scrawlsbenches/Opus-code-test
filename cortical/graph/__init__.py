"""
Semantic Knowledge Graph: Unified Orchestrator for Cognitive Architecture.

The graph package provides the SemanticKnowledgeGraph, which integrates
all components of the cognitive architecture into a unified knowledge
representation.

Components:
    - SemanticKnowledgeGraph: Main orchestrator
    - GraphNode: Node in the knowledge graph
    - GraphEdge: Edge with semantic relation
    - ConnectionType: Types of connections (Lateral, Feedforward, Feedback, Semantic)
    - RelationType: Semantic relation types

Integration Points:
    - Core: Minicolumn, Edge, Layers
    - Algorithms: PageRank, TF-IDF, BM25
    - Semantics: IsA, PartOf, HasA, SimilarTo, etc.
    - CEL: Event sourcing and persistence
    - GoT: Task and decision tracking
    - WovenMind: Dual-process cognition
    - PRISM: Attention and plasticity
    - SparkSLM: Prediction

Example:
    >>> from cortical.graph import SemanticKnowledgeGraph
    >>> skg = SemanticKnowledgeGraph()
    >>> skg.add_document("intro", "Machine learning enables AI.")
    >>> skg.build()
    >>> results = skg.search("artificial intelligence")
"""

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
    # Core
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
