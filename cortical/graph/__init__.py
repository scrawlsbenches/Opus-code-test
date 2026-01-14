"""
Graph Package: Unified Graph Architecture for Cortical.

This package provides both the SemanticKnowledgeGraph (domain-specific)
and a composable BaseGraph architecture for building custom graphs.

BaseGraph Architecture:
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

TrainableGraph (Message-Passing Graph Neural Network):
    - TrainableGraph: Graph with learnable parameters and gradient descent
    - TrainableNode: Node with learnable embedding vectors
    - TrainableEdge: Edge with learnable weights
    - Optimizers: SGD, Adam, AdaGrad, RMSprop
    - Loss functions: MSE, MAE, CrossEntropy, Huber, Contrastive
    - LR schedulers: StepLR, ExponentialLR, CosineAnnealing, ReduceOnPlateau

    Example:
        >>> from cortical.graph import TrainableGraph, Adam, MSELoss
        >>> import numpy as np
        >>> graph = TrainableGraph(embedding_dim=16)
        >>> graph.add_node("A")
        >>> graph.add_node("B")
        >>> graph.add_edge("A", "B")
        >>> optimizer = Adam(graph.parameters(), lr=0.01)
        >>> outputs = graph.forward(num_layers=2)
        >>> loss = MSELoss()(outputs["B"], target)
        >>> graph.backward({"B": MSELoss().gradient(outputs["B"], target)})
        >>> optimizer.step()

AttentionGraph (Self-Attention Graph Neural Network):
    - AttentionGraph: Graph using self-attention instead of message passing
    - AttentionNode: Node with query/key/value projections
    - AttentionEdge: Edge defining attention visibility (causal mask via structure)
    - ProcessingLayer: Abstract base for composable layers
    - TrainableGraphProtocol: Contract for experiment kernel compatibility

    Key difference: While TrainableGraph aggregates neighbor messages with fixed
    weights, AttentionGraph computes dynamic attention weights based on content.
    This preserves sequential order better for language modeling tasks.

    Example:
        >>> from cortical.graph import AttentionGraph, create_causal_attention_graph
        >>> graph = create_causal_attention_graph(seq_len=16, embedding_dim=64)
        >>> outputs = graph.forward(num_layers=2)
        >>> print(graph.visualize_attention("pos_5"))  # See what pos_5 attends to

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

from .trainable import (
    # Core types
    TrainableNode,
    TrainableEdge,
    TrainableGraph,
    Parameter,
    # Activation and aggregation
    Activation,
    Aggregation,
    apply_activation,
    activation_derivative,
    aggregate_messages,
    # Loss functions
    LossFunction,
    MSELoss,
    MAELoss,
    CrossEntropyLoss,
    BinaryCrossEntropyLoss,
    HuberLoss,
    ContrastiveLoss,
    # Optimizers
    Optimizer,
    SGD,
    Adam,
    AdaGrad,
    RMSprop,
    # Learning rate schedulers
    LRScheduler,
    StepLR,
    ExponentialLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    # Training utilities
    EarlyStopping,
    TrainingHistory,
    train_step,
    fit,
)

from .attention import (
    # Core types
    AttentionNode,
    AttentionEdge,
    AttentionGraph,
    # Processing layer abstraction
    ProcessingLayer,
    AttentionLayer,
    # Protocol for experiment kernel compatibility
    TrainableGraphProtocol,
    # Attention functions
    scaled_dot_product_attention,
    attention_backward,
    # Convenience functions
    create_causal_attention_graph,
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
    # Trainable Graph (NEW)
    'TrainableNode',
    'TrainableEdge',
    'TrainableGraph',
    'Parameter',
    'Activation',
    'Aggregation',
    'apply_activation',
    'activation_derivative',
    'aggregate_messages',
    'LossFunction',
    'MSELoss',
    'MAELoss',
    'CrossEntropyLoss',
    'BinaryCrossEntropyLoss',
    'HuberLoss',
    'ContrastiveLoss',
    'Optimizer',
    'SGD',
    'Adam',
    'AdaGrad',
    'RMSprop',
    'LRScheduler',
    'StepLR',
    'ExponentialLR',
    'CosineAnnealingLR',
    'ReduceLROnPlateau',
    'EarlyStopping',
    'TrainingHistory',
    'train_step',
    'fit',
    # Attention Graph (NEW)
    'AttentionNode',
    'AttentionEdge',
    'AttentionGraph',
    'ProcessingLayer',
    'AttentionLayer',
    'TrainableGraphProtocol',
    'scaled_dot_product_attention',
    'attention_backward',
    'create_causal_attention_graph',
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
