"""
Graph of Thought (GoT) Reasoning Framework for the Cortical Text Processor.

This package provides a comprehensive framework for complex reasoning tasks,
implementing the cognitive architecture defined in docs/complex-reasoning-workflow.md.

The framework supports:
- Multi-step reasoning with explicit dependency tracking
- QAPV cognitive loops (Question, Answer, Produce, Verify)
- Production state management with chunking
- Crisis management and recovery
- Verification protocols (unit → integration → E2E → acceptance)
- Collaboration modes (synchronous, asynchronous, semi-synchronous)
- Graph-based thought representation

Quick Start:
    >>> from cortical.reasoning import ReasoningWorkflow
    >>> workflow = ReasoningWorkflow()
    >>> ctx = workflow.start_session("Implement feature X")
    >>> workflow.begin_question_phase(ctx)
    >>> workflow.record_question(ctx, "What are the requirements?")
    >>> # ... continue through QAPV phases

Core Components:
    - ReasoningWorkflow: Main orchestrator (workflow.py)
    - ThoughtGraph: Graph-based thought representation (thought_graph.py)
    - CognitiveLoop: QAPV loop implementation (cognitive_loop.py)
    - ProductionTask: Artifact creation tracking (production_state.py)
    - CrisisManager: Failure handling (crisis_manager.py)
    - VerificationManager: Testing protocols (verification.py)
    - CollaborationManager: Human-AI coordination (collaboration.py)

Data Structures:
    - ThoughtNode: Individual reasoning units
    - ThoughtEdge: Typed relationships between thoughts
    - ThoughtCluster: Grouped thoughts for hierarchical reasoning

Pattern Factories:
    - create_investigation_graph(): For investigating questions
    - create_decision_graph(): For decision-making
    - create_debug_graph(): For debugging problems
    - create_feature_graph(): For feature planning
    - create_requirements_graph(): For requirements analysis

See Also:
    - docs/complex-reasoning-workflow.md: Full workflow documentation
    - docs/graph-of-thought.md: Graph-based reasoning patterns
"""

# =============================================================================
# CORE DATA STRUCTURES
# =============================================================================

from .graph_of_thought import (
    NodeType,
    EdgeType,
    ThoughtNode,
    ThoughtEdge,
    ThoughtCluster,
)

from .thought_graph import ThoughtGraph

# =============================================================================
# COGNITIVE LOOP SYSTEM
# =============================================================================

from .cognitive_loop import (
    # Enums
    LoopPhase,
    LoopStatus,
    TerminationReason,
    # Core classes
    PhaseContext,
    LoopTransition,
    CognitiveLoop,
    CognitiveLoopManager,
    # Stub classes (for extension)
    LoopStateSerializer,
)

from .nested_loop import (
    NestedLoopExecutor,
    LoopContext,
)

from .loop_validator import (
    LoopValidator,
    ValidationResult,
)

from .qapv_verification import (
    QAPVAnomaly,
    TransitionEvent,
    AnomalyReport,
    QAPVVerifier,
)

# =============================================================================
# PRODUCTION STATE MANAGEMENT
# =============================================================================

from .production_state import (
    # Enums
    ProductionState,
    # Core classes
    ProductionChunk,
    CommentMarker,
    ProductionTask,
    ProductionManager,
    # Stub classes
    ChunkPlanner,
    CommentCleaner,
    ProductionMetrics,
)

# =============================================================================
# CRISIS MANAGEMENT
# =============================================================================

from .crisis_manager import (
    # Enums
    CrisisLevel,
    RecoveryAction,
    # Core classes
    CrisisEvent,
    FailureAttempt,
    RepeatedFailureTracker,
    ScopeCreepDetector,
    BlockedDependency,
    CrisisManager,
    # Stub classes
    RecoveryProcedures,
    CrisisPredictor,
)

# =============================================================================
# VERIFICATION SYSTEM
# =============================================================================

from .verification import (
    # Enums
    VerificationLevel,
    VerificationPhase,
    VerificationStatus,
    # Core classes
    VerificationCheck,
    VerificationFailure,
    VerificationSuite,
    VerificationManager,
    # Factory functions
    create_drafting_checklist,
    create_refining_checklist,
    create_finalizing_checklist,
    # Stub classes
    VerificationRunner,
    FailureAnalyzer,
    RegressionDetector,
)

# =============================================================================
# COLLABORATION SYSTEM
# =============================================================================

from .collaboration import (
    # Enums
    CollaborationMode,
    BlockerType,
    ConflictType,
    AgentStatus,
    # Core classes
    StatusUpdate,
    Blocker,
    DisagreementRecord,
    ParallelWorkBoundary,
    ConflictEvent,
    ActiveWorkHandoff,
    CollaborationManager,
    AgentResult,
    AgentSpawner,
    SequentialSpawner,
    ConflictDetail,
    # Parallel coordination
    ParallelCoordinator,
    QuestionBatcher,
)

from .claude_code_spawner import (
    ClaudeCodeSpawner,
    TaskToolConfig,
    generate_parallel_task_calls,
    # Subprocess spawning
    SubprocessClaudeCodeSpawner,
    SpawnResult,
    SpawnHandle,
    SpawnMetrics,
)

# =============================================================================
# GRAPH PERSISTENCE
# =============================================================================

from .graph_persistence import (
    # Git integration
    GitAutoCommitter,
    # WAL (Write-Ahead Log)
    GraphWAL,
    GraphWALEntry,
    # Recovery
    GraphRecovery,
    GraphRecoveryResult,
    GraphSnapshot,
)

# =============================================================================
# CONTEXT POOL (Multi-Agent Coordination)
# =============================================================================

from .context_pool import (
    ContextFinding,
    ContextPool,
    ConflictResolutionStrategy,
)

# =============================================================================
# AGENT REJECTION PROTOCOL
# =============================================================================

from .rejection_protocol import (
    # Enums
    RejectionReason,
    DecisionType,
    # Core classes
    TaskRejection,
    RejectionValidator,
    RejectionDecision,
    # GoT integration
    log_rejection_to_got,
    analyze_rejection_patterns,
)

# =============================================================================
# PUB/SUB MESSAGING SYSTEM
# =============================================================================

from .pubsub import (
    # Enums
    MessageStatus,
    # Core classes
    Message,
    Subscription,
    PubSubBroker,
    # Helper functions
    create_topic_filter,
    create_payload_filter,
)

# =============================================================================
# METRICS AND OBSERVABILITY
# =============================================================================

from .metrics import (
    PhaseMetrics,
    ReasoningMetrics,
    MetricsContextManager,
    create_loop_metrics_handler,
)

# =============================================================================
# MAIN WORKFLOW ORCHESTRATOR
# =============================================================================

from .workflow import (
    WorkflowContext,
    ReasoningWorkflow,
)

# =============================================================================
# PATTERN FACTORIES
# =============================================================================

from .thought_patterns import (
    create_investigation_graph,
    create_decision_graph,
    create_debug_graph,
    create_feature_graph,
    create_requirements_graph,
    create_analysis_graph,
    create_pattern_graph,
    PATTERN_REGISTRY,
)

# =============================================================================
# PRISM-GoT: Predictive Reasoning through Incremental Synaptic Memory
# =============================================================================

from .prism_got import (
    # Core data structures
    ActivationRecord,
    ActivationTrace,
    SynapticEdge,
    PredictionResult,
    # Learning rules
    PlasticityRules,
    # Graph with synaptic memory
    SynapticMemoryGraph,
    # Incremental reasoning
    IncrementalReasoner,
)

# Convenient alias for PRISM systems
PRISMGraph = SynapticMemoryGraph

# =============================================================================
# PRISM-SLM: Statistical Language Model with Synaptic Learning
# =============================================================================

from .prism_slm import (
    # Core data structures
    SynapticTransition,
    ContextWindow,
    TransitionGraph,
    # Language model
    PRISMLanguageModel,
)

# =============================================================================
# PRISM-PLN: Probabilistic Logic Networks with Synaptic Learning
# =============================================================================

from .prism_pln import (
    # Truth values
    TruthValue,
    SynapticTruthValue,
    # Logical operations
    pln_not,
    pln_and,
    pln_or,
    pln_implication,
    # Inference rules
    deduce,
    induce,
    abduce,
    # Knowledge structures
    Atom,
    ImplicationLink,
    PLNGraph,
    PLNReasoner,
)

# =============================================================================
# THE LOOM: Dual-Process Integration Layer
# =============================================================================

from .loom import (
    # Enums
    ThinkingMode,
    TransitionTrigger,
    # Data structures
    SurpriseSignal,
    LoomConfig,
    ModeTransition,
    LoomAttentionResult,
    # Protocols
    SurpriseDetectorProtocol,
    ModeControllerProtocol,
    LoomObserverProtocol,
    # Abstract base
    LoomInterface,
    # Concrete implementations
    SurpriseDetector,
    ModeController,
    Loom,
    # PRISM integration
    LoomEnhancedAttention,
    # Type aliases
    TransitionCallback,
    SurpriseCallback,
)

# =============================================================================
# PRISM-Attention: Selective Focus Mechanisms
# =============================================================================

from .prism_attention import (
    # Core attention layers
    AttentionLayer,
    MultiHeadAttention,
    SynapticAttention,
    LearnableAttention,
    TemporalAttention,
    UnifiedAttention,
    # PLN integration
    AttentionGuidedReasoner,
    # Visualization
    AttentionVisualizer,
    AttentionHeatmap,
    # Result types
    AttentionResult,
)

# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # === Core Data Structures ===
    'NodeType',
    'EdgeType',
    'ThoughtNode',
    'ThoughtEdge',
    'ThoughtCluster',
    'ThoughtGraph',

    # === Cognitive Loop ===
    'LoopPhase',
    'LoopStatus',
    'TerminationReason',
    'PhaseContext',
    'LoopTransition',
    'CognitiveLoop',
    'CognitiveLoopManager',
    'NestedLoopExecutor',
    'LoopContext',
    'LoopStateSerializer',
    'LoopValidator',
    'ValidationResult',
    'QAPVAnomaly',
    'TransitionEvent',
    'AnomalyReport',
    'QAPVVerifier',

    # === Production State ===
    'ProductionState',
    'ProductionChunk',
    'CommentMarker',
    'ProductionTask',
    'ProductionManager',
    'ChunkPlanner',
    'CommentCleaner',
    'ProductionMetrics',

    # === Crisis Management ===
    'CrisisLevel',
    'RecoveryAction',
    'CrisisEvent',
    'FailureAttempt',
    'RepeatedFailureTracker',
    'ScopeCreepDetector',
    'BlockedDependency',
    'CrisisManager',
    'RecoveryProcedures',
    'CrisisPredictor',

    # === Verification ===
    'VerificationLevel',
    'VerificationPhase',
    'VerificationStatus',
    'VerificationCheck',
    'VerificationFailure',
    'VerificationSuite',
    'VerificationManager',
    'create_drafting_checklist',
    'create_refining_checklist',
    'create_finalizing_checklist',
    'VerificationRunner',
    'FailureAnalyzer',
    'RegressionDetector',

    # === Collaboration ===
    'CollaborationMode',
    'BlockerType',
    'ConflictType',
    'AgentStatus',
    'StatusUpdate',
    'Blocker',
    'DisagreementRecord',
    'ParallelWorkBoundary',
    'ConflictEvent',
    'ActiveWorkHandoff',
    'CollaborationManager',
    'AgentResult',
    'AgentSpawner',
    'SequentialSpawner',
    'ConflictDetail',
    'ParallelCoordinator',
    'QuestionBatcher',
    'ClaudeCodeSpawner',
    'TaskToolConfig',
    'generate_parallel_task_calls',
    'SubprocessClaudeCodeSpawner',
    'SpawnResult',
    'SpawnHandle',
    'SpawnMetrics',

    # === Graph Persistence ===
    'GitAutoCommitter',
    'GraphWAL',
    'GraphWALEntry',
    'GraphRecovery',
    'GraphRecoveryResult',
    'GraphSnapshot',

    # === Context Pool ===
    'ContextFinding',
    'ContextPool',
    'ConflictResolutionStrategy',

    # === Agent Rejection Protocol ===
    'RejectionReason',
    'DecisionType',
    'TaskRejection',
    'RejectionValidator',
    'RejectionDecision',
    'log_rejection_to_got',
    'analyze_rejection_patterns',

    # === Pub/Sub Messaging ===
    'MessageStatus',
    'Message',
    'Subscription',
    'PubSubBroker',
    'create_topic_filter',
    'create_payload_filter',

    # === Metrics and Observability ===
    'PhaseMetrics',
    'ReasoningMetrics',
    'MetricsContextManager',
    'create_loop_metrics_handler',

    # === Main Workflow ===
    'WorkflowContext',
    'ReasoningWorkflow',

    # === Pattern Factories ===
    'create_investigation_graph',
    'create_decision_graph',
    'create_debug_graph',
    'create_feature_graph',
    'create_requirements_graph',
    'create_analysis_graph',
    'create_pattern_graph',
    'PATTERN_REGISTRY',

    # === PRISM-GoT: Synaptic Memory Graph ===
    'ActivationRecord',
    'ActivationTrace',
    'SynapticEdge',
    'PredictionResult',
    'PlasticityRules',
    'SynapticMemoryGraph',
    'PRISMGraph',  # Alias for SynapticMemoryGraph
    'IncrementalReasoner',

    # === PRISM-SLM: Statistical Language Model ===
    'SynapticTransition',
    'ContextWindow',
    'TransitionGraph',
    'PRISMLanguageModel',

    # === PRISM-PLN: Probabilistic Logic Networks ===
    'TruthValue',
    'SynapticTruthValue',
    'pln_not',
    'pln_and',
    'pln_or',
    'pln_implication',
    'deduce',
    'induce',
    'abduce',
    'Atom',
    'ImplicationLink',
    'PLNGraph',
    'PLNReasoner',

    # === PRISM-Attention: Selective Focus ===
    'AttentionLayer',
    'MultiHeadAttention',
    'SynapticAttention',
    'LearnableAttention',
    'TemporalAttention',
    'UnifiedAttention',
    'AttentionGuidedReasoner',
    'AttentionVisualizer',
    'AttentionHeatmap',
    'AttentionResult',

    # === The Loom: Dual-Process Integration ===
    'ThinkingMode',
    'TransitionTrigger',
    'SurpriseSignal',
    'LoomConfig',
    'ModeTransition',
    'LoomAttentionResult',
    'SurpriseDetectorProtocol',
    'ModeControllerProtocol',
    'LoomObserverProtocol',
    'LoomInterface',
    'SurpriseDetector',
    'ModeController',
    'Loom',
    'LoomEnhancedAttention',
    'TransitionCallback',
    'SurpriseCallback',
]
