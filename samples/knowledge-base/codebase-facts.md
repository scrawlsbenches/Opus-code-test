# Codebase Facts

## File Locations

### Core Library (cortical/)

#### Main API Package
- Main processor class: `cortical/processor/__init__.py` (CorticalTextProcessor composed from mixins)
- Initialization and staleness tracking: `cortical/processor/core.py`
- Document processing: `cortical/processor/documents.py`
- Computation methods: `cortical/processor/compute.py`
- Query API: `cortical/processor/query_api.py`
- Introspection methods: `cortical/processor/introspection.py`
- Persistence API: `cortical/processor/persistence_api.py`
- SparkSLM integration: `cortical/processor/spark_api.py`

#### Graph Algorithms Package
- PageRank: `cortical/analysis/pagerank.py`
- TF-IDF: `cortical/analysis/tfidf.py`
- Clustering (Louvain): `cortical/analysis/clustering.py`
- Connection building: `cortical/analysis/connections.py`
- Activation propagation: `cortical/analysis/activation.py`
- Parallel computation: `cortical/analysis/parallel.py`
- Quality metrics: `cortical/analysis/quality.py`
- Analysis utilities: `cortical/analysis/utils.py`

#### Query and Search Package
- Query expansion: `cortical/query/expansion.py`
- Document search: `cortical/query/search.py`
- Passage retrieval: `cortical/query/passages.py`
- Text chunking: `cortical/query/chunking.py`
- Intent parsing: `cortical/query/intent.py`
- Definition search: `cortical/query/definitions.py`
- Multi-stage ranking: `cortical/query/ranking.py`
- Analogy completion: `cortical/query/analogy.py`
- Query utilities: `cortical/query/utils.py`

#### Reasoning Framework (PRISM + Woven Mind + GoT)
- WovenMind facade: `cortical/reasoning/woven_mind.py`
- The Loom (mode switching): `cortical/reasoning/loom.py`
- Hive connector (System 1): `cortical/reasoning/loom_hive.py`
- Cortex connector (System 2): `cortical/reasoning/loom_cortex.py`
- Consolidation engine: `cortical/reasoning/consolidation.py`
- PRISM language model: `cortical/reasoning/prism_slm.py`
- PRISM-GoT integration: `cortical/reasoning/prism_got.py`
- PRISM-PLN integration: `cortical/reasoning/prism_pln.py`
- PRISM attention mechanisms: `cortical/reasoning/prism_attention.py`
- Abstraction formation: `cortical/reasoning/abstraction.py`
- PLN abstractions: `cortical/reasoning/abstraction_pln.py`
- Attention router: `cortical/reasoning/attention_router.py`
- Homeostatic regulation: `cortical/reasoning/homeostasis.py`
- Goal stack: `cortical/reasoning/goal_stack.py`
- Cognitive loop (QAPV): `cortical/reasoning/cognitive_loop.py`
- Thought graph: `cortical/reasoning/thought_graph.py`
- Graph of thought structures: `cortical/reasoning/graph_of_thought.py`
- Graph persistence (WAL): `cortical/reasoning/graph_persistence.py`
- Workflow orchestration: `cortical/reasoning/workflow.py`
- Verification manager: `cortical/reasoning/verification.py`
- QAPV verification: `cortical/reasoning/qapv_verification.py`
- Crisis manager: `cortical/reasoning/crisis_manager.py`
- Parallel coordination: `cortical/reasoning/collaboration.py`
- Production state tracking: `cortical/reasoning/production_state.py`
- Loop validator: `cortical/reasoning/loop_validator.py`
- Nested loops: `cortical/reasoning/nested_loop.py`
- Thought patterns: `cortical/reasoning/thought_patterns.py`
- Rejection protocol: `cortical/reasoning/rejection_protocol.py`
- Pub/sub messaging: `cortical/reasoning/pubsub.py`
- Context pooling: `cortical/reasoning/context_pool.py`
- Reasoning metrics: `cortical/reasoning/metrics.py`
- Claude Code spawner: `cortical/reasoning/claude_code_spawner.py`

#### Graph of Thought (Task Management)
- GoT manager (main API): `cortical/got/api.py`
- Transaction manager: `cortical/got/tx_manager.py`
- Transaction primitives: `cortical/got/transaction.py`
- Versioned storage: `cortical/got/versioned_store.py`
- Write-ahead log: `cortical/got/wal.py`
- Entity types: `cortical/got/types.py`
- Entity schemas: `cortical/got/entity_schemas.py`
- Schema system: `cortical/got/schema.py`
- Recovery manager: `cortical/got/recovery.py`
- Sync manager: `cortical/got/sync.py`
- Conflict resolution: `cortical/got/conflict.py`
- GoT configuration: `cortical/got/config.py`
- GoT protocol: `cortical/got/protocol.py`
- Query builder (fluent API): `cortical/got/query_builder.py`
- Query indexing: `cortical/got/indexer.py`
- Graph walker: `cortical/got/graph_walker.py`
- Path finder: `cortical/got/path_finder.py`
- Pattern matcher: `cortical/got/pattern_matcher.py`
- Orphan detection: `cortical/got/orphan.py`
- CLAUDE.md integration: `cortical/got/claudemd.py`
- CLI commands: `cortical/got/cli/` (analyze.py, etc.)
- Error types: `cortical/got/errors.py`

#### SparkSLM (Lightweight Predictions)
- N-gram model: `cortical/spark/ngram.py`
- Alignment index: `cortical/spark/alignment.py`
- Predictor facade: `cortical/spark/predictor.py`
- Anomaly detection: `cortical/spark/anomaly.py`
- Code tokenizer: `cortical/spark/tokenizer.py`
- Diff tokenizer: `cortical/spark/diff_tokenizer.py`
- AST indexing: `cortical/spark/ast_index.py`
- Code intelligence: `cortical/spark/intelligence.py`
- Intent parser: `cortical/spark/intent_parser.py`
- Co-change model: `cortical/spark/co_change.py`
- Sample suggester: `cortical/spark/suggester.py`
- Knowledge transfer: `cortical/spark/transfer.py`
- Quality evaluation: `cortical/spark/quality.py`

#### Utility Modules
- ID generation: `cortical/utils/id_generation.py`
- Checksums: `cortical/utils/checksums.py`
- Persistence utilities: `cortical/utils/persistence.py`
- Text utilities: `cortical/utils/text.py`
- Locking utilities: `cortical/utils/locking.py`

#### Core Modules
- Base WAL entries: `cortical/wal.py`
- Semantic relations: `cortical/semantics.py`
- Tokenization: `cortical/tokenizer.py`
- Minicolumn structure: `cortical/minicolumn.py`
- Layer management: `cortical/layers.py`
- Configuration: `cortical/config.py`
- Persistence: `cortical/persistence.py`
- State storage: `cortical/state_storage.py`
- ML storage: `cortical/ml_storage.py`
- Chunk indexing: `cortical/chunk_index.py`
- Fingerprinting: `cortical/fingerprint.py`
- Embeddings: `cortical/embeddings.py`
- Gap detection: `cortical/gaps.py`
- Observability: `cortical/observability.py`
- Code concepts: `cortical/code_concepts.py`
- Pattern extraction: `cortical/patterns.py`
- Diff analysis: `cortical/diff.py`
- Fluent API: `cortical/fluent.py`
- Validation: `cortical/validation.py`
- Progress tracking: `cortical/progress.py`
- Results formatting: `cortical/results.py`
- Type definitions: `cortical/types.py`
- Constants: `cortical/constants.py`
- CLI wrapper: `cortical/cli_wrapper.py`
- Async API: `cortical/async_api.py`

#### ML Experiments
- Experiment framework: `cortical/ml_experiments/experiment.py`
- Dataset utilities: `cortical/ml_experiments/dataset.py`
- Metrics: `cortical/ml_experiments/metrics.py`
- File prediction adapter: `cortical/ml_experiments/file_prediction_adapter.py`
- Utilities: `cortical/ml_experiments/utils.py`

#### Projects
- Project CLI: `cortical/projects/cli/__init__.py`

### Scripts
- GoT utilities: `scripts/got_utils.py`
- Task utilities: `scripts/task_utils.py`
- Index codebase: `scripts/index_codebase.py`
- Search codebase: `scripts/search_codebase.py`
- ML data collector: `scripts/ml_data_collector.py`
- File prediction: `scripts/ml_file_prediction.py`
- Session memory generator: `scripts/session_memory_generator.py`
- Generate AI metadata: `scripts/generate_ai_metadata.py`
- Profile analysis: `scripts/profile_full_analysis.py`
- Reasoning demo: `scripts/reasoning_demo.py`
- Woven Mind demo: `examples/woven_mind_demo.py`
- Graph persistence demo: `examples/graph_persistence_demo.py`
- Observability demo: `examples/observability_demo.py`

### Tests
- Smoke tests: `tests/smoke/`
- Unit tests: `tests/unit/`
- Integration tests: `tests/integration/`
- Performance tests: `tests/performance/`
- Regression tests: `tests/regression/`
- Behavioral tests: `tests/behavioral/`
- Test fixtures: `tests/fixtures/`

## Module Purposes

### Core Processor
- `processor/__init__.py`: Main API composed from mixins for document processing, search, and analysis
- `processor/core.py`: Initialization, layer management, and staleness tracking for computations
- `processor/documents.py`: Add, remove, and manage documents with metadata
- `processor/compute.py`: PageRank, TF-IDF, clustering, and semantic relation computation
- `processor/query_api.py`: Search, query expansion, and document retrieval methods
- `processor/introspection.py`: Fingerprinting, gap detection, and state inspection
- `processor/persistence_api.py`: Save and load processor state in multiple formats
- `processor/spark_api.py`: First-blitz priming using SparkSLM for faster query responses

### Graph Algorithms
- `analysis/pagerank.py`: Compute term importance using PageRank algorithm with damping
- `analysis/tfidf.py`: TF-IDF scoring for term relevance and document weighting
- `analysis/clustering.py`: Louvain community detection for concept formation
- `analysis/connections.py`: Build lateral connections between minicolumns based on co-occurrence
- `analysis/activation.py`: Propagate activation through the network graph
- `analysis/parallel.py`: Parallel computation utilities for multi-core processing
- `analysis/quality.py`: Quality metrics and coherence scoring
- `analysis/utils.py`: Shared analysis utilities and helpers

### Query and Search
- `query/expansion.py`: TF-IDF weighted lateral expansion for query enrichment
- `query/search.py`: Document search with BM25 and TF-IDF scoring
- `query/passages.py`: Passage retrieval for RAG systems with chunking
- `query/chunking.py`: Text chunking with overlap for context preservation
- `query/intent.py`: Intent-based query parsing (what, where, how questions)
- `query/definitions.py`: Definition search for code and concepts
- `query/ranking.py`: Multi-stage ranking combining multiple signals
- `query/analogy.py`: Analogy completion using graph structure
- `query/utils.py`: Shared query utilities including TF-IDF scoring

### Reasoning Framework
- `reasoning/woven_mind.py`: Unified facade for dual-process cognitive architecture
- `reasoning/loom.py`: Mode switching between FAST (System 1) and SLOW (System 2) based on surprise
- `reasoning/loom_hive.py`: System 1 connector for fast pattern matching
- `reasoning/loom_cortex.py`: System 2 connector for deliberate reasoning
- `reasoning/consolidation.py`: Sleep-like memory consolidation from Hive to Cortex
- `reasoning/prism_slm.py`: Statistical language model with synaptic learning (Hebbian)
- `reasoning/prism_got.py`: Integration between PRISM SLM and Graph of Thought
- `reasoning/prism_pln.py`: Probabilistic Logic Network integration with PRISM
- `reasoning/prism_attention.py`: Attention mechanisms for PRISM
- `reasoning/abstraction.py`: Abstraction formation through pattern clustering
- `reasoning/abstraction_pln.py`: PLN-based abstraction formation
- `reasoning/attention_router.py`: Route attention based on thinking mode
- `reasoning/homeostasis.py`: Homeostatic regulation for stable learning
- `reasoning/goal_stack.py`: Goal management and prioritization
- `reasoning/cognitive_loop.py`: QAPV cycle (Question, Answer, Produce, Verify)
- `reasoning/thought_graph.py`: Graph-based thought representation with typed edges
- `reasoning/graph_of_thought.py`: Core data structures for thought nodes and edges
- `reasoning/graph_persistence.py`: WAL, snapshots, git integration for graph durability
- `reasoning/workflow.py`: Reasoning workflow orchestration
- `reasoning/verification.py`: Multi-level verification for thought quality
- `reasoning/crisis_manager.py`: Failure detection and recovery protocols
- `reasoning/collaboration.py`: Parallel agent coordination with boundary isolation
- `reasoning/production_state.py`: Track artifact creation during reasoning

### Graph of Thought
- `got/api.py`: Main GoT manager for tasks, decisions, edges, sprints, epics, handoffs
- `got/tx_manager.py`: ACID transaction manager with snapshot isolation
- `got/transaction.py`: Transaction primitives and state management
- `got/versioned_store.py`: File-based storage with checksums and versioning
- `got/wal.py`: Write-ahead log for crash recovery
- `got/types.py`: Entity types (Task, Decision, Edge, Sprint, Epic, Handoff)
- `got/entity_schemas.py`: Schema definitions for all entity types
- `got/schema.py`: Schema registry and validation system
- `got/recovery.py`: Recovery manager for corrupted or missing data
- `got/sync.py`: Sync manager for distributed GoT state
- `got/conflict.py`: Conflict resolution strategies
- `got/query_builder.py`: Fluent SQL-like query API for GoT
- `got/indexer.py`: Query indexing for performance
- `got/graph_walker.py`: Graph traversal with visitor pattern (BFS/DFS)
- `got/path_finder.py`: Shortest path and reachability algorithms
- `got/pattern_matcher.py`: Subgraph pattern matching
- `got/orphan.py`: Detect and suggest connections for orphaned entities
- `got/claudemd.py`: CLAUDE.md integration for project context

### SparkSLM
- `spark/ngram.py`: Bigram and trigram statistical language model
- `spark/alignment.py`: User-defined patterns and preferences index
- `spark/predictor.py`: Unified facade for first-blitz predictions
- `spark/anomaly.py`: Statistical and pattern-based anomaly detection
- `spark/tokenizer.py`: Code-aware tokenizer preserving operators
- `spark/diff_tokenizer.py`: Git diff tokenizer for code evolution
- `spark/ast_index.py`: AST-based structural code indexing
- `spark/intelligence.py`: Hybrid AST + N-gram code intelligence
- `spark/intent_parser.py`: Parse commit message intent for evolution model
- `spark/co_change.py`: Learn file co-change patterns from git history
- `spark/suggester.py`: Sample-based suggestion engine
- `spark/transfer.py`: Knowledge transfer utilities for portable models
- `spark/quality.py`: Quality evaluation for predictions and search

### Core Data Structures
- `minicolumn.py`: Minicolumn with typed edges (lateral, feedforward, feedback, semantic)
- `layers.py`: HierarchicalLayer with O(1) ID lookups via _id_index
- `tokenizer.py`: Tokenization with stemming, stop word removal, identifier splitting
- `semantics.py`: Relation extraction using patterns, retrofitting, inheritance
- `config.py`: CorticalConfig with validation for all parameters
- `persistence.py`: Save/load with state preservation and format migration
- `state_storage.py`: JSON-based state storage system
- `chunk_index.py`: Git-friendly chunk-based storage for collaboration
- `fingerprint.py`: Semantic fingerprinting for code similarity

### Utilities
- `utils/id_generation.py`: Canonical ID generation (task IDs, plan IDs, etc.)
- `utils/checksums.py`: Unified checksum computation and verification
- `utils/persistence.py`: Atomic file save operations
- `utils/text.py`: Text processing utilities (slugify, etc.)
- `utils/locking.py`: Process-safe file locking
- `wal.py`: Base WAL entry classes for transactional logging

### Other Core Modules
- `embeddings.py`: Graph embeddings (adjacency, spectral, random walk)
- `gaps.py`: Knowledge gap detection and anomaly analysis
- `observability.py`: Timing, metrics collection, and trace context
- `code_concepts.py`: Programming concept synonyms for code search
- `patterns.py`: Pattern extraction utilities
- `diff.py`: Diff analysis between processor states
- `fluent.py`: Fluent query API
- `validation.py`: Parameter validation utilities
- `progress.py`: Progress tracking for long operations
- `results.py`: Results formatting and presentation
- `cli_wrapper.py`: CLI wrapper for processor operations
- `async_api.py`: Async API for concurrent operations

## Key Classes

### Main API
- **CorticalTextProcessor**: Main API for text processing, composed from mixins (CoreMixin, DocumentsMixin, ComputeMixin, QueryMixin, IntrospectionMixin, PersistenceMixin, SparkMixin)

### Data Structures
- **Minicolumn**: Core unit representing a term/bigram/concept with lateral connections, typed connections (feedforward, feedback, semantic), document IDs, PageRank, TF-IDF
- **Edge**: Typed connection between minicolumns with relation type, weight, confidence, source
- **HierarchicalLayer**: Container for minicolumns with O(1) ID lookups via _id_index

### Graph Algorithms
- **PageRank**: Compute term importance using iterative power method
- **TF-IDF**: Compute term relevance using term frequency and inverse document frequency
- **Louvain**: Community detection for concept clustering

### Reasoning Framework
- **WovenMind**: Unified facade for dual-process cognitive architecture
- **Loom**: Mode switching between FAST and SLOW based on surprise detection
- **LoomHiveConnector**: System 1 connector for fast automatic processing
- **LoomCortexConnector**: System 2 connector for slow deliberate reasoning
- **ConsolidationEngine**: Sleep-like memory consolidation from Hive to Cortex
- **PRISMLanguageModel**: Statistical language model with synaptic learning
- **CognitiveLoop**: QAPV cycle implementation (Question, Answer, Produce, Verify)
- **ThoughtGraph**: Graph-based thought representation with typed edges
- **GraphWAL**: Write-ahead log for durable graph operations
- **GraphSnapshot**: Point-in-time compressed snapshots
- **GitAutoCommitter**: Automatic git versioning with protected branch safety
- **GraphRecovery**: 4-level cascade recovery (WAL → Snapshot → Git → Chunks)
- **ParallelCoordinator**: Spawn parallel sub-agents with boundary isolation
- **VerificationManager**: Multi-level testing protocols
- **CrisisManager**: Failure detection and recovery

### Graph of Thought
- **GoTManager**: High-level API for tasks, decisions, edges, sprints, epics, handoffs
- **TransactionManager**: ACID transaction manager with snapshot isolation
- **Transaction**: Transaction object with read/write operations
- **Task**: Work item with status, priority, dependencies
- **Decision**: Logged decision with rationale and consequences
- **Edge**: Typed relationship between entities (DEPENDS_ON, BLOCKS, CONTAINS, etc.)
- **Sprint**: Time-boxed work period with goals and status
- **Epic**: Large initiative spanning multiple sprints
- **Handoff**: Agent-to-agent work transfer with context and result
- **Query**: Fluent SQL-like query builder for GoT (tasks(), where(), order_by(), execute())
- **GraphWalker**: Graph traversal with visitor pattern (BFS/DFS)
- **PathFinder**: Shortest path and reachability algorithms
- **PatternMatcher**: Subgraph pattern matching
- **OrphanDetector**: Detect and suggest connections for orphaned entities

### SparkSLM
- **SparkPredictor**: Unified facade for first-blitz predictions
- **NGramModel**: Bigram/trigram statistical language model
- **AlignmentIndex**: User definitions and patterns from markdown
- **AnomalyDetector**: Prompt injection detection using statistical analysis
- **CodeTokenizer**: Code-aware tokenizer preserving punctuation
- **DiffTokenizer**: Git diff tokenizer for code evolution training
- **ASTIndex**: AST-based code indexing
- **SparkCodeIntelligence**: Hybrid AST + N-gram code intelligence
- **IntentParser**: Parse commit message intent
- **CoChangeModel**: Learn file co-change patterns from git history

### Configuration and Storage
- **CorticalConfig**: Configuration dataclass with validation for all parameters
- **StateLoader**: JSON-based state loader with format detection
- **StateSaver**: JSON-based state saver with atomic writes
- **ChunkIndex**: Git-friendly chunk-based storage for collaboration

## Architecture Layers

The Cortical Text Processor uses a 4-layer hierarchical architecture inspired by visual cortex organization:

### Layer 0: TOKENS
- **Purpose**: Individual words and terms
- **Analogy**: V1 visual cortex (detects edges)
- **ID format**: `L0_{term}` (e.g., `L0_neural`)
- **Content**: Stemmed tokens from documents
- **Connections**: Lateral connections to related tokens based on co-occurrence

### Layer 1: BIGRAMS
- **Purpose**: Word pairs representing local context
- **Analogy**: V2 visual cortex (detects patterns)
- **ID format**: `L1_{term1} {term2}` (e.g., `L1_neural networks`)
- **Content**: Consecutive word pairs (uses SPACE separator, not underscore)
- **Connections**: Lateral connections to related bigrams, feedforward to constituent tokens

### Layer 2: CONCEPTS
- **Purpose**: Semantic clusters discovered through community detection
- **Analogy**: V4 visual cortex (detects shapes/objects)
- **ID format**: `L2_concept_{N}` (e.g., `L2_concept_0`)
- **Content**: Groups of related terms/bigrams identified by Louvain clustering
- **Connections**: Feedforward to member terms/bigrams, lateral to related concepts

### Layer 3: DOCUMENTS
- **Purpose**: Complete documents as the highest-level representation
- **Analogy**: IT visual cortex (recognizes complete objects)
- **ID format**: `L3_{doc_id}` (e.g., `L3_doc1`)
- **Content**: Full documents with metadata
- **Connections**: Feedback to all terms/bigrams contained in the document

## Important Implementation Details

### Staleness Tracking
The processor tracks which computations are up-to-date vs. stale:
- **COMP_TFIDF**: TF-IDF scores
- **COMP_PAGERANK**: PageRank importance
- **COMP_ACTIVATION**: Activation propagation
- **COMP_DOC_CONNECTIONS**: Document-to-document links
- **COMP_BIGRAM_CONNECTIONS**: Bigram lateral connections
- **COMP_CONCEPTS**: Concept clusters
- **COMP_EMBEDDINGS**: Graph embeddings
- **COMP_SEMANTICS**: Semantic relations

Adding documents marks all computations stale. Each `compute_*()` method marks its computation fresh.

### Bigram Separators
Bigrams use SPACE separators throughout: `"neural networks"`, NOT `"neural_networks"`. This is critical for correct lookups.

### TF-IDF Variants
- **Global TF-IDF**: `col.tfidf` - uses total corpus occurrence count
- **Per-document TF-IDF**: `col.tfidf_per_doc[doc_id]` - true per-document TF-IDF

### O(1) Lookups
Always use `layer.get_by_id(col_id)` for O(1) access via `_id_index`, not iteration over `layer.minicolumns`.

### ID Generation
Use canonical ID generation from `cortical/utils/id_generation.py`:
- Format: `{PREFIX}-YYYYMMDD-HHMMSS-{8-char-hex}`
- Example: `T-20251222-093045-a1b2c3d4`

### Persistence Formats
- **JSON**: Recommended (secure, git-friendly, human-readable)
- **Pickle**: Deprecated (security risk, binary format)

## Reasoning Framework Architecture

### Dual-Process Cognition (Woven Mind)
The Woven Mind implements Kahneman's System 1/System 2 theory:

**System 1 (FAST mode - Hive)**:
- Automatic pattern matching
- Hebbian learning (connections strengthen with use)
- Spreading activation
- Low computational cost
- Used for familiar patterns

**System 2 (SLOW mode - Cortex)**:
- Deliberate reasoning
- Abstraction formation
- Planning and goal pursuit
- Higher computational cost
- Used for novel/surprising inputs

**The Loom**:
- Routes between FAST and SLOW based on surprise detection
- Monitors prediction errors and novelty
- Switches modes when surprise threshold is exceeded
- Manages mode transitions and context

**Consolidation**:
- Transfers patterns from Hive (System 1) to Cortex (System 2)
- Sleep-like memory consolidation
- Strengthens frequently-used patterns
- Decay factor for forgetting

### Graph of Thought (GoT)
Transactional task and decision tracking with:
- **ACID transactions**: Snapshot isolation, write-ahead logging
- **Entity types**: Tasks, Decisions, Edges, Sprints, Epics, Handoffs
- **Typed edges**: DEPENDS_ON, BLOCKS, CONTAINS, PART_OF, etc.
- **Query API**: Fluent SQL-like syntax for graph queries
- **Recovery**: Multi-level recovery from WAL, snapshots, or git history
- **Conflict resolution**: Automatic and manual conflict resolution strategies

### PRISM Integration
PRISM (statistical language model) integrates with:
- **GoT**: Use graph structure to guide predictions
- **PLN**: Probabilistic Logic Networks for reasoning
- **Attention**: Mode-based attention routing
- **Abstractions**: Pattern clustering and generalization

## Testing Architecture

### Test Categories
- **Smoke**: Quick sanity checks (<30s)
- **Unit**: Fast isolated tests with mocks
- **Integration**: Component interaction tests
- **Performance**: Timing regression tests
- **Regression**: Bug-specific regression tests
- **Behavioral**: User workflow quality tests

### Test Execution
- Use `pytest` for all tests (handles both pytest and unittest styles)
- Never run both pytest and unittest on same files (doubles CI time)
- Parallel execution: `pytest tests/unit/ -j 4` (3x faster)
- Optional dependencies marked with `@pytest.mark.optional`

### Coverage Requirements
- **Target**: 95%+ for core logic, 90%+ for error handling
- **Current baseline**: 98% on fault-tolerant systems
- **TDD required**: Write tests FIRST, then implement

## Command-Line Tools

### GoT Management
- Create task: `python scripts/got_utils.py task create "Title" --priority high`
- Start task: `python scripts/got_utils.py task start T-XXX`
- Complete task: `python scripts/got_utils.py task complete T-XXX`
- Query tasks: `python scripts/got_utils.py query "what blocks TASK_ID"`
- Dashboard: `python scripts/got_utils.py dashboard`

### Codebase Search
- Index: `python scripts/index_codebase.py --incremental`
- Search: `python scripts/search_codebase.py "query text"`
- Interactive: `python scripts/search_codebase.py --interactive`

### ML Data Collection
- Stats: `python scripts/ml_data_collector.py stats`
- Train file prediction: `python scripts/ml_file_prediction.py train`
- Predict files: `python scripts/ml_file_prediction.py predict "Add auth feature"`

### Demonstrations
- Reasoning demo: `python scripts/reasoning_demo.py --quick`
- Woven Mind demo: `python examples/woven_mind_demo.py --section all`
- Observability demo: `python examples/observability_demo.py`

## Common Patterns

### Adding Documents
```python
processor = CorticalTextProcessor()
processor.process_document("doc_id", "Text content here.")
processor.compute_all()
```

### Searching
```python
# Standard search
results = processor.find_documents_for_query("neural networks", top_n=5)

# Fast search (2-3x faster)
results = processor.fast_find_documents("neural networks")

# Graph-boosted search (hybrid)
results = processor.graph_boosted_search("neural networks", pagerank_weight=0.3)
```

### Query Expansion
```python
# Basic expansion
expanded = processor.expand_query("neural networks", max_expansions=10)

# Code-aware expansion
expanded = processor.expand_query_for_code("fetch data")
```

### Persistence
```python
# Save (JSON recommended)
processor.save("corpus_state")

# Load (auto-detects format)
processor = CorticalTextProcessor.load("corpus_state")
```

### GoT Usage
```python
from cortical.got import GoTManager, Query

manager = GoTManager()
task_id = manager.create_task("Title", priority="high")
manager.start_task(task_id)

# Query with fluent API
pending = Query(manager).tasks().where(status="pending").execute()
```

### Woven Mind Usage
```python
from cortical.reasoning.woven_mind import WovenMind

mind = WovenMind()
mind.train("neural networks process data")
result = mind.process(["neural", "networks"])
print(f"Mode: {result.mode.name}, Source: {result.source}")
```
