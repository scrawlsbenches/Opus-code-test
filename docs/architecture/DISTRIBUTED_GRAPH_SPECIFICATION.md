# Distributed Graph Specification

**Cortical Distributed Graph (CDG)**

*A unified distributed graph storage system built from first principles*

---

## Executive Summary

The Cortical Distributed Graph (CDG) is a unified graph storage and query system designed to serve all graph-related needs across the codebase. It provides:

- **Sub-100ms p95 query latency** for service provider response time requirements
- **ACID transactions** across distributed partitions
- **Unified API** that all existing graphs (GoT, ThoughtGraph, Knowledge Graph) can depend on
- **Zero external dependencies** - built entirely from first principles per our Sovereignty Principle

---

## Design Goals

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│                    DISTRIBUTED GRAPH DESIGN GOALS                        │
│                                                                          │
│  1. PERFORMANCE: p50 < 20ms, p95 < 100ms, p99 < 200ms                   │
│  2. CONSISTENCY: ACID transactions with optimistic concurrency          │
│  3. SCALABILITY: Linear scaling with partition count                    │
│  4. SOVEREIGNTY: No external dependencies - we own every line           │
│  5. UNIFICATION: Single API for all graph use cases                     │
│  6. DURABILITY: WAL-based recovery with configurable fsync modes        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Architecture Overview

```
                              ┌─────────────────────────────────────┐
                              │      CDG Query Coordinator          │
                              │                                     │
                              │  • Query parsing & planning         │
                              │  • Partition routing                │
                              │  • Result aggregation               │
                              │  • Transaction coordination         │
                              └─────────────┬───────────────────────┘
                                            │
              ┌─────────────────────────────┼─────────────────────────────┐
              │                             │                             │
              ▼                             ▼                             ▼
    ┌─────────────────┐          ┌─────────────────┐          ┌─────────────────┐
    │   Partition 0   │          │   Partition 1   │          │   Partition N   │
    │                 │          │                 │          │                 │
    │ ┌─────────────┐ │          │ ┌─────────────┐ │          │ ┌─────────────┐ │
    │ │   Shard A   │ │          │ │   Shard A   │ │          │ │   Shard A   │ │
    │ │  (primary)  │ │          │ │  (primary)  │ │          │ │  (primary)  │ │
    │ └─────────────┘ │          │ └─────────────┘ │          │ └─────────────┘ │
    │ ┌─────────────┐ │          │ ┌─────────────┐ │          │ ┌─────────────┐ │
    │ │   Shard B   │ │          │ │   Shard B   │ │          │ │   Shard B   │ │
    │ │  (replica)  │ │          │ │  (replica)  │ │          │ │  (replica)  │ │
    │ └─────────────┘ │          │ └─────────────┘ │          │ └─────────────┘ │
    │                 │          │                 │          │                 │
    │ Local WAL       │          │ Local WAL       │          │ Local WAL       │
    │ Local Index     │          │ Local Index     │          │ Local Index     │
    └─────────────────┘          └─────────────────┘          └─────────────────┘
```

---

## Core Components

### 1. Universal Node Model

All graph needs unified under a single node model:

```python
@dataclass
class DistributedNode:
    """
    Universal node model supporting all graph use cases.

    Supports:
    - GoT entities (Task, Decision, Sprint, Epic, etc.)
    - ThoughtGraph nodes (Concept, Question, Hypothesis, etc.)
    - Knowledge Graph nodes (Entity, Relation, Fact)
    - Custom domain-specific nodes
    """

    # Identity
    id: str                          # Globally unique ID (UUID v7 for time-ordering)
    partition_key: str               # Determines which partition owns this node

    # Type system
    namespace: str                   # e.g., "got", "thought", "knowledge", "custom"
    node_type: str                   # e.g., "task", "concept", "entity"

    # Content
    content: str                     # Primary content (title, description, etc.)
    properties: Dict[str, Any]       # Type-specific properties
    metadata: Dict[str, Any]         # System metadata (created_at, version, etc.)

    # Versioning (for MVCC)
    version: int                     # Monotonically increasing version
    created_at: datetime             # Creation timestamp
    updated_at: datetime             # Last update timestamp

    # Embedding (optional, for semantic queries)
    embedding: Optional[List[float]] # Vector embedding for similarity search
    embedding_model: Optional[str]   # Model used to generate embedding

    # Checksum for corruption detection
    checksum: str                    # SHA-256 of content + properties
```

### 2. Universal Edge Model

```python
@dataclass
class DistributedEdge:
    """
    Universal edge model for all relationship types.

    Edges are stored on BOTH source and target partitions
    for efficient traversal in both directions.
    """

    # Identity
    id: str                          # Globally unique edge ID

    # Endpoints
    source_id: str                   # Source node ID
    target_id: str                   # Target node ID
    source_partition: str            # Partition of source node
    target_partition: str            # Partition of target node

    # Type system
    namespace: str                   # e.g., "got", "thought", "semantic"
    edge_type: str                   # e.g., "DEPENDS_ON", "REQUIRES", "SIMILAR"

    # Weights and confidence
    weight: float                    # Relationship strength (0.0 to 1.0)
    confidence: float                # Confidence in relationship (0.0 to 1.0)

    # Directionality
    bidirectional: bool              # Whether relationship is symmetric

    # Properties
    properties: Dict[str, Any]       # Edge-specific properties
    metadata: Dict[str, Any]         # System metadata

    # Versioning
    version: int
    created_at: datetime
    updated_at: datetime
```

### 3. Partitioning Strategy

```python
class PartitionStrategy(Enum):
    """
    Partitioning strategies for distributing graph data.
    """

    # Hash-based: Consistent hashing on node ID
    # Best for: Uniform distribution, random access patterns
    HASH = "hash"

    # Namespace-based: Partition by namespace (got, thought, knowledge)
    # Best for: Workload isolation, namespace-specific optimization
    NAMESPACE = "namespace"

    # Cluster-based: Keep densely-connected nodes together
    # Best for: Traversal-heavy workloads, community detection
    CLUSTER = "cluster"

    # Temporal-based: Partition by time (daily/weekly buckets)
    # Best for: Time-series graphs, archival patterns
    TEMPORAL = "temporal"


class PartitionManager:
    """
    Manages partition assignment and routing.

    Uses consistent hashing with virtual nodes for:
    - Minimal rebalancing on partition add/remove
    - Even distribution across partitions
    - Deterministic routing without central lookup
    """

    def __init__(
        self,
        num_partitions: int = 16,
        strategy: PartitionStrategy = PartitionStrategy.HASH,
        virtual_nodes_per_partition: int = 150
    ):
        self.num_partitions = num_partitions
        self.strategy = strategy
        self.ring = self._build_hash_ring(virtual_nodes_per_partition)

    def get_partition(self, node_id: str) -> int:
        """
        Get partition number for a node ID.

        O(log n) lookup using binary search on hash ring.
        """
        hash_value = self._hash(node_id)
        return self._find_partition_on_ring(hash_value)

    def get_partitions_for_query(self, query: GraphQuery) -> List[int]:
        """
        Determine which partitions need to be queried.

        Optimizations:
        - Single node lookup: 1 partition
        - Range query on partition key: subset of partitions
        - Global query: all partitions (with parallel fan-out)
        """
        if query.is_single_node_lookup:
            return [self.get_partition(query.node_id)]
        elif query.has_partition_key_predicate:
            return self._partitions_for_key_range(query.partition_key_range)
        else:
            return list(range(self.num_partitions))
```

### 4. Distributed Transaction Protocol

```python
class DistributedTransactionManager:
    """
    Two-Phase Commit (2PC) with optimizations for graph operations.

    Optimizations we implement:
    - Single-partition transactions bypass 2PC entirely
    - Read-only transactions use snapshot isolation (no locking)
    - Multi-partition writes use pipelined 2PC with early abort
    """

    def begin_transaction(self, read_only: bool = False) -> DistributedTransaction:
        """
        Begin a new distributed transaction.

        Returns a transaction object that tracks:
        - Read set (nodes/edges read)
        - Write set (nodes/edges to write)
        - Participating partitions
        - Transaction state (ACTIVE, PREPARING, COMMITTED, ABORTED)
        """
        tx_id = generate_transaction_id()
        snapshot_version = self.global_version_counter.current()

        return DistributedTransaction(
            id=tx_id,
            snapshot_version=snapshot_version,
            read_only=read_only,
            state=TransactionState.ACTIVE,
            read_set={},
            write_set={},
            participating_partitions=set()
        )

    def commit(self, tx: DistributedTransaction) -> CommitResult:
        """
        Commit a distributed transaction.

        Protocol:
        1. PREPARE phase: Ask all participants if they can commit
        2. COMMIT phase: If all say yes, commit on all partitions
        3. ABORT: If any say no, abort everywhere

        Optimization: Single-partition transactions commit directly.
        """
        if len(tx.participating_partitions) == 1:
            return self._commit_single_partition(tx)
        else:
            return self._commit_multi_partition(tx)

    def _commit_multi_partition(self, tx: DistributedTransaction) -> CommitResult:
        """
        Multi-partition commit using 2PC.

        Timeline for 2-partition transaction:
        - Phase 1 (PREPARE): 10-20ms per partition (parallel)
        - Phase 2 (COMMIT): 5-10ms per partition (parallel)
        - Total: 15-30ms for 2PC
        """
        # Phase 1: Prepare
        prepare_results = self._parallel_prepare(tx)

        if all(r.success for r in prepare_results):
            # Phase 2: Commit
            return self._parallel_commit(tx)
        else:
            # Abort on all partitions
            return self._parallel_abort(tx, prepare_results)
```

### 5. Query Engine

```python
class DistributedQueryEngine:
    """
    Distributed graph query engine supporting:
    - Point queries (get node by ID)
    - Pattern matching (find subgraphs matching pattern)
    - Path queries (find paths between nodes)
    - Aggregations (count, sum, avg by grouping)
    - Full-text search (on content field)
    - Semantic search (on embedding field)
    """

    def execute(self, query: GraphQuery) -> QueryResult:
        """
        Execute a query across partitions.

        Query planning:
        1. Parse query to determine required partitions
        2. Generate per-partition query plans
        3. Execute in parallel with timeout
        4. Merge results and return

        Performance targets:
        - Point query: p95 < 20ms
        - Pattern match (2-3 hops): p95 < 100ms
        - Path query (up to 6 hops): p95 < 200ms
        - Aggregation (all partitions): p95 < 150ms
        """
        # Determine partitions to query
        partitions = self.partition_manager.get_partitions_for_query(query)

        # Generate per-partition plans
        plans = [self._generate_plan(query, p) for p in partitions]

        # Execute in parallel
        results = self._parallel_execute(plans, timeout_ms=500)

        # Merge results
        return self._merge_results(results, query)

    def pattern_match(self, pattern: GraphPattern) -> List[Dict[str, Node]]:
        """
        Find all subgraphs matching a pattern.

        Pattern example:
            Pattern()
                .node("a", type="task", status="pending")
                .edge("DEPENDS_ON")
                .node("b", type="task", status="completed")

        Execution:
        1. Start from most selective node predicate
        2. Expand along edges, pruning non-matches
        3. Join results from different partitions
        """
        # Find starting point (most selective predicate)
        start_predicate = self._most_selective(pattern.node_predicates)

        # Query partitions for matching start nodes
        start_nodes = self._find_matching_nodes(start_predicate)

        # Expand pattern from each start node
        matches = []
        for node in start_nodes:
            partial_matches = self._expand_pattern(node, pattern)
            matches.extend(partial_matches)

        return matches
```

### 6. Local Storage Engine

```python
class LocalStorageEngine:
    """
    Local storage engine for a single partition.

    Storage layers:
    1. MemTable: In-memory buffer for recent writes (LSM-tree style)
    2. WAL: Write-ahead log for durability
    3. SSTable: Immutable sorted files on disk
    4. Index: B-tree index for range queries, hash index for point queries

    All built from scratch - no external dependencies.
    """

    def __init__(self, partition_dir: Path, config: StorageConfig):
        self.memtable = MemTable(max_size_bytes=config.memtable_size)
        self.wal = WALManager(
            partition_dir / "wal",
            durability=config.durability_mode
        )
        self.sstables = SSTableManager(partition_dir / "data")
        self.node_index = BTreeIndex(partition_dir / "index" / "nodes")
        self.edge_index = BTreeIndex(partition_dir / "index" / "edges")

    def get_node(self, node_id: str) -> Optional[DistributedNode]:
        """
        Get a node by ID.

        Lookup order (most recent first):
        1. MemTable (in-memory, O(log n))
        2. SSTable bloom filters (eliminate non-matches)
        3. SSTable binary search (if bloom filter positive)

        Target: p99 < 5ms for single node lookup
        """
        # Check memtable first
        if node := self.memtable.get(node_id):
            return node

        # Check SSTables (most recent first)
        for sstable in self.sstables.by_recency():
            if sstable.bloom_filter.might_contain(node_id):
                if node := sstable.get(node_id):
                    return node

        return None

    def write_node(self, tx: Transaction, node: DistributedNode) -> None:
        """
        Write a node within a transaction.

        Write path:
        1. Log to WAL (fsync based on durability mode)
        2. Insert into memtable
        3. Update indexes

        When memtable reaches threshold, flush to SSTable.
        """
        # WAL first
        self.wal.log(tx.id, "WRITE_NODE", node.to_dict())

        # Then memtable
        self.memtable.put(node.id, node)

        # Update indexes
        self.node_index.insert(node.id, node.partition_key, node.node_type)
```

### 7. Index Structures

```python
class BTreeIndex:
    """
    B-tree index for range queries.

    Implemented from scratch with:
    - Page-based storage (4KB pages)
    - Write-ahead logging for crash recovery
    - Lock-free reads with copy-on-write
    - Prefix compression for space efficiency
    """

    def __init__(self, index_path: Path, page_size: int = 4096, order: int = 128):
        self.index_path = index_path
        self.page_size = page_size
        self.order = order
        self.root_page_id = 0
        self.pages = PageManager(index_path, page_size)

    def range_query(
        self,
        start_key: Optional[str],
        end_key: Optional[str],
        limit: int = 1000
    ) -> List[Tuple[str, int]]:
        """
        Range query returning (key, page_id) pairs.

        Performance: O(log n + k) where n is tree size, k is result count
        Target: < 2ms for 100-item range
        """
        results = []
        current = self._find_leaf(start_key)

        while current and len(results) < limit:
            for key, value in current.entries:
                if end_key and key > end_key:
                    return results
                if start_key is None or key >= start_key:
                    results.append((key, value))
            current = current.next_leaf

        return results


class BloomFilter:
    """
    Space-efficient probabilistic set membership.

    Used to quickly eliminate SSTable lookups.
    False positive rate: 1% with 10 bits per element.
    """

    def __init__(self, expected_elements: int, false_positive_rate: float = 0.01):
        self.size = self._optimal_size(expected_elements, false_positive_rate)
        self.num_hashes = self._optimal_hashes(self.size, expected_elements)
        self.bits = bitarray(self.size)
        self.bits.setall(0)

    def might_contain(self, key: str) -> bool:
        """Check if key might be in set. O(k) where k is num_hashes."""
        for i in range(self.num_hashes):
            if not self.bits[self._hash(key, i) % self.size]:
                return False
        return True
```

---

## Unified API

All existing graph implementations can migrate to CDG through adapters:

### GoT Adapter

```python
class GoTAdapter:
    """
    Adapter allowing GoT to use CDG as its storage backend.

    Maps GoT entities to CDG nodes:
    - Task → Node(namespace="got", node_type="task")
    - Decision → Node(namespace="got", node_type="decision")
    - Sprint → Node(namespace="got", node_type="sprint")
    - Edge → Edge(namespace="got", edge_type=original_type)
    """

    def __init__(self, cdg_client: CDGClient):
        self.cdg = cdg_client
        self.namespace = "got"

    def create_task(self, title: str, **kwargs) -> Task:
        node = DistributedNode(
            id=generate_task_id(),
            partition_key=kwargs.get("sprint_id", "default"),
            namespace=self.namespace,
            node_type="task",
            content=title,
            properties={
                "status": kwargs.get("status", "pending"),
                "priority": kwargs.get("priority", "medium"),
                **kwargs
            }
        )
        self.cdg.write_node(node)
        return Task.from_node(node)

    def transaction(self) -> GoTTransaction:
        """
        Return a transaction context that wraps CDG transactions.

        Provides same API as existing GoTManager.transaction()
        but backed by distributed CDG storage.
        """
        return GoTTransaction(self.cdg.begin_transaction())
```

### ThoughtGraph Adapter

```python
class ThoughtGraphAdapter:
    """
    Adapter allowing ThoughtGraph to use CDG as its storage backend.

    Maps ThoughtGraph nodes to CDG:
    - ThoughtNode → Node(namespace="thought", node_type=thought_type)
    - ThoughtEdge → Edge(namespace="thought", edge_type=edge_type)
    - ThoughtCluster → Node(namespace="thought", node_type="cluster")
    """

    def __init__(self, cdg_client: CDGClient):
        self.cdg = cdg_client
        self.namespace = "thought"

    def add_node(
        self,
        node_id: str,
        node_type: NodeType,
        content: str,
        **properties
    ) -> ThoughtNode:
        node = DistributedNode(
            id=node_id,
            partition_key=properties.get("cluster_id", "default"),
            namespace=self.namespace,
            node_type=node_type.value,
            content=content,
            properties=properties
        )
        self.cdg.write_node(node)
        return ThoughtNode.from_node(node)

    def bfs(self, start_id: str, max_depth: int = 10) -> List[str]:
        """
        Breadth-first search from a starting node.

        CDG optimizes this by:
        1. Fetching all edges from start node's partition
        2. Parallel fan-out to target partitions
        3. Streaming results as discovered
        """
        query = GraphQuery.bfs(
            start_id=start_id,
            max_depth=max_depth,
            namespace=self.namespace
        )
        result = self.cdg.execute(query)
        return [node.id for node in result.nodes]
```

---

## Performance Contracts

```python
"""
╔══════════════════════════════════════════════════════════════════════╗
║                    CDG PERFORMANCE CONTRACT                           ║
╠══════════════════════════════════════════════════════════════════════╣
║  Ratified:     2025-12-31                                            ║
║  Guardian:     CI Pipeline                                            ║
║  Renegotiation: Requires team review + documented justification      ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  Query Latencies (single partition, warm cache):                     ║
║  • Point query (get node by ID):     p50 < 5ms,  p95 < 20ms         ║
║  • Range query (100 results):        p50 < 10ms, p95 < 50ms         ║
║  • Pattern match (2-hop):            p50 < 20ms, p95 < 100ms        ║
║  • Path query (up to 6 hops):        p50 < 50ms, p95 < 200ms        ║
║                                                                       ║
║  Query Latencies (multi-partition):                                  ║
║  • Fan-out query (all partitions):   p50 < 50ms, p95 < 150ms        ║
║  • Cross-partition join:             p50 < 80ms, p95 < 200ms        ║
║                                                                       ║
║  Write Latencies:                                                    ║
║  • Single-partition write:           p50 < 10ms, p95 < 30ms         ║
║  • Multi-partition 2PC:              p50 < 30ms, p95 < 100ms        ║
║                                                                       ║
║  Throughput (per partition):                                         ║
║  • Reads: > 10,000 ops/sec                                          ║
║  • Writes: > 1,000 ops/sec (PARANOID durability)                    ║
║  • Writes: > 5,000 ops/sec (BALANCED durability)                    ║
║                                                                       ║
║  Scalability:                                                        ║
║  • Linear read scaling with partition count                          ║
║  • Sub-linear write scaling (due to 2PC overhead)                   ║
║                                                                       ║
╚══════════════════════════════════════════════════════════════════════╝
"""
```

---

## Resource Control & Operational Excellence

### 8. Custom Index Management

Developers have full control over index creation, lifecycle, and optimization:

```python
class IndexManager:
    """
    Programmable index management for custom workloads.

    Developers control:
    - Which fields are indexed
    - Index type selection
    - Index build timing (sync/async)
    - Index storage budget
    - Automatic vs manual maintenance
    """

    def create_index(
        self,
        name: str,
        index_type: IndexType,
        fields: List[str],
        options: IndexOptions
    ) -> IndexHandle:
        """
        Create a custom index on node/edge fields.

        Index Types:
        - BTREE: Range queries, sorting (default)
        - HASH: Point lookups, equality checks
        - INVERTED: Full-text search
        - BITMAP: Low-cardinality fields (status, type)
        - SPATIAL: Geospatial queries (future)
        - VECTOR: Embedding similarity (future)

        Options:
        - build_mode: SYNC (block until built) | ASYNC (background)
        - storage_budget_mb: Maximum index size
        - compaction_strategy: LEVELED | SIZE_TIERED | TIME_WINDOW
        - bloom_filter_fp_rate: False positive rate (default 0.01)

        Example:
            # Index for fast status lookups
            idx = index_mgr.create_index(
                name="task_status_idx",
                index_type=IndexType.BITMAP,
                fields=["properties.status"],
                options=IndexOptions(
                    build_mode=BuildMode.ASYNC,
                    partitions=["got"]  # Only for GoT namespace
                )
            )

            # Composite index for complex queries
            idx = index_mgr.create_index(
                name="task_priority_date_idx",
                index_type=IndexType.BTREE,
                fields=["properties.priority", "created_at"],
                options=IndexOptions(
                    storage_budget_mb=100
                )
            )
        """
        pass

    def drop_index(self, name: str, force: bool = False) -> None:
        """Drop an index. Use force=True to drop even if queries depend on it."""
        pass

    def rebuild_index(self, name: str, options: RebuildOptions) -> AsyncHandle:
        """
        Rebuild index in background without blocking reads.

        Useful for:
        - Fixing corruption
        - Changing index parameters
        - Compacting fragmented indexes
        """
        pass

    def analyze_index(self, name: str) -> IndexAnalysis:
        """
        Get index statistics and recommendations.

        Returns:
        - size_bytes: Current index size
        - entries: Number of indexed entries
        - fragmentation_pct: How fragmented
        - hit_rate: Cache hit rate
        - recommendations: ["consider compaction", "unused - drop?"]
        """
        pass

    def list_indexes(self, namespace: Optional[str] = None) -> List[IndexInfo]:
        """List all indexes with their metadata."""
        pass


@dataclass
class IndexOptions:
    """Fine-grained index configuration."""

    # Build behavior
    build_mode: BuildMode = BuildMode.ASYNC
    parallelism: int = 4  # Concurrent index build threads

    # Storage limits
    storage_budget_mb: Optional[int] = None  # None = unlimited
    compaction_strategy: CompactionStrategy = CompactionStrategy.LEVELED

    # Performance tuning
    bloom_filter_fp_rate: float = 0.01
    cache_size_mb: int = 64
    page_size: int = 4096

    # Filtering
    partitions: Optional[List[str]] = None  # Only index these partitions
    filter_predicate: Optional[str] = None  # Only index matching nodes

    # Maintenance
    auto_compact: bool = True
    compact_threshold: float = 0.3  # Compact when 30% fragmented
```

### 9. Memory Management & Backpressure

Prevent OOM crashes with explicit memory budgets and backpressure:

```python
class MemoryManager:
    """
    Explicit memory budgets to prevent OS crashes.

    Philosophy: Better to reject requests gracefully than crash.
    """

    def __init__(self, config: MemoryConfig):
        self.config = config
        self._allocator = BuddyAllocator(config.total_budget_mb * 1024 * 1024)
        self._pressure_callbacks: List[Callable] = []

    def configure(
        self,
        total_budget_mb: int = 1024,
        memtable_budget_mb: int = 256,
        cache_budget_mb: int = 512,
        query_budget_mb: int = 256,
        emergency_reserve_mb: int = 64
    ) -> None:
        """
        Configure memory budgets by subsystem.

        Budgets:
        - memtable: Write buffer before flush to disk
        - cache: Block cache, page cache, index cache
        - query: Per-query result buffers, intermediate state
        - emergency_reserve: Reserved for graceful degradation

        When a budget is exceeded:
        1. Backpressure applied (slow down writes/queries)
        2. Eviction triggered (LRU cache entries)
        3. If still exceeded, requests rejected with RESOURCE_EXHAUSTED

        Example:
            mem_mgr.configure(
                total_budget_mb=2048,      # 2GB total
                memtable_budget_mb=512,    # 512MB for writes
                cache_budget_mb=1024,      # 1GB for caching
                query_budget_mb=448,       # 448MB for queries
                emergency_reserve_mb=64    # 64MB reserved
            )
        """
        pass

    def allocate(
        self,
        subsystem: MemorySubsystem,
        size_bytes: int,
        priority: AllocationPriority = AllocationPriority.NORMAL
    ) -> Optional[MemoryAllocation]:
        """
        Allocate memory from a subsystem budget.

        Returns None if budget exceeded and backpressure active.
        Raises ResourceExhausted if emergency reserve depleted.
        """
        pass

    def get_pressure_level(self) -> PressureLevel:
        """
        Current memory pressure level.

        Levels:
        - NONE: < 70% budget used, all systems go
        - LOW: 70-85%, start evicting cold cache entries
        - MEDIUM: 85-95%, slow down writes, aggressive eviction
        - HIGH: 95-99%, reject new queries, emergency eviction
        - CRITICAL: > 99%, reject all requests, trigger snapshot
        """
        pass

    def on_pressure_change(self, callback: Callable[[PressureLevel], None]) -> None:
        """Register callback for pressure level changes."""
        pass

    def get_usage_stats(self) -> MemoryUsageStats:
        """
        Detailed memory usage breakdown.

        Returns:
            MemoryUsageStats(
                total_budget_mb=2048,
                total_used_mb=1536,
                by_subsystem={
                    "memtable": SubsystemUsage(budget=512, used=400, pct=78.1),
                    "cache": SubsystemUsage(budget=1024, used=900, pct=87.9),
                    "query": SubsystemUsage(budget=448, used=200, pct=44.6),
                },
                largest_allocations=[
                    Allocation("idx:task_status", 150_000_000),
                    Allocation("cache:hot_nodes", 120_000_000),
                ],
                gc_stats=GCStats(collections=42, time_ms=1200)
            )
        """
        pass


@dataclass
class MemoryConfig:
    """Memory configuration with sensible defaults."""

    total_budget_mb: int = 1024

    # Subsystem budgets (must sum to <= total - emergency)
    memtable_budget_pct: float = 0.25    # 25% for write buffers
    cache_budget_pct: float = 0.50       # 50% for caches
    query_budget_pct: float = 0.20       # 20% for query processing
    emergency_reserve_pct: float = 0.05  # 5% emergency reserve

    # Backpressure thresholds
    pressure_low_pct: float = 0.70
    pressure_medium_pct: float = 0.85
    pressure_high_pct: float = 0.95
    pressure_critical_pct: float = 0.99

    # Eviction policy
    eviction_policy: EvictionPolicy = EvictionPolicy.LRU
    eviction_batch_size: int = 100
```

### 10. Network Bandwidth Control

Rate limiting and bandwidth management for distributed operations:

```python
class NetworkManager:
    """
    Control network bandwidth to prevent saturation.

    Critical for:
    - Multi-tenant deployments
    - Preventing noisy neighbor problems
    - Graceful degradation under load
    """

    def configure(
        self,
        max_bandwidth_mbps: int = 1000,
        max_connections_per_partition: int = 100,
        request_timeout_ms: int = 5000,
        compression: CompressionConfig = CompressionConfig()
    ) -> None:
        """
        Configure network limits.

        Parameters:
        - max_bandwidth_mbps: Total outbound bandwidth limit
        - max_connections_per_partition: Connection pool size
        - request_timeout_ms: Timeout for inter-partition requests
        - compression: Compression settings for wire protocol
        """
        pass

    def create_rate_limiter(
        self,
        name: str,
        requests_per_second: int,
        burst_size: int = 10
    ) -> RateLimiter:
        """
        Create a named rate limiter for specific operations.

        Example:
            # Limit bulk imports to not overwhelm the system
            bulk_limiter = net_mgr.create_rate_limiter(
                name="bulk_import",
                requests_per_second=100,
                burst_size=50
            )

            # Use in code
            async with bulk_limiter.acquire():
                await cdg.write_batch(nodes)
        """
        pass

    def get_bandwidth_usage(self) -> BandwidthStats:
        """Current bandwidth usage by partition and operation type."""
        pass


@dataclass
class CompressionConfig:
    """Wire protocol compression settings."""

    enabled: bool = True
    algorithm: CompressionAlgorithm = CompressionAlgorithm.LZ4  # Fast
    level: int = 1  # 1-9, higher = better ratio, slower
    min_size_bytes: int = 1024  # Don't compress small messages

    # Adaptive compression
    adaptive: bool = True  # Adjust based on CPU/bandwidth tradeoff
    cpu_threshold_pct: float = 0.80  # Disable compression if CPU > 80%
```

### 11. Thread Pool Configuration

Separate thread pools for reads and writes with explicit control:

```python
class ThreadPoolManager:
    """
    Explicit thread pool management for workload isolation.

    Why separate pools?
    - Prevent write storms from starving reads
    - Tune for workload characteristics
    - Better resource utilization
    """

    def configure(
        self,
        read_threads: int = 8,
        write_threads: int = 4,
        background_threads: int = 2,
        query_threads: int = 4,
        io_threads: int = 4
    ) -> None:
        """
        Configure thread pools by purpose.

        Pools:
        - read_threads: Handle read queries (point, range, pattern)
        - write_threads: Handle writes (insert, update, delete)
        - background_threads: Compaction, index builds, snapshots
        - query_threads: Complex query execution (joins, aggregations)
        - io_threads: Disk I/O operations

        Guidelines:
        - Read-heavy: read_threads = 2 * write_threads
        - Write-heavy: write_threads = read_threads
        - CPU cores: total threads <= 2 * cores for CPU-bound
        - I/O bound: io_threads = disk count * 2

        Example for 8-core machine with SSD:
            pool_mgr.configure(
                read_threads=8,      # Handle concurrent reads
                write_threads=4,     # Writes are I/O bound
                background_threads=2, # Low priority
                query_threads=4,     # Complex queries
                io_threads=4         # SSD parallelism
            )
        """
        pass

    def set_priority(self, pool: ThreadPool, priority: Priority) -> None:
        """
        Set OS thread priority for a pool.

        Priorities:
        - REALTIME: Critical path (use sparingly)
        - HIGH: User-facing queries
        - NORMAL: Standard operations
        - LOW: Background tasks (compaction, GC)
        """
        pass

    def get_pool_stats(self) -> Dict[str, PoolStats]:
        """
        Get statistics for each thread pool.

        Returns:
            {
                "read": PoolStats(
                    threads=8,
                    active=5,
                    queued=12,
                    completed=100000,
                    avg_latency_ms=2.5,
                    p99_latency_ms=15.0
                ),
                ...
            }
        """
        pass

    def resize_pool(self, pool: ThreadPool, new_size: int) -> None:
        """Dynamically resize a thread pool (takes effect immediately)."""
        pass
```

### 12. Query Tracing & Explain Plans

Distributed tracing for debugging query performance:

```python
class QueryTracer:
    """
    Distributed query tracing for debugging and optimization.

    Every query can be traced to see:
    - Which partitions were touched
    - Time spent in each phase
    - Rows scanned vs returned
    - Index usage
    - Network hops
    """

    def trace(self, query: GraphQuery) -> TracedResult:
        """
        Execute query with full tracing enabled.

        Example:
            result = tracer.trace(
                GraphQuery.pattern_match(
                    Pattern()
                        .node("a", type="task", status="pending")
                        .edge("DEPENDS_ON")
                        .node("b", type="task")
                )
            )

            print(result.trace.summary())
            # Query: pattern_match
            # Total time: 45.2ms
            # Partitions: [0, 3, 7]
            # Phases:
            #   - parse: 0.1ms
            #   - plan: 0.5ms
            #   - partition_route: 0.2ms
            #   - partition_0_execute: 15.1ms (scanned: 1000, matched: 50)
            #   - partition_3_execute: 14.8ms (scanned: 800, matched: 30)
            #   - partition_7_execute: 12.5ms (scanned: 600, matched: 20)
            #   - merge: 2.0ms
            # Indexes used: [task_status_idx, task_type_idx]
            # Warnings: ["partition_3: sequential scan on edges"]
        """
        pass

    def explain(self, query: GraphQuery) -> QueryPlan:
        """
        Get query execution plan without executing.

        Returns:
            QueryPlan(
                steps=[
                    PlanStep(
                        operation="INDEX_SCAN",
                        target="task_status_idx",
                        predicate="status = 'pending'",
                        estimated_rows=500,
                        estimated_cost=10.0
                    ),
                    PlanStep(
                        operation="EDGE_EXPAND",
                        edge_type="DEPENDS_ON",
                        direction="OUTGOING",
                        estimated_rows=2000,
                        estimated_cost=50.0
                    ),
                    PlanStep(
                        operation="FILTER",
                        predicate="node_type = 'task'",
                        estimated_rows=1500,
                        estimated_cost=5.0
                    ),
                ],
                total_estimated_cost=65.0,
                partitions_touched=[0, 3, 7],
                indexes_used=["task_status_idx"],
                warnings=["Consider adding index on DEPENDS_ON edges"]
            )
        """
        pass

    def analyze(self, query: GraphQuery) -> QueryAnalysis:
        """
        Deep analysis with recommendations.

        Runs query multiple times to gather statistics, then provides:
        - Actual vs estimated row counts
        - Hot spots (partitions with most work)
        - Missing index recommendations
        - Query rewrite suggestions
        """
        pass


@dataclass
class TraceSpan:
    """A single span in the distributed trace."""

    span_id: str
    parent_id: Optional[str]
    operation: str
    partition: Optional[int]
    start_time: datetime
    duration_ms: float

    # Metrics
    rows_scanned: int = 0
    rows_returned: int = 0
    bytes_read: int = 0
    bytes_written: int = 0

    # Context
    index_used: Optional[str] = None
    cache_hit: bool = False

    # Errors/warnings
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
```

### 13. Debugging & Introspection

Tools for debugging hard-to-reach issues:

```python
class DebugTools:
    """
    Introspection and debugging tools for operational issues.

    Philosophy: When things go wrong at 3 AM, these tools help you
    understand what's happening without restarting the system.
    """

    def dump_partition_state(self, partition_id: int) -> PartitionDump:
        """
        Dump complete state of a partition for analysis.

        Includes:
        - Node/edge counts by type
        - Index states
        - Pending transactions
        - WAL position
        - Memory usage breakdown
        - Recent errors
        """
        pass

    def inspect_node(self, node_id: str) -> NodeInspection:
        """
        Deep inspection of a single node.

        Returns:
        - All versions (MVCC history)
        - Which indexes contain it
        - Edges in/out with their states
        - Transaction locks held
        - Last access time
        - Storage location (memtable, which SSTable)
        """
        pass

    def inspect_transaction(self, tx_id: str) -> TransactionInspection:
        """
        Inspect a transaction (active or historical).

        Returns:
        - State (ACTIVE, COMMITTED, ABORTED)
        - Read/write sets
        - Locks held
        - Participants (for 2PC)
        - Timeline of events
        - If aborted: reason
        """
        pass

    def find_locks(self, node_id: Optional[str] = None) -> List[LockInfo]:
        """
        Find all active locks, optionally filtered by node.

        Essential for debugging deadlocks and contention.
        """
        pass

    def simulate_failure(self, failure: FailureScenario) -> SimulationResult:
        """
        Simulate failure scenarios for testing recovery.

        Scenarios:
        - PARTITION_NETWORK_FAILURE: Simulate network partition
        - NODE_CRASH: Simulate sudden process death
        - DISK_FULL: Simulate disk space exhaustion
        - SLOW_DISK: Simulate degraded disk performance
        - MEMORY_PRESSURE: Simulate OOM conditions

        Example:
            result = debug.simulate_failure(
                FailureScenario.PARTITION_NETWORK_FAILURE,
                partition_id=3,
                duration_seconds=30
            )
            # System continues, can observe behavior
        """
        pass

    def enable_debug_mode(
        self,
        partition: Optional[int] = None,
        log_level: LogLevel = LogLevel.DEBUG,
        trace_all_queries: bool = False,
        record_allocations: bool = False
    ) -> DebugSession:
        """
        Enable detailed debug logging for a partition or globally.

        Warning: Significant performance impact. Use sparingly.
        """
        pass

    def health_check(self) -> HealthReport:
        """
        Comprehensive health check of the entire system.

        Checks:
        - All partitions responding
        - WAL not falling behind
        - No stuck transactions
        - Memory within limits
        - Disk space adequate
        - Index health
        - Replication lag (if applicable)
        """
        pass
```

### 14. Cortical Query Language (CQL)

A declarative query language inspired by community standards (Cypher/Gremlin/SQL):

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    CORTICAL QUERY LANGUAGE (CQL)                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Design Principles:                                                      │
│  1. Familiar to users of Cypher, Gremlin, SQL                           │
│  2. Graph-native: paths and patterns are first-class                    │
│  3. Composable: queries can be combined                                 │
│  4. Typed: catch errors at parse time                                   │
│  5. Extensible: custom functions and operators                          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

## Basic Syntax

```cql
-- Find all pending tasks
MATCH (t:Task {status: "pending"})
RETURN t

-- Find tasks with their dependencies
MATCH (t:Task)-[:DEPENDS_ON]->(dep:Task)
WHERE t.status = "pending" AND dep.status = "completed"
RETURN t.id, t.title, dep.title AS dependency

-- Path queries
MATCH path = (start:Task {id: "T-001"})-[:DEPENDS_ON*1..5]->(end:Task)
WHERE end.status = "blocked"
RETURN path, length(path) AS hops

-- Pattern matching with properties
MATCH (q:Question)-[:EXPLORES]->(h:Hypothesis)-[:SUPPORTS]->(c:Conclusion)
WHERE h.confidence > 0.8
RETURN q.content, h.content, c.content

-- Aggregations
MATCH (t:Task)
WHERE t.created_at > datetime("2025-01-01")
RETURN t.status, count(*) AS count, avg(t.priority) AS avg_priority
GROUP BY t.status

-- Mutations
CREATE (t:Task {
    id: generate_id(),
    title: "Implement CDG",
    status: "pending",
    priority: 1
})

-- Update with pattern
MATCH (t:Task {id: "T-001"})
SET t.status = "completed", t.completed_at = now()
RETURN t

-- Delete with safety
MATCH (t:Task {id: "T-001"})
WHERE NOT (t)<-[:DEPENDS_ON]-(:Task {status: "pending"})
DELETE t

-- Transactions
BEGIN TRANSACTION
    CREATE (t1:Task {id: "T-100", title: "Task A"})
    CREATE (t2:Task {id: "T-101", title: "Task B"})
    CREATE (t1)-[:DEPENDS_ON {weight: 0.9}]->(t2)
COMMIT

-- Namespace scoping
USE NAMESPACE thought
MATCH (n:Concept)-[:SIMILAR*1..3]-(related)
RETURN n, collect(related) AS related_concepts
```

```python
class CQLParser:
    """
    Parse and execute CQL queries.

    Built from scratch - no external parser generators.
    """

    def parse(self, query: str) -> CQLStatement:
        """Parse CQL string into AST."""
        pass

    def validate(self, statement: CQLStatement) -> ValidationResult:
        """Validate query against schema and permissions."""
        pass

    def execute(self, query: str, params: Dict[str, Any] = None) -> QueryResult:
        """
        Parse, validate, and execute a CQL query.

        Example:
            result = cql.execute('''
                MATCH (t:Task {status: $status})
                WHERE t.priority <= $max_priority
                RETURN t
                ORDER BY t.created_at DESC
                LIMIT 10
            ''', params={"status": "pending", "max_priority": 2})

            for row in result:
                print(row["t"])
        """
        pass

    def explain(self, query: str) -> QueryPlan:
        """Get execution plan for a query."""
        pass


# CQL can also be built programmatically
class CQLBuilder:
    """Fluent API for building CQL queries."""

    def match(self, pattern: str) -> "CQLBuilder":
        """Add MATCH clause."""
        pass

    def where(self, condition: str) -> "CQLBuilder":
        """Add WHERE clause."""
        pass

    def return_(self, *expressions: str) -> "CQLBuilder":
        """Add RETURN clause."""
        pass

    def build(self) -> str:
        """Build CQL string."""
        pass

    # Example usage:
    # query = (CQLBuilder()
    #     .match("(t:Task)-[:DEPENDS_ON]->(d:Task)")
    #     .where("t.status = 'pending'")
    #     .return_("t", "collect(d) as deps")
    #     .build())
```

### 15. Schema Evolution & Migration

Handle schema changes without downtime:

```python
class SchemaManager:
    """
    Schema versioning and migration without downtime.

    Principles:
    - Backward compatible by default
    - Lazy migration (on read/write, not big bang)
    - Rollback capability
    - Audit trail
    """

    def register_schema(
        self,
        namespace: str,
        version: int,
        schema: Schema
    ) -> None:
        """
        Register a schema version for a namespace.

        Schema defines:
        - Required properties
        - Property types
        - Constraints (unique, not null, etc.)
        - Indexes to create

        Example:
            schema_mgr.register_schema(
                namespace="got",
                version=2,
                schema=Schema(
                    node_types={
                        "task": NodeSchema(
                            required=["title", "status"],
                            properties={
                                "title": PropertyType.STRING,
                                "status": PropertyType.ENUM(["pending", "active", "completed"]),
                                "priority": PropertyType.INT,
                                "estimate_hours": PropertyType.FLOAT,  # NEW in v2
                            },
                            indexes=["status", ("priority", "created_at")]
                        )
                    },
                    edge_types={
                        "depends_on": EdgeSchema(
                            from_types=["task"],
                            to_types=["task"],
                            properties={"weight": PropertyType.FLOAT}
                        )
                    }
                )
            )
        """
        pass

    def migrate(
        self,
        namespace: str,
        from_version: int,
        to_version: int,
        migration: Migration
    ) -> MigrationHandle:
        """
        Register a migration between schema versions.

        Migrations are applied lazily:
        - On read: Transform old format to new
        - On write: Ensure new format
        - Background: Gradually migrate all data

        Example:
            schema_mgr.migrate(
                namespace="got",
                from_version=1,
                to_version=2,
                migration=Migration(
                    # Transform function for lazy migration
                    transform=lambda node: {
                        **node,
                        "estimate_hours": node.get("estimate_hours", 0.0)
                    },
                    # Validation for new writes
                    validate=lambda node: "estimate_hours" in node,
                    # Background migration (optional)
                    background=True,
                    batch_size=1000
                )
            )
        """
        pass

    def get_migration_status(self, namespace: str) -> MigrationStatus:
        """
        Check migration progress.

        Returns:
            MigrationStatus(
                namespace="got",
                current_version=2,
                nodes_migrated=50000,
                nodes_remaining=10000,
                estimated_completion="2025-01-15T10:00:00",
                errors=[]
            )
        """
        pass

    def rollback(self, namespace: str, to_version: int) -> None:
        """Rollback to a previous schema version."""
        pass
```

### 16. Observability Framework

Metrics, logging, and alerting built from scratch:

```python
class ObservabilityManager:
    """
    Built-in observability without external dependencies.

    We build our own because:
    - No dependency on Prometheus/Grafana/etc.
    - Customized for graph workloads
    - Integrated with our debug tools
    """

    def __init__(self, config: ObservabilityConfig):
        self.metrics = MetricsRegistry()
        self.logger = StructuredLogger()
        self.alerter = AlertManager()

    # ─────────────────────────────────────────────────────────────────
    # METRICS
    # ─────────────────────────────────────────────────────────────────

    def record_latency(self, operation: str, latency_ms: float, tags: Dict = None):
        """Record operation latency with histogram."""
        pass

    def increment_counter(self, name: str, value: int = 1, tags: Dict = None):
        """Increment a counter metric."""
        pass

    def set_gauge(self, name: str, value: float, tags: Dict = None):
        """Set a gauge metric."""
        pass

    def get_metrics(self, filter: Optional[str] = None) -> MetricsSnapshot:
        """
        Get current metrics snapshot.

        Built-in metrics:
        - cdg_query_latency_ms{operation, partition}
        - cdg_query_count{operation, status}
        - cdg_write_latency_ms{operation, partition}
        - cdg_transaction_count{status}
        - cdg_memory_usage_bytes{subsystem}
        - cdg_disk_usage_bytes{partition}
        - cdg_index_size_bytes{name}
        - cdg_wal_lag_bytes{partition}
        - cdg_cache_hit_ratio{cache}
        - cdg_thread_pool_queue_size{pool}
        """
        pass

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format for compatibility."""
        pass

    # ─────────────────────────────────────────────────────────────────
    # LOGGING
    # ─────────────────────────────────────────────────────────────────

    def log(self, level: LogLevel, message: str, **context):
        """
        Structured logging with context.

        Example:
            obs.log(
                LogLevel.WARN,
                "Query exceeded timeout",
                query_id="q-123",
                duration_ms=5500,
                timeout_ms=5000,
                partition=3
            )

        Output (JSON):
            {
                "timestamp": "2025-01-01T12:00:00.000Z",
                "level": "WARN",
                "message": "Query exceeded timeout",
                "query_id": "q-123",
                "duration_ms": 5500,
                "timeout_ms": 5000,
                "partition": 3
            }
        """
        pass

    def set_log_level(self, level: LogLevel, component: Optional[str] = None):
        """Set log level globally or per component."""
        pass

    # ─────────────────────────────────────────────────────────────────
    # ALERTING
    # ─────────────────────────────────────────────────────────────────

    def register_alert(
        self,
        name: str,
        condition: str,
        severity: AlertSeverity,
        action: AlertAction
    ) -> None:
        """
        Register an alert condition.

        Example:
            obs.register_alert(
                name="high_query_latency",
                condition="cdg_query_latency_ms.p99 > 200",
                severity=AlertSeverity.WARNING,
                action=AlertAction.LOG  # or WEBHOOK, EMAIL, etc.
            )

            obs.register_alert(
                name="memory_critical",
                condition="cdg_memory_usage_pct > 95",
                severity=AlertSeverity.CRITICAL,
                action=AlertAction.WEBHOOK(url="http://alerts/webhook")
            )
        """
        pass

    def get_active_alerts(self) -> List[Alert]:
        """Get currently firing alerts."""
        pass
```

---

## Implementation Phases

### Phase 1: Single-Node Foundation (Week 1-2)
- [ ] Universal node/edge models
- [ ] Local storage engine (MemTable + WAL + SSTable)
- [ ] B-tree and hash indexes
- [ ] Bloom filters for SSTable lookup
- [ ] Single-partition transactions
- [ ] Basic query engine (point queries, range queries)

### Phase 2: Distributed Coordination (Week 3-4)
- [ ] Partition manager with consistent hashing
- [ ] Multi-partition query routing
- [ ] Distributed transactions (2PC)
- [ ] Cross-partition edge storage
- [ ] Parallel query execution
- [ ] Result merging

### Phase 3: Advanced Features (Week 5-6)
- [ ] Pattern matching engine
- [ ] Path finding algorithms (BFS, DFS, Dijkstra)
- [ ] Graph analytics (PageRank, clustering)
- [ ] Semantic search (embedding support)
- [ ] Full-text search (inverted index)
- [ ] Aggregation queries

### Phase 4: Integration & Migration (Week 7-8)
- [ ] GoT adapter
- [ ] ThoughtGraph adapter
- [ ] Knowledge graph adapter
- [ ] Migration tooling
- [ ] Backward compatibility layer
- [ ] Performance validation

---

## Directory Structure

```
cortical/cdg/
├── __init__.py
├── models/
│   ├── node.py              # DistributedNode
│   ├── edge.py              # DistributedEdge
│   └── types.py             # NodeType, EdgeType enums
├── storage/
│   ├── memtable.py          # In-memory write buffer
│   ├── wal.py               # Write-ahead log
│   ├── sstable.py           # Sorted string tables
│   └── engine.py            # LocalStorageEngine
├── index/
│   ├── manager.py           # IndexManager (custom index creation)
│   ├── btree.py             # B-tree index
│   ├── hash.py              # Hash index
│   ├── bloom.py             # Bloom filter
│   ├── bitmap.py            # Bitmap index (low cardinality)
│   ├── inverted.py          # Inverted index (full-text)
│   └── vector.py            # Vector index (embeddings, future)
├── partition/
│   ├── manager.py           # PartitionManager
│   ├── router.py            # Query routing
│   └── hasher.py            # Consistent hashing
├── transaction/
│   ├── local.py             # Single-partition transactions
│   ├── distributed.py       # 2PC coordinator
│   └── mvcc.py              # Multi-version concurrency
├── query/
│   ├── cql/
│   │   ├── parser.py        # CQL parser (hand-written)
│   │   ├── lexer.py         # CQL lexer
│   │   ├── ast.py           # Abstract syntax tree
│   │   └── builder.py       # Fluent query builder
│   ├── planner.py           # Query planning
│   ├── executor.py          # Query execution
│   ├── pattern.py           # Pattern matching
│   ├── path.py              # Path finding
│   └── tracer.py            # Query tracing & explain
├── resource/
│   ├── memory.py            # MemoryManager (budgets, backpressure)
│   ├── network.py           # NetworkManager (bandwidth, rate limiting)
│   ├── threads.py           # ThreadPoolManager (read/write isolation)
│   └── allocator.py         # BuddyAllocator (memory allocation)
├── schema/
│   ├── manager.py           # SchemaManager (versioning)
│   ├── migration.py         # Migration engine
│   └── validator.py         # Schema validation
├── observability/
│   ├── metrics.py           # MetricsRegistry
│   ├── logging.py           # StructuredLogger
│   ├── alerting.py          # AlertManager
│   └── exporter.py          # Prometheus format export
├── debug/
│   ├── tools.py             # DebugTools (introspection)
│   ├── inspection.py        # Node/transaction inspection
│   ├── simulation.py        # Failure simulation
│   └── health.py            # Health checks
├── adapters/
│   ├── got.py               # GoT adapter
│   ├── thought.py           # ThoughtGraph adapter
│   ├── synaptic.py          # SynapticMemoryGraph adapter
│   ├── pln.py               # PLNGraph adapter
│   └── slm.py               # TransitionGraph adapter
├── config.py                # Configuration
└── client.py                # CDGClient API
```

---

## Existing Graph Implementation Coverage Matrix

The following matrix documents all existing graph implementations in the codebase and how CDG provides unified storage for each:

### Graph Implementation Inventory

| Graph Class | Location | Purpose | CDG Mapping |
|------------|----------|---------|-------------|
| **ThoughtGraph** | `cortical/reasoning/thought_graph.py:24` | Main Graph of Thought operations | Namespace: `thought`, Adapter: `ThoughtGraphAdapter` |
| **ThoughtGraph Protocol** | `llm_orchestration/protocols.py:198` | Interface definition | CDG implements this protocol |
| **SynapticMemoryGraph** | `cortical/reasoning/prism_got.py:473` | ThoughtGraph with synaptic plasticity | Namespace: `synaptic`, Edge: `SynapticEdge` properties |
| **PLNGraph** | `cortical/reasoning/prism_pln.py:336` | Probabilistic Logic Network | Namespace: `pln`, TruthValue in properties |
| **TransitionGraph** | `cortical/reasoning/prism_slm.py:294` | Token transition graph for LM | Namespace: `slm`, Transitions as weighted edges |
| **GraphWalker** | `cortical/got/graph_walker.py:170` | Visitor pattern traversal | CDG Query API with streaming |
| **GraphWAL** | `cortical/reasoning/graph_persistence.py:576` | Write-ahead logging | CDG WAL built-in |
| **GraphRecovery** | `cortical/reasoning/graph_persistence.py:1214` | Multi-level recovery | CDG Recovery subsystem |
| **HiveNode/HiveEdge** | `cortical/reasoning/prism_slm.py:82,166` | Hebbian learning structures | Namespace: `hive`, Traces in properties |

### Feature Coverage Matrix

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                     FEATURE COVERAGE BY GRAPH IMPLEMENTATION                         │
├──────────────────────┬──────────┬─────────┬─────────┬─────────┬──────────┬──────────┤
│ Feature              │ Thought  │Synaptic │   PLN   │ Transit │  Hive   │   CDG    │
│                      │  Graph   │ Memory  │  Graph  │  Graph  │  Node   │ Unified  │
├──────────────────────┼──────────┼─────────┼─────────┼─────────┼──────────┼──────────┤
│ Node CRUD            │    ✓     │    ✓    │    ✓    │    ✓    │    ✓    │    ✓     │
│ Edge CRUD            │    ✓     │    ✓    │    ✓    │    ✓    │    ✓    │    ✓     │
│ Typed Nodes          │    ✓     │    ✓    │    ✓    │    ✓    │    ✓    │    ✓     │
│ Typed Edges          │    ✓     │    ✓    │    ✓    │    ✓    │    ✓    │    ✓     │
│ Edge Weights         │    ✓     │    ✓    │    ✓    │    ✓    │    ✓    │    ✓     │
│ Bidirectional Edges  │    ✓     │    ✓    │    ✓    │    -    │    -    │    ✓     │
│ BFS/DFS Traversal    │    ✓     │    ✓    │    -    │    -    │    -    │    ✓     │
│ Shortest Path        │    ✓     │    ✓    │    -    │    -    │    -    │    ✓     │
│ Cycle Detection      │    ✓     │    ✓    │    -    │    -    │    -    │    ✓     │
│ Clustering           │    ✓     │    ✓    │    -    │    -    │    -    │    ✓     │
│ Pattern Matching     │    ✓     │    ✓    │    ✓    │    -    │    -    │    ✓     │
│ Activation Traces    │    -     │    ✓    │    -    │    -    │    ✓    │    ✓     │
│ Synaptic Decay       │    -     │    ✓    │    -    │    ✓    │    -    │    ✓     │
│ Hebbian Learning     │    -     │    ✓    │    -    │    ✓    │    ✓    │    ✓     │
│ Probabilistic Truth  │    -     │    -    │    ✓    │    -    │    -    │    ✓     │
│ Inference Rules      │    -     │    -    │    ✓    │    -    │    -    │    ✓     │
│ Context Windows      │    -     │    -    │    -    │    ✓    │    -    │    ✓     │
│ Spreading Activation │    -     │    -    │    -    │    ✓    │    -    │    ✓     │
│ Lateral Inhibition   │    -     │    -    │    -    │    ✓    │    -    │    ✓     │
│ WAL Persistence      │    ✓     │    -    │    ✓    │    ✓    │    -    │    ✓     │
│ Snapshot Recovery    │    ✓     │    -    │    -    │    -    │    -    │    ✓     │
│ Distributed Storage  │    -     │    -    │    -    │    -    │    -    │    ✓     │
│ ACID Transactions    │    -     │    -    │    -    │    -    │    -    │    ✓     │
│ Partition Scaling    │    -     │    -    │    -    │    -    │    -    │    ✓     │
└──────────────────────┴──────────┴─────────┴─────────┴─────────┴──────────┴──────────┘
```

### Detailed Mapping Specifications

#### 1. ThoughtGraph → CDG

```python
# ThoughtGraph node types map directly to CDG node_type
NodeType.QUESTION    → DistributedNode(namespace="thought", node_type="question")
NodeType.HYPOTHESIS  → DistributedNode(namespace="thought", node_type="hypothesis")
NodeType.EVIDENCE    → DistributedNode(namespace="thought", node_type="evidence")
NodeType.CONCLUSION  → DistributedNode(namespace="thought", node_type="conclusion")
NodeType.CONCEPT     → DistributedNode(namespace="thought", node_type="concept")

# ThoughtGraph edge types
EdgeType.SUPPORTS    → DistributedEdge(namespace="thought", edge_type="supports")
EdgeType.REFUTES     → DistributedEdge(namespace="thought", edge_type="refutes")
EdgeType.EXPLORES    → DistributedEdge(namespace="thought", edge_type="explores")
EdgeType.DERIVES     → DistributedEdge(namespace="thought", edge_type="derives")

# ThoughtGraph clusters
ThoughtCluster       → DistributedNode(namespace="thought", node_type="cluster",
                                        properties={"node_ids": [...], "name": "..."})
```

#### 2. SynapticMemoryGraph → CDG

```python
# Synaptic extensions stored in properties
SynapticEdge → DistributedEdge(
    namespace="synaptic",
    properties={
        "activation_count": 42,
        "last_activation_time": "2025-12-31T12:00:00",
        "decay_factor": 0.99,
        "prediction_accuracy": 0.85,
        "prediction_correct": 17,
        "prediction_total": 20
    }
)

# ActivationTrace stored as node metadata
ActivationTrace → DistributedNode.metadata = {
    "activation_trace": {
        "total_activations": 100,
        "history": [...],  # Bounded to max_history
        "max_history": 100
    }
}

# PlasticityRules stored as graph-level configuration
PlasticityRules → CDG partition config or metadata node
```

#### 3. PLNGraph → CDG

```python
# PLN atoms with TruthValue
Atom → DistributedNode(
    namespace="pln",
    node_type="atom",
    content="bird(tweety)",
    properties={
        "predicate": "bird",
        "arguments": ["tweety"],
        "truth_value": {
            "strength": 0.99,
            "confidence": 0.85
        }
    }
)

# PLN implication links
ImplicationLink → DistributedEdge(
    namespace="pln",
    edge_type="implies",
    properties={
        "truth_value": {"strength": 0.85, "confidence": 0.9}
    }
)

# Inference rules executed via CDG stored procedures
deduce(), induce(), abduce() → CDG Query + Transaction
```

#### 4. TransitionGraph → CDG

```python
# Token transitions as edges with context
SynapticTransition → DistributedEdge(
    namespace="slm",
    edge_type="transition",
    source_id="token:quick",
    target_id="token:brown",
    weight=0.7,
    properties={
        "count": 42,
        "decay_rate": 0.99,
        "context": ("the", "quick")  # Context tuple as key
    }
)

# Vocabulary as nodes
Token → DistributedNode(
    namespace="slm",
    node_type="token",
    content="quick",
    properties={"frequency": 100}
)

# Context-based lookup via CDG pattern query
graph.get_transitions(context) → cdg.pattern_match(
    Pattern()
        .edge("transition", context=context)
        .node(node_type="token")
)
```

#### 5. HiveNode/HiveEdge → CDG

```python
# Hebbian Hive nodes with activation state
HiveNode → DistributedNode(
    namespace="hive",
    node_type="hive_node",
    properties={
        "activation": 0.5,
        "trace": 0.8,
        "trace_decay": 0.95,
        "target_activation": 0.05,
        "excitability": 1.0,
        "activation_count": 50,
        "last_activation_step": 1000
    }
)

# STDP-inspired edges
HiveEdge → DistributedEdge(
    namespace="hive",
    edge_type="synapse",
    properties={
        "pre_trace": 0.3,
        "post_trace": 0.7,
        "co_activations": 25,
        "total_observations": 100,
        "learning_rate": 0.01
    }
)
```

### Future Extensibility

CDG is designed to support any future graph-based needs:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         EXTENSIBILITY PATTERNS                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. NEW NODE TYPES                                                       │
│     Just create with new namespace and node_type:                       │
│     DistributedNode(namespace="custom", node_type="my_entity")          │
│                                                                          │
│  2. NEW EDGE TYPES                                                       │
│     Edge types are strings, add any:                                    │
│     DistributedEdge(edge_type="CUSTOM_RELATION")                        │
│                                                                          │
│  3. CUSTOM PROPERTIES                                                    │
│     Properties dict accepts any JSON-serializable data:                 │
│     properties={"custom_field": value, "nested": {"data": [...]}}       │
│                                                                          │
│  4. CUSTOM ALGORITHMS                                                    │
│     CDG query API supports custom traversal:                            │
│     cdg.execute(CustomAlgorithm(graph_walker_spec))                     │
│                                                                          │
│  5. DOMAIN ADAPTERS                                                      │
│     Wrap CDG with domain-specific API:                                  │
│     class KnowledgeBaseAdapter(CDGAdapter): ...                         │
│                                                                          │
│  EXAMPLES OF FUTURE USE CASES:                                          │
│  • Document knowledge graphs                                            │
│  • User behavior graphs                                                  │
│  • Semantic code dependency graphs                                       │
│  • Agent collaboration networks                                          │
│  • Temporal event graphs                                                 │
│  • Multi-modal content graphs (text + code + diagrams)                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Knowledge Worker Universal Needs

CDG supports all knowledge worker graph needs:

| Knowledge Worker Need | CDG Solution |
|----------------------|--------------|
| **Store entities with relationships** | Universal Node + Edge models |
| **Query by type, property, content** | Multi-index query engine |
| **Find patterns in data** | Pattern matching with predicates |
| **Navigate connections** | BFS/DFS with depth control |
| **Discover shortest paths** | Dijkstra with weight support |
| **Cluster related items** | Built-in clustering + partition strategies |
| **Track changes over time** | MVCC versioning + temporal queries |
| **Learn from usage patterns** | Synaptic property tracking |
| **Reason with uncertainty** | TruthValue properties support |
| **Scale with data growth** | Horizontal partition scaling |
| **Survive failures** | WAL + snapshot + multi-level recovery |
| **Search semantically** | Embedding field + vector similarity |
| **Full-text search** | Inverted index on content |

---

## Conclusion

The Cortical Distributed Graph provides a unified foundation for all graph storage needs in the system. By building every component from first principles, we maintain complete sovereignty over the implementation while achieving service provider-grade performance targets.

The coverage matrix above demonstrates that CDG can unify all existing graph implementations (ThoughtGraph, SynapticMemoryGraph, PLNGraph, TransitionGraph, HiveNode/HiveEdge) through adapters, enabling a gradual migration path while preserving backward compatibility.

**Key Guarantees:**
- All existing graph operations are supported
- All synaptic/learning features are preserved
- All probabilistic reasoning capabilities are maintained
- Performance contracts meet service provider requirements
- Future extensibility is unlimited through flexible properties

---

*Built with reverence for the craft. Every line is ours.*
