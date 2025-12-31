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
│   ├── btree.py             # B-tree index
│   ├── hash.py              # Hash index
│   ├── bloom.py             # Bloom filter
│   └── inverted.py          # Inverted index (full-text)
├── partition/
│   ├── manager.py           # PartitionManager
│   ├── router.py            # Query routing
│   └── hasher.py            # Consistent hashing
├── transaction/
│   ├── local.py             # Single-partition transactions
│   ├── distributed.py       # 2PC coordinator
│   └── mvcc.py              # Multi-version concurrency
├── query/
│   ├── parser.py            # Query parsing
│   ├── planner.py           # Query planning
│   ├── executor.py          # Query execution
│   ├── pattern.py           # Pattern matching
│   └── path.py              # Path finding
├── adapters/
│   ├── got.py               # GoT adapter
│   ├── thought.py           # ThoughtGraph adapter
│   └── knowledge.py         # Knowledge graph adapter
├── config.py                # Configuration
└── client.py                # CDGClient API
```

---

## Conclusion

The Cortical Distributed Graph provides a unified foundation for all graph storage needs in the system. By building every component from first principles, we maintain complete sovereignty over the implementation while achieving service provider-grade performance targets.

All existing graph implementations (GoT, ThoughtGraph, Knowledge Graph) can adopt CDG through adapters, enabling a gradual migration path while preserving backward compatibility.

---

*Built with reverence for the craft. Every line is ours.*
