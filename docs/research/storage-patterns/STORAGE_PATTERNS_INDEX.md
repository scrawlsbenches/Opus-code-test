# Storage Tier Patterns: Complete Research Index

## Overview

This collection provides comprehensive research on storage tier patterns for file-based database systems, with specific focus on Python implementations using **git as slow storage** and **local files/memory as fast storage**.

The research is organized across 4 documents totaling **3,240 lines** of detailed patterns, code examples, and real-world implementations from your Cortical Text Processor and Graph of Thought systems.

---

## Documents

### 1. STORAGE_TIER_PATTERNS_RESEARCH.md (1,665 lines)

**Comprehensive theoretical and practical research** with academic rigor.

**Covers:**
- §1: Hot/Warm/Cold Storage Tiers
  - Tier definitions and characteristics
  - Promotion/demotion policies
  - Real examples from Cortical

- §2: Write-Behind (Write-Back) Caching
  - Time-based, size-based, and batch-based flush strategies
  - Durability with WAL (Write-Ahead Logging)
  - Consistency models (eventual vs strong)
  - GoT Manager real-world implementation

- §3: Read-Through Caching
  - LRU (Least Recently Used) eviction
  - Thread-safe cache design
  - Tiered read-through with fallback
  - LFU, TTL, and other eviction policies
  - Query expansion cache in Cortical

- §4: Storage Location Abstraction
  - Abstract StorageBackend interface
  - Memory, Disk, and Git implementations
  - Multi-tier unified abstraction
  - Cortical StateWriter/StateLoader pattern

- §5: Sync Points and Durability Guarantees
  - Durability levels (None, Loose, Sync, Committed)
  - fsync and directory sync requirements
  - Write-Ahead Logging for ACID
  - Transaction boundaries and consistency
  - GraphWAL and GitAutoCommitter patterns

- §6: Practical Implementation (Git + Local FS)
  - Reference architecture diagram
  - Minimal viable implementation (<200 lines)
  - Integration with existing systems

- §7: Performance Characteristics
  - Latency comparison table
  - Throughput with write-behind
  - Space amplification

- §8: Failure Modes and Recovery
  - Crash scenarios and impacts
  - Recovery strategies
  - Prevention techniques

**Sources:** 30+ research links from Azure, Elasticsearch, Redis, AWS, LWN, MIT, Stanford

---

### 2. STORAGE_PATTERNS_FOR_GIT_SYSTEMS.md (1,043 lines)

**Production-ready patterns** specifically for git-based systems.

**Covers:**
- Pattern 1: Multi-Tier Triage System
  - Classify data by criticality (low/medium/high/critical)
  - Route writes based on importance
  - Example: GoT tasks vs query cache

- Pattern 2: Debounced Git Commits
  - Intelligent batching strategy
  - Time bounds + size bounds
  - CommitMetrics for optimization
  - Tuning recommendations

- Pattern 3: Chunk-Based Git Storage
  - Append-only timestamp chunks
  - No merge conflicts
  - Chunk compaction (like `git gc`)
  - Rebuild corpus from chunks

- Pattern 4: Snapshot + WAL for Crash Recovery
  - Fast recovery path (0.1s)
  - Slow fallback (rebuild from chunks)
  - SnapshotManager and RecoveryManager

- Pattern 5: Safe Remote Push with Verification
  - Verify before push
  - Retry on network failure
  - Check consistency on remote

- Pattern 6: Consistency Guarantees and Transactions
  - Transaction isolation levels
  - All-or-nothing semantics
  - Conflict detection
  - Example: Atomic node transfers

- Complete Integration: ProductionStorageStack
  - Full system combining all 6 patterns
  - Startup with crash recovery
  - Create task with full durability
  - Graceful shutdown with backup

**Specific to your system:**
- GoTManager integration
- GraphWAL and GitAutoCommitter
- Chunk-based corpus storage
- Real task creation workflow

---

### 3. STORAGE_PATTERNS_QUICK_REFERENCE.md (532 lines)

**Fast lookup guide** for busy developers.

**Covers:**
- Pattern selection flowchart
- Code snippets at a glance
  - LRU cache
  - Write-behind
  - Read-through
  - WAL
  - Debounced commits
  - Snapshots
  - Chunking

- Decision matrix by use case
- Configuration cheat sheets (light/medium/heavy)
- Failure scenarios and recovery
- Real-world numbers for Cortical
- When to use each pattern
- Testing patterns
- Monitoring and observability
- Debugging tips
- Link to full guides

**Perfect for:** Implementing a new storage system, troubleshooting, configuration tuning

---

## Quick Navigation

### By Your Question

**"How do I implement hot/warm/cold tiers?"**
→ STORAGE_TIER_PATTERNS_RESEARCH.md §1

**"How do I buffer writes but keep them safe?"**
→ STORAGE_TIER_PATTERNS_RESEARCH.md §2

**"How do I avoid keeping everything in memory?"**
→ STORAGE_TIER_PATTERNS_RESEARCH.md §3

**"How do I abstract storage backends?"**
→ STORAGE_TIER_PATTERNS_RESEARCH.md §4

**"What durability guarantees do I need?"**
→ STORAGE_TIER_PATTERNS_RESEARCH.md §5

**"How do I implement this in production?"**
→ STORAGE_PATTERNS_FOR_GIT_SYSTEMS.md (all patterns)

**"I just need the code!"**
→ STORAGE_PATTERNS_QUICK_REFERENCE.md

---

### By Implementation Pattern

| Pattern | Research | Production | Quick Ref |
|---------|----------|-----------|-----------|
| Hot/Warm/Cold | §1 | Pattern 1 (Triage) | ✓ |
| Write-Behind | §2 | Pattern 2 (Commit) | ✓ |
| Read-Through | §3 | Pattern 3 (Chunk) | ✓ |
| Abstraction | §4 | Pattern 6 (Tx) | ✓ |
| Durability | §5 | Pattern 4 (WAL) | ✓ |
| Git Integration | §6 | Pattern 2, 3, 5 | ✓ |
| Failure Recovery | §8 | Pattern 4 | ✓ |

---

### By Implementation Complexity

**Simple (Quick Reference only):**
- LRU Cache
- Write buffering
- Read-through

**Intermediate (Quick Reference + Pattern doc):**
- Write-behind + fsync
- Debounced commits
- Chunked storage

**Advanced (All documents):**
- Transactions + isolation
- Crash recovery
- Multi-tier system
- Remote backup

---

## How to Use This Research

### Option 1: Learn the Theory First
1. Start with **STORAGE_TIER_PATTERNS_RESEARCH.md**
2. Read sections in order
3. Follow the code examples
4. Check the sources for deep dives
5. Implement based on your needs

### Option 2: Implement ASAP
1. Go to **STORAGE_PATTERNS_QUICK_REFERENCE.md**
2. Find your use case in decision matrix
3. Copy the code snippet
4. Reference full guides as needed
5. Test using the testing patterns

### Option 3: Production Integration
1. Read **STORAGE_PATTERNS_FOR_GIT_SYSTEMS.md**
2. Understand all 6 patterns
3. Choose which patterns you need
4. Implement ProductionStorageStack
5. Configure using cheat sheet
6. Monitor using metrics

---

## Key Insights from Your Codebase

### What You're Already Doing Well

**1. Cortical Text Processor**
- ✓ Memory (hot) + disk (warm) + git (cold) tiers
- ✓ Chunk-based JSON storage (git-friendly)
- ✓ Query expansion cache with LRU eviction
- ✓ StateWriter/StateLoader for abstraction

**2. Graph of Thought (GoT)**
- ✓ Write-Ahead Logging (WAL) for durability
- ✓ Transaction semantics (all-or-nothing)
- ✓ Task criticality classification
- ✓ Debounced auto-commit via GitAutoCommitter

**3. Graph Persistence**
- ✓ GraphWAL for crash recovery
- ✓ Snapshot + WAL hybrid approach
- ✓ Protected branch detection
- ✓ Safe push with verification

### Recommendations for Enhancement

1. **Triage Router** (Pattern 1)
   - Currently implicit in GoTManager
   - Make explicit: classify data by criticality
   - Route to different durability levels

2. **Chunk Compaction** (Pattern 3)
   - Implement periodic compaction
   - Prevent infinite git history growth
   - Like your documented corpus_chunks pattern

3. **Snapshot Metrics** (Pattern 4)
   - Track recovery times
   - Monitor snapshot freshness
   - Alert if old (>N hours)

4. **Transaction Isolation** (Pattern 6)
   - Currently serializable
   - Consider weaker isolation for read-heavy
   - Document isolation guarantees

5. **Chaos Testing**
   - Simulate crashes mid-write
   - Test WAL recovery
   - Verify snapshot consistency

---

## Architecture Diagrams

### Three-Tier Flow
```
Application Layer
    ↓
HOT TIER (Memory, <1ms)
├─ In-memory dict
├─ LRU cache
└─ Current session

WRITE-BEHIND BUFFER
├─ Buffered writes
└─ Debounced flush (5s)
    ↓
WARM TIER (Local Disk, 1-10ms)
├─ JSON files
├─ WAL entries
└─ Snapshots

PERIODIC COMMIT
├─ git add
├─ git commit (debounced)
└─ fsync
    ↓
COLD TIER (Git, 100-500ms)
├─ .git/objects/ (history)
├─ Reasoning graphs
└─ Archives
    ↓
REMOTE BACKUP
├─ git push
└─ Replicated copies
```

### Recovery Paths
```
Normal Operation
    ├─ Read: hot → warm → cold
    ├─ Write: hot + buffer
    └─ Flush: async to warm/cold

After Crash/Restart
    ├─ Fast path:
    │  ├─ Load latest snapshot (0.05s)
    │  ├─ Replay WAL since snapshot (0.01s)
    │  └─ Resume operations (0.1s total)
    │
    └─ Fallback path:
       ├─ Rebuild from chunks (5s)
       ├─ Verify checksums
       ├─ Create fresh snapshot
       └─ Resume operations
```

---

## Configuration Templates

### Development (Can Lose Data)
```python
HOT_CAPACITY_MB = 256
WRITE_BUFFER_TIME_SEC = 30
GIT_COMMIT_DEBOUNCE = 10
DURABILITY = 'none'
```

### Staging (Important Data)
```python
HOT_CAPACITY_MB = 1024
WRITE_BUFFER_TIME_SEC = 5
GIT_COMMIT_DEBOUNCE = 5
DURABILITY = 'sync + fsync'
SNAPSHOTS_EVERY_N_OPS = 1000
BACKUP_AFTER_COMMITS = 10
```

### Production (Critical Data)
```python
HOT_CAPACITY_MB = 2048
WRITE_BUFFER_TIME_SEC = 1
GIT_COMMIT_DEBOUNCE = 'immediate'
DURABILITY = 'committed'
SNAPSHOTS_EVERY_N_OPS = 100
BACKUP_AFTER_COMMITS = 1
REMOTE_REPLICATION = 2
```

---

## Testing Checklist

- [ ] LRU cache hit rate >90%
- [ ] Write-behind latency <5s
- [ ] Git commits batch 100-1000 changes
- [ ] Recovery time <1s (snapshot path)
- [ ] WAL replay 100% accurate
- [ ] fsync works on your OS
- [ ] git push retries on network error
- [ ] Transaction isolation prevents corruption
- [ ] Chunk compaction reduces disk usage
- [ ] Snapshot compression works
- [ ] Remote backup validated
- [ ] Crash recovery tested

---

## Performance Targets

| Operation | Target | Measurement |
|-----------|--------|-------------|
| Hot read | <1ms | Cache hit |
| Hot write | <1ms | Hot tier only |
| Buffered write | 0-5ms avg | Deferred flush |
| Warm read | 1-10ms | Disk I/O |
| Cold read | 100-500ms | Git checkout |
| Commit latency | 100-500ms | git commit |
| Recovery time (restart) | <1s | Snapshot + WAL |
| Recovery time (rebuild) | <10s | From chunks |

---

## Monitoring Recommendations

### Per-Tier Metrics
```python
# Hot tier
hot_cache.hit_rate()          # Target: >90%
hot_cache.size_mb()           # Alert if >90% capacity
hot_cache.eviction_rate       # Alert if >10/sec

# Warm tier
warm_disk_usage_mb()          # Alert if >80% capacity
warm_read_latency_ms()        # Alert if >100ms
warm_write_latency_ms()       # Alert if >50ms

# Cold tier (Git)
git_repo_size_mb()            # Alert if growing >1GB/week
commits_per_day()             # Alert if <10 or >1000
push_failures_count()         # Alert if >0
```

### System Health
```python
recovery_time_ms()            # Alert if >5000ms
wal_replay_accuracy()         # Alert if <100%
snapshot_freshness_hours()    # Alert if >24h old
transaction_conflicts()       # Alert if growing
```

---

## Further Research

### Academic Papers
- "The Design and Implementation of a Write-Ahead Logging DBMS"
- "Crashmonkey: An OS-agnostic Synthesizing File System Workload"
- "Durability-Aware Semantics of LevelDB and RocksDB"

### Industry Best Practices
- PostgreSQL: WAL + fsync + MVCC
- RocksDB: LSM trees + WAL + compression
- Git: Object storage + packed refs
- Redis: RDB snapshots + AOF (Append-Only File)

### Tools for Measurement
- `strace`: Trace system calls (fsync, write)
- `fio`: Benchmark disk I/O
- `pyperf`: Benchmark Python code
- `py-spy`: Profiling hot spots

---

## Document Statistics

```
Total Lines:       3,240
Code Examples:     50+
Performance Tests: 10+
Configuration Templates: 15+
References:        30+ curated sources
Real Implementations: 8 from your codebase
Recovery Scenarios: 8 with solutions
```

---

## Usage Agreement

These patterns are designed for the Cortical Text Processor and Graph of Thought systems but are general enough to apply to any Python file-based database.

**Attribution:** Based on research from Azure, Elasticsearch, Redis, AWS, PostgreSQL, and academic literature. See individual documents for sources.

---

*Last Updated: December 2025*
*Comprehensive research compiled from your codebase and leading industry sources*
