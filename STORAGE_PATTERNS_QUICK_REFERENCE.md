# Storage Tier Patterns: Quick Reference

## Pattern Selection Flowchart

```
Need to store data?
    ↓
    ├─ HOT only?
    │  ├─ Cache/temp data
    │  ├─ Session state
    │  └─ Config overrides
    │      → Use: Dict, OrderedDict, lru_cache
    │
    ├─ HOT + WARM?
    │  ├─ Medium data (doc indices)
    │  ├─ Intermediate state
    │  └─ Analytics data
    │      → Use: LRU cache + disk JSON
    │
    └─ HOT + WARM + COLD?
       ├─ User data, tasks, graphs
       ├─ Mission-critical state
       └─ Audit trail
           → Use: Write-behind + WAL + Git

Need durability?
    ↓
    ├─ Optional (eventual consistency OK)
    │  → Use: Write-behind cache, debounced commits
    │
    ├─ Important (should survive restart)
    │  → Use: Write-ahead log, fsync
    │
    └─ Critical (survive power loss)
        → Use: WAL + fsync + git + remote
```

## Code Patterns at a Glance

### 1. LRU Cache (Hot Tier)

**Fastest possible reads:**
```python
from collections import OrderedDict

class LRU(OrderedDict):
    def __init__(self, max_size=1000):
        super().__init__()
        self.max_size = max_size

    def get(self, key):
        if key in self: self.move_to_end(key)
        return super().get(key)

    def set(self, key, value):
        if key in self: self.move_to_end(key)
        else:
            self[key] = value
            if len(self) > self.max_size: self.popitem(last=False)
```

**Performance:** <1ms lookup, O(1) operations

---

### 2. Write-Behind Cache (Buffered Writes)

**Fast writes with eventual durability:**
```python
class WriteBuffer:
    def __init__(self, flush_interval=5):
        self.buffer = {}
        self.flush_interval = flush_interval

    def write(self, key, value):
        # Accept immediately (hot tier)
        self.buffer[key] = value

        # Schedule async flush
        Timer(self.flush_interval, self.flush).start()

    def flush(self):
        # Persist to disk/git asynchronously
        persist_async(self.buffer)
        self.buffer.clear()
```

**Performance:** 0ms accept, flush every N seconds

---

### 3. Read-Through Cache (Lazy Loading)

**Automatic tiering on read:**
```python
def read(key):
    # Try hot
    if key in hot_cache: return hot_cache[key]

    # Try warm
    if path.exists(warm_path / f"{key}.json"):
        with open(...) as f: value = json.load(f)
        hot_cache[key] = value  # Promote
        return value

    # Try cold
    result = subprocess.run(['git', 'show', f'HEAD:{key}.json'])
    if result.returncode == 0:
        value = json.loads(result.stdout)
        hot_cache[key] = value  # Promote
        return value

    return None
```

**Performance:** <1ms (hot hit), 1-10ms (warm hit), 100-500ms (cold hit)

---

### 4. Write-Ahead Log (Durability)

**Crash-safe writes:**
```python
def write_durable(key, value):
    # STEP 1: Log to WAL (durable)
    wal_file.write(json.dumps({'op': 'write', 'key': key, 'value': value}) + '\n')
    wal_file.flush()
    os.fsync(wal_file.fileno())

    # STEP 2: Apply in memory (safe now)
    hot_cache[key] = value

    return True

def recover_after_crash():
    for line in open('wal.log'):
        entry = json.loads(line)
        if entry['op'] == 'write':
            hot_cache[entry['key']] = entry['value']
```

**Durability guarantee:** Data survives process crash

---

### 5. Debounced Git Commits

**Batch commits automatically:**
```python
class Committer:
    def __init__(self, debounce_sec=5, max_age_sec=60):
        self.debounce_sec = debounce_sec
        self.max_age_sec = max_age_sec
        self.changes = set()
        self.first_change_time = None

    def mark_changed(self, key):
        self.changes.add(key)
        if self.first_change_time is None:
            self.first_change_time = time.time()

        age = time.time() - self.first_change_time

        if age >= self.max_age_sec:
            self._commit()  # Force commit
        else:
            Timer(self.debounce_sec, self._commit).start()

    def _commit(self):
        # Write files, git add, git commit
        ...
```

**Pattern:** Multiple writes → 1 git commit

---

### 6. Snapshot + WAL (Fast Recovery)

**Best of both worlds:**
```python
class Recovery:
    def recover(self):
        # Fast path: load snapshot + WAL
        latest_snapshot = find_latest_snapshot()
        if latest_snapshot:
            state = load_snapshot(latest_snapshot)
            for wal_entry in get_wal_after(latest_snapshot):
                apply_entry(state, wal_entry)
            return state

        # Slow path: rebuild from chunks
        state = rebuild_from_chunks()
        save_snapshot(state)
        return state
```

**Recovery time:** 0.1s (snapshot path) vs 5s (rebuild path)

---

### 7. Chunk-Based Storage (Git-Friendly)

**Append-only chunks for git:**
```python
"""
corpus_chunks/
├── 2025-12-10_21-53-45_a1b2.json  # +50 docs
├── 2025-12-10_22-15-30_c3d4.json  # +30 docs
└── 2025-12-10_23-00-00_e5f6.json  # +10 docs

Each chunk is ~20KB, total git history small
vs single 100MB corpus_dev.json
"""

def flush_chunk():
    chunk = {
        'timestamp': now.isoformat(),
        'session_id': uuid.uuid4().hex,
        'operations': [
            {'op': 'add', 'doc_id': 'x', 'content': '...'},
            {'op': 'modify', 'doc_id': 'y', 'content': '...'},
        ]
    }
    save_json(f"corpus_chunks/{timestamp}_{session}.json", chunk)
    git_commit(f"chunk: {len(operations)} ops")
```

**Benefit:** No merge conflicts, clean git history

---

## Decision Matrix

Choose based on your needs:

| Need | Pattern | Time Cost | Space Cost | Complexity |
|------|---------|-----------|-----------|-----------|
| Speed | LRU cache | <1ms | Medium | Low |
| High throughput | Write-behind | 0ms + async flush | High | Medium |
| Crash recovery | WAL | 1-10ms | High | Medium |
| Durability | fsync + git | 50-500ms | Very high | High |
| Git history | Chunks | 1-10ms | Low | Medium |
| Fast recovery | Snapshots | 0.1s | High | Medium |
| ACID | Transactions | 10-100ms | High | High |

---

## Configuration Cheat Sheet

```python
# LIGHT (development, can lose data):
HOT_CAPACITY = 256 MB
WRITE_BUFFER_FLUSH = 30 sec
GIT_COMMIT = debounced (10 sec)
DURABILITY = none

# MEDIUM (production, can survive restart):
HOT_CAPACITY = 1 GB
WRITE_BUFFER_FLUSH = 5 sec
GIT_COMMIT = debounced (5 sec)
DURABILITY = WAL + fsync
SNAPSHOTS = every 1000 ops

# HEAVY (critical, survive anything):
HOT_CAPACITY = 2+ GB
WRITE_BUFFER_FLUSH = 1 sec
GIT_COMMIT = immediate + debounced
DURABILITY = WAL + fsync + git
SNAPSHOTS = every 100 ops
REMOTE_BACKUP = after every N commits
REPLICATION = 2+ replicas
```

---

## Common Failure Scenarios and Recovery

| Failure | Impact | Recovery Time | Prevention |
|---------|--------|---------------|-----------|
| Process crash | Lose hot | 0.1s (snapshot) | WAL + snapshots |
| Disk crash | Lose warm/cold | Manual restore | Remote backup |
| Git corruption | Lose cold | Manual restore | Remote backup + fsync |
| Unflushed write | Data loss | N/A | WAL or transaction log |
| Concurrent corruption | Invalid state | Depends | Transactions + locking |

---

## Real-World Numbers

### Your Cortical System

```
Hot tier (memory):       256 MB  → 1000 graphs
Warm tier (disk):        2 GB    → Full state
Cold tier (git):         500 MB  → History + backups

Write latency:           <1ms    (hot) → 5-10ms (buffer) → 50-500ms (git)
Read latency:            <1ms    (hot) → 1-10ms (warm) → 100-500ms (cold)
Recovery time (restart): 0.5s    (snapshots) vs 10s (rebuild)
```

### Typical Workload

```
100 graphs, each ~1MB:
  Hot cache:    256 MB (all 100 graphs fit)
  Write throughput: 10K updates/sec (buffered) → 100 commits/sec (git)
  Commit batching: ~100 changes per commit

Large corpus analysis (1GB state):
  Hot cache:    Not feasible (only 256 MB)
  Warm cache:   2 GB (fits on SSD)
  Cold storage: Git (incremental backups)
  Recovery:     Load warm + replay WAL (0.5s) vs rebuild (5s)
```

---

## When to Use Each Pattern

### Use LRU Cache When:
- ✅ Data is read-heavy
- ✅ Memory is plentiful
- ✅ Data loss is acceptable
- ❌ Need durability

### Use Write-Behind When:
- ✅ Write-heavy workload
- ✅ Can tolerate consistency lag
- ✅ Performance critical
- ❌ Need immediate durability

### Use WAL When:
- ✅ Can't afford data loss
- ✅ Process crashes are possible
- ✅ Rebuilding state is slow
- ❌ Simple systems (overkill)

### Use Git + fsync When:
- ✅ Mission-critical data
- ✅ Audit trail required
- ✅ Distributed backups needed
- ❌ High-frequency writes (too slow)

### Use Snapshots When:
- ✅ Recovery time matters
- ✅ State is large
- ✅ Crashes are frequent
- ❌ Simple systems (overkill)

### Use Transactions When:
- ✅ Multiple related updates
- ✅ Concurrent access likely
- ✅ Consistency critical
- ❌ Single-threaded or simple cases

### Use Chunking When:
- ✅ Using git as storage
- ✅ History grows continuously
- ✅ Merge conflicts a problem
- ❌ Static state (not append-only)

---

## Testing Your Patterns

### Test Hot Tier Performance
```python
import time
cache = LRU(max_size=1000)

# Add 1000 items
start = time.perf_counter()
for i in range(1000):
    cache.set(f'key:{i}', f'value {i}')
elapsed = time.perf_counter() - start
print(f"Writes: {elapsed/1000*1000:.2f}ms per item")

# Read with working set
start = time.perf_counter()
for i in range(10000):
    cache.get(f'key:{i % 1000}')
elapsed = time.perf_counter() - start
print(f"Reads: {elapsed/10000*1000000:.2f}µs per item")
```

Expected: <10µs reads, <1ms writes

### Test Write-Behind Batching
```python
buffer = WriteBuffer(flush_interval=5)

# Rapid writes
start = time.perf_counter()
for i in range(1000):
    buffer.write(f'key:{i}', f'value {i}')
elapsed = time.perf_counter() - start
print(f"Buffered writes: {elapsed/1000*1000:.2f}µs per write")

# Wait for flush
time.sleep(6)
```

Expected: <10µs buffered, 1 git commit for 1000 writes

### Test Crash Recovery
```python
# Write with WAL
for i in range(100):
    write_durable(f'key:{i}', f'value {i}')

# Simulate crash
hot_cache.clear()

# Recover
recover_from_wal()

# Verify
assert len(hot_cache) == 100
```

Expected: 100% recovery, <100ms recovery time

---

## Monitoring Patterns

### LRU Cache Hit Rate
```python
hit_rate = cache.hits / (cache.hits + cache.misses)
print(f"Hit rate: {hit_rate:.1%}")  # Target: >90%
```

### Write-Behind Latency
```python
batch_age = time.time() - first_write_time
print(f"Oldest unbuffered write: {batch_age:.1f}s")  # Target: <5s
```

### Commit Batch Size
```python
avg_batch_size = total_changes / num_commits
print(f"Changes per commit: {avg_batch_size:.0f}")  # Target: 100-1000
```

### Recovery Time
```python
start = time.perf_counter()
state = storage.recover()
elapsed = time.perf_counter() - start
print(f"Recovery time: {elapsed:.2f}s")  # Target: <1s
```

### Space Utilization
```python
hot_usage = sum(sys.getsizeof(v) for v in hot_cache.values())
warm_usage = sum(f.stat().st_size for f in warm_path.glob('*.json'))
cold_usage = repo.git_size()

print(f"Hot: {hot_usage/1024/1024:.0f} MB / {HOT_CAPACITY}")
print(f"Warm: {warm_usage/1024/1024:.0f} MB")
print(f"Cold: {cold_usage/1024/1024:.0f} MB")
```

---

## Debugging Tips

**Cache not working?**
```python
# Check hit rate
print(cache.hit_rate())  # Should be >80% for cached workloads

# Check eviction
print(f"Evicted: {cache.evicted_count}")

# Check size
print(f"Size: {len(cache)} items, {cache.size_bytes()} bytes")
```

**Writes not persisting?**
```python
# Check write buffer
print(f"Buffered: {len(write_buffer)} items")

# Check fsync
import os
fd = os.open('test.txt', os.O_RDONLY)
os.fsync(fd)  # If this fails, fsync is broken
os.close(fd)
```

**Recovery broken?**
```python
# Check WAL
with open('wal.log') as f:
    for line in f:
        entry = json.loads(line)
        print(entry)  # Each entry should be valid JSON

# Check snapshots
import subprocess
result = subprocess.run(['ls', '-lh', 'snapshots/'])
print(result.stdout)  # Should have recent snapshots
```

**Git commits not happening?**
```python
# Check git status
subprocess.run(['git', 'status'])

# Check committer state
print(f"Pending changes: {len(committer.changes)}")
print(f"Time since first change: {committer.age_seconds()}s")

# Force commit
committer._commit_batch()
subprocess.run(['git', 'log', '--oneline', '-5'])
```

---

## Further Reading

See full guides:
- `STORAGE_TIER_PATTERNS_RESEARCH.md` - Comprehensive research with web sources
- `STORAGE_PATTERNS_FOR_GIT_SYSTEMS.md` - Production patterns and integration

---

*Last Updated: December 2025*
