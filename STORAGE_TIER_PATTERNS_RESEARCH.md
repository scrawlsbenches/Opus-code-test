# Storage Tier Patterns Research

## Overview

This document provides comprehensive patterns for building file-based database systems with multi-tier storage, focusing on Python implementations using git as slow storage and local files/memory as fast storage.

---

## 1. Hot/Warm/Cold Storage Tiers

### 1.1 Tier Definition and Characteristics

| Tier | Access Frequency | Performance | Cost | Example Usage |
|------|------------------|-------------|------|----------------|
| **Hot** | Frequent (every operation) | Ultra-fast (memory/NVMe) | High | Active reasoning graphs, current reasoning state |
| **Warm** | Occasional (hours/days) | Medium (local SSD) | Medium | Recently used documents, snapshots, WAL files |
| **Cold** | Rare (weeks/months) | Slow (network/archive) | Low | Old versions, archived reasoning, git history |

### 1.2 Tier Implementation Architecture

```python
class StorageTierManager:
    """
    Manages data movement across storage tiers.

    Tier promotion/demotion is based on:
    - Access frequency (hits per unit time)
    - Recency of access (last accessed time)
    - Configured time thresholds
    - Storage capacity constraints
    """

    def __init__(self,
                 hot_capacity_mb: int = 256,
                 warm_capacity_mb: int = 2048,
                 cold_path: str = ".git"):
        """
        Initialize three-tier storage.

        Args:
            hot_capacity_mb: Max size for memory cache
            warm_capacity_mb: Max size for local disk cache
            cold_path: Path to slow storage (git, remote, archive)
        """
        self.hot_storage = {}  # In-memory cache
        self.hot_capacity = hot_capacity_mb * 1024 * 1024
        self.hot_size = 0

        self.warm_storage = Path(warm_path or "corpus_warm/")
        self.warm_capacity = warm_capacity_mb * 1024 * 1024
        self.cold_path = Path(cold_path)

        # Access tracking for promotion/demotion
        self.access_counts = {}
        self.last_access_time = {}

    def promote_to_hot(self, key: str, value: Any, size_bytes: int):
        """
        Promote data to hot (memory) tier.

        Triggers demotion if at capacity.
        """
        self._ensure_hot_capacity(size_bytes)
        self.hot_storage[key] = value
        self.hot_size += size_bytes
        self._record_access(key)

    def demote_to_warm(self, key: str, value: Any) -> Path:
        """
        Demote data to warm (disk) tier.

        Serializes to JSON, returns path.
        """
        path = self.warm_storage / f"{key}.json"
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(value, f)

        return path

    def archive_to_cold(self, key: str, source_path: Path):
        """
        Archive to cold tier (git).

        Commits to git, possibly with compression.
        """
        # In the actual system, this is handled by GitAutoCommitter
        subprocess.run(['git', 'add', str(source_path)])
        subprocess.run(['git', 'commit', '-m', f'archive: {key}'])
```

### 1.3 Promotion/Demotion Policies

**Time-Based Demotion:**
- Hot → Warm: After 1 hour without access
- Warm → Cold: After 24 hours without access

**Access-Frequency Demotion:**
- Track hits per hour
- Demote items with declining access patterns
- Keep hot items that show sustained usage

**Capacity-Triggered Demotion:**
- When tier reaches capacity, evict LRU items
- Batch demotion to amortize overhead
- Reserve 10% capacity for sudden bursts

### 1.4 Real-World Example: Cortical Text Processor Tiers

The existing codebase implements this pattern:

```
HOT (Memory):
├── ThoughtGraph nodes and edges (active reasoning)
├── Current session state
└── Computed PageRank values

WARM (Local Disk):
├── corpus_dev.json/ (processed document state)
├── reasoning_wal/ (write-ahead log entries)
├── corpus_chunks/ (session-timestamped chunks)
└── snapshots/ (periodic state snapshots)

COLD (Git Repository):
├── .git/objects/ (history of all changes)
├── reasoning/ (committed graph states)
└── samples/memories/ (knowledge archives)
```

---

## 2. Write-Behind (Write-Back) Caching

### 2.1 Pattern Overview

Write-behind caching optimizes write performance by:
1. Accepting writes immediately in fast tier (hot)
2. Buffering writes in a queue
3. Asynchronously flushing to slow tier (warm/cold)
4. Maintaining durability guarantees

```
Application
    ↓
Write Accepted (fast)
    ↓
Buffer in Hot Tier
    ↓
Queue for Background Flush
    ↓
Periodically Flush to Warm/Cold Tier
    ↓
Durability Checkpoint
```

### 2.2 Buffering Strategies

**Option A: Time-Based Flush**
```python
class TimeBasedWriteBuffer:
    """Flush buffer after N seconds of activity or M seconds max."""

    def __init__(self, flush_interval_sec: int = 5, max_age_sec: int = 30):
        self.buffer = {}
        self.flush_interval = flush_interval_sec
        self.max_age = max_age_sec
        self._flush_timer = None
        self._first_write_time = None

    def write(self, key: str, value: Any):
        """Buffer a write."""
        self.buffer[key] = value

        if self._first_write_time is None:
            self._first_write_time = time.time()

        # Schedule flush if not already scheduled
        self._schedule_flush()

    def _schedule_flush(self):
        """Schedule debounced flush."""
        if self._flush_timer is not None:
            self._flush_timer.cancel()

        # Check for max age violation
        age = time.time() - (self._first_write_time or time.time())
        if age >= self.max_age:
            self.flush()  # Immediate flush
        else:
            # Debounced flush after quiet period
            self._flush_timer = Timer(self.flush_interval, self.flush)
            self._flush_timer.start()

    def flush(self):
        """Write buffered data to storage."""
        if not self.buffer:
            return

        data = self.buffer.copy()
        self.buffer.clear()
        self._first_write_time = None

        # Async persistence
        self._persist_async(data)
```

**Option B: Size-Based Flush**
```python
class SizeBasedWriteBuffer:
    """Flush buffer when reaching byte threshold."""

    def __init__(self, max_buffer_bytes: int = 1024 * 1024):
        self.buffer = {}
        self.buffer_size = 0
        self.max_buffer_bytes = max_buffer_bytes

    def write(self, key: str, value: Any, size_bytes: int):
        """Buffer write, flush if size exceeded."""
        self.buffer[key] = value
        self.buffer_size += size_bytes

        if self.buffer_size >= self.max_buffer_bytes:
            self.flush()

    def flush(self):
        """Persist when buffer is full."""
        data = self.buffer.copy()
        self.buffer.clear()
        self.buffer_size = 0
        self._persist_async(data)
```

**Option C: Batch-Based Flush**
```python
class BatchWriteBuffer:
    """Flush buffer when reaching operation count."""

    def __init__(self, batch_size: int = 100):
        self.buffer = {}
        self.batch_size = batch_size

    def write(self, key: str, value: Any):
        """Buffer write, flush if batch full."""
        self.buffer[key] = value

        if len(self.buffer) >= self.batch_size:
            self.flush()
```

### 2.3 Handling Durability in Write-Behind

**Challenge:** Data loss if cache crashes before flush.

**Solution: Write-Ahead Logging (WAL)**

```python
class DurableWriteBuffer:
    """Write-behind with durability guarantees."""

    def __init__(self, wal_dir: str = "wal"):
        self.buffer = {}
        self.wal = WALWriter(wal_dir)
        self.flush_interval = 5
        self._flush_timer = None

    def write(self, key: str, value: Any):
        """
        Write with durability:
        1. Log to WAL immediately (durable)
        2. Buffer in memory (fast)
        3. Flush asynchronously to storage
        """
        # STEP 1: Write to WAL for crash recovery
        self.wal.append(WALEntry(
            operation='write',
            payload={'key': key, 'value': value}
        ))

        # STEP 2: Buffer in hot storage
        self.buffer[key] = value

        # STEP 3: Schedule async flush
        self._schedule_flush()

    def flush(self):
        """Flush buffer to durable storage."""
        if not self.buffer:
            return

        data = self.buffer.copy()

        # Persist to warm/cold
        self._persist_to_disk(data)

        # Only clear buffer after successful persist
        self.buffer.clear()

        # Truncate WAL now that data is persisted
        self.wal.truncate()

    def recover_after_crash(self):
        """Replay WAL after crash."""
        for entry in self.wal.get_all_entries():
            if entry.operation == 'write':
                key = entry.payload['key']
                value = entry.payload['value']
                self.buffer[key] = value
```

### 2.4 Consistency Model

**Eventual Consistency (Weaker):**
- Acceptable data loss: Yes (if cache crashes before flush)
- Use case: Analytics, indexing, non-critical caches

**Strong Consistency (with WAL):**
- Acceptable data loss: No (WAL replays lost writes)
- Use case: Critical data, transaction logs, reasoning graphs

### 2.5 Real-World Implementation: GoT Manager

The Graph of Thought (GoT) system in the codebase uses write-behind with WAL:

```python
# From cortical/got/manager.py
class GoTManager:
    def __init__(self, ...):
        self.wal = TransactionWAL(...)  # Write-ahead log for durability
        self.buffer = {}  # In-memory write buffer
        self.flush_timer = None

    def create_task(self, title: str, **kwargs):
        """Create task with write-behind."""
        task_id = generate_task_id()

        # STEP 1: Log to WAL (durable)
        self.wal.log_task_created(task_id, title, kwargs)

        # STEP 2: Update in-memory state (fast)
        self.buffer[task_id] = Task(task_id, title, **kwargs)

        # STEP 3: Schedule auto-flush via git
        self._schedule_auto_commit()

        return task_id

    def _schedule_auto_commit(self):
        """Debounced git commit."""
        # See GitAutoCommitter.commit_on_save() for full pattern
        if self._flush_timer:
            self._flush_timer.cancel()

        self._flush_timer = Timer(5, self._flush_to_git)
        self._flush_timer.start()
```

---

## 3. Read-Through Caching

### 3.1 Pattern Overview

Read-through caching optimizes read performance by:
1. Checking fast tier (hot) first
2. If miss, loading from warm/cold tier
3. Caching in hot for future reads
4. Managing eviction when capacity exceeded

```
Read Request
    ↓
Check Hot (Memory)
    ├─ Hit: Return immediately
    └─ Miss: Load from Warm/Cold
           ↓
        Cache in Hot
        ↓
        Evict LRU if needed
```

### 3.2 LRU (Least Recently Used) Eviction

**Core Data Structures:**

```python
from collections import OrderedDict
from typing import Any, Optional

class LRUCache:
    """
    O(1) LRU cache using OrderedDict.

    OrderedDict maintains insertion order. By moving accessed items
    to end, we efficiently track recency.
    """

    def __init__(self, max_size: int = 1000):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get item, updating recency."""
        if key not in self.cache:
            self.misses += 1
            return None

        # Move to end (mark as recently used)
        self.cache.move_to_end(key)
        self.hits += 1
        return self.cache[key]

    def put(self, key: str, value: Any):
        """Put item, evicting LRU if needed."""
        if key in self.cache:
            # Update existing, move to end
            self.cache.move_to_end(key)

        self.cache[key] = value

        # Evict LRU if over capacity
        if len(self.cache) > self.max_size:
            # popitem(last=False) removes oldest (first) item
            evicted_key, evicted_value = self.cache.popitem(last=False)
            return evicted_key, evicted_value

        return None, None

    def hit_rate(self) -> float:
        """Compute cache effectiveness."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
```

**Thread-Safe LRU (with locks):**

```python
from threading import RLock

class ThreadSafeCache:
    """LRU cache safe for concurrent access."""

    def __init__(self, max_size: int = 1000):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.lock = RLock()

    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key not in self.cache:
                return None
            self.cache.move_to_end(key)
            return self.cache[key]

    def put(self, key: str, value: Any):
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = value

            if len(self.cache) > self.max_size:
                self.cache.popitem(last=False)
```

### 3.3 Read-Through with Fallback to Disk

```python
class TieredReadCache:
    """
    Implements lazy-loading read-through caching.

    Hot → Warm → Cold
    """

    def __init__(self,
                 hot_max_size: int = 1000,
                 warm_path: str = "corpus_warm/"):
        self.hot_cache = LRUCache(max_size=hot_max_size)
        self.warm_path = Path(warm_path)

    def get(self, key: str) -> Optional[Any]:
        """
        Get value with automatic tiering.

        1. Check hot (memory)
        2. Check warm (disk)
        3. Check cold (git)
        """
        # TIER 1: Hot (memory)
        value = self.hot_cache.get(key)
        if value is not None:
            return value

        # TIER 2: Warm (disk)
        warm_file = self.warm_path / f"{key}.json"
        if warm_file.exists():
            with open(warm_file) as f:
                value = json.load(f)

            # Promote to hot for next access
            self.hot_cache.put(key, value)
            return value

        # TIER 3: Cold (git)
        value = self._load_from_git(key)
        if value is not None:
            # Promote to hot
            self.hot_cache.put(key, value)
            return value

        return None

    def _load_from_git(self, key: str) -> Optional[Any]:
        """Load from git history."""
        try:
            result = subprocess.run(
                ['git', 'show', f'HEAD:{key}.json'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return json.loads(result.stdout)
        except Exception:
            pass
        return None
```

### 3.4 Eviction Policies Beyond LRU

**LFU (Least Frequently Used):**
```python
class LFUCache:
    """Evicts least frequently accessed items."""

    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.freq = {}  # Key → access count
        self.max_size = max_size

    def get(self, key: str):
        if key not in self.cache:
            return None

        # Increment frequency
        self.freq[key] = self.freq.get(key, 0) + 1
        return self.cache[key]

    def put(self, key: str, value: Any):
        if key in self.cache:
            self.freq[key] += 1
        else:
            self.freq[key] = 1

        self.cache[key] = value

        # Evict least frequent
        if len(self.cache) > self.max_size:
            lfu_key = min(self.freq, key=self.freq.get)
            del self.cache[lfu_key]
            del self.freq[lfu_key]
```

**Time-To-Live (TTL):**
```python
from datetime import datetime, timedelta

class TTLCache:
    """Evicts expired items."""

    def __init__(self, default_ttl_sec: int = 3600):
        self.cache = {}
        self.expiry = {}  # Key → expiration time
        self.ttl = default_ttl_sec

    def get(self, key: str):
        if key not in self.cache:
            return None

        if datetime.now() >= self.expiry.get(key, datetime.now()):
            # Expired, evict
            del self.cache[key]
            del self.expiry[key]
            return None

        return self.cache[key]

    def put(self, key: str, value: Any, ttl_sec: int = None):
        ttl = ttl_sec or self.ttl
        self.cache[key] = value
        self.expiry[key] = datetime.now() + timedelta(seconds=ttl)
```

### 3.5 Real-World Implementation: Cortical Query Expansion Cache

```python
# From cortical/query/expansion.py
class QueryExpansionCache:
    """Cache query expansions with LRU eviction."""

    def __init__(self, max_cached: int = 10000):
        self._cache = OrderedDict()
        self._max_cached = max_cached

    def get_expansion(self, query: str) -> Optional[Dict[str, float]]:
        """Get cached expansion or None."""
        if query not in self._cache:
            return None

        # Move to end (LRU)
        self._cache.move_to_end(query)
        return self._cache[query]

    def cache_expansion(self, query: str, expansion: Dict[str, float]):
        """Cache expansion result."""
        if query in self._cache:
            self._cache.move_to_end(query)

        self._cache[query] = expansion

        # Evict LRU if over capacity
        if len(self._cache) > self._max_cached:
            self._cache.popitem(last=False)
```

---

## 4. Storage Location Abstraction

### 4.1 Abstract Storage Interface

Design principle: Same API regardless of underlying storage.

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class StorageBackend(ABC):
    """Abstract storage interface."""

    @abstractmethod
    def read(self, key: str) -> Optional[Any]:
        """Read value by key."""
        pass

    @abstractmethod
    def write(self, key: str, value: Any) -> bool:
        """Write value, return success."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete value, return success."""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass

    @abstractmethod
    def list_keys(self, prefix: str = "") -> List[str]:
        """List all keys (optionally filtered by prefix)."""
        pass

    @abstractmethod
    def get_size_bytes(self) -> int:
        """Get total storage size."""
        pass
```

### 4.2 Concrete Implementations

**Memory Storage:**
```python
class MemoryStorage(StorageBackend):
    """In-memory storage (fast, volatile)."""

    def __init__(self):
        self.data = {}

    def read(self, key: str) -> Optional[Any]:
        return self.data.get(key)

    def write(self, key: str, value: Any) -> bool:
        self.data[key] = value
        return True

    def delete(self, key: str) -> bool:
        if key in self.data:
            del self.data[key]
            return True
        return False

    def exists(self, key: str) -> bool:
        return key in self.data

    def list_keys(self, prefix: str = "") -> List[str]:
        return [k for k in self.data.keys() if k.startswith(prefix)]

    def get_size_bytes(self) -> int:
        # Rough estimate
        return sum(len(json.dumps(v).encode()) for v in self.data.values())
```

**Disk Storage (JSON):**
```python
class DiskStorage(StorageBackend):
    """File-based storage (warm tier)."""

    def __init__(self, base_path: str = "storage"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_path(self, key: str) -> Path:
        """Safely map key to file path."""
        # Prevent path traversal attacks
        safe_key = key.replace('/', '_').replace('\\', '_')
        return self.base_path / f"{safe_key}.json"

    def read(self, key: str) -> Optional[Any]:
        path = self._get_path(key)
        if not path.exists():
            return None

        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

    def write(self, key: str, value: Any) -> bool:
        path = self._get_path(key)
        try:
            with open(path, 'w') as f:
                json.dump(value, f)
            return True
        except IOError:
            return False

    def delete(self, key: str) -> bool:
        path = self._get_path(key)
        try:
            path.unlink()
            return True
        except FileNotFoundError:
            return False

    def exists(self, key: str) -> bool:
        return self._get_path(key).exists()

    def list_keys(self, prefix: str = "") -> List[str]:
        keys = [f.stem for f in self.base_path.glob("*.json")]
        return [k for k in keys if k.startswith(prefix)]

    def get_size_bytes(self) -> int:
        return sum(f.stat().st_size for f in self.base_path.glob("*.json"))
```

**Git Storage (Cold tier):**
```python
class GitStorage(StorageBackend):
    """Git-based storage (cold/archive tier)."""

    def __init__(self, repo_path: str = ".", data_dir: str = "data"):
        self.repo_path = Path(repo_path)
        self.data_dir = self.repo_path / data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def read(self, key: str) -> Optional[Any]:
        """Read from git (may checkout old version)."""
        path = self.data_dir / f"{key}.json"

        if path.exists():
            try:
                with open(path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return None

        # Try to read from git history
        try:
            result = subprocess.run(
                ['git', 'show', f'HEAD:{self.data_dir.name}/{key}.json'],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return json.loads(result.stdout)
        except Exception:
            pass

        return None

    def write(self, key: str, value: Any) -> bool:
        """Write and commit to git."""
        path = self.data_dir / f"{key}.json"

        try:
            with open(path, 'w') as f:
                json.dump(value, f, indent=2)

            # Commit to git
            subprocess.run(
                ['git', 'add', str(path)],
                cwd=self.repo_path,
                check=True,
                timeout=10
            )
            subprocess.run(
                ['git', 'commit', '-m', f'storage: update {key}'],
                cwd=self.repo_path,
                check=True,
                timeout=10
            )
            return True
        except Exception:
            return False

    def delete(self, key: str) -> bool:
        """Delete and commit removal to git."""
        path = self.data_dir / f"{key}.json"

        try:
            if path.exists():
                path.unlink()
                subprocess.run(
                    ['git', 'add', str(path)],
                    cwd=self.repo_path,
                    check=True,
                    timeout=10
                )
                subprocess.run(
                    ['git', 'commit', '-m', f'storage: delete {key}'],
                    cwd=self.repo_path,
                    check=True,
                    timeout=10
                )
            return True
        except Exception:
            return False

    def exists(self, key: str) -> bool:
        return (self.data_dir / f"{key}.json").exists()

    def list_keys(self, prefix: str = "") -> List[str]:
        keys = [f.stem for f in self.data_dir.glob("*.json")]
        return [k for k in keys if k.startswith(prefix)]

    def get_size_bytes(self) -> int:
        return sum(f.stat().st_size for f in self.data_dir.glob("*.json"))
```

### 4.3 Multi-Tier Abstraction

```python
class TieredStorage(StorageBackend):
    """
    Unified storage interface abstracting hot/warm/cold tiers.

    Routes operations to appropriate backend based on tier.
    """

    def __init__(self,
                 hot: Optional[StorageBackend] = None,
                 warm: Optional[StorageBackend] = None,
                 cold: Optional[StorageBackend] = None):
        self.hot = hot or MemoryStorage()
        self.warm = warm or DiskStorage("corpus_warm/")
        self.cold = cold or GitStorage()

    def read(self, key: str) -> Optional[Any]:
        """Read from fastest available tier."""
        # Try hot first
        value = self.hot.read(key)
        if value is not None:
            return value

        # Try warm
        value = self.warm.read(key)
        if value is not None:
            # Promote to hot
            self.hot.write(key, value)
            return value

        # Try cold
        value = self.cold.read(key)
        if value is not None:
            # Promote to hot
            self.hot.write(key, value)
            return value

        return None

    def write(self, key: str, value: Any) -> bool:
        """Write to hot, schedule promotion to warm/cold."""
        # Always write to hot
        self.hot.write(key, value)

        # Asynchronously write to warm
        threading.Thread(
            target=self.warm.write,
            args=(key, value),
            daemon=True
        ).start()

        # Schedule write to cold (less frequently)
        # In practice, this is done via batch commits
        return True

    def delete(self, key: str) -> bool:
        """Delete from all tiers."""
        results = [
            self.hot.delete(key),
            self.warm.delete(key),
            self.cold.delete(key),
        ]
        return any(results)

    def exists(self, key: str) -> bool:
        """Check existence in any tier."""
        return (self.hot.exists(key) or
                self.warm.exists(key) or
                self.cold.exists(key))

    def list_keys(self, prefix: str = "") -> List[str]:
        """Union of keys across all tiers."""
        keys = set()
        keys.update(self.hot.list_keys(prefix))
        keys.update(self.warm.list_keys(prefix))
        keys.update(self.cold.list_keys(prefix))
        return sorted(list(keys))

    def get_size_bytes(self) -> int:
        """Total size across all tiers."""
        return (self.hot.get_size_bytes() +
                self.warm.get_size_bytes() +
                self.cold.get_size_bytes())
```

### 4.4 Real-World Implementation: Cortical State Storage

The codebase implements storage abstraction via `state_storage.py`:

```python
# Pattern: StateWriter/StateLoader abstraction
class StateWriter:
    """Write processor state to storage (hot → warm)."""

    def __init__(self, path: str):
        self.path = Path(path)

    def save_config(self, config: Dict, doc_lengths: Dict, avg_length: float):
        """Save to warm tier (disk)."""
        config_path = self.path / "config.json"
        with open(config_path, 'w') as f:
            json.dump({'config': config, 'doc_lengths': doc_lengths}, f)

    def save_all(self, layers, documents, document_metadata, ...):
        """Batch save all components."""
        # Writes to JSON files in warm tier
        # Can be promoted to cold tier via git

class StateLoader:
    """Load processor state from storage (cold/warm → hot)."""

    def load_all(self, validate: bool = True):
        """Load from warm/cold, populate hot tier."""
        # Reads JSON files
        # Populates in-memory structures
        # Returns hot-tier objects
```

---

## 5. Sync Points and Durability Guarantees

### 5.1 Durability Levels (Git-Inspired)

Git defines durability in terms of "sync points":

| Level | Guarantee | Use Case | Performance Impact |
|-------|-----------|----------|-------------------|
| **None** | Data in memory only | Development/testing | None (fastest) |
| **Loose** | Data on disk, not fsync'd | Development | Minimal |
| **Loose + fsync** | Data on disk + flushed | Non-critical data | 5-10x slower |
| **Committed** | Data in git, fsync'd, backed up | Critical data | 20-50x slower |

### 5.2 fsync and Durability

The Linux filesystem provides these guarantees:

```python
import os
import fsync as _fsync

class DurableFile:
    """Write with explicit durability guarantees."""

    def write_loose(self, path: str, data: str):
        """
        Write data to disk.

        Guarantees: Data in filesystem cache. May be lost on power failure.
        Performance: Fast
        """
        with open(path, 'w') as f:
            f.write(data)
        # Data is now in kernel cache but not guaranteed on disk

    def write_sync(self, path: str, data: str):
        """
        Write data with filesystem sync.

        Guarantees: Data on disk, accessible after reboot.
        BUT: Not durable against power loss during fsync itself.
        Performance: Slow (~5-10x)
        """
        with open(path, 'w') as f:
            f.write(data)
            f.flush()  # Flush to OS cache
            os.fsync(f.fileno())  # Sync to disk

    def write_synced_dir(self, path: str, data: str):
        """
        Write data with full POSIX durability.

        Guarantees: Data durable after power loss OR OS crash.
        Requires: fsync on both file AND parent directory.
        Performance: Very slow (~20-50x)

        Why directory sync is needed:
        - UNIX allows hard links and renames
        - Directory entry itself must be durable
        - Without dir sync, file may disappear after reboot
        """
        # Write and sync file
        with open(path, 'w') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())

        # Sync parent directory
        dir_fd = os.open(os.path.dirname(path), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
```

### 5.3 Write-Ahead Logging for Durability

**Key insight:** Without WAL, all writes must be synchronous (slow).
**With WAL:** Async writes become safe.

```python
class WALDurability:
    """
    Implement ACID properties using Write-Ahead Logging.

    Pattern:
    1. Write operation to WAL (sequential, fast)
    2. fsync WAL (small file, fast)
    3. Apply operation in memory (instant)
    4. Return success
    5. Later: Flush memory to cold storage (can fail, but WAL protects)
    """

    def __init__(self, wal_path: str = "wal.log"):
        self.wal_path = Path(wal_path)
        self.wal_fd = None
        self.state = {}

    def __enter__(self):
        # Open WAL in append mode
        self.wal_fd = open(self.wal_path, 'a')
        return self

    def __exit__(self, *args):
        if self.wal_fd:
            self.wal_fd.close()

    def write_durable(self, key: str, value: Any) -> bool:
        """Write with durability guarantees."""

        # STEP 1: Write to WAL
        wal_entry = {
            'op': 'write',
            'key': key,
            'value': value,
            'timestamp': time.time()
        }

        self.wal_fd.write(json.dumps(wal_entry) + '\n')
        self.wal_fd.flush()

        # STEP 2: Sync WAL to disk (small, fast)
        os.fsync(self.wal_fd.fileno())

        # STEP 3: Apply to in-memory state (now safe)
        self.state[key] = value

        return True

    def recover_after_crash(self):
        """Recover from WAL after crash."""
        if not self.wal_path.exists():
            return

        with open(self.wal_path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    if entry['op'] == 'write':
                        self.state[entry['key']] = entry['value']
                except json.JSONDecodeError:
                    pass  # Corrupted entry, skip
```

### 5.4 Transaction Boundaries

Define clear sync points for consistency:

```python
class Transaction:
    """
    Transactional writes with explicit sync points.

    Usage:
        with Transaction(manager) as tx:
            tx.write('key1', value1)
            tx.write('key2', value2)
            # Auto-commit and sync on exit
    """

    def __init__(self, manager, sync_level: str = 'committed'):
        self.manager = manager
        self.sync_level = sync_level  # 'none', 'loose', 'sync', 'committed'
        self.writes = []

    def write(self, key: str, value: Any):
        """Queue write in transaction."""
        self.writes.append((key, value))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Commit transaction with specified sync level."""
        if exc_type is not None:
            # Rollback on exception
            self.writes.clear()
            return

        # Commit
        if self.sync_level == 'none':
            self._commit_loose()
        elif self.sync_level == 'loose':
            self._commit_loose_fsync()
        elif self.sync_level == 'sync':
            self._commit_sync()
        elif self.sync_level == 'committed':
            self._commit_full()

    def _commit_loose(self):
        """Memory only."""
        for key, value in self.writes:
            self.manager.hot_cache[key] = value

    def _commit_loose_fsync(self):
        """Memory + disk cache."""
        for key, value in self.writes:
            self.manager.hot_cache[key] = value
            self.manager.warm_storage[key] = value
        # No fsync

    def _commit_sync(self):
        """Memory + disk + fsync."""
        for key, value in self.writes:
            self.manager.hot_cache[key] = value
            self.manager.warm_storage_with_fsync[key] = value

    def _commit_full(self):
        """Memory + disk + fsync + git commit."""
        for key, value in self.writes:
            self.manager.hot_cache[key] = value
            self.manager.warm_storage_with_fsync[key] = value
        # Commit to git (which does fsync internally)
        self.manager.git_commit(self.writes)
```

### 5.5 Real-World Implementation: GraphWAL and GitAutoCommitter

The codebase implements durability via layered sync points:

```
Application
    ↓
Hot Tier (in-memory)
    ↓
WAL Entry (persistent + checksummed)
    ↓
Snapshot (periodic, compressed)
    ↓
Git Commit (debounced, backed by fsync)
    ↓
(Optional) Push to remote
```

**From cortical/reasoning/graph_persistence.py:**
```python
class GraphWAL:
    """
    Write-Ahead Log for graph with crash recovery.

    Durability: WAL entries + snapshots provide 4-level recovery:
    1. Latest snapshot + WAL entries
    2. Previous snapshot + WAL entries
    3. Git history
    4. Manual reconstruction
    """

    def log_add_node(self, node_id: str, node_type: str, content: str):
        """Log graph operation to WAL."""
        entry = GraphWALEntry(
            operation='add_node',
            node_id=node_id,
            node_type=node_type,
            payload={'content': content}
        )
        self._append_to_wal(entry)

class GitAutoCommitter:
    """
    Auto-commit with configurable sync guarantees.

    Modes:
    - 'immediate': Commit + sync immediately (high durability, slow)
    - 'debounced': Commit after idle period (good balance)
    - 'manual': Only validate (fast, low durability)
    """

    def commit_on_save(self, graph_path: str, graph: ThoughtGraph):
        """Commit with specified durability level."""
        if self.mode == 'immediate':
            # Sync immediately
            self.auto_commit(graph_path, graph)
            if self.auto_push:
                self.push_if_safe()  # Also backs up to remote

        elif self.mode == 'debounced':
            # Debounce to batch commits, but within time window
            self._schedule_debounced_commit(graph_path, graph)
```

### 5.6 Sync Point Decision Matrix

When to use each sync level:

```python
# Decision matrix based on data criticality and performance needs

SYNC_POINTS = {
    'Development/Testing': 'none',
    'Cached query results': 'loose',
    'Indexed documents': 'loose_fsync',
    'User data': 'sync',
    'Transaction logs': 'committed',
    'Reasoning graphs': 'committed',
    'Critical reasoning state': 'committed + remote',
}

def choose_sync_level(data_type: str, criticality: str):
    """Recommend sync level based on data characteristics."""

    if criticality == 'low':
        return 'loose'
    elif criticality == 'medium':
        return 'sync'
    else:  # 'high'
        return 'committed + remote'
```

---

## 6. Practical Implementation: Git + Local File System

### 6.1 Reference Architecture

```
Your Python Application
    │
    ├─ HOT TIER (Memory)
    │  ├─ Dict: Current state
    │  ├─ LRUCache: Query results
    │  └─ Thread-safe locks
    │
    ├─ WARM TIER (Local Disk)
    │  ├─ corpus_dev.json/     (JSON files)
    │  ├─ reasoning_wal/       (WAL log)
    │  └─ snapshots/           (Periodic snapshots)
    │
    └─ COLD TIER (Git Repository)
       ├─ .git/objects/        (All history)
       ├─ reasoning/           (Committed graphs)
       └─ samples/             (Archives)
```

### 6.2 Minimal Viable Implementation

```python
import json
import os
import subprocess
from pathlib import Path
from collections import OrderedDict
from threading import Timer, RLock
from typing import Any, Optional, Dict

class MinimalStorageStack:
    """
    Minimal but complete storage tier system.

    Demonstrates all 5 patterns with <200 lines.
    """

    def __init__(self, warm_path: str = "data", wal_path: str = "wal"):
        # HOT: Memory cache with LRU
        self.hot = OrderedDict()
        self.hot_max_size = 1000

        # WARM: Local disk
        self.warm_path = Path(warm_path)
        self.warm_path.mkdir(exist_ok=True)

        # WAL: Write-ahead log for durability
        self.wal_path = Path(wal_path)
        self.wal_path.mkdir(exist_ok=True)
        self.wal_file = None

        # Write buffer (write-behind)
        self.write_buffer = {}
        self.flush_timer = None

        # Thread safety
        self.lock = RLock()

    # PATTERN 1: Hot/Warm/Cold Tiers
    def promote_to_hot(self, key: str, value: Any):
        """Promote to hot tier (LRU eviction)."""
        with self.lock:
            if key in self.hot:
                self.hot.move_to_end(key)
            else:
                self.hot[key] = value
                if len(self.hot) > self.hot_max_size:
                    # Evict LRU
                    evicted_key, evicted_value = self.hot.popitem(last=False)
                    self.demote_to_warm(evicted_key, evicted_value)

    def demote_to_warm(self, key: str, value: Any):
        """Demote to warm tier (disk)."""
        with self.lock:
            warm_file = self.warm_path / f"{key}.json"
            with open(warm_file, 'w') as f:
                json.dump(value, f)

    def archive_to_cold(self, key: str):
        """Archive to cold tier (git)."""
        try:
            subprocess.run(['git', 'add', str(self.warm_path)],
                          check=True, timeout=5)
            subprocess.run(['git', 'commit', '-m', f'archive: {key}'],
                          check=True, timeout=5)
        except Exception as e:
            print(f"Archive failed: {e}")

    # PATTERN 2: Write-Behind Caching
    def write_buffered(self, key: str, value: Any):
        """Buffer write, flush asynchronously."""
        with self.lock:
            # Log to WAL immediately (durability)
            self._log_to_wal(key, value)

            # Buffer in memory
            self.write_buffer[key] = value

            # Schedule flush
            self._schedule_flush()

    def _log_to_wal(self, key: str, value: Any):
        """Write to WAL for crash recovery."""
        if self.wal_file is None:
            self.wal_file = open(self.wal_path / 'wal.log', 'a')

        entry = {'op': 'write', 'key': key, 'value': value}
        self.wal_file.write(json.dumps(entry) + '\n')
        self.wal_file.flush()
        os.fsync(self.wal_file.fileno())

    def _schedule_flush(self):
        """Debounce flush."""
        if self.flush_timer:
            self.flush_timer.cancel()
        self.flush_timer = Timer(5, self._do_flush)
        self.flush_timer.start()

    def _do_flush(self):
        """Flush buffer to warm tier."""
        with self.lock:
            for key, value in self.write_buffer.items():
                self.demote_to_warm(key, value)
            self.write_buffer.clear()

    # PATTERN 3: Read-Through Caching
    def read(self, key: str) -> Optional[Any]:
        """Read with automatic tiering."""
        with self.lock:
            # Check hot
            if key in self.hot:
                self.hot.move_to_end(key)
                return self.hot[key]

            # Check warm
            warm_file = self.warm_path / f"{key}.json"
            if warm_file.exists():
                with open(warm_file) as f:
                    value = json.load(f)
                self.promote_to_hot(key, value)
                return value

            # Check cold (git)
            try:
                result = subprocess.run(
                    ['git', 'show', f'HEAD:data/{key}.json'],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    value = json.loads(result.stdout)
                    self.promote_to_hot(key, value)
                    return value
            except Exception:
                pass

            return None

    # PATTERN 4: Storage Abstraction
    def exists(self, key: str) -> bool:
        """Storage-agnostic existence check."""
        with self.lock:
            return key in self.hot or (self.warm_path / f"{key}.json").exists()

    def list_keys(self, prefix: str = "") -> list:
        """List keys across all tiers."""
        keys = set()
        with self.lock:
            keys.update(k for k in self.hot.keys() if k.startswith(prefix))
            keys.update(f.stem for f in self.warm_path.glob("*.json")
                       if f.stem.startswith(prefix))
        return sorted(list(keys))

    # PATTERN 5: Sync Points
    def sync_loose(self):
        """Write to disk cache (not fsync)."""
        self._do_flush()

    def sync_durable(self):
        """Write to disk with fsync."""
        self._do_flush()
        # Sync warm directory
        for file in self.warm_path.glob("*.json"):
            fd = os.open(str(file), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)

    def sync_committed(self):
        """Write, sync, and commit to git."""
        self.sync_durable()
        self.archive_to_cold("*")

    def recover_from_wal(self):
        """Recover after crash."""
        wal_file = self.wal_path / 'wal.log'
        if not wal_file.exists():
            return

        with open(wal_file) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    if entry['op'] == 'write':
                        self.promote_to_hot(entry['key'], entry['value'])
                except json.JSONDecodeError:
                    pass
```

### 6.3 Integration with Existing Systems

Connect to the codebase:

```python
from cortical.reasoning.graph_persistence import GraphWAL, GitAutoCommitter
from cortical.wal import WALWriter, SnapshotManager
from cortical.got.manager import GoTManager

class IntegratedStorageStack(MinimalStorageStack):
    """
    Production storage stack integrating:
    - GraphWAL for reasoning graphs
    - GitAutoCommitter for durability
    - GoT for task management
    - All five storage patterns
    """

    def __init__(self):
        super().__init__()

        # Graph persistence
        self.graph_wal = GraphWAL("reasoning_wal")
        self.committer = GitAutoCommitter(mode='debounced')

        # Task management
        self.got = GoTManager()

    def save_graph_durable(self, graph_id: str, graph_state: Dict):
        """Save reasoning graph with all durability guarantees."""

        # STEP 1: Write to hot (fast)
        self.promote_to_hot(f"graph:{graph_id}", graph_state)

        # STEP 2: Log to WAL (durable)
        self.write_buffered(f"graph:{graph_id}", graph_state)

        # STEP 3: Commit to git (backed up)
        self.committer.commit_on_save(
            f"graphs/{graph_id}.json",
            graph_state
        )
```

---

## 7. Performance Characteristics

### 7.1 Latency Comparison

| Operation | Hot | Warm | Cold | Cold (cached) |
|-----------|-----|------|------|---------------|
| Read | <1ms | 1-10ms | 100-500ms | <1ms (hit) |
| Write | <1ms | 10-50ms | 500ms-2s | N/A |
| List keys | 1-10ms | 50-500ms | 1-5s | N/A |

### 7.2 Throughput with Write-Behind

```
Without write-behind (sync writes):
  - Limited by disk I/O (~100-1000 writes/sec)

With write-behind (buffered):
  - Hot tier accepts ~100K writes/sec
  - Batch flushed to warm every 5s
  - Effective throughput: 20K writes/sec sustained
```

### 7.3 Space Amplification

- Hot: Data once (memory)
- Warm: Data once (disk)
- Cold: Data + history (git, grows over time)

Total space ≈ 3x data size initially, growing with history.

---

## 8. Failure Modes and Recovery

### 8.1 Crash Scenarios

| Failure | Impact | Recovery |
|---------|--------|----------|
| Process crash | Lose hot, WAL recovers | Replay WAL on startup |
| Disk failure | Lose warm, cold intact | Restore from git, rebuild warm |
| Git corruption | Lose cold, warm+hot intact | Restore from remote, rebuild cold |
| Unflushed fsync | Lost data, corruption risk | WAL or transaction replay |
| Concurrent writes | Race conditions, corruption | Use locks and transactions |

### 8.2 Recovery Strategy

```python
def startup_with_recovery():
    """Standard startup with crash recovery."""

    storage = MinimalStorageStack()

    # STEP 1: Recover from WAL (fast)
    storage.recover_from_wal()

    # STEP 2: Check warm tier consistency
    # (Verify checksums, rebuild if needed)

    # STEP 3: Validate against git (slow but safe)
    # (For critical data only)

    # STEP 4: Resume normal operation
    return storage
```

---

## References & Sources

### Storage Tiering
- [Access tiers for blob data - Azure Storage](https://learn.microsoft.com/en-us/azure/storage/blobs/access-tiers-overview)
- [Elasticsearch data tiers: hot, warm, cold, and frozen storage explained](https://www.elastic.co/docs/manage-data/lifecycle/data-tiers)
- [Tiered Storage Guide for Data Archival](https://www.nakivo.com/blog/storage-tiering/)
- [Hot Storage vs Cold Storage - Medium](https://aronbrand.medium.com/hot-storage-vs-cold-storage-choosing-the-right-tier-for-your-data-12fa7c30959d)

### Caching Patterns
- [Understanding write-through, write-around and write-back caching](https://shahriar.svbtle.com/Understanding-writethrough-writearound-and-writeback-caching-with-python)
- [Caching Strategy: Write-Behind (Write-Back) Pattern](https://www.enjoyalgorithms.com/blog/write-behind-caching-pattern/)
- [Write-behind caching - Redis Docs](https://redis.io/docs/latest/operate/oss_and_stack/stack-with-enterprise/gears-v1/python/recipes/write-behind/)
- [Python Cache: Effective Caching](https://oxylabs.io/blog/python-cache-how-to-use-effectively)
- [Caching Patterns - Hazelcast](https://hazelcast.com/blog/a-hitchhikers-guide-to-caching-patterns/)
- [LRU Cache Best Practices - Imperva](https://www.imperva.com/learn/application-security/lru-caching/)
- [Cache-Aside Pattern - GeeksforGeeks](https://www.geeksforgeeks.org/system-design/cache-aside-pattern/)

### Storage Abstraction
- [Database abstraction layer - Wikipedia](https://en.wikipedia.org/wiki/Database_abstraction_layer)
- [PyFilesystem2 - Python filesystem abstraction](https://github.com/PyFilesystem/pyfilesystem2)
- [Repository Pattern - Cosmic Python](https://www.cosmicpython.com/book/chapter_02_repository.html)

### Durability & Sync Points
- [Durability: Linux File APIs](https://www.evanjones.ca/durability-filesystem.html)
- [Filesystems and crash resistance - LWN](https://lwn.net/Articles/788938/)
- [Git fsync Documentation](https://patchwork.kernel.org/project/git/patch/20220315191245.17990-1-neerajsi@microsoft.com/)
- [Everything About fsync()](https://blog.httrack.com/blog/2013/11/15/everything-you-always-wanted-to-know-about-fsync/)
- [File System Performance and Durability - MIT](https://pdos.csail.mit.edu/6.828/2010/lec/l-sync.html)
- [Files are Hard - Dan Luu](https://danluu.com/file-consistency/)
- [Fsyncgate: Errors on fsync](https://danluu.com/fsyncgate/)

---

## Quick Reference: Pattern Selection

| Pattern | When | Trade-offs |
|---------|------|-----------|
| **Hot/Warm/Cold** | Optimize cost, manage lifecycle | Complexity, migration overhead |
| **Write-Behind** | High write throughput needed | Risk of data loss (mitigate with WAL) |
| **Read-Through** | Hot data changes infrequently | Memory/disk space trade-off |
| **Storage Abstraction** | Multi-backend support needed | Reduced performance, increased abstraction |
| **Sync Points** | Durability requirements exist | Performance impact (10-50x slower) |

---

*Last Updated: December 2025*
