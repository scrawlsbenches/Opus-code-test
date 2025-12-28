# Thread-Safety in Cortical Text Processor

**Last Updated:** 2025-12-27

---

## Overview

The Cortical Text Processor codebase has **mixed thread-safety guarantees**. Most components are designed for single-threaded use, with specific exceptions where thread-safety is explicitly required.

**Key Principle:** Unless explicitly documented as thread-safe, assume components are NOT thread-safe and require external synchronization when used across multiple threads.

---

## Thread-Safe Components

### 1. GitAutoCommitter

**Location:** `cortical/reasoning/graph_persistence.py`

**Thread-Safety Mechanism:** Uses `threading.Lock` to protect debouncing state.

```python
# Thread-safe debouncing state
self._lock = threading.Lock()
self._debounce_timer: Optional[Timer] = None
self._pending_commit: Optional[tuple] = None
```

**Protected Operations:**
- Debounced commit scheduling
- Timer cancellation
- Pending commit state management

**Usage:**
```python
from cortical.reasoning.graph_persistence import GitAutoCommitter

# Safe to use from multiple threads
committer = GitAutoCommitter(mode='debounced', debounce_seconds=5)

# Thread A
committer.commit_on_save('/path/to/graph1.json')

# Thread B (concurrent access is safe)
committer.commit_on_save('/path/to/graph2.json')
```

**Internal Locking Example:**
```python
with self._lock:
    # Cancel pending commit if any
    if self._debounce_timer is not None:
        self._debounce_timer.cancel()
    # Schedule new commit
    self._debounce_timer = Timer(self.debounce_seconds, _do_debounced_commit)
    self._debounce_timer.start()
    self._pending_commit = (message, files, graph)
```

---

### 2. ProcessLock

**Location:** `cortical/utils/locking.py`

**Thread-Safety Mechanism:** Uses both `fcntl.flock()` (process-level) and `threading.Lock` (thread-level).

```python
# Thread-safe lock state
self._thread_lock = threading.Lock()
self._fd = None
self._lock_count = 0
```

**Protection Levels:**
- **Process-level:** `fcntl.flock()` prevents race conditions between separate processes
- **Thread-level:** `threading.Lock()` prevents race conditions within the same process

**Usage:**
```python
from cortical.utils.locking import ProcessLock
from pathlib import Path

lock = ProcessLock(Path("/path/to/.lock"))

# Safe from multiple threads in same process
with lock:
    # Critical section protected across threads AND processes
    pass
```

**Reentrant Locking:**
```python
# Same thread can re-acquire (reentrant=True by default)
with lock:
    with lock:  # Safe - same thread
        pass
```

**Why Both Locks?**
- `fcntl.flock()` alone doesn't protect against threads in the same process
- `threading.Lock()` alone doesn't protect against separate processes
- Combined, they provide full multi-process, multi-thread safety

---

## NOT Thread-Safe Components

The following components are designed for **single-threaded use only** and require external synchronization if shared across threads.

### 1. ThoughtGraph

**Location:** `cortical/reasoning/thought_graph.py`

**Thread-Safety:** None - in-memory graph with no internal locking

**Reason:** Performance - graph operations are CPU-bound and locking would add overhead for the common single-threaded case.

**Unsafe Example:**
```python
graph = ThoughtGraph()

# Thread A
graph.add_node("Q1", NodeType.QUESTION, "What is X?")

# Thread B (RACE CONDITION!)
graph.add_node("Q2", NodeType.QUESTION, "What is Y?")
```

**Safe Usage:** See "Recommendations" section below.

---

### 2. GoTManager

**Location:** `cortical/got/api.py`

**Thread-Safety:** None - file-based operations with no locking

**Reason:** GoT is designed for sequential agent workflows. Concurrent mutations would violate transaction semantics.

**Unsafe Example:**
```python
manager = GoTManager("/path/to/.got")

# Thread A
task_id = manager.create_task("Task A", priority="high")

# Thread B (FILE CORRUPTION RISK!)
other_id = manager.create_task("Task B", priority="medium")
```

**Safe Usage:** Use separate `GoTManager` instances per thread, or use external locking.

---

### 3. WovenMind and Loom System

**Components:**
- `WovenMind` (`cortical/reasoning/woven_mind.py`)
- `Loom` (`cortical/reasoning/loom.py`)
- `LoomHiveConnector` (`cortical/reasoning/loom_hive.py`)
- `LoomCortexConnector` (`cortical/reasoning/loom_cortex.py`)

**Thread-Safety:** None - designed for single-threaded processing

**Reason:** State machines with internal mode transitions that assume sequential execution.

**Unsafe Example:**
```python
mind = WovenMind()

# Thread A
result_a = mind.process(["neural", "networks"])

# Thread B (STATE CORRUPTION!)
result_b = mind.process(["machine", "learning"])
```

**Safe Usage:** Create separate `WovenMind` instances per thread.

---

### 4. CorticalTextProcessor

**Location:** `cortical/processor/`

**Thread-Safety:** None - mutable graph state with no locking

**Reason:** Designed for batch processing and indexing workflows where threading adds complexity without benefit.

**Unsafe Example:**
```python
processor = CorticalTextProcessor()

# Thread A
processor.process_document("doc1", "Text A")

# Thread B (RACE CONDITION!)
processor.process_document("doc2", "Text B")
```

**Safe Usage:** Use process-based parallelism (see "Recommendations").

---

### 5. GraphWAL

**Location:** `cortical/reasoning/graph_persistence.py`

**Thread-Safety:** File-based with atomic writes (process-safe via `fcntl`, NOT thread-safe)

**Process-Safety:** ✅ Multiple processes can safely use the WAL (file locking via `fcntl`)

**Thread-Safety:** ❌ Multiple threads in same process need external synchronization

**Reason:** Uses `ProcessLock` which provides process-level safety but requires external coordination for multi-threaded access within a process.

**Safe (Multi-Process):**
```python
# Process A
wal_a = GraphWAL("reasoning_wal")
wal_a.log_add_node("Q1", NodeType.QUESTION, "What is X?")

# Process B (different process - safe via fcntl)
wal_b = GraphWAL("reasoning_wal")
wal_b.log_add_node("Q2", NodeType.QUESTION, "What is Y?")
```

**Unsafe (Multi-Thread):**
```python
wal = GraphWAL("reasoning_wal")

# Thread A
wal.log_add_node("Q1", NodeType.QUESTION, "What is X?")

# Thread B (NEEDS EXTERNAL LOCK!)
wal.log_add_node("Q2", NodeType.QUESTION, "What is Y?")
```

---

## Recommendations

### 1. Default to Single-Threaded Design

For most use cases, keep it simple:

```python
# Good - simple and safe
processor = CorticalTextProcessor()
for doc_id, text in documents:
    processor.process_document(doc_id, text)
processor.compute_all()
```

### 2. Use Process-Based Parallelism

When you need parallelism, prefer multiprocessing over threading:

```python
from multiprocessing import Pool

def index_batch(doc_batch):
    """Each process gets its own processor instance."""
    processor = CorticalTextProcessor()
    for doc_id, text in doc_batch:
        processor.process_document(doc_id, text)
    processor.compute_all()
    return processor.save(f"batch_{os.getpid()}.json")

# Safe - separate processes, separate memory
with Pool(processes=4) as pool:
    results = pool.map(index_batch, document_batches)
```

**Why Prefer Processes?**
- No shared memory = no race conditions
- Better CPU utilization (no GIL)
- Simpler reasoning about correctness
- Natural isolation for error recovery

### 3. Use External Locks for Thread-Based Parallelism

If you must use threads, add explicit synchronization:

```python
import threading

graph = ThoughtGraph()
graph_lock = threading.Lock()

def worker(node_id, content):
    with graph_lock:
        graph.add_node(node_id, NodeType.QUESTION, content)

# Now safe for multi-threaded access
threads = [
    threading.Thread(target=worker, args=(f"Q{i}", f"Question {i}"))
    for i in range(10)
]
for t in threads:
    t.start()
for t in threads:
    t.join()
```

### 4. Use Separate Instances Per Thread

Often the simplest approach:

```python
import threading

def worker(thread_id, documents):
    # Each thread gets its own instance
    processor = CorticalTextProcessor()
    for doc_id, text in documents:
        processor.process_document(doc_id, text)
    processor.compute_all()
    processor.save(f"corpus_thread_{thread_id}.json")

threads = [
    threading.Thread(target=worker, args=(i, doc_batches[i]))
    for i in range(4)
]
```

### 5. For Parallel Agents, Use Separate Instances

When spawning parallel sub-agents (via `ParallelCoordinator`):

```python
from cortical.reasoning.collaboration import ParallelCoordinator

coordinator = ParallelCoordinator()

# Each sub-agent gets isolated state
coordinator.spawn_agent(
    agent_id="agent-1",
    task="Index module A",
    initial_context={
        # No shared mutable state
        "module": "cortical/processor",
    }
)

coordinator.spawn_agent(
    agent_id="agent-2",
    task="Index module B",
    initial_context={
        "module": "cortical/analysis",
    }
)
```

---

## Example: Safe Multi-Threaded Usage

Here's a complete example showing safe multi-threaded usage with external locking:

```python
import threading
from cortical import CorticalTextProcessor

class ThreadSafeProcessor:
    """
    Thread-safe wrapper for CorticalTextProcessor.

    Uses external locking to coordinate access from multiple threads.
    """

    def __init__(self):
        self._processor = CorticalTextProcessor()
        self._lock = threading.Lock()

    def process_document(self, doc_id: str, text: str):
        """Thread-safe document processing."""
        with self._lock:
            self._processor.process_document(doc_id, text)

    def compute_all(self):
        """Thread-safe computation."""
        with self._lock:
            self._processor.compute_all()

    def find_documents(self, query: str):
        """Thread-safe search."""
        with self._lock:
            return self._processor.find_documents_for_query(query)

    def save(self, path: str):
        """Thread-safe save."""
        with self._lock:
            return self._processor.save(path)


# Usage
processor = ThreadSafeProcessor()

def indexer_worker(doc_id, text):
    processor.process_document(doc_id, text)

# Spawn worker threads
threads = [
    threading.Thread(target=indexer_worker, args=(f"doc{i}", f"Text {i}"))
    for i in range(100)
]

for t in threads:
    t.start()
for t in threads:
    t.join()

# Compute and save
processor.compute_all()
processor.save("corpus.json")
```

**Trade-offs:**
- **Pro:** Simple to use, all state in one place
- **Con:** Locks serialize access, reducing parallelism benefits
- **Con:** Fine-grained locking (per-document) would be complex and error-prone

**Better Alternative (Process-Based):**
```python
from multiprocessing import Pool, Manager

def worker(doc_batch, output_queue):
    """Each process gets isolated processor."""
    processor = CorticalTextProcessor()
    for doc_id, text in doc_batch:
        processor.process_document(doc_id, text)
    processor.compute_all()

    # Send results back
    output_queue.put(processor.export_to_dict())

# Much better parallelism - no lock contention
manager = Manager()
queue = manager.Queue()

with Pool(processes=4) as pool:
    pool.starmap(worker, [(batch, queue) for batch in doc_batches])
```

---

## Testing Thread-Safety

If you add new components that need thread-safety, add tests:

```python
import threading
import pytest

def test_component_thread_safety():
    """Verify component is thread-safe."""
    component = ThreadSafeComponent()
    results = []
    errors = []

    def worker(i):
        try:
            result = component.operation(i)
            results.append(result)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(100)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # No errors, all operations succeeded
    assert len(errors) == 0
    assert len(results) == 100
    # Verify results are consistent (no race conditions)
    assert len(set(results)) == 100  # All unique
```

---

## Summary

| Component | Thread-Safe? | Mechanism | Recommendation |
|-----------|--------------|-----------|----------------|
| `GitAutoCommitter` | ✅ Yes | `threading.Lock` | Safe to share across threads |
| `ProcessLock` | ✅ Yes | `fcntl.flock()` + `threading.Lock` | Safe for multi-process and multi-thread |
| `GraphWAL` | ⚠️ Process-safe only | `fcntl.flock()` (via ProcessLock) | Use external lock for threads |
| `ThoughtGraph` | ❌ No | None | Use external lock or separate instances |
| `GoTManager` | ❌ No | None | Use separate instances per thread |
| `WovenMind` | ❌ No | None | Use separate instances per thread |
| `Loom` | ❌ No | None | Use separate instances per thread |
| `CorticalTextProcessor` | ❌ No | None | Prefer multiprocessing over threading |

**Default Strategy:** Use process-based parallelism (`multiprocessing.Pool`) instead of threading when performance matters.

**When Threading is Required:** Wrap components with external locks or use separate instances per thread.

---

## Platform Notes

**POSIX Systems (Linux, macOS):**
- Full support for `fcntl.flock()` process locking
- All locking features work as documented

**Windows:**
- ❌ NOT SUPPORTED
- `fcntl.flock()` is not available on Windows
- Components using `ProcessLock` will fail at runtime
- See `cortical/utils/locking.py` module docstring

---

## Further Reading

- Python `threading` module: https://docs.python.org/3/library/threading.html
- Python `multiprocessing` module: https://docs.python.org/3/library/multiprocessing.html
- `fcntl` documentation: https://docs.python.org/3/library/fcntl.html
- Thread-safety in Python: https://docs.python.org/3/glossary.html#term-thread-safe

---

**Questions or Issues?**

If you encounter thread-safety issues or need clarification, check the source code comments or open an issue with:
1. Component name
2. Expected behavior
3. Observed behavior (race conditions, deadlocks, etc.)
4. Minimal reproducible example
