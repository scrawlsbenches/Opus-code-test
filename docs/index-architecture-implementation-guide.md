# Index Architecture Implementation Guide

**Practical patterns for file-based database indexing**

---

## Quick Start: Pick Your Pattern

### Decision Tree

```
Do you need O(1) lookups by ID?
├─ YES → Use PRIMARY INDEX (Section 1.1)
└─ NO ↓

Do you frequently filter by a specific field (status, priority, etc)?
├─ YES → Use SECONDARY INDEX (Section 1.2)
└─ NO ↓

Do you need probabilistic deduplication checks?
├─ YES → Use BLOOM FILTER (Section 3)
└─ NO ↓

Do you need full-text search or "who references X"?
├─ YES → Use INVERTED INDEX (Section 4)
└─ NO ↓

Are you unsure about access patterns?
├─ YES → Use PROVISIONAL INDEX (Section 2)
└─ NO → You might not need indexing yet
```

---

## Implementation Checklist

### Phase 1: Foundation (Week 1)
- [ ] Choose primary key structure
- [ ] Implement FileBasedPrimaryIndex
- [ ] Add primary index to entity manager
- [ ] Test: `test.get_by_id(entity_id)` returns O(1)

### Phase 2: Query Performance (Week 2)
- [ ] Identify top 3 query patterns
- [ ] Create secondary indexes for those fields
- [ ] Measure query latency before/after
- [ ] Document expected improvement (e.g., "50x faster")

### Phase 3: Deduplication (Week 3)
- [ ] Implement BloomFilter with appropriate sizing
- [ ] Integrate into entity add workflow
- [ ] Handle false positives (fallback to disk check)
- [ ] Test with real data volume

### Phase 4: Full-Text Search (Week 4)
- [ ] Choose tokenization strategy
- [ ] Build SimpleInvertedIndex or RankedInvertedIndex
- [ ] Implement query interface
- [ ] Add relevance ranking if needed

### Phase 5: Adaptive Optimization (Ongoing)
- [ ] Monitor query patterns
- [ ] Create provisional indexes for hot fields
- [ ] Collect metrics (query count, latency)
- [ ] Promote to permanent when ROI is clear

---

## Integration Examples

### Example 1: GoT Task Manager with Indexes

```python
from pathlib import Path
import json
from typing import Dict, Set, Optional

class GoTTaskManagerWithIndexes:
    """
    Graph of Thought task manager with comprehensive indexing.
    """

    def __init__(self, got_dir: str):
        self.got_dir = Path(got_dir)
        self.tasks_dir = self.got_dir / "entities" / "tasks"
        self.tasks_dir.mkdir(parents=True, exist_ok=True)

        # Initialize all indexes
        self.primary_index = FileBasedPrimaryIndex(str(self.got_dir), "tasks")
        self.secondary_index = FileBasedSecondaryIndex(
            str(self.got_dir), "tasks",
            field_names=["status", "priority", "category"]
        )
        self.inverted_index = SimpleInvertedIndex(str(self.got_dir), "tasks")
        self.dependency_index = DependencyInvertedIndex(str(self.got_dir), "tasks")
        self.bloom_filter = BloomFilter(100000, 7)
        self.staleness = IndexStalenessDetector(str(self.got_dir))

        # Load indexes at startup
        self._load_indexes()

    def _load_indexes(self) -> None:
        """Load all indexes from disk."""
        self.primary_index.build()
        self.secondary_index.load()
        self.inverted_index.load()
        self.dependency_index.load()

    def _save_indexes(self) -> None:
        """Persist all indexes to disk."""
        self.secondary_index.save()
        self.inverted_index.save()
        self.dependency_index.save()

    def create_task(self, task_id: str, title: str, **kwargs) -> Dict:
        """Create new task with indexed fields."""
        task = {
            "id": task_id,
            "title": title,
            "status": kwargs.get("status", "pending"),
            "priority": kwargs.get("priority", "normal"),
            "category": kwargs.get("category", "feature"),
            "content": kwargs.get("content", ""),
            "depends_on": kwargs.get("depends_on", []),
            "tags": kwargs.get("tags", [])
        }

        # Save to disk
        task_file = self.tasks_dir / f"{task_id}.json"
        task_file.write_text(json.dumps(task, indent=2))

        # Update indexes
        self.primary_index.add(task_id, task, task_file)
        self.secondary_index.update(task_id, task)
        self.inverted_index.build()  # Rebuild inverted (could be incremental)
        self.dependency_index.build()
        self.bloom_filter.add(task_id)

        self._save_indexes()
        return task

    def get_task(self, task_id: str) -> Optional[Dict]:
        """Get task by ID (O(1))."""
        return self.primary_index.get(task_id)

    def find_tasks_by_status(self, status: str) -> Set[str]:
        """Find all tasks with given status."""
        return self.secondary_index.find("status", status)

    def find_tasks_by_priority(self, priority: str) -> Set[str]:
        """Find all tasks with given priority."""
        return self.secondary_index.find("priority", priority)

    def search_tasks(self, query: str) -> Set[str]:
        """Search tasks by content."""
        return self.inverted_index.find_by_terms_any(
            self.inverted_index.tokenize(query)
        )

    def find_blocking_tasks(self, task_id: str) -> Set[str]:
        """Find all tasks that depend on this one."""
        return self.dependency_index.find_dependents(task_id)

    def update_task(self, task_id: str, updates: Dict) -> None:
        """Update task and refresh indexes."""
        # Load current task
        task = self.get_task(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")

        # Update fields
        old_task = task.copy()
        task.update(updates)

        # Save to disk
        task_file = self.tasks_dir / f"{task_id}.json"
        task_file.write_text(json.dumps(task, indent=2))

        # Update indexes incrementally
        self.secondary_index.update(task_id, task)
        # For inverted, we'd need incremental update (omitted for brevity)
        self.inverted_index.build()

        self._save_indexes()
        self.staleness.mark_fresh("all_indexes")
```

**Usage:**
```python
# Initialize
manager = GoTTaskManagerWithIndexes(".got")

# Create tasks
manager.create_task(
    "T-20251223-093045-a1b2",
    "Fix auth bug",
    status="pending",
    priority="high",
    content="OAuth token refresh issue"
)
manager.create_task(
    "T-20251223-100000-b2c3",
    "Add tests",
    status="pending",
    priority="normal",
    depends_on=["T-20251223-093045-a1b2"]
)

# Query by indexes
print("High priority tasks:", manager.find_tasks_by_priority("high"))
print("Pending tasks:", manager.find_tasks_by_status("pending"))
print("Search 'OAuth':", manager.search_tasks("OAuth"))
print("Tasks depending on auth:", manager.find_blocking_tasks("T-20251223-093045-a1b2"))
```

### Example 2: Incremental Indexing in Cortical Text Processor

```python
class CorticalTextProcessorWithIndexing:
    """
    Extend CorticalTextProcessor with efficient indexing.
    """

    def __init__(self, processor, data_dir: str):
        self.processor = processor
        self.data_dir = Path(data_dir)

        # Create indexes for document metadata
        self.doc_index = FileBasedPrimaryIndex(str(data_dir), "documents")
        self.metadata_secondary = FileBasedSecondaryIndex(
            str(data_dir), "documents",
            field_names=["category", "language", "source"]
        )
        self.term_inverted = RankedInvertedIndex(str(data_dir), "documents")

    def add_document_indexed(self, doc_id: str, text: str, **metadata) -> None:
        """Add document with automatic indexing."""
        # Save metadata
        doc_metadata = {
            "id": doc_id,
            "text": text,
            "category": metadata.get("category"),
            "language": metadata.get("language", "en"),
            "source": metadata.get("source"),
            "terms": self.processor.tokenizer.tokenize(text)
        }

        # Process in main corpus
        self.processor.process_document(doc_id, text)

        # Update indexes
        metadata_file = self.data_dir / "documents" / f"{doc_id}.json"
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        metadata_file.write_text(json.dumps(doc_metadata, indent=2))

        self.doc_index.add(doc_id, doc_metadata, metadata_file)
        self.metadata_secondary.update(doc_id, doc_metadata)

    def search_by_category(self, category: str):
        """Fast search by category using secondary index."""
        doc_ids = self.metadata_secondary.find("category", category)
        return {doc_id: self.processor.layers[CorticalLayer.DOCUMENTS].get_minicolumn(doc_id)
                for doc_id in doc_ids}

    def search_ranked(self, query: str, top_n: int = 10):
        """Search with TF-IDF ranking using inverted index."""
        results = self.term_inverted.search(query, top_n=top_n)
        return results

    def get_document_metadata(self, doc_id: str) -> Optional[Dict]:
        """Get document metadata (O(1) via primary index)."""
        return self.doc_index.get(doc_id)
```

---

## Performance Tuning Guide

### Benchmark Before Optimizing

```python
import time

def benchmark_queries(manager, queries: List[str], iterations: int = 100):
    """Benchmark query performance."""
    start = time.time()
    for _ in range(iterations):
        for query in queries:
            manager.find_tasks_by_status(query)
    elapsed = time.time() - start
    return elapsed / iterations

# Run before creating indexes
baseline = benchmark_queries(manager, ["pending", "completed", "in_progress"])
print(f"Baseline latency: {baseline*1000:.2f}ms")

# Create index
manager.secondary_index.build()

# Run again
optimized = benchmark_queries(manager, ["pending", "completed", "in_progress"])
print(f"Optimized latency: {optimized*1000:.2f}ms")
print(f"Speedup: {baseline/optimized:.1f}x")
```

### Memory vs Storage Trade-off

```python
def measure_index_overhead(manager):
    """Measure index storage overhead."""
    import os

    # Measure entity files
    entity_dir = manager.tasks_dir
    entity_size = sum(f.stat().st_size for f in entity_dir.glob("*.json"))

    # Measure index files
    index_files = [
        manager.secondary_index.index_file if manager.secondary_index.index_file.exists() else None,
        manager.inverted_index.index_file if manager.inverted_index.index_file.exists() else None,
        manager.dependency_index.index_file if manager.dependency_index.index_file.exists() else None,
    ]
    index_size = sum(f.stat().st_size for f in index_files if f)

    print(f"Entity data: {entity_size / 1024 / 1024:.2f} MB")
    print(f"Index overhead: {index_size / 1024 / 1024:.2f} MB ({100*index_size/entity_size:.1f}%)")
```

### Staleness Detection in Practice

```python
def monitor_staleness(manager, entity_type: str, check_interval_hours: int = 24):
    """Monitor and rebuild stale indexes."""
    detector = IndexStalenessDetector(manager.got_dir)
    entity_dir = manager.got_dir / "entities" / entity_type

    indexes = ["primary", "secondary", "inverted", "dependency"]

    for index_name in indexes:
        if detector.is_stale_by_age(index_name, max_age_hours=check_interval_hours):
            print(f"Index {index_name} is stale (> {check_interval_hours}h old), rebuilding...")
            if index_name == "primary":
                manager.primary_index.rebuild()
            elif index_name == "secondary":
                manager.secondary_index.build()
            elif index_name == "inverted":
                manager.inverted_index.build()
            elif index_name == "dependency":
                manager.dependency_index.build()

            detector.mark_fresh(index_name)
```

---

## Testing Strategy

### Unit Tests

```python
import unittest

class TestIndexes(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.manager = GoTTaskManagerWithIndexes(str(self.temp_dir))

    def test_primary_index_o1_lookup(self):
        """Verify O(1) primary index lookup."""
        task = self.manager.create_task("T-001", "Task 1")
        retrieved = self.manager.get_task("T-001")
        self.assertEqual(retrieved["id"], "T-001")

    def test_secondary_index_filtering(self):
        """Verify secondary index filtering."""
        self.manager.create_task("T-001", "Task 1", status="pending")
        self.manager.create_task("T-002", "Task 2", status="pending")
        self.manager.create_task("T-003", "Task 3", status="completed")

        pending = self.manager.find_tasks_by_status("pending")
        self.assertEqual(len(pending), 2)

    def test_inverted_index_search(self):
        """Verify inverted index search."""
        self.manager.create_task("T-001", "OAuth fix", content="Fix OAuth bug")
        results = self.manager.search_tasks("OAuth")
        self.assertIn("T-001", results)

    def test_bloom_filter_no_false_negatives(self):
        """Verify Bloom filter never says 'absent' when present."""
        self.manager.create_task("T-001", "Task 1")
        self.assertTrue(self.manager.bloom_filter.might_contain("T-001"))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
```

### Integration Tests

```python
def test_concurrent_indexing():
    """Test that indexing works correctly under concurrent writes."""
    import concurrent.futures

    manager = GoTTaskManagerWithIndexes(".got")

    def create_many_tasks(start_id: int):
        for i in range(100):
            task_id = f"T-{start_id:03d}-{i:03d}"
            manager.create_task(
                task_id,
                f"Task {task_id}",
                status="pending" if i % 2 == 0 else "completed"
            )

    # Create tasks in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(create_many_tasks, i*100) for i in range(4)]
        concurrent.futures.wait(futures)

    # Verify index consistency
    all_pending = manager.find_tasks_by_status("pending")
    assert len(all_pending) == 200  # 50% of 400 tasks

    # Verify search
    search_results = manager.search_tasks("Task")
    assert len(search_results) > 0
```

---

## Monitoring & Metrics

```python
class IndexMetrics:
    """Track index performance and health."""

    def __init__(self, manager):
        self.manager = manager
        self.metrics = {
            "query_count": {},
            "query_latency_ms": {},
            "index_sizes": {},
            "rebuild_times_ms": {}
        }

    def record_query(self, index_name: str, latency_ms: float):
        """Record a query."""
        if index_name not in self.metrics["query_count"]:
            self.metrics["query_count"][index_name] = 0
            self.metrics["query_latency_ms"][index_name] = []

        self.metrics["query_count"][index_name] += 1
        self.metrics["query_latency_ms"][index_name].append(latency_ms)

    def record_rebuild(self, index_name: str, elapsed_ms: float):
        """Record an index rebuild."""
        if index_name not in self.metrics["rebuild_times_ms"]:
            self.metrics["rebuild_times_ms"][index_name] = []

        self.metrics["rebuild_times_ms"][index_name].append(elapsed_ms)

    def get_summary(self) -> Dict:
        """Get metrics summary."""
        summary = {}
        for index_name, latencies in self.metrics["query_latency_ms"].items():
            if latencies:
                summary[index_name] = {
                    "query_count": self.metrics["query_count"][index_name],
                    "avg_latency_ms": sum(latencies) / len(latencies),
                    "min_latency_ms": min(latencies),
                    "max_latency_ms": max(latencies),
                    "avg_rebuild_ms": sum(self.metrics["rebuild_times_ms"].get(index_name, [0])) / max(1, len(self.metrics["rebuild_times_ms"].get(index_name, [])))
                }
        return summary

# Usage:
metrics = IndexMetrics(manager)

# Track queries
start = time.time()
result = manager.find_tasks_by_status("pending")
elapsed = (time.time() - start) * 1000
metrics.record_query("secondary_status", elapsed)

# Get summary
print(metrics.get_summary())
```

---

## Troubleshooting

### Problem: Indexes are out of sync with data

**Solution:** Rebuild all indexes
```python
manager.staleness.mark_stale("all", "Data inconsistency detected")
manager.primary_index.rebuild()
manager.secondary_index.build()
manager.inverted_index.build()
manager.dependency_index.build()
manager.staleness.mark_fresh("all")
```

### Problem: High memory usage with large indexes

**Solution:** Use lazy loading
```python
# Instead of loading all data, load only ID→path mappings
class LazySecondaryIndex(FileBasedSecondaryIndex):
    def load(self) -> None:
        """Load only metadata, not full index."""
        if self.index_file.exists():
            metadata = json.loads(self.index_file.read_text())
            # Store only counts, not full sets
            self.index_counts = metadata
```

### Problem: Bloom filter has too many false positives

**Solution:** Rebuild with larger size
```python
if false_positive_rate > 0.01:  # More than 1% FP
    detector.rebuild_larger()
```

### Problem: Indexes are stale after crash

**Solution:** Detect and rebuild on startup
```python
class RobustIndexManager:
    def __init__(self, got_dir: str):
        self.manager = GoTTaskManagerWithIndexes(got_dir)

        # Check if indexes need recovery
        try:
            # Verify index integrity
            self.manager._load_indexes()
        except Exception as e:
            print(f"Index corruption detected: {e}, rebuilding...")
            self.manager.primary_index.rebuild()
            self.manager.secondary_index.build()
            self.manager.inverted_index.build()
            self.manager.dependency_index.build()
            self.manager._save_indexes()
```

---

## Migration Path: From Scan to Indexed

### Stage 1: Baseline (Week 1)
```python
# Current approach: full scan
def find_pending_tasks(manager):
    result = []
    for task_id, task in manager.tasks.items():
        if task["status"] == "pending":
            result.append(task_id)
    return result  # O(n)
```

### Stage 2: Add Primary Index (Week 2)
```python
# Now O(1) for ID lookups
task = manager.get_task("T-001")  # O(1) instead of O(n)
```

### Stage 3: Add Secondary Index (Week 3)
```python
# Find pending tasks: now O(1) with secondary index
pending = manager.find_tasks_by_status("pending")  # O(1) instead of O(n)
```

### Stage 4: Add Inverted Index (Week 4)
```python
# Search: now ranked by relevance
results = manager.search_tasks("OAuth authentication")
```

### Stage 5: Adaptive Optimization (Ongoing)
```python
# Monitor which queries are slow, create indexes as needed
manager.adaptive_indexer.monitor_and_optimize()
```

---

## Reference: Time Complexity Comparison

| Operation | No Index | With Index | Storage |
|-----------|----------|-----------|---------|
| Get by ID | O(n) | O(1) | +10-20% |
| Filter by field | O(n) | O(1) | +20-50% |
| Full-text search | O(n*m) | O(n) + ranking | +20-40% |
| Dedup check | O(n) | O(1) probabilistic | +1-3% |
| Find dependencies | O(n) | O(1) + traversal | +15-25% |

