# File-Based Index Architecture Patterns for JSON Databases

**Research Document**
**Date:** 2025-12-23
**Scope:** Python file-based systems with JSON storage

---

## Executive Summary

File-based databases with JSON storage require careful index architecture to balance:
- **Query performance** (O(1) lookups vs O(n) scans)
- **Storage overhead** (indexes consume disk space)
- **Maintenance cost** (keeping indexes consistent with data)
- **Startup time** (loading indexes from disk)

This document presents five core patterns suitable for Python file-based systems:

| Pattern | Use Case | Cost | Benefit |
|---------|----------|------|---------|
| **Primary Index** | ID lookups, identity | Low | O(1) access by key |
| **Secondary Index** | Field filtering, range queries | Medium | Fast filtered queries |
| **Bloom Filter** | Deduplication checks | Very Low | Probabilistic pre-filter |
| **Inverted Index** | Full-text search, "who references X" | Medium | Semantic queries |
| **Provisional Index** | Adaptive optimization | Medium | Learns from access patterns |

---

## 1. Primary vs Secondary Indexes

### 1.1 Primary Index: Identity-Based Access

**Purpose:** Direct lookup by unique identifier.

**Trade-offs:**
| Aspect | Cost | Benefit |
|--------|------|---------|
| Storage | ~10-20% of data size | O(1) lookups |
| Write latency | +5-10ms per write | Amortized across reads |
| Startup time | +100-500ms | Single load, used until close |
| Memory | ~100MB per 1M records | Blazing fast access |

**When to use:**
- ✅ You frequently look up records by ID (`processor.get_task(task_id)`)
- ✅ ID is the natural clustering key
- ✅ Startup performance matters (load once, use repeatedly)

**When NOT to use:**
- ❌ Records are accessed mostly by content, not ID
- ❌ IDs are sequential/predictable (don't need indexing)
- ❌ Memory is severely constrained

**Implementation Pattern:**
```python
class FileBasedPrimaryIndex:
    """
    O(1) lookup by ID using in-memory dict or file-based btree.
    For JSON storage, this is essential infrastructure.
    """

    def __init__(self, data_dir: str, entity_type: str):
        self.data_dir = Path(data_dir)
        self.entity_type = entity_type
        self.id_to_path: Dict[str, Path] = {}  # ID -> file path mapping
        self._loaded = False

    def build(self) -> None:
        """Build index from JSON files (once at startup)."""
        entity_dir = self.data_dir / self.entity_type
        for json_file in entity_dir.glob("*.json"):
            data = json.loads(json_file.read_text())
            entity_id = data.get("id")
            if entity_id:
                self.id_to_path[entity_id] = json_file
        self._loaded = True

    def get(self, entity_id: str) -> Optional[Dict]:
        """O(1) lookup."""
        if not self._loaded:
            self.build()

        path = self.id_to_path.get(entity_id)
        if not path or not path.exists():
            return None

        return json.loads(path.read_text())

    def add(self, entity_id: str, data: Dict, path: Path) -> None:
        """Add to index when writing new entity."""
        self.id_to_path[entity_id] = path

    def remove(self, entity_id: str) -> None:
        """Remove from index when deleting entity."""
        self.id_to_path.pop(entity_id, None)

    def rebuild(self) -> None:
        """Rebuild when detecting inconsistency."""
        self.id_to_path.clear()
        self.build()
```

**Optimization: Lazy Loading**
```python
class LazyPrimaryIndex(FileBasedPrimaryIndex):
    """
    For systems with millions of entities, don't load all IDs at startup.
    Instead, load only the ID→path mapping (metadata), not the data.
    """

    def build(self) -> None:
        """Load only ID→path mappings, not full data."""
        entity_dir = self.data_dir / self.entity_type
        for json_file in entity_dir.glob("*.json"):
            # Read only first line (usually contains ID)
            # OR store metadata in a separate index file
            with open(json_file) as f:
                first_line = f.readline()
                try:
                    # Assume ID is at top level
                    data = json.loads(first_line + "}")
                    entity_id = data.get("id")
                    if entity_id:
                        self.id_to_path[entity_id] = json_file
                except:
                    # Fallback: load full file
                    f.seek(0)
                    data = json.load(f)
                    entity_id = data.get("id")
                    if entity_id:
                        self.id_to_path[entity_id] = json_file
        self._loaded = True
```

### 1.2 Secondary Index: Field-Based Filtering

**Purpose:** Fast filtering on non-ID fields (e.g., "find all tasks with status=pending").

**Trade-offs:**
| Aspect | Cost | Benefit |
|--------|------|---------|
| Storage | +20-50% of data size | Fast filtered queries |
| Write latency | +10-20ms per write | Amortized |
| Maintenance | Must update on field changes | Avoids full scans |
| Memory | ~200MB per 1M records + index | O(1) to O(log n) filtered access |

**When to use:**
- ✅ Frequently query by field (e.g., `status`, `priority`, `category`)
- ✅ Field has moderate cardinality (10-10,000 distinct values)
- ✅ You need to list all matching records

**When NOT to use:**
- ❌ Field has very high cardinality (nearly unique values)
- ❌ Queries are rare or access patterns unclear
- ❌ Storage is severely constrained

**Cardinality Guidelines:**

| Field Cardinality | Recommendation | Reason |
|-------------------|----------------|--------|
| < 10 (boolean, enum) | Always index | Huge fan-out, storage cheap |
| 10-1,000 | Index if queried often | Good ROI |
| 1,000-10,000 | Index only if frequent queries | Storage overhead rises |
| > 10,000 (nearly unique) | Rarely index | Not selective, poor ROI |

**Implementation Pattern:**
```python
class FileBasedSecondaryIndex:
    """
    Maps field values to lists of entity IDs.

    Structure:
    {
        "status": {
            "pending": ["task-001", "task-003"],
            "completed": ["task-002"]
        },
        "priority": {
            "high": ["task-001"],
            "low": ["task-002", "task-003"]
        }
    }
    """

    def __init__(self, data_dir: str, entity_type: str, field_names: List[str]):
        self.data_dir = Path(data_dir)
        self.entity_type = entity_type
        self.field_names = field_names
        self.index: Dict[str, Dict[Any, Set[str]]] = {}
        self._index_file = self.data_dir / f"{entity_type}_secondary.json"

    def build(self) -> None:
        """Build index from all JSON files."""
        self.index = {field: {} for field in self.field_names}

        entity_dir = self.data_dir / self.entity_type
        for json_file in entity_dir.glob("*.json"):
            data = json.loads(json_file.read_text())
            entity_id = data.get("id")
            if not entity_id:
                continue

            # Index each tracked field
            for field in self.field_names:
                value = data.get(field)
                if value is not None:
                    # Handle list values (e.g., tags)
                    values = value if isinstance(value, list) else [value]
                    for v in values:
                        # Convert unhashable types to strings
                        v_key = json.dumps(v, sort_keys=True) if isinstance(v, dict) else str(v)
                        if v_key not in self.index[field]:
                            self.index[field][v_key] = set()
                        self.index[field][v_key].add(entity_id)

    def find(self, field: str, value: Any) -> Set[str]:
        """Find all entity IDs with field=value."""
        if field not in self.index:
            raise ValueError(f"Field {field} not indexed")

        v_key = json.dumps(value, sort_keys=True) if isinstance(value, dict) else str(value)
        return self.index[field].get(v_key, set())

    def find_in(self, field: str, values: List[Any]) -> Set[str]:
        """Find all entity IDs where field in [values]."""
        result = set()
        for value in values:
            result.update(self.find(field, value))
        return result

    def update(self, entity_id: str, data: Dict) -> None:
        """Update index when entity is modified."""
        # First, remove old entries for this entity
        self.remove(entity_id)

        # Then add new entries
        for field in self.field_names:
            value = data.get(field)
            if value is not None:
                values = value if isinstance(value, list) else [value]
                for v in values:
                    v_key = json.dumps(v, sort_keys=True) if isinstance(v, dict) else str(v)
                    if v_key not in self.index[field]:
                        self.index[field][v_key] = set()
                    self.index[field][v_key].add(entity_id)

    def remove(self, entity_id: str) -> None:
        """Remove entity from all index entries."""
        for field in self.field_names:
            for value_set in self.index[field].values():
                value_set.discard(entity_id)

    def save(self) -> None:
        """Persist index to JSON (periodic snapshots)."""
        # Convert sets to lists for JSON serialization
        serializable = {}
        for field, value_map in self.index.items():
            serializable[field] = {k: list(v) for k, v in value_map.items()}

        self._index_file.write_text(json.dumps(serializable, indent=2))

    def load(self) -> None:
        """Load index from JSON."""
        if self._index_file.exists():
            data = json.loads(self._index_file.read_text())
            self.index = {field: {k: set(v) for k, v in data.get(field, {}).items()}
                         for field in self.field_names}
```

**Optimization: Compound Index**
```python
class CompoundSecondaryIndex:
    """
    Index multiple fields together for complex queries.
    Example: Find all high-priority pending tasks.
    """

    def __init__(self, data_dir: str, entity_type: str, field_combinations: List[Tuple[str, ...]]):
        self.data_dir = Path(data_dir)
        self.entity_type = entity_type
        self.field_combinations = field_combinations
        self.index: Dict[Tuple[str, ...], Dict[Tuple, Set[str]]] = {}

    def build(self) -> None:
        """Build compound indexes."""
        for field_combo in self.field_combinations:
            self.index[field_combo] = {}

        entity_dir = self.data_dir / self.entity_type
        for json_file in entity_dir.glob("*.json"):
            data = json.loads(json_file.read_text())
            entity_id = data.get("id")
            if not entity_id:
                continue

            for field_combo in self.field_combinations:
                values = tuple(str(data.get(f)) for f in field_combo)
                if values not in self.index[field_combo]:
                    self.index[field_combo][values] = set()
                self.index[field_combo][values].add(entity_id)

    def find(self, **kwargs) -> Set[str]:
        """Find by compound fields: find(status="pending", priority="high")."""
        field_combo = tuple(sorted(kwargs.keys()))
        if field_combo not in self.index:
            raise ValueError(f"No index for fields {field_combo}")

        values = tuple(str(kwargs[f]) for f in field_combo)
        return self.index[field_combo].get(values, set())
```

---

## 2. Provisional Indexes: Adaptive Optimization

**Purpose:** Indexes that are created on-demand based on observed query patterns, may become permanent if usage justifies it.

**Philosophy:** Don't pre-index everything. Instead, monitor which fields are frequently filtered on, and create indexes only when the ROI is clear.

**Pattern: Query-Driven Index Creation**

```python
class AdaptiveIndexManager:
    """
    Monitor query patterns and create indexes when a field is queried frequently.
    Similar to database query plan caching (PostgreSQL, MySQL, SQL Server).
    """

    def __init__(self, data_dir: str, entity_type: str):
        self.data_dir = Path(data_dir)
        self.entity_type = entity_type
        self.query_frequency: Dict[str, int] = {}  # field -> query count
        self.indexes: Dict[str, FileBasedSecondaryIndex] = {}
        self.config_file = self.data_dir / f"{entity_type}_index_config.json"
        self.threshold = 10  # Create index after 10 queries on same field

    def query(self, field: str, value: Any) -> Set[str]:
        """Execute a query, tracking field access patterns."""
        self.query_frequency[field] = self.query_frequency.get(field, 0) + 1

        # If index exists, use it
        if field in self.indexes:
            return self.indexes[field].find(field, value)

        # Otherwise, do a full scan
        result = self._full_scan(field, value)

        # Check if we should create an index for this field
        if self.query_frequency[field] >= self.threshold:
            self._create_index_for_field(field)

        return result

    def _full_scan(self, field: str, value: Any) -> Set[str]:
        """Fallback: scan all entities."""
        result = set()
        entity_dir = self.data_dir / self.entity_type
        for json_file in entity_dir.glob("*.json"):
            data = json.loads(json_file.read_text())
            if data.get(field) == value:
                result.add(data.get("id"))
        return result

    def _create_index_for_field(self, field: str) -> None:
        """Create and build a secondary index for the field."""
        print(f"Creating index for {self.entity_type}.{field} (queried {self.query_frequency[field]}x)")
        index = FileBasedSecondaryIndex(self.data_dir, self.entity_type, [field])
        index.build()
        self.indexes[field] = index
        self._save_config()

    def _save_config(self) -> None:
        """Persist which indexes are permanent."""
        config = {
            "permanent_indexes": list(self.indexes.keys()),
            "creation_dates": {f: str(dt.datetime.now()) for f in self.indexes}
        }
        self.config_file.write_text(json.dumps(config, indent=2))

    def load_config(self) -> None:
        """Load saved index configuration at startup."""
        if self.config_file.exists():
            config = json.loads(self.config_file.read_text())
            for field in config.get("permanent_indexes", []):
                index = FileBasedSecondaryIndex(self.data_dir, self.entity_type, [field])
                index.load()  # Load from saved index file
                self.indexes[field] = index
```

**Decision Logic: When to Make an Index Permanent**

```python
class IndexROICalculator:
    """
    Calculate if an index is worth keeping based on:
    - Query frequency
    - Scan cost (how many files to read)
    - Index creation cost (one-time)
    - Index maintenance cost (per write)
    """

    def should_index(self,
                     query_count: int,
                     entity_count: int,
                     field_cardinality: int,
                     avg_result_size: float = 0.1) -> bool:
        """
        Decide if an index should be created.

        Args:
            query_count: Number of times field was queried
            entity_count: Total number of entities
            field_cardinality: Number of distinct values
            avg_result_size: Expected fraction of entities matched (0.0-1.0)

        Returns:
            True if creating the index is worth it
        """

        # Full scan cost: read all entities, filter in Python
        full_scan_cost = entity_count * 0.001  # ms per entity
        indexed_query_cost = field_cardinality * 0.01  # ms for index lookup

        # Break-even point: how many queries before index pays for itself?
        index_creation_cost = 50  # ms to build index
        queries_for_breakeven = index_creation_cost / (full_scan_cost - indexed_query_cost)

        # Add 20% safety margin
        break_even_with_margin = queries_for_breakeven * 1.2

        # Create index if we've already exceeded break-even
        if query_count > break_even_with_margin:
            return True

        # Also consider cardinality: high-cardinality fields are poor indexes
        if field_cardinality > entity_count * 0.5:
            return False

        return False
```

**Workflow: Monitor → Analyze → Promote**

```
┌─────────────────────────────────────────────────────────────┐
│              INDEX LIFECYCLE (Adaptive)                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  1. QUERY MONITORING                                         │
│     └── Count queries on each field                          │
│     └── Track full scans vs indexed queries                  │
│                                                               │
│  2. DECISION: Create Index?                                  │
│     └── If: query_count > 10 AND field_cardinality < 1000   │
│     └── Then: Create provisional index                       │
│                                                               │
│  3. METRICS: Track ROI                                       │
│     └── Query latency before/after                           │
│     └── Disk usage (index size)                              │
│     └── Write latency impact                                 │
│                                                               │
│  4. DECISION: Keep Index?                                    │
│     └── If: queries_per_day > 50 AND latency_improved > 10% │
│     └── Then: Mark as permanent (won't be deleted)           │
│     └── Else: Delete index (it's not worth it)               │
│                                                               │
│  5. ARCHIVAL (Optional)                                      │
│     └── Write metrics to samples/memories/                   │
│     └── Include in session reports                           │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

**Storage Format: Lightweight Provisional Index**

```python
# .got/indexes/provisional/task_status.json
{
  "field": "status",
  "entity_type": "task",
  "created_at": "2025-12-23T10:30:00Z",
  "query_count": 47,
  "promotion_status": "provisional",  # or "permanent"
  "metrics": {
    "avg_query_time_ms": 2.5,
    "full_scan_time_ms": 145.0,
    "queries_per_day": 23,
    "maintenance_cost_ms": 5.0
  },
  "index_data": {
    "pending": ["T-20251221-093045-a1b2", "T-20251222-101230-c3d4"],
    "completed": ["T-20251220-155423-e5f6"]
  }
}
```

---

## 3. Bloom Filters for Deduplication

**Purpose:** Quick probabilistic check "might be a duplicate" before expensive lookups.

**Key Property:** No false negatives (if Bloom says "not present," it's definitely not), but can have false positives.

### 3.1 Sizing Formulas

Given:
- `n` = expected number of elements
- `p` = acceptable false positive rate
- `m` = number of bits needed
- `k` = number of hash functions

**Formula 1: Calculate bits needed for target FP rate**
```
m = -n * ln(p) / (ln(2))²
  ≈ n * 1.44 * log₂(1/p)
```

**Formula 2: Optimal number of hash functions**
```
k = (m / n) * ln(2)
  ≈ 0.693 * (m / n)
```

**Formula 3: Actual FP rate for given m, n, k**
```
p ≈ (1 - e^(-kn/m))^k
```

**Examples:**
```python
import math

def calculate_bloom_params(n: int, target_fp_rate: float) -> tuple:
    """Calculate m (bits) and k (hash functions) for Bloom filter."""

    # Bits needed
    m = -n * math.log(target_fp_rate) / (math.log(2) ** 2)

    # Optimal number of hash functions
    k = (m / n) * math.log(2)

    return int(m), int(k)

# Examples:
configs = [
    (1_000_000, 0.01),      # 1M elements, 1% FP rate
    (1_000_000, 0.001),     # 1M elements, 0.1% FP rate
    (10_000_000, 0.0001),   # 10M elements, 0.01% FP rate
]

for n, target_fp in configs:
    m, k = calculate_bloom_params(n, target_fp)
    bits_per_element = m / n
    print(f"n={n:,}, target_fp={target_fp:.4f}")
    print(f"  → m={int(m):,} bits, k={k}")
    print(f"  → {bits_per_element:.2f} bits/element, {bits_per_element/8:.2f} bytes/element\n")
```

**Output:**
```
n=1,000,000, target_fp=0.01
  → m=9,585,058 bits, k=7
  → 9.58 bits/element, 1.20 bytes/element

n=1,000,000, target_fp=0.001
  → m=14,377,588 bits, k=10
  → 14.38 bits/element, 1.80 bytes/element

n=10,000,000, target_fp=0.0001
  → m=191,701,763 bits, k=13
  → 19.17 bits/element, 2.40 bytes/element
```

**Quick Sizing Table:**
| Target FP Rate | Bits per Element | Bytes per 1M Items | Hash Functions |
|---|---|---|---|
| 1% (0.01) | 9.6 | 1.2 MB | 7 |
| 0.1% (0.001) | 14.4 | 1.8 MB | 10 |
| 0.01% (0.0001) | 19.2 | 2.4 MB | 13 |
| 0.001% (0.00001) | 24.0 | 3.0 MB | 16 |

### 3.2 Bloom Filter Implementation

```python
from collections.abc import Hashable
from hashlib import md5, sha256, sha512
import json

class BloomFilter:
    """
    Simple Bloom filter implementation for deduplication.
    Use case: Check if a record likely exists before expensive disk lookup.
    """

    def __init__(self, num_bits: int, num_hashes: int):
        self.num_bits = num_bits
        self.num_hashes = num_hashes
        self.bits = bytearray(num_bits // 8 + 1)  # Bit array

    def _hash(self, item: str, seed: int) -> int:
        """Generate hash for item with seed."""
        # Use different hash functions by varying the hash input
        combined = f"{item}:{seed}".encode()
        h = int(md5(combined).hexdigest(), 16)
        return h % self.num_bits

    def add(self, item: str) -> None:
        """Add item to Bloom filter."""
        for seed in range(self.num_hashes):
            bit_position = self._hash(item, seed)
            byte_index = bit_position // 8
            bit_offset = bit_position % 8
            self.bits[byte_index] |= (1 << bit_offset)

    def might_contain(self, item: str) -> bool:
        """Check if item might be in the set (can have false positives)."""
        for seed in range(self.num_hashes):
            bit_position = self._hash(item, seed)
            byte_index = bit_position // 8
            bit_offset = bit_position % 8
            if not (self.bits[byte_index] & (1 << bit_offset)):
                return False  # Definitely not in set
        return True  # Probably in set (but might be false positive)

    def save(self, filepath: str) -> None:
        """Persist Bloom filter to file."""
        data = {
            "num_bits": self.num_bits,
            "num_hashes": self.num_hashes,
            "bits": self.bits.hex()  # Hex-encode bits
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)

    @classmethod
    def load(cls, filepath: str) -> "BloomFilter":
        """Load Bloom filter from file."""
        with open(filepath) as f:
            data = json.load(f)

        bf = cls(data["num_bits"], data["num_hashes"])
        bf.bits = bytearray.fromhex(data["bits"])
        return bf
```

### 3.3 Using Bloom Filters for Deduplication

```python
class DeduplicationManager:
    """
    Use Bloom filter to quickly check for duplicates before expensive validation.

    Workflow:
    1. Bloom filter says "NOT present" → Definitely not a duplicate, add it
    2. Bloom filter says "MIGHT be present" → Do expensive check (disk lookup)
    """

    def __init__(self, data_dir: str, entity_type: str,
                 target_fp_rate: float = 0.001):
        self.data_dir = Path(data_dir)
        self.entity_type = entity_type
        self.target_fp_rate = target_fp_rate

        # Calculate Bloom filter size based on entity count
        import os
        entity_dir = self.data_dir / entity_type
        entity_count = len(list(entity_dir.glob("*.json"))) if entity_dir.exists() else 1000

        # Reserve 20% extra capacity
        n = int(entity_count * 1.2)
        m, k = calculate_bloom_params(n, target_fp_rate)

        self.bloom = BloomFilter(m, k)
        self.bloom_file = self.data_dir / f"{entity_type}_dedup.bloom"
        self._load_or_build_bloom()

    def _load_or_build_bloom(self) -> None:
        """Load existing Bloom filter or build from scratch."""
        if self.bloom_file.exists():
            self.bloom = BloomFilter.load(self.bloom_file)
        else:
            # Build from existing entities
            entity_dir = self.data_dir / self.entity_type
            if entity_dir.exists():
                for json_file in entity_dir.glob("*.json"):
                    data = json.loads(json_file.read_text())
                    dedup_key = self._get_dedup_key(data)
                    self.bloom.add(dedup_key)
            self.bloom.save(self.bloom_file)

    def _get_dedup_key(self, entity_data: Dict) -> str:
        """Extract deduplication key from entity."""
        # Example: for tasks, use (title, status) tuple
        key_fields = ["title", "status"]
        key_parts = [str(entity_data.get(f, "")) for f in key_fields]
        return "|".join(key_parts)

    def check_and_add(self, entity_data: Dict) -> bool:
        """
        Check if entity is likely a duplicate.

        Returns:
            True if can safely add (definitely not duplicate)
            False if might be duplicate (needs manual verification)
        """
        dedup_key = self._get_dedup_key(entity_data)

        # Quick Bloom check
        if not self.bloom.might_contain(dedup_key):
            # Definitely not a duplicate, safe to add
            self.bloom.add(dedup_key)
            self.bloom.save(self.bloom_file)
            return True

        # Might be a duplicate, requires expensive check
        return False

    def manual_add(self, entity_data: Dict) -> None:
        """After manual verification, add to Bloom filter."""
        dedup_key = self._get_dedup_key(entity_data)
        self.bloom.add(dedup_key)
        self.bloom.save(self.bloom_file)
```

### 3.4 False Positive Handling

```python
class BloomFilterWithRecovery:
    """
    Bloom filter with false positive recovery:
    If user reports a false positive, rebuild with larger size.
    """

    def __init__(self, data_dir: str, entity_type: str):
        self.data_dir = Path(data_dir)
        self.entity_type = entity_type
        self.metadata_file = self.data_dir / f"{entity_type}_bloom_meta.json"

        self.num_bits = 100_000  # Start conservative
        self.num_hashes = 7
        self.false_positives = 0  # Count reported FPs
        self.load_or_create()

    def load_or_create(self) -> None:
        """Load metadata and Bloom filter."""
        if self.metadata_file.exists():
            meta = json.loads(self.metadata_file.read_text())
            self.num_bits = meta["num_bits"]
            self.num_hashes = meta["num_hashes"]
            self.false_positives = meta.get("false_positives", 0)

        self.bloom_file = self.data_dir / f"{entity_type}_dedup.bloom"
        if self.bloom_file.exists():
            self.bloom = BloomFilter.load(self.bloom_file)
        else:
            self.bloom = BloomFilter(self.num_bits, self.num_hashes)

    def report_false_positive(self, entity_id: str) -> None:
        """User reported a false positive."""
        self.false_positives += 1

        # If false positive rate exceeds 5%, rebuild with larger size
        target_fp_rate = 0.001
        actual_fp_rate = self.false_positives / max(1, self.bloom.num_bits // 100)

        if actual_fp_rate > 0.05:  # 5% actual FP rate
            print(f"Rebuilding Bloom filter (FP rate: {actual_fp_rate:.1%})")
            self._rebuild_larger()

    def _rebuild_larger(self) -> None:
        """Rebuild Bloom filter with 2x more bits."""
        old_bloom = self.bloom
        self.num_bits *= 2
        self.bloom = BloomFilter(self.num_bits, self.num_hashes)

        # Re-add all items to new filter
        entity_dir = self.data_dir / self.entity_type
        if entity_dir.exists():
            for json_file in entity_dir.glob("*.json"):
                data = json.loads(json_file.read_text())
                # Assume data has an "id" field
                self.bloom.add(data.get("id", ""))

        self.bloom.save(self.bloom_file)
        self._save_metadata()

    def _save_metadata(self) -> None:
        """Save Bloom filter metadata."""
        meta = {
            "num_bits": self.num_bits,
            "num_hashes": self.num_hashes,
            "false_positives": self.false_positives,
            "target_fp_rate": 0.001
        }
        self.metadata_file.write_text(json.dumps(meta, indent=2))
```

---

## 4. Inverted Indexes for Queries

**Purpose:** Answer "which entities reference X" efficiently. Essential for full-text search and dependency tracking.

### 4.1 Simple Inverted Index

```python
class SimpleInvertedIndex:
    """
    Maps terms to entity IDs that contain them.

    Structure:
    {
        "neural": ["doc-1", "doc-3", "doc-5"],
        "networks": ["doc-1", "doc-2"],
        "learning": ["doc-2", "doc-4"]
    }
    """

    def __init__(self, data_dir: str, entity_type: str):
        self.data_dir = Path(data_dir)
        self.entity_type = entity_type
        self.index: Dict[str, Set[str]] = {}
        self.index_file = self.data_dir / f"{entity_type}_inverted.json"

    def tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase, split on whitespace."""
        return text.lower().split()

    def build(self) -> None:
        """Build inverted index from all entities."""
        self.index.clear()

        entity_dir = self.data_dir / self.entity_type
        for json_file in entity_dir.glob("*.json"):
            data = json.loads(json_file.read_text())
            entity_id = data.get("id")
            if not entity_id:
                continue

            # Tokenize content fields (assume "content" and "title")
            for field in ["content", "title"]:
                text = data.get(field, "")
                if text:
                    tokens = self.tokenize(text)
                    for token in tokens:
                        if token not in self.index:
                            self.index[token] = set()
                        self.index[token].add(entity_id)

    def find_by_term(self, term: str) -> Set[str]:
        """Find all entities containing term."""
        return self.index.get(term.lower(), set())

    def find_by_terms_any(self, terms: List[str]) -> Set[str]:
        """Find entities containing ANY of the terms (OR query)."""
        result = set()
        for term in terms:
            result.update(self.find_by_term(term))
        return result

    def find_by_terms_all(self, terms: List[str]) -> Set[str]:
        """Find entities containing ALL terms (AND query)."""
        if not terms:
            return set()

        # Start with first term, intersect with others
        result = self.find_by_term(terms[0])
        for term in terms[1:]:
            result &= self.find_by_term(term)
        return result

    def save(self) -> None:
        """Persist index to JSON."""
        serializable = {term: list(ids) for term, ids in self.index.items()}
        self.index_file.write_text(json.dumps(serializable, indent=2))

    def load(self) -> None:
        """Load index from JSON."""
        if self.index_file.exists():
            data = json.loads(self.index_file.read_text())
            self.index = {term: set(ids) for term, ids in data.items()}
```

### 4.2 Ranked Inverted Index (with TF-IDF)

```python
class RankedInvertedIndex:
    """
    Inverted index that stores relevance scores (TF-IDF).

    Structure:
    {
        "neural": {
            "doc-1": 0.85,  # TF-IDF score
            "doc-3": 0.72
        },
        "networks": {
            "doc-1": 0.91,
            "doc-2": 0.43
        }
    }
    """

    def __init__(self, data_dir: str, entity_type: str):
        self.data_dir = Path(data_dir)
        self.entity_type = entity_type
        self.index: Dict[str, Dict[str, float]] = {}  # term -> {entity_id -> score}
        self.doc_lengths: Dict[str, int] = {}  # entity_id -> token count
        self.index_file = self.data_dir / f"{entity_type}_ranked.json"

    def tokenize(self, text: str) -> List[str]:
        """Tokenization with stopword removal."""
        stopwords = {"the", "a", "an", "and", "or", "is", "to", "of", "in"}
        tokens = text.lower().split()
        return [t for t in tokens if t not in stopwords]

    def build(self) -> None:
        """Build index with TF-IDF scores."""
        self.index.clear()
        self.doc_lengths.clear()

        # First pass: collect term frequencies
        term_frequency: Dict[str, Dict[str, int]] = {}  # term -> {entity_id -> count}

        entity_dir = self.data_dir / self.entity_type
        entities = list(entity_dir.glob("*.json")) if entity_dir.exists() else []

        for json_file in entities:
            data = json.loads(json_file.read_text())
            entity_id = data.get("id")
            if not entity_id:
                continue

            # Tokenize and count
            text = data.get("content", "") + " " + data.get("title", "")
            tokens = self.tokenize(text)
            self.doc_lengths[entity_id] = len(tokens)

            token_counts: Dict[str, int] = {}
            for token in tokens:
                token_counts[token] = token_counts.get(token, 0) + 1

            # Add to term frequency
            for token, count in token_counts.items():
                if token not in term_frequency:
                    term_frequency[token] = {}
                term_frequency[token][entity_id] = count

        # Second pass: compute TF-IDF
        num_docs = len(entities)

        for term, doc_freqs in term_frequency.items():
            self.index[term] = {}

            # Inverse document frequency
            idf = math.log(num_docs / len(doc_freqs))

            for entity_id, count in doc_freqs.items():
                # Term frequency (normalized)
                tf = count / self.doc_lengths[entity_id]

                # TF-IDF score
                score = tf * idf
                self.index[term][entity_id] = score

    def search(self, query: str, top_n: int = 5) -> List[Tuple[str, float]]:
        """Search for query, return top N entities by relevance."""
        query_tokens = self.tokenize(query)

        # Accumulate scores for each entity
        scores: Dict[str, float] = {}
        for token in query_tokens:
            if token in self.index:
                for entity_id, score in self.index[token].items():
                    scores[entity_id] = scores.get(entity_id, 0) + score

        # Sort by score and return top N
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return ranked[:top_n]

    def save(self) -> None:
        """Persist ranked index to JSON."""
        serializable = {term: list(ids.items()) for term, ids in self.index.items()}
        self.index_file.write_text(json.dumps(serializable, indent=2))

    def load(self) -> None:
        """Load ranked index from JSON."""
        if self.index_file.exists():
            data = json.loads(self.index_file.read_text())
            self.index = {term: dict(items) for term, items in data.items()}
```

### 4.3 Graph-Based Inverted Index (for dependencies)

```python
class DependencyInvertedIndex:
    """
    Specialized inverted index for tracking dependencies.

    Use case: "Find all tasks that depend on task X"

    Structure:
    {
        "T-20251221-093045-a1b2": {  # task_id
            "depends_on": ["T-20251221-080000-b2c3"],  # IDs of tasks it depends on
            "depended_by": ["T-20251222-100000-c3d4"]  # IDs of tasks depending on it
        }
    }
    """

    def __init__(self, data_dir: str, entity_type: str = "task"):
        self.data_dir = Path(data_dir)
        self.entity_type = entity_type
        self.index: Dict[str, Dict[str, Set[str]]] = {}  # entity_id -> {relation -> set of IDs}
        self.index_file = self.data_dir / f"{entity_type}_dependencies.json"

    def build(self) -> None:
        """Build index from all entities."""
        self.index.clear()

        entity_dir = self.data_dir / self.entity_type
        for json_file in entity_dir.glob("*.json"):
            data = json.loads(json_file.read_text())
            entity_id = data.get("id")
            if not entity_id:
                continue

            self.index[entity_id] = {
                "depends_on": set(data.get("depends_on", [])),
                "depended_by": set(data.get("depended_by", [])),
                "blocks": set(data.get("blocks", [])),
                "blocked_by": set(data.get("blocked_by", []))
            }

    def find_dependents(self, entity_id: str) -> Set[str]:
        """Find all entities that depend on this one."""
        if entity_id not in self.index:
            return set()
        return self.index[entity_id].get("depended_by", set())

    def find_dependencies(self, entity_id: str) -> Set[str]:
        """Find all entities this one depends on."""
        if entity_id not in self.index:
            return set()
        return self.index[entity_id].get("depends_on", set())

    def add_dependency(self, from_id: str, to_id: str, relation: str = "depends_on") -> None:
        """Add a dependency relationship."""
        # Forward relationship
        if from_id not in self.index:
            self.index[from_id] = {"depends_on": set(), "depended_by": set(), "blocks": set(), "blocked_by": set()}
        self.index[from_id][relation].add(to_id)

        # Reverse relationship
        reverse_relation = self._reverse_relation(relation)
        if to_id not in self.index:
            self.index[to_id] = {"depends_on": set(), "depended_by": set(), "blocks": set(), "blocked_by": set()}
        self.index[to_id][reverse_relation].add(from_id)

    def _reverse_relation(self, relation: str) -> str:
        """Get the reverse relationship."""
        reverses = {
            "depends_on": "depended_by",
            "depended_by": "depends_on",
            "blocks": "blocked_by",
            "blocked_by": "blocks"
        }
        return reverses.get(relation, relation)

    def find_transitive_dependents(self, entity_id: str, max_depth: int = 10) -> Set[str]:
        """Find all entities that transitively depend on this one (BFS)."""
        visited = set()
        queue = [entity_id]
        depth = 0

        while queue and depth < max_depth:
            depth += 1
            next_queue = []

            for eid in queue:
                dependents = self.find_dependents(eid)
                for dep in dependents:
                    if dep not in visited:
                        visited.add(dep)
                        next_queue.append(dep)

            queue = next_queue

        return visited

    def save(self) -> None:
        """Persist index to JSON."""
        serializable = {
            eid: {rel: list(ids) for rel, ids in rels.items()}
            for eid, rels in self.index.items()
        }
        self.index_file.write_text(json.dumps(serializable, indent=2))

    def load(self) -> None:
        """Load index from JSON."""
        if self.index_file.exists():
            data = json.loads(self.index_file.read_text())
            self.index = {
                eid: {rel: set(ids) for rel, ids in rels.items()}
                for eid, rels in data.items()
            }
```

---

## 5. Index Invalidation Strategies

**Problem:** When underlying data changes, indexes become stale. How to detect and rebuild efficiently?

### 5.1 Staleness Detection

```python
class IndexStalenessDetector:
    """
    Track staleness of indexes using multiple strategies.
    """

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.metadata_file = self.data_dir / "index_metadata.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load staleness metadata."""
        if self.metadata_file.exists():
            return json.loads(self.metadata_file.read_text())
        return {}

    def _save_metadata(self) -> None:
        """Persist staleness metadata."""
        self.metadata_file.write_text(json.dumps(self.metadata, indent=2))

    # STRATEGY 1: Timestamp-based staleness
    def mark_fresh(self, index_name: str) -> None:
        """Mark index as freshly built."""
        if index_name not in self.metadata:
            self.metadata[index_name] = {}

        self.metadata[index_name]["built_at"] = dt.datetime.now().isoformat()
        self._save_metadata()

    def is_stale_by_age(self, index_name: str, max_age_hours: int = 24) -> bool:
        """Check if index is older than max_age_hours."""
        if index_name not in self.metadata:
            return True  # Unknown index is stale

        built_at_str = self.metadata[index_name].get("built_at")
        if not built_at_str:
            return True

        built_at = dt.datetime.fromisoformat(built_at_str)
        age = dt.datetime.now() - built_at
        return age > dt.timedelta(hours=max_age_hours)

    # STRATEGY 2: Hash-based change detection
    def compute_source_hash(self, entity_dir: Path) -> str:
        """Compute hash of all source files."""
        import hashlib

        file_hashes = []
        for json_file in sorted(entity_dir.glob("*.json")):
            content = json_file.read_bytes()
            file_hash = hashlib.md5(content).hexdigest()
            file_hashes.append(file_hash)

        combined = "".join(file_hashes).encode()
        return hashlib.md5(combined).hexdigest()

    def is_stale_by_hash(self, index_name: str, entity_dir: Path) -> bool:
        """Check if source files changed since index was built."""
        current_hash = self.compute_source_hash(entity_dir)
        previous_hash = self.metadata.get(index_name, {}).get("source_hash")

        changed = current_hash != previous_hash
        if not changed:
            self.mark_fresh(index_name)  # Still fresh
        return changed

    def mark_fresh_with_hash(self, index_name: str, entity_dir: Path) -> None:
        """Mark index as fresh and store source hash."""
        if index_name not in self.metadata:
            self.metadata[index_name] = {}

        self.metadata[index_name]["built_at"] = dt.datetime.now().isoformat()
        self.metadata[index_name]["source_hash"] = self.compute_source_hash(entity_dir)
        self._save_metadata()

    # STRATEGY 3: Version-based staleness
    def mark_stale(self, index_name: str, reason: str = "") -> None:
        """Explicitly mark index as stale."""
        if index_name not in self.metadata:
            self.metadata[index_name] = {}

        self.metadata[index_name]["stale"] = True
        self.metadata[index_name]["stale_reason"] = reason
        self._save_metadata()

    def is_stale_explicit(self, index_name: str) -> bool:
        """Check if index was explicitly marked stale."""
        return self.metadata.get(index_name, {}).get("stale", False)
```

### 5.2 Incremental Index Updates

Instead of rebuilding indexes from scratch, update them incrementally when entities change.

```python
class IncrementalIndexUpdater:
    """
    Update indexes incrementally rather than rebuilding.
    Called after entities are added, modified, or deleted.
    """

    def __init__(self, index: SimpleInvertedIndex,
                 secondary_index: FileBasedSecondaryIndex,
                 dependency_index: DependencyInvertedIndex):
        self.inverted = index
        self.secondary = secondary_index
        self.dependency = dependency_index

    def handle_add(self, entity_id: str, entity_data: Dict) -> None:
        """Add new entity to all indexes."""
        # Add to inverted index
        tokens = self.inverted.tokenize(entity_data.get("content", ""))
        for token in tokens:
            if token not in self.inverted.index:
                self.inverted.index[token] = set()
            self.inverted.index[token].add(entity_id)

        # Add to secondary indexes
        self.secondary.update(entity_id, entity_data)

        # Add to dependency index
        for dep_id in entity_data.get("depends_on", []):
            self.dependency.add_dependency(entity_id, dep_id, "depends_on")

    def handle_modify(self, entity_id: str, old_data: Dict, new_data: Dict) -> None:
        """Update indexes when entity is modified."""
        # Remove old entries from inverted index
        old_tokens = self.inverted.tokenize(old_data.get("content", ""))
        for token in old_tokens:
            if token in self.inverted.index:
                self.inverted.index[token].discard(entity_id)
                if not self.inverted.index[token]:
                    del self.inverted.index[token]

        # Add new entries
        new_tokens = self.inverted.tokenize(new_data.get("content", ""))
        for token in new_tokens:
            if token not in self.inverted.index:
                self.inverted.index[token] = set()
            self.inverted.index[token].add(entity_id)

        # Update secondary indexes
        self.secondary.update(entity_id, new_data)

        # Update dependencies if changed
        old_deps = set(old_data.get("depends_on", []))
        new_deps = set(new_data.get("depends_on", []))

        if old_deps != new_deps:
            # Remove old dependencies
            for dep_id in old_deps - new_deps:
                self.dependency.index[entity_id]["depends_on"].discard(dep_id)
                self.dependency.index[dep_id]["depended_by"].discard(entity_id)

            # Add new dependencies
            for dep_id in new_deps - old_deps:
                self.dependency.add_dependency(entity_id, dep_id, "depends_on")

    def handle_delete(self, entity_id: str, entity_data: Dict) -> None:
        """Remove entity from all indexes."""
        # Remove from inverted index
        tokens = self.inverted.tokenize(entity_data.get("content", ""))
        for token in tokens:
            if token in self.inverted.index:
                self.inverted.index[token].discard(entity_id)

        # Remove from secondary indexes
        self.secondary.remove(entity_id)

        # Remove from dependency index
        if entity_id in self.dependency.index:
            del self.dependency.index[entity_id]
            for eid, rels in self.dependency.index.items():
                for rel_set in rels.values():
                    rel_set.discard(entity_id)
```

### 5.3 Lazy Rebuild with Background Jobs

```python
import threading
from queue import Queue

class IndexRebuildManager:
    """
    Coordinate lazy rebuilding of indexes.
    Use background thread to rebuild stale indexes without blocking.
    """

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.rebuild_queue: Queue = Queue()
        self.staleness_detector = IndexStalenessDetector(data_dir)
        self.rebuild_thread = None
        self.stop_event = threading.Event()

    def start_rebuild_worker(self) -> None:
        """Start background thread for rebuilding indexes."""
        self.stop_event.clear()
        self.rebuild_thread = threading.Thread(target=self._rebuild_worker, daemon=True)
        self.rebuild_thread.start()

    def _rebuild_worker(self) -> None:
        """Background worker that rebuilds indexes from queue."""
        while not self.stop_event.is_set():
            try:
                # Wait for up to 1 second for a rebuild request
                index_name, rebuild_func = self.rebuild_queue.get(timeout=1)

                print(f"Rebuilding index: {index_name}")
                rebuild_func()
                self.staleness_detector.mark_fresh(index_name)
                print(f"Rebuilt index: {index_name}")

            except Exception:
                continue

    def request_rebuild(self, index_name: str, rebuild_func) -> None:
        """Request index rebuild (async)."""
        self.rebuild_queue.put((index_name, rebuild_func))

    def check_and_rebuild_if_stale(self, index_name: str, rebuild_func) -> None:
        """Check if stale, rebuild synchronously if needed."""
        if self.staleness_detector.is_stale_explicit(index_name):
            print(f"Index {index_name} is stale, rebuilding...")
            rebuild_func()
            self.staleness_detector.mark_fresh(index_name)

    def stop(self) -> None:
        """Stop rebuild worker."""
        self.stop_event.set()
        if self.rebuild_thread:
            self.rebuild_thread.join(timeout=5)
```

### 5.4 Comprehensive Index Rebuilding Pattern

```python
class ComprehensiveIndexManager:
    """
    Unified interface for managing all indexes.
    Handles creation, updates, staleness detection, and rebuilding.
    """

    def __init__(self, data_dir: str, entity_type: str):
        self.data_dir = Path(data_dir)
        self.entity_type = entity_type

        # All indexes
        self.primary = FileBasedPrimaryIndex(data_dir, entity_type)
        self.secondary = FileBasedSecondaryIndex(data_dir, entity_type, ["status", "priority"])
        self.inverted = SimpleInvertedIndex(data_dir, entity_type)
        self.dependency = DependencyInvertedIndex(data_dir, entity_type)
        self.bloom = BloomFilter(100000, 7)

        # Utilities
        self.staleness = IndexStalenessDetector(data_dir)
        self.incremental = IncrementalIndexUpdater(self.inverted, self.secondary, self.dependency)
        self.rebuild_manager = IndexRebuildManager(data_dir)

        # Load all indexes at startup
        self._load_all()

    def _load_all(self) -> None:
        """Load all indexes from disk."""
        self.primary.build()
        self.secondary.load()
        self.inverted.load()
        self.dependency.load()

        if (self.data_dir / f"{self.entity_type}_dedup.bloom").exists():
            self.bloom = BloomFilter.load(self.data_dir / f"{self.entity_type}_dedup.bloom")

    def _save_all(self) -> None:
        """Save all indexes to disk."""
        self.secondary.save()
        self.inverted.save()
        self.dependency.save()
        self.bloom.save(self.data_dir / f"{self.entity_type}_dedup.bloom")

    def add_entity(self, entity_id: str, entity_data: Dict) -> None:
        """Add new entity, updating all indexes."""
        self.incremental.handle_add(entity_id, entity_data)
        self.bloom.add(entity_id)
        self._save_all()

    def modify_entity(self, entity_id: str, old_data: Dict, new_data: Dict) -> None:
        """Modify entity, updating all indexes."""
        self.incremental.handle_modify(entity_id, old_data, new_data)
        self._save_all()

    def delete_entity(self, entity_id: str, entity_data: Dict) -> None:
        """Delete entity, updating all indexes."""
        self.incremental.handle_delete(entity_id, entity_data)
        self._save_all()

    def ensure_fresh(self, force_rebuild: bool = False) -> None:
        """Ensure all indexes are up-to-date."""
        entity_dir = self.data_dir / self.entity_type

        indexes_to_check = [
            ("primary", self.primary.build),
            ("secondary", self.secondary.build),
            ("inverted", self.inverted.build),
            ("dependency", self.dependency.build)
        ]

        for index_name, rebuild_func in indexes_to_check:
            if force_rebuild or self.staleness.is_stale_by_hash(index_name, entity_dir):
                print(f"Rebuilding {index_name} index...")
                rebuild_func()
                self.staleness.mark_fresh_with_hash(index_name, entity_dir)

    def get_entity(self, entity_id: str) -> Optional[Dict]:
        """Get entity by ID (using primary index)."""
        return self.primary.get(entity_id)

    def find_by_field(self, field: str, value: Any) -> Set[str]:
        """Find entities by field value (using secondary index)."""
        return self.secondary.find(field, value)

    def search(self, query: str) -> Set[str]:
        """Search for entities (using inverted index)."""
        tokens = self.inverted.tokenize(query)
        return self.inverted.find_by_terms_all(tokens)

    def find_dependents(self, entity_id: str) -> Set[str]:
        """Find entities that depend on this one (using dependency index)."""
        return self.dependency.find_dependents(entity_id)
```

---

## 6. Complete Workflow Example

```python
# Initialize system
manager = ComprehensiveIndexManager(
    data_dir="/home/user/Opus-code-test/.got",
    entity_type="task"
)

# Ensure indexes are fresh at startup
manager.ensure_fresh()

# Add a new task
new_task = {
    "id": "T-20251223-093045-a1b2",
    "title": "Fix authentication bug",
    "content": "Resolve OAuth token refresh issue in login flow",
    "status": "pending",
    "priority": "high",
    "depends_on": ["T-20251222-000000-c3d4"]
}
manager.add_entity(new_task["id"], new_task)

# Query examples
print("Find all pending tasks:", manager.find_by_field("status", "pending"))
print("Find by search:", manager.search("authentication OAuth"))
print("Find dependents:", manager.find_dependents("T-20251222-000000-c3d4"))

# Check deduplication
is_dupe = not manager.bloom.might_contain(new_task["id"])
print(f"Is duplicate: {is_dupe}")

# Mark as needing rebuild later
manager.staleness.mark_stale("secondary", reason="new status value added")

# Rebuild when needed
manager.ensure_fresh()
```

---

## Summary Table: When to Use Each Pattern

| Pattern | Use When | Cost | Benefit | Rebuild Frequency |
|---------|----------|------|---------|-------------------|
| **Primary Index** | Frequent ID lookups | Low (10-20%) | O(1) access | Once, update incrementally |
| **Secondary Index** | Field filtering, moderate cardinality | Medium (20-50%) | Fast queries | Incremental updates, rebuild weekly |
| **Provisional Index** | Unknown access patterns | Medium (starts small) | Adaptive cost | Learn from queries, no scheduled rebuild |
| **Bloom Filter** | Dedup checks | Very Low (1-3%) | Probabilistic filter | Rebuild when FP rate rises |
| **Inverted Index** | Full-text search, dependencies | Medium (20-40%) | Complex queries possible | Incremental updates, rebuild monthly |

**Golden Rules:**

1. **Load indexes once at startup**, keep in memory
2. **Update indexes incrementally** when data changes
3. **Detect staleness** via timestamps, hashes, or explicit marks
4. **Rebuild lazily** in background thread, not blocking
5. **Measure ROI** before creating new indexes
6. **Use Bloom filters** as probabilistic pre-filters, not authoritative
7. **Compress indexes** periodically (weekly for provisional, monthly for permanent)

