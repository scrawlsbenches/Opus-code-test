# Design: Space-Efficient Data Structures for Git Training

**Date:** 2025-12-27
**Status:** Proposal
**Related:** `samples/memories/2025-12-27-knowledge-transfer-sparkslm-git-training.md`

---

## Problem Statement

Training SparkSLM on git history requires:
1. Processing potentially millions of commits
2. Deduplicating across branches (by SHA)
3. Looking up branch weights efficiently
4. Storing quality signals per commit
5. Accumulating weighted n-gram counts
6. Supporting streaming/incremental updates

Standard approaches (dicts, sets, lists) work but don't scale well for large repositories.

---

## Proposed Data Structures

### 1. Bloom Filter for SHA Deduplication

**Problem:** Check if a commit SHA has been seen before.

**Standard approach:** `seen: Set[str]` - O(1) but ~40 bytes per SHA (40-char string + set overhead)

**Proposed:** Bloom filter - O(1) with ~10 bits (1.25 bytes) per element

```python
class CommitBloomFilter:
    """Space-efficient commit deduplication using Bloom filter."""

    def __init__(self, expected_commits: int = 100_000, fp_rate: float = 0.01):
        """
        Args:
            expected_commits: Expected number of unique commits
            fp_rate: Acceptable false positive rate (0.01 = 1%)
        """
        # Calculate optimal size: m = -n*ln(p) / (ln(2)^2)
        import math
        self.size = int(-expected_commits * math.log(fp_rate) / (math.log(2) ** 2))
        self.num_hashes = int(self.size / expected_commits * math.log(2))
        self.bits = bytearray((self.size + 7) // 8)

    def _hashes(self, sha: str) -> list:
        """Generate k hash positions using double hashing."""
        import hashlib
        h1 = int(hashlib.md5(sha.encode()).hexdigest(), 16)
        h2 = int(hashlib.sha1(sha.encode()).hexdigest(), 16)
        return [(h1 + i * h2) % self.size for i in range(self.num_hashes)]

    def add(self, sha: str) -> None:
        """Add a SHA to the filter."""
        for pos in self._hashes(sha):
            self.bits[pos // 8] |= (1 << (pos % 8))

    def __contains__(self, sha: str) -> bool:
        """Check if SHA might be in the filter."""
        return all(
            self.bits[pos // 8] & (1 << (pos % 8))
            for pos in self._hashes(sha)
        )

# Memory comparison for 100K commits:
# Set[str]: ~4MB (40 bytes × 100K)
# BloomFilter: ~120KB (10 bits × 100K)
# Savings: 97%
```

**Trade-off:** 1% false positives mean we might skip 1 in 100 commits. Acceptable for training (not for deduplication logs).

---

### 2. Radix Trie for Branch Weight Lookup

**Problem:** Map branch names to weights, with prefix matching.

**Standard approach:** Linear scan through prefix patterns

**Proposed:** Radix trie (compressed prefix tree)

```python
class BranchWeightTrie:
    """Radix trie for efficient branch weight lookup."""

    def __init__(self):
        self.root = {'_weight': 0.5}  # Default weight

    def add_prefix(self, prefix: str, weight: float) -> None:
        """Add a branch prefix with its weight."""
        node = self.root
        for char in prefix.lower():
            if char not in node:
                node[char] = {}
            node = node[char]
        node['_weight'] = weight

    def get_weight(self, branch: str) -> float:
        """Get weight for branch, matching longest prefix."""
        node = self.root
        best_weight = self.root.get('_weight', 0.5)

        for char in branch.lower():
            if char not in node:
                break
            node = node[char]
            if '_weight' in node:
                best_weight = node['_weight']

        return best_weight

# Usage:
trie = BranchWeightTrie()
trie.add_prefix('main', 1.0)
trie.add_prefix('master', 1.0)
trie.add_prefix('feature/', 0.6)
trie.add_prefix('claude/', 0.4)
trie.add_prefix('release/', 0.9)

trie.get_weight('claude/sparkslm-training-abc')  # → 0.4
trie.get_weight('feature/auth-system')           # → 0.6
trie.get_weight('unknown-branch')                # → 0.5 (default)
```

**Note:** For the typical case (<20 branch prefixes), a simple dict with linear scan is probably fine. Trie shines if you have hundreds of complex patterns.

---

### 3. Packed Bit Vector for Quality Signals

**Problem:** Store 8+ boolean quality signals per commit.

**Standard approach:** Dict with boolean values - ~200 bytes per commit

**Proposed:** Pack signals into single byte

```python
from dataclasses import dataclass
from typing import Dict
import struct

class QualitySignals:
    """Bit-packed quality signals for commits."""

    # Signal bit positions
    MERGED = 0b00000001
    HAS_TESTS = 0b00000010
    CI_PASSED = 0b00000100
    REVERTED = 0b00001000
    BUG_FIX = 0b00010000
    BREAKING = 0b00100000
    DOCS_ONLY = 0b01000000
    REFACTOR = 0b10000000

    # Weight multipliers for each signal
    MULTIPLIERS = {
        MERGED: 1.2,
        HAS_TESTS: 1.1,
        CI_PASSED: 1.1,
        REVERTED: 0.0,
        BUG_FIX: 1.1,
        BREAKING: 0.8,
        DOCS_ONLY: 0.7,
        REFACTOR: 0.9,
    }

    def __init__(self, capacity: int = 100_000):
        """Pre-allocate storage for expected commits."""
        self.signals = bytearray(capacity)
        self.sha_to_idx: Dict[str, int] = {}
        self.next_idx = 0

    def register(self, sha: str) -> int:
        """Register a commit, return its index."""
        if sha in self.sha_to_idx:
            return self.sha_to_idx[sha]
        idx = self.next_idx
        self.sha_to_idx[sha] = idx
        self.next_idx += 1
        return idx

    def set_signal(self, sha: str, signal: int) -> None:
        """Set a quality signal for a commit."""
        idx = self.sha_to_idx.get(sha)
        if idx is not None:
            self.signals[idx] |= signal

    def get_multiplier(self, sha: str) -> float:
        """Compute combined weight multiplier from signals."""
        idx = self.sha_to_idx.get(sha)
        if idx is None:
            return 1.0

        byte = self.signals[idx]
        multiplier = 1.0

        for signal, mult in self.MULTIPLIERS.items():
            if byte & signal:
                if mult == 0.0:  # Reverted - instant zero
                    return 0.0
                multiplier *= mult

        return multiplier

# Memory comparison for 100K commits:
# Dict[str, Dict[str, bool]]: ~20MB
# QualitySignals: ~4MB (SHA index) + 100KB (signals) = ~4.1MB
# Savings: 80%
```

---

### 4. Count-Min Sketch for Approximate N-gram Counting

**Problem:** Count n-grams at scale without storing exact counts.

**Standard approach:** `Dict[tuple, Counter]` - unbounded memory growth

**Proposed:** Count-min sketch for approximate frequency estimation

```python
import hashlib
from typing import List, Tuple

class CountMinSketch:
    """
    Probabilistic n-gram counter with bounded memory.

    Guarantees: count(x) <= true_count(x) + epsilon * total_count
    """

    def __init__(self, width: int = 10_000, depth: int = 5):
        """
        Args:
            width: Number of counters per row (~error rate)
            depth: Number of hash functions (~confidence)
        """
        self.width = width
        self.depth = depth
        self.table = [[0.0] * width for _ in range(depth)]
        self.total = 0.0

    def _hashes(self, key: Tuple[str, ...]) -> List[int]:
        """Generate d hash values for a key."""
        key_str = '|'.join(key)
        hashes = []
        for i in range(self.depth):
            h = hashlib.md5(f"{i}:{key_str}".encode()).hexdigest()
            hashes.append(int(h, 16) % self.width)
        return hashes

    def add(self, context: Tuple[str, ...], word: str, weight: float = 1.0) -> None:
        """Add weighted count for (context, word) pair."""
        key = context + (word,)
        for i, pos in enumerate(self._hashes(key)):
            self.table[i][pos] += weight
        self.total += weight

    def get(self, context: Tuple[str, ...], word: str) -> float:
        """Get estimated count (minimum across all hash positions)."""
        key = context + (word,)
        return min(
            self.table[i][pos]
            for i, pos in enumerate(self._hashes(key))
        )

    def probability(self, context: Tuple[str, ...], word: str) -> float:
        """Estimate P(word | context)."""
        count = self.get(context, word)
        context_total = sum(
            self.get(context, w) for w in self._get_seen_words(context)
        )
        if context_total == 0:
            return 0.0
        return count / context_total

# Memory: width × depth × 8 bytes = 400KB for 10K × 5
# vs. exact Dict: potentially unbounded (10MB+ typical)
```

**Trade-off:** Approximate counts with bounded error. Good for:
- Very large corpora
- When exact counts aren't critical
- Memory-constrained environments

**Not recommended when:** You need exact probabilities or small vocab size.

---

### 5. Streaming Iterator Pattern

**Problem:** Process git history without loading everything into memory.

**Proposed:** Generator-based pipeline

```python
from typing import Iterator, Generator
from dataclasses import dataclass
from datetime import datetime

@dataclass
class CommitRecord:
    sha: str
    message: str
    branch: str
    timestamp: datetime
    files: List[str]
    diff: Optional[str] = None

class GitHistoryStream:
    """Streaming git history processor."""

    def __init__(self, repo_path: str = "."):
        self.repo_path = repo_path
        self.bloom = CommitBloomFilter()
        self.weights = BranchWeightTrie()
        self.signals = QualitySignals()

    def stream_commits(
        self,
        branches: Optional[List[str]] = None,
        since: Optional[datetime] = None,
        batch_size: int = 100,
    ) -> Generator[CommitRecord, None, None]:
        """Stream commits without loading all into memory."""

        if branches is None:
            branches = self._get_all_branches()

        for branch in branches:
            # Use git log with pagination
            skip = 0
            while True:
                commits = self._fetch_batch(branch, skip, batch_size, since)
                if not commits:
                    break

                for commit in commits:
                    # Deduplicate via bloom filter
                    if commit.sha in self.bloom:
                        continue
                    self.bloom.add(commit.sha)

                    # Yield for processing
                    yield commit

                skip += batch_size

    def stream_weighted(
        self,
        half_life_months: float = 6.0
    ) -> Generator[Tuple[CommitRecord, float], None, None]:
        """Stream commits with pre-computed weights."""

        for commit in self.stream_commits():
            # Compute weight
            branch_weight = self.weights.get_weight(commit.branch)
            signal_mult = self.signals.get_multiplier(commit.sha)
            temporal = self._temporal_decay(commit.timestamp, half_life_months)

            weight = branch_weight * signal_mult * temporal

            if weight > 0:
                yield commit, weight

# Usage - memory-efficient training:
stream = GitHistoryStream()

for commit, weight in stream.stream_weighted():
    tokens = tokenize(commit.message)
    for context, word in get_ngrams(tokens):
        model.counts[context][word] += weight
        model.context_totals[context] += weight
```

---

### 6. Composite Structure: GitTrainingIndex

Combining all structures into a cohesive index:

```python
@dataclass
class GitTrainingIndex:
    """
    Space-efficient index for git history training.

    Memory budget for 100K commits:
    - Bloom filter: ~120KB
    - Branch trie: ~1KB
    - Quality signals: ~4.1MB
    - Temporal index: ~800KB
    - Total: ~5MB (vs. ~25MB naive)
    """

    bloom: CommitBloomFilter
    branch_weights: BranchWeightTrie
    signals: QualitySignals
    timestamps: Dict[str, float]  # SHA → Unix timestamp

    @classmethod
    def create(cls, expected_commits: int = 100_000) -> 'GitTrainingIndex':
        """Factory with sensible defaults."""
        index = cls(
            bloom=CommitBloomFilter(expected_commits),
            branch_weights=BranchWeightTrie(),
            signals=QualitySignals(expected_commits),
            timestamps={},
        )

        # Initialize standard branch weights
        for prefix, weight in [
            ('main', 1.0), ('master', 1.0),
            ('release/', 0.9), ('hotfix/', 1.1),
            ('feature/', 0.6), ('develop/', 0.7),
            ('claude/', 0.4), ('experimental/', 0.2),
        ]:
            index.branch_weights.add_prefix(prefix, weight)

        return index

    def should_process(self, sha: str) -> bool:
        """Check if commit should be processed (not seen before)."""
        if sha in self.bloom:
            return False
        return True

    def register_commit(
        self,
        sha: str,
        timestamp: datetime,
        **quality_flags
    ) -> None:
        """Register a commit in the index."""
        self.bloom.add(sha)
        self.signals.register(sha)
        self.timestamps[sha] = timestamp.timestamp()

        # Set quality signals
        signal_map = {
            'merged': QualitySignals.MERGED,
            'has_tests': QualitySignals.HAS_TESTS,
            'ci_passed': QualitySignals.CI_PASSED,
            'reverted': QualitySignals.REVERTED,
            'bug_fix': QualitySignals.BUG_FIX,
            'breaking': QualitySignals.BREAKING,
        }
        for flag, signal in signal_map.items():
            if quality_flags.get(flag):
                self.signals.set_signal(sha, signal)

    def get_weight(
        self,
        sha: str,
        branch: str,
        half_life_months: float = 6.0
    ) -> float:
        """Compute training weight for a commit."""
        branch_weight = self.branch_weights.get_weight(branch)
        signal_mult = self.signals.get_multiplier(sha)

        # Temporal decay
        if sha in self.timestamps:
            age_seconds = datetime.now().timestamp() - self.timestamps[sha]
            age_months = age_seconds / (30 * 24 * 3600)
            temporal = 0.5 ** (age_months / half_life_months)
        else:
            temporal = 0.5  # Unknown age, assume moderate

        return branch_weight * signal_mult * temporal

    def memory_usage(self) -> Dict[str, int]:
        """Report memory usage of each component."""
        import sys
        return {
            'bloom': len(self.bloom.bits),
            'signals': len(self.signals.signals),
            'timestamps': sys.getsizeof(self.timestamps),
            'total_commits': self.signals.next_idx,
        }
```

---

## Recommendation

For this project's scale (~10K commits currently, potentially 100K+):

| Component | Recommendation | Reason |
|-----------|---------------|--------|
| SHA dedup | Bloom filter | 97% memory savings, 1% FP acceptable |
| Branch weights | Simple dict | <20 prefixes, O(n) is fine |
| Quality signals | Packed bytes | Clean abstraction, 80% savings |
| N-gram counts | Exact dict | Vocab is manageable |
| Processing | Streaming | Memory-bounded, incremental |

**Start simple, optimize as needed.** The streaming pattern is the most important for scalability.

---

## Alternative: LSM-Tree Style Storage

For persistent, incremental training across sessions:

```python
class LSMTrainingStore:
    """
    Log-Structured Merge Tree for training data.

    - Write: Append to current segment (fast)
    - Read: Check segments newest-first
    - Compact: Merge segments periodically
    """

    def __init__(self, path: str):
        self.path = Path(path)
        self.current_segment = []
        self.segment_size = 10_000

    def add_commit(self, commit: CommitRecord, weight: float):
        self.current_segment.append((commit, weight))
        if len(self.current_segment) >= self.segment_size:
            self._flush_segment()

    def _flush_segment(self):
        """Write current segment to disk."""
        segment_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = self.path / f"segment_{segment_id}.jsonl"
        # Write segment...
        self.current_segment = []

    def compact(self):
        """Merge old segments into larger ones."""
        # Similar to git gc or LSM compaction
        pass
```

---

## Next Steps

1. Implement `CommitBloomFilter` in `cortical/utils/`
2. Implement `QualitySignals` in `cortical/spark/`
3. Create `GitTrainingIndex` as the main interface
4. Integrate streaming pattern into `GitHistoryTrainer`
5. Benchmark memory usage vs. naive approach

---

*These data structures are optimizations - start with the simple approach in the knowledge transfer and optimize if/when memory becomes a bottleneck.*
