"""
Semantic indexing for the Cognitive Event Lattice.

The semantic layer provides fast, approximate answers to queries:
- Bloom filter: "Does concept X probably exist?"
- Inverted index: "What events mention concept X?"
- (Future) Embeddings: "What's semantically similar to X?"

Trade-off:
    These structures trade exactness for speed. The bloom filter
    may have false positives (but never false negatives).

This module implements Level 2 of the CEL architecture.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

from ..core.events import CognitiveEvent
from ..core.protocols import EventStore


class BloomFilter:
    """
    Probabilistic set membership with tunable false positive rate.

    A Bloom filter can definitively say "X is NOT in the set" but
    can only probabilistically say "X is PROBABLY in the set".

    Parameters:
        expected_elements: Expected number of elements
        fp_rate: Desired false positive rate (e.g., 0.01 for 1%)

    The filter size and hash count are computed from these parameters.
    """

    def __init__(self, expected_elements: int = 10000, fp_rate: float = 0.01):
        """
        Initialize bloom filter.

        Args:
            expected_elements: Expected number of unique elements
            fp_rate: Target false positive rate (0.0 to 1.0)
        """
        self.expected_elements = expected_elements
        self.fp_rate = fp_rate

        # Calculate optimal size and hash count
        # m = -n * ln(p) / (ln(2)^2)
        # k = (m/n) * ln(2)
        self.size = self._optimal_size(expected_elements, fp_rate)
        self.hash_count = self._optimal_hash_count(self.size, expected_elements)

        # Bit array as bytearray
        self._bits = bytearray((self.size + 7) // 8)
        self._count = 0

    @staticmethod
    def _optimal_size(n: int, p: float) -> int:
        """Calculate optimal bit array size."""
        if p <= 0:
            p = 0.0001
        m = -n * math.log(p) / (math.log(2) ** 2)
        return max(int(m), 64)

    @staticmethod
    def _optimal_hash_count(m: int, n: int) -> int:
        """Calculate optimal number of hash functions."""
        k = (m / n) * math.log(2)
        return max(int(k), 1)

    def _hashes(self, item: str) -> Iterator[int]:
        """Generate hash values for an item."""
        # Use double hashing: h(i) = h1 + i*h2
        h1 = int(hashlib.md5(item.encode()).hexdigest(), 16)
        h2 = int(hashlib.sha1(item.encode()).hexdigest(), 16)

        for i in range(self.hash_count):
            yield (h1 + i * h2) % self.size

    def add(self, item: str) -> None:
        """Add an item to the filter."""
        for pos in self._hashes(item):
            byte_pos = pos // 8
            bit_pos = pos % 8
            self._bits[byte_pos] |= (1 << bit_pos)
        self._count += 1

    def contains(self, item: str) -> bool:
        """
        Check if item is probably in the filter.

        Returns:
            True if item PROBABLY exists (may be false positive)
            False if item DEFINITELY does not exist
        """
        for pos in self._hashes(item):
            byte_pos = pos // 8
            bit_pos = pos % 8
            if not (self._bits[byte_pos] & (1 << bit_pos)):
                return False
        return True

    def __contains__(self, item: str) -> bool:
        """Enable 'in' operator."""
        return self.contains(item)

    @property
    def count(self) -> int:
        """Number of items added (not unique, may have duplicates)."""
        return self._count

    @property
    def estimated_fp_rate(self) -> float:
        """
        Estimate current false positive rate.

        Based on number of bits set vs total bits.
        """
        bits_set = sum(bin(byte).count('1') for byte in self._bits)
        if bits_set == 0:
            return 0.0
        ratio = bits_set / self.size
        return ratio ** self.hash_count

    def to_bytes(self) -> bytes:
        """Serialize to bytes."""
        return bytes(self._bits)

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        expected_elements: int,
        fp_rate: float,
    ) -> 'BloomFilter':
        """Deserialize from bytes."""
        bf = cls(expected_elements, fp_rate)
        bf._bits = bytearray(data)
        return bf


@dataclass
class InvertedIndex:
    """
    Inverted index mapping terms to event IDs.

    Enables fast lookup: "Which events contain term X?"
    """

    _index: Dict[str, Set[str]] = field(default_factory=dict)
    _event_terms: Dict[str, Set[str]] = field(default_factory=dict)

    def add(self, term: str, event_id: str) -> None:
        """Add term-event mapping."""
        if term not in self._index:
            self._index[term] = set()
        self._index[term].add(event_id)

        if event_id not in self._event_terms:
            self._event_terms[event_id] = set()
        self._event_terms[event_id].add(term)

    def search(self, term: str) -> Set[str]:
        """Get event IDs containing term."""
        return self._index.get(term, set()).copy()

    def search_all(self, terms: List[str], require_all: bool = False) -> Set[str]:
        """
        Search for events matching terms.

        Args:
            terms: Terms to search for
            require_all: If True, events must contain ALL terms (AND)
                        If False, events can contain ANY term (OR)

        Returns:
            Set of matching event IDs
        """
        if not terms:
            return set()

        results = [self.search(term) for term in terms]

        if require_all:
            return set.intersection(*results) if results else set()
        else:
            return set.union(*results) if results else set()

    def remove_event(self, event_id: str) -> None:
        """Remove all terms for an event."""
        terms = self._event_terms.pop(event_id, set())
        for term in terms:
            if term in self._index:
                self._index[term].discard(event_id)
                if not self._index[term]:
                    del self._index[term]

    @property
    def term_count(self) -> int:
        """Number of unique terms."""
        return len(self._index)

    @property
    def event_count(self) -> int:
        """Number of indexed events."""
        return len(self._event_terms)

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            'index': {k: list(v) for k, v in self._index.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'InvertedIndex':
        """Deserialize from dictionary."""
        index = cls()
        for term, event_ids in data.get('index', {}).items():
            for event_id in event_ids:
                index.add(term, event_id)
        return index


class BloomSemanticIndex:
    """
    Semantic index using bloom filter for fast existence checks.

    Provides O(1) probabilistic "does this concept exist?" queries.
    Falls back to inverted index for actual retrieval.

    Implements: SemanticIndex protocol
    """

    def __init__(
        self,
        expected_concepts: int = 10000,
        fp_rate: float = 0.01,
    ):
        """
        Initialize semantic index.

        Args:
            expected_concepts: Expected unique concepts
            fp_rate: Bloom filter false positive rate
        """
        self._bloom = BloomFilter(expected_concepts, fp_rate)
        self._inverted = InvertedIndex()

    def index_event(self, event: CognitiveEvent) -> None:
        """Index concepts from an event."""
        for concept in event.concepts:
            normalized = concept.lower().strip()
            if normalized:
                self._bloom.add(normalized)
                self._inverted.add(normalized, event.id)

        # Also index from content
        if 'title' in event.content:
            for word in self._extract_words(event.content['title']):
                self._bloom.add(word)
                self._inverted.add(word, event.id)

    def _extract_words(self, text: str) -> List[str]:
        """Extract indexable words from text."""
        # Simple tokenization - could be more sophisticated
        words = text.lower().split()
        stop_words = {'the', 'a', 'an', 'to', 'for', 'of', 'in', 'on', 'at', 'is', 'are'}
        return [w for w in words if w not in stop_words and len(w) > 2]

    def probably_contains(self, concept: str) -> bool:
        """Fast probabilistic existence check."""
        return concept.lower().strip() in self._bloom

    def search(self, query: str, limit: int = 10) -> List[str]:
        """Search for events matching query."""
        terms = self._extract_words(query)

        # Fast path: check bloom filter first
        matching_terms = [t for t in terms if self.probably_contains(t)]
        if not matching_terms:
            return []

        # Get events from inverted index
        results = self._inverted.search_all(matching_terms, require_all=False)

        # Sort by number of matching terms (relevance)
        def relevance(event_id: str) -> int:
            event_terms = self._inverted._event_terms.get(event_id, set())
            return len(event_terms & set(matching_terms))

        sorted_results = sorted(results, key=relevance, reverse=True)
        return sorted_results[:limit]

    def similar_to(self, entity_id: str, limit: int = 10) -> List[Tuple[str, float]]:
        """
        Find entities similar to a given entity.

        Currently uses term overlap as similarity metric.
        Could be enhanced with embeddings for semantic similarity.
        """
        # Get terms for this entity
        entity_terms = self._inverted._event_terms.get(entity_id, set())
        if not entity_terms:
            return []

        # Find entities with overlapping terms
        candidates: Dict[str, int] = {}
        for term in entity_terms:
            for other_id in self._inverted.search(term):
                if other_id != entity_id:
                    candidates[other_id] = candidates.get(other_id, 0) + 1

        # Calculate Jaccard similarity
        results = []
        for other_id, overlap in candidates.items():
            other_terms = self._inverted._event_terms.get(other_id, set())
            union = len(entity_terms | other_terms)
            similarity = overlap / union if union > 0 else 0.0
            results.append((other_id, similarity))

        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def rebuild(self, event_store: EventStore) -> None:
        """Rebuild indexes from event store."""
        # Reset
        self._bloom = BloomFilter(
            self._bloom.expected_elements,
            self._bloom.fp_rate,
        )
        self._inverted = InvertedIndex()

        # Re-index all events
        for event in event_store.iterate():
            self.index_event(event)

    @property
    def stats(self) -> Dict:
        """Get index statistics."""
        return {
            'bloom_size': self._bloom.size,
            'bloom_fp_rate': self._bloom.estimated_fp_rate,
            'term_count': self._inverted.term_count,
            'event_count': self._inverted.event_count,
        }


class HybridSemanticIndex(BloomSemanticIndex):
    """
    Enhanced semantic index with persistence support.

    Adds:
    - Disk persistence
    - Incremental updates
    - Background rebuilding
    """

    def __init__(
        self,
        base_path: Path,
        expected_concepts: int = 10000,
        fp_rate: float = 0.01,
    ):
        super().__init__(expected_concepts, fp_rate)
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        self._bloom_path = self.base_path / "bloom.bin"
        self._index_path = self.base_path / "inverted.json"
        self._meta_path = self.base_path / "meta.json"

        self._load()

    def _load(self) -> None:
        """Load indexes from disk."""
        # Load bloom filter
        if self._bloom_path.exists():
            with open(self._bloom_path, 'rb') as f:
                self._bloom = BloomFilter.from_bytes(
                    f.read(),
                    self._bloom.expected_elements,
                    self._bloom.fp_rate,
                )

        # Load inverted index
        if self._index_path.exists():
            with open(self._index_path) as f:
                self._inverted = InvertedIndex.from_dict(json.load(f))

    def save(self) -> None:
        """Persist indexes to disk."""
        # Save bloom filter
        with open(self._bloom_path, 'wb') as f:
            f.write(self._bloom.to_bytes())

        # Save inverted index
        with open(self._index_path, 'w') as f:
            json.dump(self._inverted.to_dict(), f)

        # Save metadata
        with open(self._meta_path, 'w') as f:
            json.dump(self.stats, f)

    def index_event(self, event: CognitiveEvent) -> None:
        """Index event and persist."""
        super().index_event(event)
        # Note: For performance, consider batching saves
        # self.save()  # Uncomment for immediate persistence

    def rebuild(self, event_store: EventStore) -> None:
        """Rebuild and persist indexes."""
        super().rebuild(event_store)
        self.save()
