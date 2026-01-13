"""
Text-to-Atoms Bridge: Converting Raw Text into Cognitive Graph Atoms.

This module provides BPE-inspired tokenization and text-to-atoms conversion
for feeding textual data into the CognitiveAgent.

Design Philosophy:
    - Start simple, make expandable
    - BPE (Byte Pair Encoding) approach: learn frequent subword pairs
    - Each token becomes a WORD atom
    - Co-occurrence creates SIMILARITY links
    - Document structure preserved via metadata

Why BPE-Style?
    Traditional word tokenization has fixed vocabulary. BPE learns subwords
    from data, handling rare/unknown words gracefully. We adapt this for
    the cognitive graph where:
    - Frequent patterns → stronger atoms (higher LTI)
    - Co-occurring words → linked atoms
    - Subwords → compositional understanding

Example:
    >>> bridge = TextToAtomsBridge(agent)
    >>> bridge.learn_vocabulary(["The cat sat on the mat."])
    >>> bridge.feed_text("The cat sat on the mat.", doc_id="sample1")
    # Creates atoms: "the", "cat", "sat", "on", "mat"
    # Creates SIMILARITY links between co-occurring words

Grounding: BPE algorithm (Sennrich et al., 2016) adapted for cognitive graphs.
"""

from __future__ import annotations

import json
import math
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from cortical.cognitive.graph import CognitiveAgent, CognitiveGraph, Atom

# Import at runtime to avoid circular imports
from cortical.cognitive.graph import AtomType, TruthValue
from cortical.common.filesystem import FileSystem
from cortical.tokenizer import split_identifier


# =============================================================================
# Progress Reporter
# =============================================================================


class ProgressReporter:
    """
    Reports progress with ETA for long-running operations.

    Usage:
        with ProgressReporter(total=100, desc="Loading") as progress:
            for i in range(100):
                do_work()
                progress.update(1)
    """

    def __init__(
        self,
        total: int,
        desc: str = "Processing",
        file=None,
        min_update_interval: float = 0.5,
    ):
        self.total = total
        self.desc = desc
        self.file = file or sys.stderr
        self.min_update_interval = min_update_interval
        self.current = 0
        self.start_time: Optional[float] = None
        self.last_update_time: float = 0

    def __enter__(self):
        self.start_time = time.time()
        self.last_update_time = 0
        self._print_progress()
        return self

    def __exit__(self, *args):
        self.current = self.total
        self._print_progress(force=True)
        print(file=self.file)  # Newline at end

    def update(self, n: int = 1):
        """Update progress by n items."""
        self.current += n
        now = time.time()
        if now - self.last_update_time >= self.min_update_interval:
            self._print_progress()
            self.last_update_time = now

    def _print_progress(self, force: bool = False):
        """Print progress bar with ETA."""
        if self.total == 0:
            return

        elapsed = time.time() - (self.start_time or time.time())
        pct = min(100, self.current * 100 // self.total)

        # Calculate ETA
        if self.current > 0 and elapsed > 0:
            rate = self.current / elapsed
            remaining = (self.total - self.current) / rate if rate > 0 else 0
            eta_str = self._format_time(remaining)
        else:
            eta_str = "calculating..."

        # Progress bar
        bar_width = 30
        filled = bar_width * self.current // self.total
        bar = "█" * filled + "░" * (bar_width - filled)

        # Print (carriage return to overwrite)
        status = f"\r{self.desc}: [{bar}] {pct:3d}% ({self.current}/{self.total}) ETA: {eta_str}  "
        print(status, end="", file=self.file, flush=True)

    def _format_time(self, seconds: float) -> str:
        """Format seconds as human-readable time."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"


# =============================================================================
# BPE-Style Tokenizer
# =============================================================================


@dataclass
class BPETokenizer:
    """
    Byte Pair Encoding inspired tokenizer for cognitive graphs.

    Unlike full BPE which operates at byte/character level, this operates
    at word level for simplicity, but learns to merge frequent word pairs
    into compound concepts.

    Attributes:
        vocab: Set of known tokens
        merges: Ordered list of (pair, merged_form) learned from data
        min_frequency: Minimum frequency for a pair to be merged
        max_vocab_size: Maximum vocabulary size

    Design Choice:
        We start at word level (not character) because:
        1. Simpler to understand and debug
        2. Atoms typically represent concepts, not characters
        3. Can extend to character-level BPE later if needed

    The tokenizer is trainable: feed it text, it learns frequent patterns.
    """

    vocab: Set[str] = field(default_factory=set)
    merges: List[Tuple[Tuple[str, str], str]] = field(default_factory=list)
    min_frequency: int = 2
    max_vocab_size: int = 10000

    # Internal tracking
    _pair_counts: Counter = field(default_factory=Counter)
    _word_counts: Counter = field(default_factory=Counter)

    # IDF tracking
    _doc_frequency: Dict[str, int] = field(default_factory=dict)
    _total_docs: int = 0

    def __post_init__(self):
        """Initialize counters if not provided."""
        if not hasattr(self, '_pair_counts') or self._pair_counts is None:
            self._pair_counts = Counter()
        if not hasattr(self, '_word_counts') or self._word_counts is None:
            self._word_counts = Counter()
        if not hasattr(self, '_doc_frequency') or self._doc_frequency is None:
            self._doc_frequency = {}
        if not hasattr(self, '_total_docs') or self._total_docs is None:
            self._total_docs = 0

    def tokenize(self, text: str) -> List[str]:
        """
        Convert text to tokens.

        Process:
            1. Normalize (lowercase, basic cleaning)
            2. Split into words
            3. Apply learned merges (if any)

        Args:
            text: Raw input text

        Returns:
            List of tokens

        Example:
            >>> tok = BPETokenizer()
            >>> tok.tokenize("The cat sat on the mat.")
            ['the', 'cat', 'sat', 'on', 'the', 'mat']
        """
        # Step 1: Normalize
        text = self._normalize(text)

        # Step 2: Split into words (basic word tokenization)
        words = self._split_words(text)

        # Step 3: Apply learned merges
        # For now, we don't merge at tokenize time (simple mode)
        # Future: Apply merges to create compound tokens

        return words

    def _normalize(self, text: str) -> str:
        """
        Normalize text for consistent tokenization.

        Uses cortical.tokenizer.split_identifier to properly handle:
            - CamelCase (TextToAtomsBridge -> text to atoms bridge)
            - PascalCase (UserCredentials -> user credentials)
            - underscore_style (get_user_data -> get user data)
            - Acronyms (XMLParser -> xml parser, parseHTTPResponse -> parse http response)

        Why split identifiers?
            Code class names like "TextToAtomsBridge" become isolated tokens that
            have no semantic links. Splitting them allows queries for "bridge" or
            "atoms" to find the class. This significantly improves code search.
        """
        # Find identifier-like patterns (CamelCase, underscore_style, etc.)
        # Pattern matches: word2vec, getUserData, get_user_data, XMLParser
        identifier_pattern = re.compile(r'\b_*[a-zA-Z][a-zA-Z0-9_]*\b')

        def replace_identifier(match: re.Match) -> str:
            """Replace identifier with full form AND split components.

            We preserve BOTH:
            - Full identifier (texttoatomsbridge) for exact matches
            - Split parts (text to atoms bridge) for semantic search

            This allows queries for either "TextToAtomsBridge" or "atoms bridge"
            to find the same code.
            """
            identifier = match.group(0)
            # Check if it looks like an identifier (has underscore or internal capitals)
            if '_' in identifier or any(c.isupper() for c in identifier[1:]):
                parts = split_identifier(identifier)
                full_lower = identifier.lower().replace('_', '')
                # Include both: full identifier AND split parts
                return f"{full_lower} {' '.join(parts)}"
            return identifier.lower()

        text = identifier_pattern.sub(replace_identifier, text)
        return text.strip()

    def _split_words(self, text: str) -> List[str]:
        """
        Split text into words.

        Uses simple regex: word characters only.
        Filters out very short tokens and pure numbers.

        Why filter short tokens?
            Single characters rarely carry semantic meaning.
            Pure numbers are better handled separately.
        """
        # Split on non-word characters
        raw_tokens = re.findall(r'\b[a-z][a-z]+\b', text)

        # Filter: min length 2, not pure number
        tokens = [t for t in raw_tokens if len(t) >= 2]

        return tokens

    def learn_from_texts(self, texts: List[str], n_merges: int = 100, incremental: bool = False) -> None:
        """
        Learn vocabulary and merges from a corpus.

        BPE Algorithm (simplified):
            1. Count all word frequencies
            2. Count all adjacent word pair frequencies
            3. Merge the most frequent pair into a single token
            4. Repeat until n_merges reached or no frequent pairs

        Args:
            texts: List of text documents
            n_merges: Maximum number of merge operations
            incremental: If True, add to existing counts instead of resetting

        Note:
            This is a simplified version. Full BPE would:
            - Operate at character level
            - Update counts after each merge
            We keep it simple for clarity and extensibility.
        """
        # Reset counters only if not incremental
        if not incremental:
            self._word_counts = Counter()
            self._pair_counts = Counter()
            self._doc_frequency = {}
            self._total_docs = 0

        # Count words and pairs
        for text in texts:
            tokens = self.tokenize(text)
            self._word_counts.update(tokens)

            # Track document frequency (count of docs containing each word)
            unique_words = set(tokens)
            for word in unique_words:
                self._doc_frequency[word] = self._doc_frequency.get(word, 0) + 1
            self._total_docs += 1

            # Count adjacent pairs
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                self._pair_counts[pair] += 1

        # Build vocabulary from word counts
        # In incremental mode, ADD to existing vocab; otherwise replace
        if incremental:
            self.vocab.update(self._word_counts.keys())
        else:
            self.vocab = set(self._word_counts.keys())

        # Learn merges (frequent pairs that could become compound concepts)
        # We store these for future use in compound atom creation
        self.merges = []
        for pair, count in self._pair_counts.most_common(n_merges):
            if count >= self.min_frequency:
                merged = f"{pair[0]}_{pair[1]}"
                self.merges.append((pair, merged))

        # Limit vocabulary size (in both incremental and non-incremental modes)
        # Without this check, incremental training can cause unbounded vocabulary growth
        # This was a bug: vocabulary grew from 10k to 30k+ across multiple incremental sessions
        if len(self.vocab) > self.max_vocab_size:
            top_words = [w for w, _ in self._word_counts.most_common(self.max_vocab_size)]
            self.vocab = set(top_words)

    def learn_vocabulary(self, texts: List[str], n_merges: int = 100, incremental: bool = False) -> None:
        """Alias for learn_from_texts - trains tokenizer on corpus."""
        self.learn_from_texts(texts, n_merges, incremental)

    def get_vocabulary(self) -> List[str]:
        """Get sorted vocabulary list."""
        return sorted(self.vocab)

    def get_word_frequency(self, word: str) -> int:
        """Get frequency count for a word."""
        return self._word_counts.get(word, 0)

    def get_pair_frequency(self, word1: str, word2: str) -> int:
        """Get co-occurrence frequency for a word pair."""
        return self._pair_counts.get((word1, word2), 0)

    def get_top_pairs(self, n: int = 20) -> List[Tuple[Tuple[str, str], int]]:
        """Get most frequent word pairs."""
        return self._pair_counts.most_common(n)

    def get_idf(self, word: str) -> float:
        """
        Get IDF (Inverse Document Frequency) for a word.

        Uses smoothed IDF formula: log((N + 1) / (df + 1))
        where N = total documents, df = documents containing the word.

        Args:
            word: The word to get IDF for

        Returns:
            IDF value, or 0.0 if word not in vocabulary
        """
        if word not in self.vocab:
            return 0.0
        df = self._doc_frequency.get(word, 0)
        return math.log((self._total_docs + 1) / (df + 1))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize tokenizer state to dictionary."""
        return {
            "vocab": list(self.vocab),
            "merges": [[[p[0], p[1]], m] for p, m in self.merges],
            "min_frequency": self.min_frequency,
            "max_vocab_size": self.max_vocab_size,
            "word_counts": dict(self._word_counts),
            "pair_counts": {f"{k[0]}|{k[1]}": v for k, v in self._pair_counts.items()},
            "doc_frequency": dict(self._doc_frequency),
            "total_docs": self._total_docs,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BPETokenizer':
        """Deserialize tokenizer from dictionary."""
        tok = cls(
            min_frequency=data.get("min_frequency", 2),
            max_vocab_size=data.get("max_vocab_size", 10000),
        )
        tok.vocab = set(data.get("vocab", []))
        tok.merges = [(tuple(p), m) for p, m in data.get("merges", [])]
        tok._word_counts = Counter(data.get("word_counts", {}))
        tok._pair_counts = Counter({
            tuple(k.split("|")): v
            for k, v in data.get("pair_counts", {}).items()
        })
        # Restore IDF data (backward compatible - defaults to empty/0 if missing)
        tok._doc_frequency = dict(data.get("doc_frequency", {}))
        tok._total_docs = data.get("total_docs", 0)
        return tok

    def save(self, path: Path, filesystem: FileSystem) -> None:
        """Save tokenizer to JSON file."""
        path = Path(path)
        filesystem.write_text(path, json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path, filesystem: FileSystem) -> 'BPETokenizer':
        """Load tokenizer from JSON file."""
        path = Path(path)
        data = json.loads(filesystem.read_text(path))
        return cls.from_dict(data)


# =============================================================================
# Text-to-Atoms Bridge
# =============================================================================


@dataclass
class TextToAtomsBridge:
    """
    Bridge between raw text and CognitiveGraph atoms.

    Converts text documents into:
        - WORD atoms for each token
        - SIMILARITY links for co-occurring words
        - Optional CONTEXT atoms for document metadata

    The bridge maintains its own tokenizer and learns patterns from data.

    Attributes:
        graph: The CognitiveGraph to populate
        tokenizer: BPE-style tokenizer for text processing
        window_size: Co-occurrence window for creating links
        min_link_strength: Minimum strength for SIMILARITY links

    Design Decisions:
        - Each unique word → one WORD atom (content-addressed)
        - Co-occurrence within window → SIMILARITY link
        - Link strength based on co-occurrence frequency
        - LTI (long-term importance) based on word frequency

    Usage:
        >>> bridge = TextToAtomsBridge(agent.graph)
        >>> bridge.learn_vocabulary([text1, text2, text3])
        >>> bridge.feed_text(text1, doc_id="doc1")
        >>> bridge.feed_text(text2, doc_id="doc2")
    """

    graph: 'CognitiveGraph'
    tokenizer: BPETokenizer = field(default_factory=BPETokenizer)
    window_size: int = 5  # Words within this window are "co-occurring"
    min_link_strength: float = 0.1
    max_links_per_doc: int = 500  # Limit links per document for performance

    # IDF-based relative link limiting
    # Base limit for atoms with maximum IDF (rare, semantically rich words)
    base_max_links: int = 500
    # Minimum links allowed even for low-IDF words (ubiquitous words)
    min_links_floor: int = 50

    # Statistics tracking
    _documents_fed: int = 0
    _atoms_created: int = 0
    _links_created: int = 0
    _links_skipped_by_limit: int = 0

    # Link count tracking per atom (atom_id -> current link count)
    _atom_link_counts: Dict[str, int] = field(default_factory=dict)

    def learn_vocabulary(self, texts: List[str], n_merges: int = 100, incremental: bool = False) -> None:
        """
        Train the tokenizer on a corpus of texts.

        Call this before feed_text() to establish vocabulary
        and learn frequent patterns.

        Args:
            texts: List of text documents
            n_merges: Number of merge operations for BPE
            incremental: If True, add to existing vocabulary instead of replacing
        """
        self.tokenizer.learn_from_texts(texts, n_merges, incremental=incremental)

    def get_max_links_for_word(self, word: str) -> int:
        """
        Calculate maximum allowed links for a word based on its IDF.

        Words with high IDF (rare, semantically rich) get more links.
        Words with low IDF (ubiquitous like 'the', 'and') get fewer links.

        This implements relative link limiting proportional to semantic depth,
        preventing hub atoms from dominating graph traversal while preserving
        them for natural language generation.

        Formula:
            max_links = min_links_floor + (base_max_links - min_links_floor) * normalized_idf

        Where normalized_idf = idf / max_possible_idf

        Args:
            word: The word to calculate limit for

        Returns:
            Maximum number of links this word's atom should have
        """
        idf = self.tokenizer.get_idf(word)

        # Calculate max possible IDF for normalization
        # max_idf = log((N + 1) / 2) where N = total docs
        # This occurs for a word appearing in exactly 1 document
        total_docs = self.tokenizer._total_docs or 1
        max_idf = math.log((total_docs + 1) / 2) if total_docs > 1 else 1.0

        # Normalize IDF to [0, 1] range
        normalized_idf = min(1.0, idf / max_idf) if max_idf > 0 else 0.0

        # Calculate max links: low IDF gets min_links_floor, high IDF gets base_max_links
        max_links = int(
            self.min_links_floor +
            (self.base_max_links - self.min_links_floor) * normalized_idf
        )

        return max_links

    def _can_add_link_to_atom(self, atom_id: str, word: str) -> bool:
        """
        Check if we can add another link to this atom based on IDF limit.

        Args:
            atom_id: The atom's ID
            word: The word this atom represents

        Returns:
            True if link can be added, False if at limit
        """
        current_count = self._atom_link_counts.get(atom_id, 0)
        max_allowed = self.get_max_links_for_word(word)
        return current_count < max_allowed

    def _increment_atom_link_count(self, atom_id: str) -> None:
        """Increment the link count for an atom."""
        self._atom_link_counts[atom_id] = self._atom_link_counts.get(atom_id, 0) + 1

    def feed_text(
        self,
        text: str,
        doc_id: Optional[str] = None,
        initial_sti: float = 0.1,
    ) -> List['Atom']:
        """
        Convert text to atoms and add to the graph.

        Process:
            1. Tokenize text
            2. Create WORD atoms for each unique token
            3. Create SIMILARITY links for co-occurring words
            4. Optionally create CONTEXT atom for document

        Args:
            text: The text to process
            doc_id: Optional document identifier
            initial_sti: Starting attention for new atoms

        Returns:
            List of created/updated atoms

        Example:
            >>> atoms = bridge.feed_text("The cat sat on the mat.", "sample1")
            >>> print([a.name for a in atoms if a.name])
            ['the', 'cat', 'sat', 'on', 'mat']
        """
        tokens = self.tokenizer.tokenize(text)
        if not tokens:
            return []

        created_atoms = []

        # Step 1: Create WORD atoms for each unique token
        token_atoms = {}
        for token in set(tokens):
            atom = self._get_or_create_word_atom(token, initial_sti)
            token_atoms[token] = atom
            created_atoms.append(atom)

        # Step 2: Create SIMILARITY links based on co-occurrence
        # Track pairs we've already linked in this document
        linked_pairs: Set[Tuple[str, str]] = set()
        links_created_this_doc = 0

        for i, token in enumerate(tokens):
            # Respect max links limit for performance
            if links_created_this_doc >= self.max_links_per_doc:
                break

            # Look at window of following words
            window_end = min(i + self.window_size + 1, len(tokens))
            for j in range(i + 1, window_end):
                if links_created_this_doc >= self.max_links_per_doc:
                    break

                other_token = tokens[j]
                if token == other_token:
                    continue

                # Normalize pair order for consistency
                pair = tuple(sorted([token, other_token]))
                if pair in linked_pairs:
                    continue
                linked_pairs.add(pair)

                # Create or strengthen SIMILARITY link
                self._create_similarity_link(
                    token_atoms[token],
                    token_atoms[other_token],
                )
                links_created_this_doc += 1

        # Step 3: Create FOLLOWS links for adjacent words (for prediction)
        # FOLLOWS is directional: A -> B means B follows A
        for i in range(len(tokens) - 1):
            current_token = tokens[i]
            next_token = tokens[i + 1]

            if current_token == next_token:
                continue

            # Create directional FOLLOWS link (not canonicalized)
            self._create_follows_link(
                token_atoms[current_token],
                token_atoms[next_token],
            )

        self._documents_fed += 1
        return created_atoms

    def _get_or_create_word_atom(
        self,
        word: str,
        initial_sti: float = 0.1,
    ) -> 'Atom':
        """
        Get existing WORD atom or create new one.

        LTI (long-term importance) is set based on word frequency
        if the tokenizer has learned from a corpus.

        Args:
            word: The word to represent
            initial_sti: Starting attention value

        Returns:
            WORD atom for this word
        """
        existing = self.graph.get_node(word)
        if existing is not None:
            # Update access time (implicit in graph operations)
            return existing

        # Calculate LTI based on word frequency
        freq = self.tokenizer.get_word_frequency(word)
        total_words = sum(self.tokenizer._word_counts.values()) or 1

        # LTI: frequent words get higher importance (capped at 0.8)
        # Formula: log-scaled frequency, normalized
        lti = min(0.8, math.log1p(freq) / math.log1p(total_words) + 0.1)

        # Create WORD atom with computed importance
        atom = self.graph.node(word, atom_type=AtomType.WORD)
        atom.sti = initial_sti
        atom.lti = lti

        # Save updates
        self.graph._storage.save(atom)
        self._atoms_created += 1

        return atom

    def _create_similarity_link(
        self,
        atom1: 'Atom',
        atom2: 'Atom',
        base_strength: float = 0.3,
    ) -> Optional['Atom']:
        """
        Create or strengthen SIMILARITY link between atoms.

        Link strength is based on co-occurrence frequency.
        If link already exists, merge evidence (strength increases).

        The link stores dual strength values in metadata:
            - raw_strength: Co-occurrence based strength (backward compatible)
            - idf_strength: raw_strength * min(idf_word1, idf_word2)
            - idf_epoch: Training epoch when IDF was computed

        IDF-based link limiting:
            Each atom has a maximum link count based on its IDF value.
            Low-IDF words (ubiquitous like 'the', 'and') get fewer links.
            High-IDF words (rare, semantically rich) get more links.
            This prevents hub atoms from dominating graph traversal.

        Args:
            atom1: First atom
            atom2: Second atom
            base_strength: Base strength for new links

        Returns:
            The SIMILARITY link atom, or None if skipped
        """
        # Canonicalize atom order for SIMILARITY (bidirectional) links
        # This ensures [A,B] and [B,A] map to the same link
        if atom1.name > atom2.name:
            atom1, atom2 = atom2, atom1

        # Check IDF-based link limits for both atoms
        # Only create link if BOTH atoms can accept more links
        can_add_1 = self._can_add_link_to_atom(atom1.id, atom1.name)
        can_add_2 = self._can_add_link_to_atom(atom2.id, atom2.name)

        if not (can_add_1 and can_add_2):
            self._links_skipped_by_limit += 1
            return None

        # Calculate strength based on co-occurrence frequency
        pair_freq = self.tokenizer.get_pair_frequency(atom1.name, atom2.name)
        pair_freq += self.tokenizer.get_pair_frequency(atom2.name, atom1.name)

        # Strength scales with frequency (capped)
        raw_strength = min(0.9, base_strength + math.log1p(pair_freq) * 0.1)

        if raw_strength < self.min_link_strength:
            return None

        # Get IDF values for both words
        # IDF=0 is valid: means word appears in all documents (no discriminative power)
        idf1 = self.tokenizer.get_idf(atom1.name) if atom1.name else 0.0
        idf2 = self.tokenizer.get_idf(atom2.name) if atom2.name else 0.0

        # Calculate IDF-weighted strength using minimum IDF of the pair
        # If either word is ubiquitous (IDF=0), the link has low discriminative value
        idf_strength = raw_strength * min(idf1, idf2)

        # Create link with raw_strength in truth value
        tv = TruthValue(strength=raw_strength, confidence=0.3)
        link = self.graph.link(AtomType.SIMILARITY, [atom1, atom2], tv)

        # Store dual values in metadata
        link.metadata['raw_strength'] = raw_strength
        link.metadata['idf_strength'] = idf_strength
        link.metadata['idf_epoch'] = getattr(self, '_current_epoch', 0)

        # Save the updated link with metadata
        self.graph._storage.save(link)

        # Update link counts for both atoms
        self._increment_atom_link_count(atom1.id)
        self._increment_atom_link_count(atom2.id)

        self._links_created += 1
        return link

    def _create_follows_link(
        self,
        from_atom: 'Atom',
        to_atom: 'Atom',
    ) -> Optional['Atom']:
        """
        Create or strengthen a directional FOLLOWS link.

        FOLLOWS links are directional: from_atom → to_atom means
        to_atom follows from_atom in text sequences.

        Unlike SIMILARITY links:
        - FOLLOWS is directional (order matters)
        - No IDF weighting (we care about transition frequency)
        - Strength represents transition probability

        IDF-based link limiting also applies to FOLLOWS links to prevent
        hub atoms from accumulating excessive transition links.

        Args:
            from_atom: The preceding word atom
            to_atom: The following word atom

        Returns:
            The FOLLOWS link atom, or None if below threshold or at limit
        """
        # Check for existing FOLLOWS link (directional - order matters)
        # Use find_link_by_outgoing with [from_id, to_id] - order matters
        existing = self.graph._storage.find_link_by_outgoing(
            AtomType.FOLLOWS,
            [from_atom.id, to_atom.id],
        )

        if existing:
            # Strengthen existing link (no limit check for strengthening)
            link = existing
            link.metadata['count'] = link.metadata.get('count', 1) + 1
            # Increase strength with each observation (diminishing returns)
            count = link.metadata['count']
            link.tv.strength = min(0.95, 0.3 + math.log1p(count) * 0.1)
            self.graph._storage.save(link)
            return link

        # Check IDF-based link limits for both atoms before creating NEW link
        can_add_from = self._can_add_link_to_atom(from_atom.id, from_atom.name)
        can_add_to = self._can_add_link_to_atom(to_atom.id, to_atom.name)

        if not (can_add_from and can_add_to):
            self._links_skipped_by_limit += 1
            return None

        # Create new FOLLOWS link
        tv = TruthValue(strength=0.3, confidence=0.2)  # Start uncertain
        link = self.graph.link(AtomType.FOLLOWS, [from_atom, to_atom], tv)

        # Initialize metadata
        link.metadata['count'] = 1
        link.metadata['direction'] = 'forward'  # Explicit direction marker

        self.graph._storage.save(link)

        # Update link counts for both atoms
        self._increment_atom_link_count(from_atom.id)
        self._increment_atom_link_count(to_atom.id)

        self._links_created += 1
        return link

    def get_statistics(self) -> Dict[str, int]:
        """Get processing statistics."""
        return {
            "documents_fed": self._documents_fed,
            "atoms_created": self._atoms_created,
            "links_created": self._links_created,
            "links_skipped_by_limit": self._links_skipped_by_limit,
            "vocabulary_size": len(self.tokenizer.vocab),
            "learned_merges": len(self.tokenizer.merges),
        }

    def get_similarity_links(self) -> List['Atom']:
        """
        Get all SIMILARITY links in the graph.

        Returns:
            List of SIMILARITY link atoms with their metadata.
        """
        from cortical.cognitive.graph import AtomType
        return self.graph.find_by_type(AtomType.SIMILARITY)

    def reindex_idf(self) -> Dict[str, Any]:
        """
        Recalculate idf_strength for all SIMILARITY links using current IDF values.

        This should be called after incremental training to update stale link
        weights. Links created before vocabulary updates will have idf_strength
        computed with old IDF values - this method fixes that.

        Performance: O(L) where L = number of SIMILARITY links.
        Each link requires 2 atom lookups (O(1) with hash storage) and
        2 IDF lookups (O(1) dict access).

        Returns:
            Dict with reindex statistics:
                - links_updated: Number of links processed
                - time_ms: Time taken in milliseconds
                - new_epoch: The new IDF epoch number
        """
        import time
        from cortical.cognitive.graph import AtomType

        start = time.perf_counter()

        # Increment epoch counter
        if not hasattr(self, '_idf_epoch'):
            self._idf_epoch = 0
        self._idf_epoch += 1
        new_epoch = self._idf_epoch

        storage = self.graph._storage
        links = storage.find_by_type(AtomType.SIMILARITY)
        links_updated = 0

        for link in links:
            # Get the two connected word atoms
            if len(link.outgoing) != 2:
                continue

            atom1 = storage.load(link.outgoing[0])
            atom2 = storage.load(link.outgoing[1])

            if not atom1 or not atom2:
                continue

            # Get current IDF values
            idf1 = self.tokenizer.get_idf(atom1.name) if atom1.name else 0.0
            idf2 = self.tokenizer.get_idf(atom2.name) if atom2.name else 0.0

            # Recalculate idf_strength using raw_strength from metadata or tv.strength
            raw = link.metadata.get('raw_strength', link.tv.strength)
            link.metadata['idf_strength'] = raw * min(idf1, idf2)
            link.metadata['idf_epoch'] = new_epoch

            # Persist updated link
            storage.save(link)
            links_updated += 1

        elapsed_ms = (time.perf_counter() - start) * 1000

        return {
            'links_updated': links_updated,
            'time_ms': round(elapsed_ms, 2),
            'new_epoch': new_epoch,
        }

    def get_idf_epoch(self) -> int:
        """Return current IDF epoch number."""
        return getattr(self, '_idf_epoch', 0)

    def add_documents(
        self,
        texts: List[str],
        doc_ids: Optional[List[str]] = None,
        show_progress: bool = True,
    ) -> Dict[str, int]:
        """
        Incrementally add documents to an already-trained bridge.

        This is the recommended way to add more data after initial training
        or after loading a saved model.

        Args:
            texts: List of text documents to add
            doc_ids: Optional list of document IDs (defaults to "doc_N")
            show_progress: Whether to show progress bar

        Returns:
            Statistics about what was added

        Example:
            >>> # Load existing model
            >>> bridge = TextToAtomsBridge.load(path, graph)
            >>> # Add more documents
            >>> bridge.add_documents(new_texts)
            >>> # Save updated model
            >>> bridge.save(path)
        """
        if not texts:
            return {"documents_added": 0}

        # Generate doc_ids if not provided
        if doc_ids is None:
            start_idx = self._documents_fed
            doc_ids = [f"doc_{start_idx + i}" for i in range(len(texts))]

        # Track stats before
        atoms_before = self._atoms_created
        links_before = self._links_created

        # Incrementally learn vocabulary from new texts
        if show_progress:
            print(f"Learning vocabulary from {len(texts)} new documents...", file=sys.stderr)
        self.learn_vocabulary(texts, incremental=True)
        if show_progress:
            print(f"  Vocabulary now: {len(self.tokenizer.vocab)} words", file=sys.stderr)

        # Feed documents
        if show_progress:
            with ProgressReporter(len(texts), desc="Adding documents") as progress:
                for text, doc_id in zip(texts, doc_ids):
                    self.feed_text(text, doc_id=doc_id)
                    progress.update(1)
        else:
            for text, doc_id in zip(texts, doc_ids):
                self.feed_text(text, doc_id=doc_id)

        return {
            "documents_added": len(texts),
            "atoms_added": self._atoms_created - atoms_before,
            "links_added": self._links_created - links_before,
            "total_documents": self._documents_fed,
            "total_vocabulary": len(self.tokenizer.vocab),
        }

    def save(self, path: Path, filesystem: FileSystem) -> None:
        """
        Save bridge state (tokenizer + graph) to directory.

        Creates:
            path/tokenizer.json - Tokenizer vocabulary and stats
            path/meta.json - Graph metadata
            path/atoms_*.json - Sharded graph atoms by type

        Uses ShardedGraphStorage to keep files under 50MB for git.

        Args:
            path: Directory to save to (created if doesn't exist)
            filesystem: FileSystem for I/O operations
        """
        from cortical.cognitive.graph_storage import ShardedGraphStorage

        path = Path(path)
        filesystem.mkdir(path, parents=True, exist_ok=True)

        # Save tokenizer
        self.tokenizer.save(path / "tokenizer.json", filesystem)

        # Save graph using sharded storage
        storage = ShardedGraphStorage()
        result = storage.save(self.graph, path)

        # Also save stats in meta.json (append to existing)
        meta_path = path / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
        else:
            meta = {}

        meta["bridge_stats"] = self.get_statistics()
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"Saved bridge to {path}/")
        print(f"  Shards: {result['shards_written']}, Atoms: {result['total_atoms']}")

    @classmethod
    def load(cls, path: Path, graph: 'CognitiveGraph', filesystem: FileSystem) -> 'TextToAtomsBridge':
        """
        Load bridge state from directory.

        Supports both sharded format (atoms_*.json) and legacy format (graph.json).

        Args:
            path: Directory containing saved state
            graph: CognitiveGraph to populate
            filesystem: FileSystem for I/O operations

        Returns:
            Loaded TextToAtomsBridge
        """
        from cortical.cognitive.graph_storage import ShardedGraphStorage

        path = Path(path)

        # Load tokenizer
        tokenizer = BPETokenizer.load(path / "tokenizer.json", filesystem)

        # Create bridge with loaded tokenizer
        bridge = cls(graph=graph, tokenizer=tokenizer)

        # Load graph data using sharded storage (handles legacy format too)
        storage = ShardedGraphStorage()
        atoms_data = storage.load(path)

        # Separate nodes and links
        nodes = [a for a in atoms_data if not a.get("outgoing")]
        links = [a for a in atoms_data if a.get("outgoing")]

        # Pass 1: Restore nodes first (so link targets exist)
        id_map = {}  # old_id -> new_atom
        for atom_data in nodes:
            atom_type = AtomType[atom_data["atom_type"]]
            # Handle both old format (tv_strength) and new format (tv.strength)
            if "tv" in atom_data:
                tv = TruthValue(atom_data["tv"]["strength"], atom_data["tv"]["confidence"])
            else:
                tv = TruthValue(atom_data.get("tv_strength", 0.5), atom_data.get("tv_confidence", 0.5))
            atom = graph.node(atom_data["name"], atom_type=atom_type, tv=tv)
            atom.sti = atom_data.get("sti", 0.0)
            atom.lti = atom_data.get("lti", 0.0)
            atom.metadata = atom_data.get("metadata", {})
            graph._storage.save(atom)
            id_map[atom_data["id"]] = atom

        # Pass 2: Restore links
        for atom_data in links:
            atom_type = AtomType[atom_data["atom_type"]]
            # Handle both old format (tv_strength) and new format (tv.strength)
            if "tv" in atom_data:
                tv = TruthValue(atom_data["tv"]["strength"], atom_data["tv"]["confidence"])
            else:
                tv = TruthValue(atom_data.get("tv_strength", 0.5), atom_data.get("tv_confidence", 0.5))

            # Resolve target atoms
            targets = []
            for old_id in atom_data["outgoing"]:
                if old_id in id_map:
                    targets.append(id_map[old_id])

            if len(targets) == len(atom_data["outgoing"]):
                link = graph.link(atom_type, targets, tv)
                link.sti = atom_data.get("sti", 0.0)
                link.lti = atom_data.get("lti", 0.0)
                link.metadata = atom_data.get("metadata", {})
                graph._storage.save(link)

        # Restore stats from meta.json if available
        meta_path = path / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            stats = meta.get("bridge_stats", {})
        else:
            stats = {}

        bridge._documents_fed = stats.get("documents_fed", 0)
        bridge._atoms_created = stats.get("atoms_created", 0)
        bridge._links_created = stats.get("links_created", 0)

        # Compute atom link counts from the loaded graph
        # This ensures IDF-based link limiting works correctly after reload
        # (Consistent with Tier 2 rebuildable philosophy - compute from graph, don't persist)
        bridge._atom_link_counts = {}
        for atom_data in links:
            for old_id in atom_data.get("outgoing", []):
                if old_id in id_map:
                    new_atom = id_map[old_id]
                    bridge._atom_link_counts[new_atom.id] = bridge._atom_link_counts.get(new_atom.id, 0) + 1

        print(f"Loaded bridge from {path}/")
        print(f"  Nodes: {len(nodes)}, Links: {len(links)}")
        return bridge


# =============================================================================
# File/Directory Loading Utilities
# =============================================================================


def load_text_file(path: Path, filesystem: FileSystem) -> str:
    """
    Load text from a file with encoding fallback.

    Tries UTF-8 first, falls back to latin-1 for real filesystems.

    Args:
        path: File path to read
        filesystem: FileSystem for I/O operations
    """
    return filesystem.read_text(path)


def iter_text_files(
    directory: Path,
    filesystem: FileSystem,
    pattern: str = "*.txt",
    recursive: bool = True,
) -> Iterator[Tuple[Path, str]]:
    """
    Iterate over text files in a directory.

    Args:
        directory: Directory to scan
        filesystem: FileSystem for I/O operations
        pattern: Glob pattern for files
        recursive: Whether to search subdirectories

    Yields:
        (path, content) tuples

    Example:
        >>> for path, text in iter_text_files(Path("samples"), filesystem):
        ...     print(f"Loaded {path.name}: {len(text)} chars")
    """
    glob_pattern = f"**/{pattern}" if recursive else pattern
    files = filesystem.glob(directory, glob_pattern)

    for path in sorted(files):
        if not filesystem.is_dir(path):
            try:
                content = load_text_file(path, filesystem)
                yield path, content
            except Exception as e:
                # Skip files that can't be read
                print(f"Warning: Could not read {path}: {e}")


def load_directory_to_bridge(
    directory: Path,
    bridge: TextToAtomsBridge,
    filesystem: FileSystem,
    pattern: str = "*.txt",
    max_files: Optional[int] = None,
    learn_first: bool = True,
    show_progress: bool = True,
) -> Dict[str, int]:
    """
    Load all text files from a directory into the bridge.

    Two-pass approach:
        1. First pass: Learn vocabulary from all texts (if learn_first=True)
        2. Second pass: Feed texts to create atoms and links

    Args:
        directory: Directory containing text files
        bridge: TextToAtomsBridge to use
        filesystem: FileSystem for I/O operations
        pattern: Glob pattern for files
        max_files: Maximum files to process (None = all)
        learn_first: Whether to learn vocabulary first
        show_progress: Whether to show progress bar with ETA

    Returns:
        Statistics dictionary

    Example:
        >>> bridge = TextToAtomsBridge(agent.graph)
        >>> stats = load_directory_to_bridge(Path("samples"), bridge, filesystem, max_files=10)
        >>> print(f"Loaded {stats['files_processed']} files")
    """
    directory = Path(directory)

    # Collect files
    files = list(iter_text_files(directory, filesystem, pattern))
    if max_files:
        files = files[:max_files]

    if not files:
        return {"files_processed": 0, "error": "No files found"}

    # Pass 1: Learn vocabulary (optional but recommended)
    if learn_first:
        if show_progress:
            print(f"Learning vocabulary from {len(files)} files...", file=sys.stderr)
        texts = [content for _, content in files]
        bridge.learn_vocabulary(texts)
        if show_progress:
            print(f"  Vocabulary: {len(bridge.tokenizer.vocab)} words", file=sys.stderr)

    # Pass 2: Feed texts to create atoms
    if show_progress:
        with ProgressReporter(len(files), desc="Feeding documents") as progress:
            for path, content in files:
                doc_id = path.stem
                bridge.feed_text(content, doc_id=doc_id)
                progress.update(1)
    else:
        for path, content in files:
            doc_id = path.stem
            bridge.feed_text(content, doc_id=doc_id)

    stats = bridge.get_statistics()
    stats["files_processed"] = len(files)

    return stats
