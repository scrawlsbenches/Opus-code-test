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

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from cortical.cognitive.graph import CognitiveAgent, CognitiveGraph, Atom

# Import at runtime to avoid circular imports
from cortical.cognitive.graph import AtomType, TruthValue


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

    def __post_init__(self):
        """Initialize counters if not provided."""
        if not hasattr(self, '_pair_counts') or self._pair_counts is None:
            self._pair_counts = Counter()
        if not hasattr(self, '_word_counts') or self._word_counts is None:
            self._word_counts = Counter()

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

        Currently:
            - Lowercase
            - Remove excessive whitespace

        Future extensions:
            - Unicode normalization
            - Accent handling
            - Custom normalization rules
        """
        return text.lower().strip()

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

    def learn_from_texts(self, texts: List[str], n_merges: int = 100) -> None:
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

        Note:
            This is a simplified version. Full BPE would:
            - Operate at character level
            - Update counts after each merge
            We keep it simple for clarity and extensibility.
        """
        # Reset counters
        self._word_counts = Counter()
        self._pair_counts = Counter()

        # Count words and pairs
        for text in texts:
            tokens = self.tokenize(text)
            self._word_counts.update(tokens)

            # Count adjacent pairs
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                self._pair_counts[pair] += 1

        # Build vocabulary from word counts
        self.vocab = set(self._word_counts.keys())

        # Learn merges (frequent pairs that could become compound concepts)
        # We store these for future use in compound atom creation
        self.merges = []
        for pair, count in self._pair_counts.most_common(n_merges):
            if count >= self.min_frequency:
                merged = f"{pair[0]}_{pair[1]}"
                self.merges.append((pair, merged))

        # Limit vocabulary size
        if len(self.vocab) > self.max_vocab_size:
            top_words = [w for w, _ in self._word_counts.most_common(self.max_vocab_size)]
            self.vocab = set(top_words)

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

    # Statistics tracking
    _documents_fed: int = 0
    _atoms_created: int = 0
    _links_created: int = 0

    def learn_vocabulary(self, texts: List[str], n_merges: int = 100) -> None:
        """
        Train the tokenizer on a corpus of texts.

        Call this before feed_text() to establish vocabulary
        and learn frequent patterns.

        Args:
            texts: List of text documents
            n_merges: Number of merge operations for BPE
        """
        self.tokenizer.learn_from_texts(texts, n_merges)

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

        for i, token in enumerate(tokens):
            # Look at window of following words
            window_end = min(i + self.window_size + 1, len(tokens))
            for j in range(i + 1, window_end):
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
        import math
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
    ) -> 'Atom':
        """
        Create or strengthen SIMILARITY link between atoms.

        Link strength is based on co-occurrence frequency.
        If link already exists, merge evidence (strength increases).

        Args:
            atom1: First atom
            atom2: Second atom
            base_strength: Base strength for new links

        Returns:
            The SIMILARITY link atom
        """
        # Calculate strength based on co-occurrence frequency
        pair_freq = self.tokenizer.get_pair_frequency(atom1.name, atom2.name)
        pair_freq += self.tokenizer.get_pair_frequency(atom2.name, atom1.name)

        # Strength scales with frequency (capped)
        import math
        strength = min(0.9, base_strength + math.log1p(pair_freq) * 0.1)

        if strength < self.min_link_strength:
            return None

        # Create link with computed strength
        tv = TruthValue(strength=strength, confidence=0.3)
        link = self.graph.link(AtomType.SIMILARITY, [atom1, atom2], tv)

        self._links_created += 1
        return link

    def get_statistics(self) -> Dict[str, int]:
        """Get processing statistics."""
        return {
            "documents_fed": self._documents_fed,
            "atoms_created": self._atoms_created,
            "links_created": self._links_created,
            "vocabulary_size": len(self.tokenizer.vocab),
            "learned_merges": len(self.tokenizer.merges),
        }


# =============================================================================
# File/Directory Loading Utilities
# =============================================================================


def load_text_file(path: Path) -> str:
    """
    Load text from a file with encoding fallback.

    Tries UTF-8 first, falls back to latin-1.
    """
    try:
        return path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        return path.read_text(encoding='latin-1')


def iter_text_files(
    directory: Path,
    pattern: str = "*.txt",
    recursive: bool = True,
) -> Iterator[Tuple[Path, str]]:
    """
    Iterate over text files in a directory.

    Args:
        directory: Directory to scan
        pattern: Glob pattern for files
        recursive: Whether to search subdirectories

    Yields:
        (path, content) tuples

    Example:
        >>> for path, text in iter_text_files(Path("samples")):
        ...     print(f"Loaded {path.name}: {len(text)} chars")
    """
    if recursive:
        files = directory.rglob(pattern)
    else:
        files = directory.glob(pattern)

    for path in sorted(files):
        if path.is_file():
            try:
                content = load_text_file(path)
                yield path, content
            except Exception as e:
                # Skip files that can't be read
                print(f"Warning: Could not read {path}: {e}")


def load_directory_to_bridge(
    directory: Path,
    bridge: TextToAtomsBridge,
    pattern: str = "*.txt",
    max_files: Optional[int] = None,
    learn_first: bool = True,
) -> Dict[str, int]:
    """
    Load all text files from a directory into the bridge.

    Two-pass approach:
        1. First pass: Learn vocabulary from all texts (if learn_first=True)
        2. Second pass: Feed texts to create atoms and links

    Args:
        directory: Directory containing text files
        bridge: TextToAtomsBridge to use
        pattern: Glob pattern for files
        max_files: Maximum files to process (None = all)
        learn_first: Whether to learn vocabulary first

    Returns:
        Statistics dictionary

    Example:
        >>> bridge = TextToAtomsBridge(agent.graph)
        >>> stats = load_directory_to_bridge(Path("samples"), bridge, max_files=10)
        >>> print(f"Loaded {stats['files_processed']} files")
    """
    directory = Path(directory)

    # Collect files
    files = list(iter_text_files(directory, pattern))
    if max_files:
        files = files[:max_files]

    if not files:
        return {"files_processed": 0, "error": "No files found"}

    # Pass 1: Learn vocabulary (optional but recommended)
    if learn_first:
        texts = [content for _, content in files]
        bridge.learn_vocabulary(texts)

    # Pass 2: Feed texts to create atoms
    for path, content in files:
        doc_id = path.stem  # Use filename without extension as ID
        bridge.feed_text(content, doc_id=doc_id)

    stats = bridge.get_statistics()
    stats["files_processed"] = len(files)

    return stats
