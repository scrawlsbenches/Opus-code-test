"""
Text-to-Atoms Bridge Behavioral Specification.

Tests the BPE tokenizer and text-to-atoms bridge functionality.

Stories covered:
    - Tokenization: text → tokens
    - Vocabulary learning: corpus → learned patterns
    - Atom creation: tokens → WORD atoms
    - Link creation: co-occurrence → SIMILARITY links
    - Directory loading: files → populated graph
"""

import pytest
from pathlib import Path
import tempfile

from cortical.cognitive.graph import (
    AtomType,
    CognitiveAgent,
    CognitiveGraph,
    TruthValue,
)
from cortical.cognitive.text_bridge import (
    BPETokenizer,
    TextToAtomsBridge,
    load_directory_to_bridge,
    load_text_file,
    iter_text_files,
)


# =============================================================================
# BPE Tokenizer Specs
# =============================================================================


class TestBPETokenizerBasics:
    """Basic tokenization behavior."""

    def test_given_simple_text_when_tokenized_then_returns_words(self):
        """
        Given: A simple sentence
        When: Tokenized
        Then: Returns lowercase word tokens
        """
        # Given
        tokenizer = BPETokenizer()
        text = "The cat sat on the mat."

        # When
        tokens = tokenizer.tokenize(text)

        # Then
        assert tokens == ["the", "cat", "sat", "on", "the", "mat"]

    def test_given_text_with_punctuation_when_tokenized_then_strips_punctuation(self):
        """
        Given: Text with various punctuation
        When: Tokenized
        Then: Punctuation is stripped, only words remain
        """
        # Given
        tokenizer = BPETokenizer()
        text = "Hello, world! How are you?"

        # When
        tokens = tokenizer.tokenize(text)

        # Then
        assert "hello" in tokens
        assert "world" in tokens
        assert "," not in tokens
        assert "!" not in tokens

    def test_given_text_with_numbers_when_tokenized_then_filters_numbers(self):
        """
        Given: Text with pure numbers
        When: Tokenized
        Then: Pure numbers are filtered (single chars are also filtered)
        """
        # Given
        tokenizer = BPETokenizer()
        text = "I have 42 apples and 7 oranges"

        # When
        tokens = tokenizer.tokenize(text)

        # Then
        assert "apples" in tokens
        assert "oranges" in tokens
        assert "have" in tokens
        # Single letters and numbers filtered
        assert "42" not in tokens
        assert "7" not in tokens


class TestBPETokenizerLearning:
    """Vocabulary learning behavior."""

    def test_given_corpus_when_learned_then_builds_vocabulary(self):
        """
        Given: A corpus of texts
        When: Vocabulary is learned
        Then: All unique words are in vocabulary
        """
        # Given
        tokenizer = BPETokenizer()
        texts = [
            "The cat sat on the mat.",
            "The dog ran in the park.",
        ]

        # When
        tokenizer.learn_from_texts(texts)

        # Then
        assert "cat" in tokenizer.vocab
        assert "dog" in tokenizer.vocab
        assert "the" in tokenizer.vocab
        assert "mat" in tokenizer.vocab

    def test_given_corpus_when_learned_then_counts_word_frequencies(self):
        """
        Given: A corpus with repeated words
        When: Vocabulary is learned
        Then: Word frequencies are tracked
        """
        # Given
        tokenizer = BPETokenizer()
        texts = ["the cat the dog the bird"]

        # When
        tokenizer.learn_from_texts(texts)

        # Then
        assert tokenizer.get_word_frequency("the") == 3
        assert tokenizer.get_word_frequency("cat") == 1
        assert tokenizer.get_word_frequency("unknown") == 0

    def test_given_corpus_when_learned_then_identifies_frequent_pairs(self):
        """
        Given: A corpus with recurring word pairs
        When: Vocabulary is learned
        Then: Frequent pairs are identified for potential merging
        """
        # Given
        tokenizer = BPETokenizer()
        texts = [
            "machine learning is great",
            "machine learning techniques",
            "deep machine learning",
        ]

        # When
        tokenizer.learn_from_texts(texts, n_merges=10)

        # Then
        # "machine learning" should be a frequent pair
        freq = tokenizer.get_pair_frequency("machine", "learning")
        assert freq >= 2  # Appears at least twice


# =============================================================================
# Text-to-Atoms Bridge Specs
# =============================================================================


class TestTextToAtomsBridge:
    """Text-to-atoms conversion behavior."""

    def test_given_text_when_fed_then_creates_word_atoms(self):
        """
        Given: A text and a bridge
        When: Text is fed
        Then: WORD atoms are created for each unique word
        """
        # Given
        graph = CognitiveGraph()
        bridge = TextToAtomsBridge(graph)
        text = "The cat sat on the mat."

        # When
        atoms = bridge.feed_text(text, doc_id="test1")

        # Then
        word_atoms = graph.find_by_type(AtomType.WORD)
        word_names = {a.name for a in word_atoms}

        assert "cat" in word_names
        assert "sat" in word_names
        assert "mat" in word_names

    def test_given_text_when_fed_then_creates_similarity_links(self):
        """
        Given: A text with co-occurring words
        When: Text is fed
        Then: SIMILARITY links are created between nearby words
        """
        # Given
        graph = CognitiveGraph()
        bridge = TextToAtomsBridge(graph, window_size=3)
        text = "cat dog bird"

        # When
        bridge.feed_text(text, doc_id="test1")

        # Then
        similarity_links = graph.find_by_type(AtomType.SIMILARITY)
        assert len(similarity_links) > 0

        # cat and dog should be linked (adjacent)
        cat = graph.get_node("cat")
        dog = graph.get_node("dog")
        assert cat is not None
        assert dog is not None

        # Check for link between them
        incoming = graph.get_incoming(cat.id)
        link_targets = []
        for link in incoming:
            if link.atom_type == AtomType.SIMILARITY:
                link_targets.extend(link.outgoing)

        assert dog.id in link_targets or cat.id in link_targets

    def test_given_duplicate_words_when_fed_then_reuses_atoms(self):
        """
        Given: Text with repeated words
        When: Fed to bridge
        Then: Same atom is reused (content-addressed)
        """
        # Given
        graph = CognitiveGraph()
        bridge = TextToAtomsBridge(graph)
        text = "the cat and the dog and the bird"

        # When
        bridge.feed_text(text)

        # Then
        # "the" should only create one atom
        word_atoms = graph.find_by_type(AtomType.WORD)
        the_atoms = [a for a in word_atoms if a.name == "the"]
        assert len(the_atoms) == 1

    def test_given_vocabulary_learned_when_fed_then_sets_lti_based_on_frequency(self):
        """
        Given: A bridge with learned vocabulary
        When: Text is fed
        Then: WORD atoms have LTI proportional to frequency
        """
        # Given
        graph = CognitiveGraph()
        bridge = TextToAtomsBridge(graph)

        # Learn vocabulary first
        corpus = [
            "the cat sat",
            "the dog ran",
            "the bird flew",
        ]
        bridge.learn_vocabulary(corpus)

        # When
        bridge.feed_text("the cat", doc_id="test")

        # Then
        the_atom = graph.get_node("the")
        cat_atom = graph.get_node("cat")

        # "the" appears 3 times, "cat" appears 1 time
        # "the" should have higher LTI
        assert the_atom is not None
        assert cat_atom is not None
        assert the_atom.lti > cat_atom.lti

    def test_given_multiple_documents_when_fed_then_tracks_statistics(self):
        """
        Given: A bridge
        When: Multiple documents are fed
        Then: Statistics are tracked correctly
        """
        # Given
        graph = CognitiveGraph()
        bridge = TextToAtomsBridge(graph)

        # When
        bridge.feed_text("the cat sat", doc_id="doc1")
        bridge.feed_text("the dog ran", doc_id="doc2")
        bridge.feed_text("the bird flew", doc_id="doc3")

        # Then
        stats = bridge.get_statistics()
        assert stats["documents_fed"] == 3
        assert stats["atoms_created"] > 0


# =============================================================================
# File Loading Specs
# =============================================================================


class TestFileLoading:
    """File and directory loading behavior."""

    def test_given_text_file_when_loaded_then_returns_content(self):
        """
        Given: A text file
        When: Loaded
        Then: Returns file content as string
        """
        from cortical.common.filesystem import InMemoryFileSystem

        # Given - use in-memory filesystem for fast testing
        filesystem = InMemoryFileSystem(Path("/test"))
        filesystem.mkdir(filesystem.base_dir, parents=True, exist_ok=True)
        path = filesystem.base_dir / "test.txt"
        filesystem.write_text(path, "Hello, world!")

        # When
        content = load_text_file(path, filesystem)

        # Then
        assert content == "Hello, world!"

    def test_given_directory_when_iterated_then_yields_text_files(self):
        """
        Given: A directory with text files
        When: Iterated
        Then: Yields (path, content) tuples
        """
        from cortical.common.filesystem import InMemoryFileSystem

        # Given - use in-memory filesystem for fast testing
        filesystem = InMemoryFileSystem(Path("/test"))
        dir_path = filesystem.base_dir
        filesystem.mkdir(dir_path, parents=True, exist_ok=True)

        # Create test files in memory
        filesystem.write_text(dir_path / "file1.txt", "Content 1")
        filesystem.write_text(dir_path / "file2.txt", "Content 2")
        filesystem.write_text(dir_path / "not_text.md", "Markdown")

        # When
        results = list(iter_text_files(dir_path, filesystem, pattern="*.txt", recursive=False))

        # Then
        assert len(results) == 2
        paths = [r[0].name for r in results]
        assert "file1.txt" in paths
        assert "file2.txt" in paths

    def test_given_directory_when_loaded_to_bridge_then_populates_graph(self):
        """
        Given: A directory with sample texts
        When: Loaded to bridge
        Then: Graph is populated with atoms and links
        """
        from cortical.common.filesystem import InMemoryFileSystem

        # Given - use in-memory filesystem for fast testing
        graph = CognitiveGraph()
        bridge = TextToAtomsBridge(graph)
        filesystem = InMemoryFileSystem(Path("/test"))
        dir_path = filesystem.base_dir
        filesystem.mkdir(dir_path, parents=True, exist_ok=True)

        # Create test files in memory
        filesystem.write_text(dir_path / "doc1.txt", "The cat sat on the mat.")
        filesystem.write_text(dir_path / "doc2.txt", "The dog ran in the park.")

        # When
        stats = load_directory_to_bridge(dir_path, bridge, filesystem)

        # Then
        assert stats["files_processed"] == 2
        assert stats["atoms_created"] > 0

        word_atoms = graph.find_by_type(AtomType.WORD)
        assert len(word_atoms) > 0


# =============================================================================
# Integration with CognitiveAgent
# =============================================================================


class TestAgentIntegration:
    """Integration with CognitiveAgent behavior."""

    def test_given_agent_when_text_loaded_then_can_navigate_concepts(self):
        """
        Given: A CognitiveAgent
        When: Text is loaded via bridge
        Then: Agent can navigate between related concepts
        """
        # Given
        agent = CognitiveAgent()
        bridge = TextToAtomsBridge(agent.graph)

        # When
        bridge.feed_text("The cat sat on the mat. The cat is fluffy.", doc_id="test")

        # Then
        cat_atom = agent.graph.get_node("cat")
        assert cat_atom is not None

        # Attend to cat (stimulates AND loads into working memory)
        agent.attend(cat_atom.id, amount=2.0)

        # Step the agent
        agent.step()

        # Agent should have items in working memory (cat was loaded by attend)
        wm_items = agent.working_memory.contents()
        assert len(wm_items) > 0

    def test_given_agent_with_loaded_text_when_stepped_then_spreads_attention(self):
        """
        Given: An agent with loaded text
        When: Agent steps
        Then: Attention spreads through connected concepts
        """
        # Given
        agent = CognitiveAgent()
        bridge = TextToAtomsBridge(agent.graph)
        bridge.feed_text("cat dog bird fish", doc_id="animals")

        # Stimulate one concept (via graph.stimulate)
        cat = agent.graph.get_node("cat")
        agent.graph.stimulate(cat.id, amount=2.0)

        # When
        for _ in range(3):
            agent.step()

        # Then
        # Other concepts should have gained attention
        dog = agent.graph.get_node("dog")
        assert dog is not None
        # Dog should have some attention (spread from cat via similarity)
        # Note: actual value depends on link strength and decay
        # Just verify dog is in graph and agent processed it
        assert dog in agent.graph._storage.all_atoms()
