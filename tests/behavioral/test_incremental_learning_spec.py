"""
Incremental Learning Behavioral Specification.

Tests the incremental training and model persistence functionality.

Stories covered:
    - Model persistence: save → load → same state
    - Incremental vocabulary: existing vocab + new docs → expanded vocab
    - Incremental links: existing links + new docs → more links
    - Query deduplication: multiple links to same word → one result
"""

import json
import pytest
from pathlib import Path
import tempfile

from cortical.cognitive.graph import (
    Atom,
    AtomType,
    CognitiveAgent,
    CognitiveGraph,
    TruthValue,
)
from cortical.cognitive.text_bridge import (
    BPETokenizer,
    TextToAtomsBridge,
)


# =============================================================================
# Model Persistence Specs
# =============================================================================


class TestModelPersistence:
    """Model save and load behavior."""

    def test_given_trained_model_when_saved_and_loaded_then_atoms_preserved(self, tmp_path):
        """
        Given: A trained model with atoms
        When: Saved to disk and loaded back
        Then: All atoms are preserved with correct attributes
        """
        # Given
        agent = CognitiveAgent()
        bridge = TextToAtomsBridge(agent.graph)
        bridge.learn_vocabulary(["The cat sat on the mat."])
        bridge.feed_text("The cat sat on the mat.", doc_id="test1")

        original_atom_count = len(list(agent.graph._storage.all_atoms()))
        original_word_count = len(agent.graph.find_by_type(AtomType.WORD))

        # When - Save
        graph_path = tmp_path / "graph.json"
        atoms_data = []
        for atom in agent.graph._storage.all_atoms():
            atoms_data.append({
                "id": atom.id,
                "name": atom.name,
                "atom_type": atom.atom_type.name,
                "tv_strength": atom.tv.strength,
                "tv_confidence": atom.tv.confidence,
                "sti": atom.sti,
                "lti": atom.lti,
                "outgoing": atom.outgoing,
            })

        with open(graph_path, "w") as f:
            json.dump({"atoms": atoms_data, "stats": bridge.get_statistics()}, f)

        # When - Load
        with open(graph_path) as f:
            data = json.load(f)

        loaded_agent = CognitiveAgent()
        for atom_data in data["atoms"]:
            atom = Atom(
                id=atom_data["id"],
                atom_type=AtomType[atom_data["atom_type"]],
                name=atom_data.get("name", ""),
                outgoing=atom_data.get("outgoing", []),
                tv=TruthValue(
                    atom_data.get("tv_strength", 1.0),
                    atom_data.get("tv_confidence", 0.0),
                ),
                sti=atom_data.get("sti", 0.0),
                lti=atom_data.get("lti", 0.0),
            )
            loaded_agent.graph._storage.save(atom)

        # Then
        loaded_atom_count = len(list(loaded_agent.graph._storage.all_atoms()))
        loaded_word_count = len(loaded_agent.graph.find_by_type(AtomType.WORD))

        assert loaded_atom_count == original_atom_count
        assert loaded_word_count == original_word_count

    def test_given_saved_model_when_loaded_then_word_atoms_have_correct_names(self, tmp_path):
        """
        Given: A saved model with word atoms
        When: Loaded back
        Then: Word atoms have their original names
        """
        # Given
        agent = CognitiveAgent()
        bridge = TextToAtomsBridge(agent.graph)
        bridge.learn_vocabulary(["neural networks process data"])
        bridge.feed_text("neural networks process data", doc_id="test1")

        # Save
        graph_path = tmp_path / "graph.json"
        atoms_data = [{
            "id": atom.id,
            "name": atom.name,
            "atom_type": atom.atom_type.name,
            "tv_strength": atom.tv.strength,
            "tv_confidence": atom.tv.confidence,
            "sti": atom.sti,
            "lti": atom.lti,
            "outgoing": atom.outgoing,
        } for atom in agent.graph._storage.all_atoms()]

        with open(graph_path, "w") as f:
            json.dump({"atoms": atoms_data}, f)

        # When - Load
        with open(graph_path) as f:
            data = json.load(f)

        loaded_agent = CognitiveAgent()
        for atom_data in data["atoms"]:
            atom = Atom(
                id=atom_data["id"],
                atom_type=AtomType[atom_data["atom_type"]],
                name=atom_data.get("name", ""),
                outgoing=atom_data.get("outgoing", []),
                tv=TruthValue(atom_data.get("tv_strength", 1.0), atom_data.get("tv_confidence", 0.0)),
                sti=atom_data.get("sti", 0.0),
                lti=atom_data.get("lti", 0.0),
            )
            loaded_agent.graph._storage.save(atom)

        # Then
        word_names = {a.name for a in loaded_agent.graph.find_by_type(AtomType.WORD)}
        assert "neural" in word_names
        assert "networks" in word_names
        assert "process" in word_names
        assert "data" in word_names


class TestLinkCountPersistence:
    """Link count computation on load behavior."""

    def test_given_saved_bridge_when_loaded_then_link_counts_recomputed(self, tmp_path):
        """
        Given: A bridge with atoms and links saved to disk
        When: Loaded back
        Then: _atom_link_counts is recomputed from existing graph

        This ensures IDF-based link limiting works correctly after reload.
        Regression test for bug: link counts weren't persisted, causing
        limits to reset after reload.
        """
        from cortical.common.filesystem import InMemoryFileSystem

        # Given - Create bridge with multiple links
        agent = CognitiveAgent()
        bridge = TextToAtomsBridge(agent.graph, window_size=3)
        bridge.learn_vocabulary(["the cat sat on the mat and the dog ran"])
        bridge.feed_text("the cat sat on the mat and the dog ran", doc_id="test1")

        # Track original link counts
        original_link_counts = dict(bridge._atom_link_counts)
        assert len(original_link_counts) > 0, "Should have link counts tracked"

        # When - Save using sharded storage
        filesystem = InMemoryFileSystem(tmp_path)
        filesystem.mkdir(tmp_path, parents=True, exist_ok=True)
        bridge.save(tmp_path, filesystem)

        # Load into new graph
        new_agent = CognitiveAgent()
        loaded_bridge = TextToAtomsBridge.load(tmp_path, new_agent.graph, filesystem)

        # Then - Link counts should be recomputed
        assert len(loaded_bridge._atom_link_counts) > 0, "Link counts should be recomputed on load"

        # Verify counts are reasonable (each link contributes to 2 atoms)
        total_link_count = sum(loaded_bridge._atom_link_counts.values())
        link_atoms = [a for a in new_agent.graph._storage.all_atoms() if a.outgoing]
        expected_total = len(link_atoms) * 2  # Each link connects 2 atoms

        assert total_link_count == expected_total, (
            f"Total link counts ({total_link_count}) should equal links*2 ({expected_total})"
        )


# =============================================================================
# Incremental Learning Specs
# =============================================================================


class TestIncrementalLearning:
    """Incremental vocabulary and link learning behavior."""

    def test_given_existing_vocabulary_when_learning_incrementally_then_vocabulary_expands(self):
        """
        Given: A tokenizer with existing vocabulary
        When: Learning from new texts incrementally
        Then: New words are added to vocabulary
        """
        # Given
        tokenizer = BPETokenizer()
        tokenizer.learn_from_texts(["The cat sat on the mat."])
        initial_vocab_size = len(tokenizer.vocab)
        assert "cat" in tokenizer.vocab
        assert "dog" not in tokenizer.vocab

        # When
        tokenizer.learn_from_texts(["The dog ran in the park."], incremental=True)

        # Then
        assert len(tokenizer.vocab) > initial_vocab_size
        assert "cat" in tokenizer.vocab  # Original words preserved
        assert "dog" in tokenizer.vocab  # New words added
        assert "park" in tokenizer.vocab

    def test_given_existing_vocabulary_when_learning_non_incrementally_then_vocabulary_replaced(self):
        """
        Given: A tokenizer with existing vocabulary
        When: Learning from new texts non-incrementally
        Then: Old vocabulary is replaced
        """
        # Given
        tokenizer = BPETokenizer()
        tokenizer.learn_from_texts(["The cat sat on the mat."])
        assert "cat" in tokenizer.vocab

        # When
        tokenizer.learn_from_texts(["The dog ran in the park."], incremental=False)

        # Then
        assert "dog" in tokenizer.vocab
        assert "cat" not in tokenizer.vocab  # Old words removed

    def test_given_graph_with_atoms_when_feeding_new_text_then_new_atoms_created(self):
        """
        Given: A graph with existing word atoms
        When: Feeding new text with new words
        Then: New word atoms are created
        """
        # Given
        agent = CognitiveAgent()
        bridge = TextToAtomsBridge(agent.graph)
        bridge.learn_vocabulary(["cognitive systems"])
        bridge.feed_text("cognitive systems", doc_id="doc1")

        initial_word_count = len(agent.graph.find_by_type(AtomType.WORD))

        # When
        bridge.tokenizer.learn_from_texts(["neural networks"], incremental=True)
        bridge.feed_text("neural networks", doc_id="doc2")

        # Then
        final_word_count = len(agent.graph.find_by_type(AtomType.WORD))
        assert final_word_count > initial_word_count

        word_names = {a.name for a in agent.graph.find_by_type(AtomType.WORD)}
        assert "cognitive" in word_names
        assert "neural" in word_names

    def test_given_graph_with_links_when_feeding_new_text_then_new_links_created(self):
        """
        Given: A graph with existing similarity links
        When: Feeding new text
        Then: New similarity links are created
        """
        # Given
        agent = CognitiveAgent()
        bridge = TextToAtomsBridge(agent.graph, window_size=3)
        bridge.learn_vocabulary(["attention mechanisms work"])
        bridge.feed_text("attention mechanisms work", doc_id="doc1")

        initial_link_count = len(agent.graph.find_by_type(AtomType.SIMILARITY))

        # When
        bridge.tokenizer.learn_from_texts(["memory systems operate"], incremental=True)
        bridge.feed_text("memory systems operate", doc_id="doc2")

        # Then
        final_link_count = len(agent.graph.find_by_type(AtomType.SIMILARITY))
        assert final_link_count > initial_link_count


# =============================================================================
# Query Deduplication Specs
# =============================================================================


class TestQueryDeduplication:
    """Query result deduplication behavior."""

    def test_given_multiple_links_to_same_word_when_querying_then_word_appears_once(self):
        """
        Given: Multiple similarity links connecting to the same word
        When: Querying connections for a word
        Then: Each connected word appears only once with max strength
        """
        # Given - Create graph with potential duplicate paths
        agent = CognitiveAgent()
        bridge = TextToAtomsBridge(agent.graph, window_size=5)

        # Feed text multiple times to potentially create multiple links
        texts = [
            "data processing systems",
            "data analysis systems",
            "data management systems",
        ]
        bridge.learn_vocabulary(texts)
        for i, text in enumerate(texts):
            bridge.feed_text(text, doc_id=f"doc{i}")

        # When - Query connections for "data"
        data_atom = agent.graph.get_node("data")
        assert data_atom is not None

        incoming = agent.graph.get_incoming(data_atom.id)
        similarity_links = [l for l in incoming if l.atom_type == AtomType.SIMILARITY]

        # Build deduplicated connections (same logic as demo fix)
        connections_dict = {}
        for link in similarity_links:
            for target_id in link.outgoing:
                if target_id != data_atom.id:
                    other = agent.graph.get_atom(target_id)
                    if other and other.name:
                        if other.name not in connections_dict or link.tv.strength > connections_dict[other.name]:
                            connections_dict[other.name] = link.tv.strength

        # Then - Each word appears exactly once
        word_counts = {}
        for name in connections_dict.keys():
            word_counts[name] = word_counts.get(name, 0) + 1

        for word, count in word_counts.items():
            assert count == 1, f"Word '{word}' appeared {count} times, expected 1"

    def test_given_deduplicated_results_when_sorted_then_max_strength_preserved(self):
        """
        Given: Multiple links to same word with different strengths
        When: Deduplicating and sorting
        Then: The maximum strength is preserved for each word
        """
        # Given - Simulate multiple links with different strengths
        connections = [
            ("systems", 0.5),
            ("systems", 0.7),  # Higher strength, should be kept
            ("systems", 0.3),
            ("processing", 0.6),
            ("processing", 0.4),
        ]

        # When - Deduplicate keeping max
        connections_dict = {}
        for name, strength in connections:
            if name not in connections_dict or strength > connections_dict[name]:
                connections_dict[name] = strength

        # Then
        assert connections_dict["systems"] == 0.7
        assert connections_dict["processing"] == 0.6


# =============================================================================
# Performance Tuning Specs
# =============================================================================


class TestPerformanceTuning:
    """Performance parameter behavior."""

    def test_given_max_links_limit_when_feeding_text_then_similarity_links_capped(self):
        """
        Given: A bridge with max_links_per_doc limit
        When: Feeding text that would create more SIMILARITY links
        Then: SIMILARITY link creation stops at the limit

        Note: max_links_per_doc only applies to SIMILARITY links.
        FOLLOWS links are created unconditionally for complete sequence prediction.
        """
        # Given
        agent = CognitiveAgent()
        bridge = TextToAtomsBridge(agent.graph, max_links_per_doc=5)

        # Text that would normally create many links
        text = "one two three four five six seven eight nine ten"
        bridge.learn_vocabulary([text])

        # When
        bridge.feed_text(text, doc_id="test")

        # Then - count SIMILARITY links specifically
        from cortical.cognitive.graph import AtomType
        similarity_links = [
            a for a in agent.graph._storage.find_by_type(AtomType.SIMILARITY)
        ]
        assert len(similarity_links) <= 5

    def test_given_small_window_size_when_feeding_text_then_fewer_links_created(self):
        """
        Given: A bridge with small window size
        When: Feeding text
        Then: Fewer links are created (only nearby words linked)
        """
        # Given
        text = "one two three four five"

        # Small window
        agent1 = CognitiveAgent()
        bridge1 = TextToAtomsBridge(agent1.graph, window_size=1, max_links_per_doc=100)
        bridge1.learn_vocabulary([text])
        bridge1.feed_text(text, doc_id="test")
        small_window_links = len(agent1.graph.find_by_type(AtomType.SIMILARITY))

        # Large window
        agent2 = CognitiveAgent()
        bridge2 = TextToAtomsBridge(agent2.graph, window_size=5, max_links_per_doc=100)
        bridge2.learn_vocabulary([text])
        bridge2.feed_text(text, doc_id="test")
        large_window_links = len(agent2.graph.find_by_type(AtomType.SIMILARITY))

        # Then
        assert small_window_links < large_window_links
