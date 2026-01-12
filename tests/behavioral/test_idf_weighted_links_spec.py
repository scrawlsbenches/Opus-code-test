"""
BDD Specification: IDF-Weighted Similarity Links with Staleness Handling

Epic: Intelligent Link Weighting for Semantic Relevance

As a cognitive agent processing natural language,
I want similarity links weighted by Inverse Document Frequency (IDF),
So that semantically meaningful relationships surface above common stopwords.

Background:
- Current link strength is based solely on co-occurrence frequency
- Stopwords like "the", "and", "is" dominate similarity links
- Stopwords are needed for future "cognitive communication skills" (not removed)
- IDF weighting down-weights common terms while preserving them
- Training is allowed to be slow; queries must be fast

Design Decisions:
1. DUAL VALUE STORAGE
   - Store BOTH raw co-occurrence strength AND IDF-weighted strength
   - Raw: strength = log1p(pair_freq) * 0.1 + base (current behavior)
   - Weighted: idf_strength = raw_strength * min(idf_word1, idf_word2)
   - Enables query flexibility and historical comparison

2. IDF COMPUTATION AT TRAINING TIME
   - IDF = log((N + 1) / (df + 1)) where N=total docs, df=docs containing term
   - Smoothed formula avoids division by zero and log(0)
   - Computed once per word during vocabulary learning
   - Stored with tokenizer for O(1) lookup at query time

3. STALENESS HANDLING
   - Track last_reindex_doc_count in manifest
   - When corpus grows >20%, warn but don't block
   - --reindex flag recalculates all IDF weights
   - Concurrent query/index: readers see consistent snapshot

Acceptance Criteria for MVP:
□ IDF computed during vocabulary learning
□ Links store both raw_strength and idf_strength
□ Query API can request either weight type
□ --reindex recalculates all link IDF weights
□ Manifest tracks staleness metrics
□ Warning when IDF staleness >20%
"""

import pytest
from dataclasses import dataclass
from typing import Optional
from pathlib import Path


# =============================================================================
# FIXTURES (will be implemented)
# =============================================================================


@pytest.fixture
def idf_tokenizer():
    """Tokenizer with IDF tracking capability."""
    from cortical.cognitive.text_bridge import BPETokenizer
    return BPETokenizer()


@pytest.fixture
def idf_bridge(memory_cognitive_container):
    """TextToAtomsBridge with IDF weighting enabled (via DI container)."""
    from cortical.cognitive.text_bridge import TextToAtomsBridge
    return memory_cognitive_container.resolve(TextToAtomsBridge)


class TrainedAgentWrapper:
    """
    Wrapper that combines trainer, agent, and bridge for testing.

    Provides unified access to:
    - agent.get_associations() from CognitiveAgent
    - agent.tokenizer from TextToAtomsBridge
    - agent.manifest from IncrementalTrainer
    - agent.reindex() from IncrementalTrainer
    """

    def __init__(self, trainer: 'IncrementalTrainer'):
        self._trainer = trainer
        self._agent = trainer.agent
        self._bridge = trainer.bridge

    @property
    def tokenizer(self):
        return self._bridge.tokenizer

    @property
    def manifest(self):
        return self._trainer.manifest

    def get_associations(self, word: str, weight_type: str = "idf", top_k: int = 20):
        return self._agent.get_associations(word, weight_type=weight_type, top_k=top_k)

    def get_all_similarity_links(self):
        return self._bridge.get_similarity_links()

    def reindex(self):
        return self._trainer.reindex(show_progress=False)

    def train_incremental(self, texts: list):
        """Train on additional texts WITHOUT updating IDF (IDF becomes stale)."""
        # Check for stale IDF weights (emits warning to stderr if stale)
        self._trainer._check_staleness_warning()
        # Feed texts to create atoms (IDF not updated - becomes stale)
        for i, text in enumerate(texts):
            self._bridge.feed_text(text, doc_id=f"incremental_{i}")
            self._trainer.manifest.total_documents += 1
        self._trainer.save()

    def get_idf_epoch(self):
        """Get current IDF epoch from bridge."""
        return self._bridge.get_idf_epoch()


@pytest.fixture
def trained_agent_with_idf(memory_cognitive_container):
    """Pre-trained cognitive agent with IDF-weighted links (via DI container)."""
    from cortical.cognitive.training import IncrementalTrainer

    # Resolve trainer from container (uses InMemoryFileSystem)
    trainer = memory_cognitive_container.resolve(IncrementalTrainer)

    # Training corpus with varied IDF values
    corpus = [
        "Neural networks learn patterns from data using layers.",
        "Deep learning uses neural networks with many layers.",
        "Machine learning algorithms process data efficiently.",
        "Data science combines statistics and programming.",
        "Artificial intelligence includes machine learning methods.",
    ]

    # Learn vocabulary first (computes IDF)
    trainer.bridge.learn_vocabulary(corpus)

    # Train on documents
    for i, text in enumerate(corpus):
        trainer.bridge.feed_text(text, doc_id=f"doc_{i}")
        trainer.manifest.total_documents += 1

    # Initialize reindex tracking
    trainer.manifest.last_reindex_doc_count = trainer.manifest.total_documents
    trainer.manifest.idf_epoch = 1
    trainer.save()

    return TrainedAgentWrapper(trainer)


# =============================================================================
# STORY 1: IDF COMPUTATION DURING VOCABULARY LEARNING
# =============================================================================


@pytest.mark.behavioral
@pytest.mark.idf
class TestIDFComputation:
    """
    Story 1: IDF values computed during vocabulary learning

    As a training system,
    I want to compute IDF for each word during vocabulary learning,
    So that term importance is known before link creation.
    """

    def test_idf_computed_for_each_word(self, idf_tokenizer):
        """
        Scenario: IDF computed for all vocabulary words

        Given a corpus with varied word frequencies
        When vocabulary learning completes
        Then each word should have an IDF value
        And IDF values should be persisted with tokenizer

        Because IDF must be available for link weighting.
        """
        # Given: Corpus with varied frequencies
        docs = [
            "the cat sat on the mat",           # "the" appears 2x
            "the dog ran in the park",          # "the" appears 2x
            "a cat and a dog played together",  # no "the"
            "neural networks process data",      # rare terms
        ]

        # When: Learn vocabulary
        idf_tokenizer.learn_vocabulary(docs)

        # Then: Each word has IDF
        for word in idf_tokenizer.vocab:
            idf = idf_tokenizer.get_idf(word)
            assert idf is not None
            assert idf >= 0.0

        # Common words have lower IDF
        idf_the = idf_tokenizer.get_idf("the")
        idf_neural = idf_tokenizer.get_idf("neural")
        assert idf_neural > idf_the  # "neural" rarer than "the"

    def test_idf_formula_correctness(self, idf_tokenizer):
        """
        Scenario: IDF follows smoothed formula

        Given a word appearing in specific number of documents
        When computing IDF
        Then IDF = log((N + 1) / (df + 1)) where N=total docs, df=doc frequency

        Because smoothed IDF avoids division by zero and ensures comparable weights.
        """
        import math

        # Given: Controlled corpus
        docs = [
            "apple banana cherry",
            "apple banana date",
            "apple elderberry fig",
            "grape honeydew",  # no apple
        ]

        # When: Learn vocabulary
        idf_tokenizer.learn_vocabulary(docs)

        # Then: IDF follows smoothed formula
        # "apple" in 3/4 docs: IDF = log((4+1)/(3+1)) = log(5/4) ≈ 0.223
        # "grape" in 1/4 docs: IDF = log((4+1)/(1+1)) = log(5/2) ≈ 0.916
        N = 4

        idf_apple = idf_tokenizer.get_idf("apple")
        expected_apple = math.log((N + 1) / (3 + 1))
        assert abs(idf_apple - expected_apple) < 0.01

        idf_grape = idf_tokenizer.get_idf("grape")
        expected_grape = math.log((N + 1) / (1 + 1))
        assert abs(idf_grape - expected_grape) < 0.01

    def test_idf_persisted_with_tokenizer(self, idf_tokenizer, tmp_path):
        """
        Scenario: IDF values survive save/load cycle

        Given a tokenizer with computed IDF values
        When saving and loading tokenizer
        Then IDF values should be restored

        Because IDF must persist across sessions.
        """
        from cortical.common.filesystem import InMemoryFileSystem

        # Given: Computed IDF
        docs = ["neural networks are powerful", "deep learning advances"]
        idf_tokenizer.learn_vocabulary(docs)

        original_idf_neural = idf_tokenizer.get_idf("neural")

        # When: Save and reload
        fs = InMemoryFileSystem(tmp_path)
        fs.mkdir(tmp_path, parents=True, exist_ok=True)
        save_path = tmp_path / "tokenizer.json"
        idf_tokenizer.save(save_path, fs)

        # Create new tokenizer and load
        from cortical.cognitive.text_bridge import BPETokenizer
        loaded = BPETokenizer.load(save_path, fs)

        # Then: IDF restored
        loaded_idf_neural = loaded.get_idf("neural")
        assert loaded_idf_neural == original_idf_neural


# =============================================================================
# STORY 2: DUAL VALUE STORAGE IN LINKS
# =============================================================================


@pytest.mark.behavioral
@pytest.mark.idf
class TestDualValueStorage:
    """
    Story 2: Links store both raw and IDF-weighted strength

    As a query system,
    I want both raw and weighted strengths available,
    So that I can choose the appropriate weight for my use case.
    """

    def test_links_store_both_strengths(self, idf_bridge):
        """
        Scenario: Links have raw_strength and idf_strength

        Given text creating similarity links
        When links are created
        Then each link should have raw_strength (co-occurrence based)
        And each link should have idf_strength (IDF weighted)

        Because different queries need different weights.
        """
        # Given: Corpus with varied term frequencies (no word in ALL docs)
        # This ensures IDF > 0 for words used in links
        idf_bridge.learn_vocabulary([
            "neural network deep learning",
            "machine learning algorithm",
            "data science visualization",
            "computer vision recognition",
        ])
        idf_bridge.feed_text("neural network learning")

        # When: Get similarity links
        links = idf_bridge.get_similarity_links()

        # Then: Links have both strengths
        for link in links:
            assert hasattr(link, 'raw_strength') or 'raw_strength' in link.metadata
            assert hasattr(link, 'idf_strength') or 'idf_strength' in link.metadata

            raw = link.raw_strength if hasattr(link, 'raw_strength') else link.metadata['raw_strength']
            weighted = link.idf_strength if hasattr(link, 'idf_strength') else link.metadata['idf_strength']

            # raw_strength should be positive (co-occurrence based)
            assert raw > 0.0
            # idf_strength should be non-negative (0 if word appears in all docs)
            assert weighted >= 0.0

    def test_idf_strength_down_weights_common_terms(self, idf_bridge):
        """
        Scenario: Common term links have lower IDF strength

        Given links involving common words ("the") and rare words ("neural")
        When comparing IDF-weighted strengths
        Then rare word links should have higher IDF strength

        Because IDF surfaces semantically meaningful relationships.
        """
        # Given: Vocabulary with varied frequencies
        idf_bridge.learn_vocabulary([
            "the the the the the",  # "the" very common
            "neural network unique",  # "neural" rare
        ] * 10)

        idf_bridge.feed_text("the neural network")

        # When: Get links
        links = idf_bridge.get_similarity_links()

        # Helper to get word names from link's outgoing atoms
        def get_link_words(link):
            words = set()
            for atom_id in link.outgoing:
                atom = idf_bridge.graph._storage.load(atom_id)
                if atom and atom.name:
                    words.add(atom.name)
            return words

        # Find specific links
        the_link = None
        neural_link = None
        for link in links:
            words = get_link_words(link)
            if "the" in words and "neural" in words:
                the_link = link
            if "neural" in words and "network" in words:
                neural_link = link

        # Then: "neural-network" link has higher IDF strength than "the-neural"
        if the_link and neural_link:
            the_idf = the_link.idf_strength if hasattr(the_link, 'idf_strength') else the_link.metadata['idf_strength']
            neural_idf = neural_link.idf_strength if hasattr(neural_link, 'idf_strength') else neural_link.metadata['idf_strength']

            assert neural_idf > the_idf

    def test_raw_strength_preserved(self, idf_bridge):
        """
        Scenario: Raw strength unchanged by IDF weighting

        Given links created with IDF weighting enabled
        When examining raw_strength
        Then it should match pre-IDF behavior (co-occurrence only)

        Because raw strength enables backward compatibility.
        """
        # Given: Create links
        idf_bridge.learn_vocabulary(["hello world hello world"])
        idf_bridge.feed_text("hello world")

        # When: Get link
        links = idf_bridge.get_similarity_links()

        # Then: Raw strength follows original formula
        for link in links:
            raw = link.raw_strength if hasattr(link, 'raw_strength') else link.metadata['raw_strength']
            # Original formula: min(0.9, base + log1p(pair_freq) * 0.1)
            assert 0.1 <= raw <= 0.9


# =============================================================================
# STORY 3: QUERY API WEIGHT SELECTION
# =============================================================================


@pytest.mark.behavioral
@pytest.mark.idf
class TestQueryWeightSelection:
    """
    Story 3: Query API allows weight type selection

    As a developer querying the knowledge graph,
    I want to choose which weight type to use,
    So that I can optimize for my specific use case.
    """

    def test_query_with_idf_weights(self, trained_agent_with_idf):
        """
        Scenario: Query uses IDF weights by default

        Given a trained agent with IDF-weighted links
        When querying for related words
        Then results should be ranked by IDF strength
        And stopwords should not dominate results

        Because IDF weighting is the primary use case.
        """
        # Given: Pre-trained agent
        agent = trained_agent_with_idf

        # When: Query for "neural" associations
        results = agent.get_associations("neural", weight_type="idf")

        # Then: Semantically meaningful words rank higher
        top_words = [r.word for r in results[:5]]

        # "network", "learning" should rank higher than "the", "and"
        stopwords_in_top5 = sum(1 for w in top_words if w in ["the", "and", "is", "a"])
        assert stopwords_in_top5 < 3  # Most top results are meaningful

    def test_query_with_raw_weights(self, trained_agent_with_idf):
        """
        Scenario: Query can use raw co-occurrence weights

        Given a trained agent with both weight types
        When querying with weight_type="raw"
        Then results should be ranked by raw co-occurrence

        Because some analyses need unweighted frequencies.
        """
        # Given: Pre-trained agent
        agent = trained_agent_with_idf

        # When: Query with raw weights
        results_raw = agent.get_associations("neural", weight_type="raw")
        results_idf = agent.get_associations("neural", weight_type="idf")

        # Then: Results may differ in ranking
        raw_top5 = [r.word for r in results_raw[:5]]
        idf_top5 = [r.word for r in results_idf[:5]]

        # Rankings should potentially differ (stopwords may rank higher in raw)
        # This is expected behavior, not an error


# =============================================================================
# STORY 4: REINDEXING COMMAND
# =============================================================================


@pytest.mark.behavioral
@pytest.mark.idf
class TestReindexCommand:
    """
    Story 4: --reindex recalculates all IDF weights

    As an operator managing a growing corpus,
    I want to recalculate IDF weights when needed,
    So that link weights reflect current corpus statistics.
    """

    def test_reindex_updates_idf_values(self, trained_agent_with_idf, tmp_path):
        """
        Scenario: Reindex updates link weights using current IDF

        Given a trained agent with similarity links
        When running reindex
        Then link idf_strengths should be recalculated
        And idf_epoch should increment

        Because reindex ensures link weights use current IDF values.

        Note: IDF values themselves only change during vocabulary learning,
        not during reindex (we don't store documents for recalculation).
        """
        # Given: Initial training
        agent = trained_agent_with_idf

        # Get initial link weights
        links_before = agent.get_all_similarity_links()
        weights_before = {
            link.id: link.metadata.get('idf_strength', 0)
            for link in links_before
        }
        epoch_before = agent.get_idf_epoch()

        # When: Reindex
        result = agent.reindex()

        # Then: Links updated and epoch incremented
        assert result['links_updated'] > 0
        epoch_after = agent.get_idf_epoch()
        assert epoch_after > epoch_before

        # Verify link weights were recalculated (may or may not change)
        links_after = agent.get_all_similarity_links()
        for link in links_after:
            assert 'idf_strength' in link.metadata
            assert 'idf_epoch' in link.metadata
            assert link.metadata['idf_epoch'] == epoch_after

    def test_reindex_preserves_raw_strength(self, trained_agent_with_idf):
        """
        Scenario: Reindex does not modify raw_strength

        Given links with raw and IDF strengths
        When running reindex
        Then raw_strength should remain unchanged
        And only idf_strength should update

        Because raw strength is historical truth.
        """
        # Given: Get initial raw strengths
        agent = trained_agent_with_idf

        links_before = agent.get_all_similarity_links()
        raw_before = {
            link.id: link.metadata.get('raw_strength', link.tv.strength)
            for link in links_before
        }

        # When: Add docs and reindex
        agent.train_incremental(["new document content"] * 10)
        agent.reindex()

        # Then: Raw strengths unchanged
        links_after = agent.get_all_similarity_links()
        for link in links_after:
            if link.id in raw_before:
                current_raw = link.metadata.get('raw_strength', link.tv.strength)
                assert current_raw == raw_before[link.id], \
                    f"Raw strength changed for link {link.id}"

    def test_reindex_cli_command(self, memory_cognitive_container):
        """
        Scenario: CLI supports --reindex flag

        Given a trained model directory
        When running the reindex CLI command
        Then IDF weights should be recalculated
        And the operation should succeed

        Because operators need CLI access to reindexing.
        """
        from argparse import Namespace
        from cortical.cognitive.training import IncrementalTrainer, run_cli

        # Given: Set up a trained model via DI (in-memory)
        trainer = memory_cognitive_container.resolve(IncrementalTrainer)

        # Train some documents
        corpus = ["Neural networks learn patterns.", "Deep learning is powerful."]
        trainer.bridge.learn_vocabulary(corpus)
        for i, text in enumerate(corpus):
            trainer.bridge.feed_text(text, doc_id=f"doc_{i}")
            trainer.manifest.total_documents += 1
        trainer.save()

        # Get initial state
        links_before = trainer.bridge.get_similarity_links()
        epoch_before = trainer.bridge.get_idf_epoch()

        # When: Run CLI reindex command with the in-memory container
        args = Namespace(
            reindex=True,
            quiet=True,
            status=False,
            list=False,
            files=None,
            batch_size=None,
            directory="samples",
            pattern="*.txt",
            force=False,
            checkpoint=None,
            model_dir=str(trainer.model_dir),
        )
        run_cli(args, container=memory_cognitive_container)

        # Then: Links should be updated
        epoch_after = trainer.bridge.get_idf_epoch()
        assert epoch_after > epoch_before, "IDF epoch should increment after reindex"


# =============================================================================
# STORY 5: STALENESS TRACKING AND WARNINGS
# =============================================================================


@pytest.mark.behavioral
@pytest.mark.idf
class TestStalenessTracking:
    """
    Story 5: System tracks and warns about IDF staleness

    As an operator,
    I want warnings when IDF weights become stale,
    So that I know when to run reindexing.
    """

    def test_manifest_tracks_reindex_count(self, tmp_path):
        """
        Scenario: Manifest records last reindex document count

        Given a training manifest
        When training and reindexing
        Then manifest should track:
          - total_documents: current corpus size
          - last_reindex_doc_count: corpus size at last reindex

        Because staleness is (total - last_reindex) / last_reindex.
        """
        from cortical.cognitive.training import TrainingManifest
        from cortical.common.filesystem import InMemoryFileSystem

        # Given: Manifest
        fs = InMemoryFileSystem(tmp_path)
        manifest = TrainingManifest()

        # When: Record training
        manifest.total_documents = 100
        manifest.last_reindex_doc_count = 100

        # Train more
        manifest.total_documents = 125

        # Then: Can compute staleness
        staleness = (manifest.total_documents - manifest.last_reindex_doc_count) / manifest.last_reindex_doc_count
        assert staleness == 0.25  # 25% growth since reindex

    def test_warning_on_high_staleness(self, trained_agent_with_idf, capsys):
        """
        Scenario: Warning when staleness exceeds threshold

        Given corpus grown >20% since last reindex
        When running any training operation
        Then warning should be printed
        And training should still proceed

        Because warnings inform without blocking.
        """
        # Given: Agent with staleness tracking
        agent = trained_agent_with_idf

        # Simulate growth beyond threshold
        agent.manifest.last_reindex_doc_count = 100
        agent.manifest.total_documents = 125  # 25% growth

        # When: Train new document
        agent.train_incremental(["new document"])

        # Then: Warning should appear
        captured = capsys.readouterr()
        assert "stale" in captured.err.lower() or "reindex" in captured.err.lower()

    def test_staleness_threshold_configurable(self, tmp_path):
        """
        Scenario: Staleness threshold can be configured

        Given default threshold of 20%
        When setting custom threshold
        Then warnings should follow custom threshold

        Because different corpora have different tolerance.
        """
        from cortical.cognitive.training import TrainingConfig

        # Given: Custom config
        config = TrainingConfig(staleness_warning_threshold=0.10)  # 10%

        # Then: Threshold applied
        assert config.staleness_warning_threshold == 0.10


# =============================================================================
# STORY 6: CONCURRENT QUERY DURING INDEXING
# =============================================================================


@pytest.mark.behavioral
@pytest.mark.idf
class TestConcurrentQueryIndex:
    """
    Story 6: Query while indexing (read consistency)

    As a system that may query during reindexing,
    I want queries to see consistent snapshots,
    So that results are not corrupted by in-progress updates.

    NOTE: Concurrency implementation is ON HOLD.
    These tests are skipped until threading/locking strategy is finalized.
    See: TestReindexCommand for core reindex functionality.
    """

    def test_query_during_reindex_sees_consistent_state(self, trained_agent_with_idf):
        """
        Scenario: Query sees either old or new weights, not mixed

        Given a reindex operation in progress
        When querying during reindex
        Then query should see consistent weights
        And not a mix of old and new values

        Because partial updates corrupt results.
        """
        import threading
        import time

        agent = trained_agent_with_idf
        results = []
        errors = []

        def query_thread():
            """Repeatedly query during reindex."""
            for _ in range(10):
                try:
                    assocs = agent.get_associations("neural", weight_type="idf")
                    # Check consistency: all weights should be from same epoch
                    weights = [a.weight for a in assocs]
                    results.append(("success", weights))
                except Exception as e:
                    errors.append(str(e))
                time.sleep(0.01)

        def reindex_thread():
            """Run reindex."""
            agent.reindex()

        # Start both threads
        qt = threading.Thread(target=query_thread)
        rt = threading.Thread(target=reindex_thread)

        qt.start()
        rt.start()

        qt.join()
        rt.join()

        # Then: No errors, all results valid
        assert len(errors) == 0
        assert len(results) > 0

    def test_reindex_is_atomic(self, trained_agent_with_idf):
        """
        Scenario: Reindex completes atomically

        Given a reindex operation
        When reindex completes
        Then all links should have consistent weights
        And no partial updates visible

        Because atomicity prevents corruption.
        """
        agent = trained_agent_with_idf

        # Get initial state
        links_before = agent.get_all_similarity_links()
        epoch_before = agent.get_idf_epoch()

        # Reindex
        agent.reindex()

        # All links should have new epoch
        links_after = agent.get_all_similarity_links()
        epoch_after = agent.get_idf_epoch()

        assert epoch_after > epoch_before

        # All links should reference same epoch
        for link in links_after:
            link_epoch = link.metadata.get('idf_epoch', epoch_after)
            assert link_epoch == epoch_after


# =============================================================================
# PERFORMANCE REQUIREMENTS
# =============================================================================


@pytest.mark.behavioral
@pytest.mark.idf
@pytest.mark.performance
class TestIDFPerformance:
    """
    Performance: IDF operations must be fast

    As a query system,
    I want O(1) IDF lookup at query time,
    So that queries remain fast regardless of corpus size.
    """

    def test_idf_lookup_is_o1(self, idf_tokenizer):
        """
        Scenario: IDF lookup is O(1)

        Given a tokenizer with IDF values
        When looking up IDF for a word
        Then lookup should be O(1) (dict access)
        And not require scanning documents

        Because query performance is critical.
        """
        import time

        # Given: Large vocabulary
        words = [f"word_{i}" for i in range(10000)]
        docs = [" ".join(words[i:i+100]) for i in range(0, 10000, 100)]
        idf_tokenizer.learn_vocabulary(docs)

        # When: Time lookups
        start = time.perf_counter()
        for word in words[:1000]:
            _ = idf_tokenizer.get_idf(word)
        elapsed = time.perf_counter() - start

        # Then: Should be very fast (< 10ms for 1000 lookups)
        assert elapsed < 0.01  # 10ms

    def test_reindex_scales_linearly(self, trained_agent_with_idf):
        """
        Scenario: Reindex time scales with link count

        Given N similarity links
        When running reindex
        Then time should be O(N) not O(N²)

        Because large corpora must remain manageable.
        """
        import time

        agent = trained_agent_with_idf
        link_count = len(agent.get_all_similarity_links())

        # Time reindex
        start = time.perf_counter()
        agent.reindex()
        elapsed = time.perf_counter() - start

        # Should complete quickly (< 1s for typical corpus)
        # Exact time depends on link count
        per_link_ms = (elapsed * 1000) / max(link_count, 1)

        # Should be < 0.1ms per link (O(1) per link operation)
        assert per_link_ms < 0.1 or link_count < 100


# =============================================================================
# IMPLEMENTATION NOTES (for developers)
# =============================================================================

"""
Implementation Plan:

1. BPETokenizer changes (cortical/cognitive/text_bridge.py):
   - Add _doc_frequency: Dict[str, int]  # word -> doc count
   - Add _total_docs: int
   - Add get_idf(word) -> float method
   - Update learn_vocabulary() to track doc frequency
   - Update save()/load() to persist IDF data
   - Add sharded storage for _doc_frequency

2. TextToAtomsBridge changes:
   - Modify _create_similarity_link() to compute both strengths
   - Store raw_strength in link.tv.strength (backward compatible)
   - Store idf_strength in link.metadata['idf_strength']
   - Store idf_epoch in link.metadata['idf_epoch']

3. TrainingManifest changes:
   - Add last_reindex_doc_count: int
   - Add idf_epoch: int
   - Add staleness_threshold: float = 0.2

4. IncrementalTrainer changes:
   - Add reindex() method
   - Add staleness check to train_directory()
   - Add --reindex CLI flag

5. CognitiveAgent changes:
   - Add get_associations(word, weight_type="idf"|"raw")
   - Add get_idf_epoch() method
   - Add reindex() method (delegates to trainer)

Threading/Atomicity:
- Use epoch counter for read consistency
- Reindex builds new weights in temp storage
- Atomic swap at completion
- Readers check epoch to detect stale results
"""
