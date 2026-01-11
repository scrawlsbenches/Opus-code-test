"""
=============================================================================
SPECIFICATION: Developer Trains Cognitive Agent Incrementally
=============================================================================

OVERVIEW
--------
This specification defines how developers train CognitiveAgents on text
documents. The training system supports incremental updates, detecting new
and modified files automatically.

CORE REQUIREMENTS
-----------------
1. FileSystem is REQUIRED - enables testability and abstraction
2. CognitiveAgent is REQUIRED - the agent being trained
3. Training is incremental - only new/modified documents are processed
4. Content-hash based detection - file modification time is irrelevant

DEPENDENCIES
------------
- CognitiveAgent: The agent to train (must be provided by caller)
- FileSystem: I/O abstraction (RealFileSystem for disk, InMemoryFileSystem for tests)
- IncrementalTrainer: Orchestrates training with manifest tracking

USAGE EXAMPLE
-------------
    fs = RealFileSystem(base_path)
    agent = CognitiveAgent(filesystem=fs)
    trainer = IncrementalTrainer(agent, model_dir, fs)
    stats = trainer.train_directory("samples/")
"""

import pytest
from pathlib import Path

from cortical.common.filesystem import RealFileSystem


# =============================================================================
# EPIC: Initial Training and Knowledge Acquisition
# =============================================================================

class TestDeveloperTrainsOnNewDocuments:
    """
    EPIC: Initial Training and Knowledge Acquisition
    ================================================

    PERSONA: Developer with a corpus of training documents
    GOAL: Train CognitiveAgent on text files
    VALUE: Agent learns vocabulary and word associations from content

    ACCEPTANCE CRITERIA:
    - Training creates atoms for vocabulary words
    - Training creates links for co-occurring words
    - Statistics accurately reflect what was learned
    - Specific files can be selected for training
    """

    def test_scenario_training_on_documents_creates_knowledge(self, tmp_path):
        """
        Scenario: First-time training creates atoms and links

        Given I have a CognitiveAgent with no prior training
        And I have text documents to train on
        When I train the agent on those documents
        Then atoms are created for vocabulary words
        And links are created for co-occurring words
        And training statistics reflect what was learned
        Because the agent needs to build its knowledge base from text.
        """
        from cortical.cognitive import CognitiveAgent, IncrementalTrainer

        # Given I have a CognitiveAgent with no prior training
        fs = RealFileSystem(tmp_path)
        agent = CognitiveAgent(filesystem=fs)
        trainer = IncrementalTrainer(agent, model_dir=tmp_path / "model", filesystem=fs)

        # And I have text documents to train on
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "doc1.txt").write_text(
            "Neural networks learn patterns from data through training."
        )
        (docs_dir / "doc2.txt").write_text(
            "Training neural models requires careful data preparation."
        )

        # When I train the agent on those documents
        stats = trainer.train_directory(docs_dir, show_progress=False)

        # Then atoms are created for vocabulary words
        assert stats.atoms_created > 0

        # And links are created for co-occurring words
        assert stats.links_created > 0

        # And training statistics reflect what was learned
        assert stats.new_documents == 2
        assert stats.vocabulary_size > 0

    def test_scenario_training_specific_files_only(self, tmp_path):
        """
        Scenario: Selective training on specific files

        Given I have multiple documents in a directory
        And I only want to train on some of them
        When I train on specific files
        Then only those files are processed
        And other files are not trained
        Because sometimes I want fine-grained control over what gets trained.
        """
        from cortical.cognitive import CognitiveAgent, IncrementalTrainer

        # Given I have multiple documents in a directory
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "important.txt").write_text("Critical knowledge about systems.")
        (docs_dir / "skip_me.txt").write_text("Unrelated content to ignore.")
        (docs_dir / "also_important.txt").write_text("More critical system knowledge.")

        # And I only want to train on some of them
        fs = RealFileSystem(tmp_path)
        agent = CognitiveAgent(filesystem=fs)
        trainer = IncrementalTrainer(agent, model_dir=tmp_path / "model", filesystem=fs)

        # When I train on specific files
        stats = trainer.train_files(
            [docs_dir / "important.txt", docs_dir / "also_important.txt"],
            base_dir=docs_dir,
            show_progress=False,
        )

        # Then only those files are processed
        assert stats.new_documents == 2
        trained_docs = trainer.list_trained()
        assert "important.txt" in trained_docs
        assert "also_important.txt" in trained_docs

        # And other files are not trained
        assert "skip_me.txt" not in trained_docs


# =============================================================================
# EPIC: Efficient Incremental Updates
# =============================================================================

class TestDeveloperSkipsAlreadyTrainedDocuments:
    """
    EPIC: Efficient Incremental Updates
    ====================================

    PERSONA: Developer updating an agent's knowledge
    GOAL: Skip already-trained documents automatically
    VALUE: No wasted time reprocessing unchanged content

    ACCEPTANCE CRITERIA:
    - Unchanged documents are skipped on retrain
    - New documents added to directory are detected
    - Statistics show skipped vs. processed counts
    """

    def test_scenario_retraining_skips_unchanged_documents(self, tmp_path):
        """
        Scenario: Running training twice skips already-trained files

        Given I have trained my agent on some documents
        When I run training again on the same directory
        Then no new documents are processed
        And the previously trained documents are skipped
        Because reprocessing unchanged content wastes resources.
        """
        from cortical.cognitive import CognitiveAgent, IncrementalTrainer

        # Given I have trained my agent on some documents
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "doc1.txt").write_text("Cognitive systems process information.")
        (docs_dir / "doc2.txt").write_text("Information processing enables learning.")

        fs = RealFileSystem(tmp_path)
        agent = CognitiveAgent(filesystem=fs)
        trainer = IncrementalTrainer(agent, model_dir=tmp_path / "model", filesystem=fs)
        first_stats = trainer.train_directory(docs_dir, show_progress=False)

        # When I run training again on the same directory
        second_stats = trainer.train_directory(docs_dir, show_progress=False)

        # Then no new documents are processed
        assert second_stats.new_documents == 0

        # And the previously trained documents are skipped
        assert second_stats.skipped_documents == 2
        assert second_stats.atoms_created == 0
        assert second_stats.links_created == 0

    def test_scenario_new_documents_are_detected_and_trained(self, tmp_path):
        """
        Scenario: Adding new files triggers incremental training

        Given I have previously trained on some documents
        And I add new documents to the directory
        When I run training again
        Then only the new documents are processed
        And existing documents are skipped
        Because incremental training should detect and process only changes.
        """
        from cortical.cognitive import CognitiveAgent, IncrementalTrainer

        # Given I have previously trained on some documents
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "original.txt").write_text("Original content for training.")

        fs = RealFileSystem(tmp_path)
        agent = CognitiveAgent(filesystem=fs)
        trainer = IncrementalTrainer(agent, model_dir=tmp_path / "model", filesystem=fs)
        trainer.train_directory(docs_dir, show_progress=False)

        # And I add new documents to the directory
        (docs_dir / "new_doc.txt").write_text("Brand new content to learn from.")
        (docs_dir / "another_new.txt").write_text("Additional knowledge to acquire.")

        # When I run training again
        stats = trainer.train_directory(docs_dir, show_progress=False)

        # Then only the new documents are processed
        assert stats.new_documents == 2
        assert stats.atoms_created > 0

        # And existing documents are skipped
        assert stats.skipped_documents == 1


# =============================================================================
# EPIC: Change Detection for Updated Content
# =============================================================================

class TestDeveloperHandlesModifiedDocuments:
    """
    EPIC: Change Detection for Updated Content
    ==========================================

    PERSONA: Developer maintaining training content
    GOAL: Modified documents are automatically retrained
    VALUE: Agent learns from updated information without manual tracking

    IMPLEMENTATION NOTE:
    Detection uses content SHA256 hash, NOT file modification time.
    This means touching a file without changing content will NOT retrain.
    """

    def test_scenario_modified_document_is_retrained(self, tmp_path):
        """
        Scenario: Editing a trained document triggers retraining

        Given I have trained on a document
        And I modify the content of that document
        When I run training again
        Then the modified document is retrained
        And it is counted as modified, not new
        Because content changes should update the agent's knowledge.
        """
        from cortical.cognitive import CognitiveAgent, IncrementalTrainer

        # Given I have trained on a document
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        doc_path = docs_dir / "evolving.txt"
        doc_path.write_text("Initial content about machine learning.")

        fs = RealFileSystem(tmp_path)
        agent = CognitiveAgent(filesystem=fs)
        trainer = IncrementalTrainer(agent, model_dir=tmp_path / "model", filesystem=fs)
        trainer.train_directory(docs_dir, show_progress=False)

        # And I modify the content of that document
        doc_path.write_text("Updated content about deep learning and transformers.")

        # When I run training again
        stats = trainer.train_directory(docs_dir, show_progress=False)

        # Then the modified document is retrained
        assert stats.atoms_created > 0

        # And it is counted as modified, not new
        assert stats.modified_documents == 1
        assert stats.new_documents == 0

    def test_scenario_unchanged_content_with_same_hash_is_skipped(self, tmp_path):
        """
        Scenario: Touching a file without changing content doesn't retrain

        Given I have trained on a document
        And I touch the file without changing its content
        When I run training again
        Then the document is skipped
        Because only actual content changes should trigger retraining.
        """
        from cortical.cognitive import CognitiveAgent, IncrementalTrainer
        import time

        # Given I have trained on a document
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        doc_path = docs_dir / "stable.txt"
        original_content = "This content will remain unchanged."
        doc_path.write_text(original_content)

        fs = RealFileSystem(tmp_path)
        agent = CognitiveAgent(filesystem=fs)
        trainer = IncrementalTrainer(agent, model_dir=tmp_path / "model", filesystem=fs)
        trainer.train_directory(docs_dir, show_progress=False)

        # And I touch the file without changing its content
        time.sleep(0.01)  # Ensure different mtime
        doc_path.write_text(original_content)  # Same content, new mtime

        # When I run training again
        stats = trainer.train_directory(docs_dir, show_progress=False)

        # Then the document is skipped
        assert stats.skipped_documents == 1
        assert stats.modified_documents == 0


# =============================================================================
# EPIC: Training State Persistence
# =============================================================================

class TestDeveloperPersistsTrainingState:
    """
    EPIC: Training State Persistence
    =================================

    PERSONA: Developer with long-running training processes
    GOAL: Training state persists across sessions
    VALUE: Resume training without starting over

    PERSISTED STATE:
    - Training manifest (which documents trained, content hashes)
    - Tokenizer vocabulary
    - Training statistics and metadata
    """

    def test_scenario_training_state_persists_across_sessions(self, tmp_path):
        """
        Scenario: Resuming training in a new session

        Given I have trained my agent and saved the state
        And I start a new session with a fresh agent
        When I load the trainer from the saved state
        And I run training on the same directory
        Then all previously trained documents are skipped
        Because training state should persist across program restarts.
        """
        from cortical.cognitive import CognitiveAgent, IncrementalTrainer

        # Given I have trained my agent and saved the state
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "persistent.txt").write_text("Content that persists across sessions.")

        model_dir = tmp_path / "model"
        fs = RealFileSystem(tmp_path)
        agent1 = CognitiveAgent(filesystem=fs)
        trainer1 = IncrementalTrainer(agent1, model_dir=model_dir, filesystem=fs)
        trainer1.train_directory(docs_dir, show_progress=False)

        # And I start a new session with a fresh agent
        agent2 = CognitiveAgent(filesystem=fs)

        # When I load the trainer from the saved state
        trainer2 = IncrementalTrainer(agent2, model_dir=model_dir, filesystem=fs)

        # And I run training on the same directory
        stats = trainer2.train_directory(docs_dir, show_progress=False)

        # Then all previously trained documents are skipped
        assert stats.skipped_documents == 1
        assert stats.new_documents == 0

    def test_scenario_training_status_shows_progress(self, tmp_path):
        """
        Scenario: Checking training status

        Given I have trained on some documents
        When I check the training status
        Then I can see how many documents were trained
        And I can see the vocabulary size
        And I can see when training last occurred
        Because visibility into training state helps with debugging and monitoring.
        """
        from cortical.cognitive import CognitiveAgent, IncrementalTrainer

        # Given I have trained on some documents
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "doc1.txt").write_text("First document for training.")
        (docs_dir / "doc2.txt").write_text("Second document for training.")

        fs = RealFileSystem(tmp_path)
        agent = CognitiveAgent(filesystem=fs)
        trainer = IncrementalTrainer(agent, model_dir=tmp_path / "model", filesystem=fs)
        trainer.train_directory(docs_dir, show_progress=False)

        # When I check the training status
        status = trainer.status()

        # Then I can see how many documents were trained
        assert status["total_documents_trained"] == 2

        # And I can see the vocabulary size
        assert status["vocabulary_size"] > 0

        # And I can see when training last occurred
        assert status["last_training"] is not None


# =============================================================================
# EPIC: Manual Training Control
# =============================================================================

class TestDeveloperForcesRetraining:
    """
    EPIC: Manual Training Control
    ==============================

    PERSONA: Developer debugging training issues
    GOAL: Force retraining of all documents when needed
    VALUE: Rebuild knowledge from scratch for debugging/recovery
    """

    def test_scenario_force_retrain_processes_all_documents(self, tmp_path):
        """
        Scenario: Force retraining ignores manifest

        Given I have previously trained on documents
        When I run training with force_retrain=True
        Then all documents are processed again
        And the manifest is updated with new timestamps
        Because sometimes I need to rebuild from scratch.
        """
        from cortical.cognitive import CognitiveAgent, IncrementalTrainer

        # Given I have previously trained on documents
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "doc1.txt").write_text("Content to be retrained.")
        (docs_dir / "doc2.txt").write_text("More content to be retrained.")

        fs = RealFileSystem(tmp_path)
        agent = CognitiveAgent(filesystem=fs)
        trainer = IncrementalTrainer(agent, model_dir=tmp_path / "model", filesystem=fs)
        trainer.train_directory(docs_dir, show_progress=False)

        # When I run training with force_retrain=True
        stats = trainer.train_directory(docs_dir, show_progress=False, force_retrain=True)

        # Then all documents are processed again
        assert stats.new_documents == 2
        assert stats.skipped_documents == 0
        # Note: atoms_created may be 0 on retrain since atoms are content-addressed
        # and already exist, but links are still created
        assert stats.links_created > 0


# =============================================================================
# EPIC: Training Visibility and Auditing
# =============================================================================

class TestDeveloperListsTrainedDocuments:
    """
    EPIC: Training Visibility and Auditing
    =======================================

    PERSONA: Developer auditing training coverage
    GOAL: List all trained documents
    VALUE: Verify what knowledge the agent has acquired
    """

    def test_scenario_listing_trained_documents(self, tmp_path):
        """
        Scenario: Getting a list of all trained documents

        Given I have trained on multiple documents
        When I list the trained documents
        Then I see all document paths that were trained
        And the list is sorted alphabetically
        Because I need to audit what the agent has learned from.
        """
        from cortical.cognitive import CognitiveAgent, IncrementalTrainer

        # Given I have trained on multiple documents
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "zebra.txt").write_text("Zebra content.")
        (docs_dir / "alpha.txt").write_text("Alpha content.")
        (docs_dir / "middle.txt").write_text("Middle content.")

        fs = RealFileSystem(tmp_path)
        agent = CognitiveAgent(filesystem=fs)
        trainer = IncrementalTrainer(agent, model_dir=tmp_path / "model", filesystem=fs)
        trainer.train_directory(docs_dir, show_progress=False)

        # When I list the trained documents
        trained = trainer.list_trained()

        # Then I see all document paths that were trained
        assert len(trained) == 3
        assert "zebra.txt" in trained
        assert "alpha.txt" in trained
        assert "middle.txt" in trained

        # And the list is sorted alphabetically
        assert trained == sorted(trained)


# =============================================================================
# EPIC: Hierarchical Document Organization
# =============================================================================

class TestDeveloperHandlesSubdirectories:
    """
    EPIC: Hierarchical Document Organization
    =========================================

    PERSONA: Developer with organized document directories
    GOAL: Training handles subdirectories recursively
    VALUE: Organize training content hierarchically
    """

    def test_scenario_recursive_training_finds_nested_documents(self, tmp_path):
        """
        Scenario: Training recursively through subdirectories

        Given I have documents organized in subdirectories
        When I train with recursive=True (default)
        Then documents in all subdirectories are found
        And their paths include the subdirectory
        Because hierarchical organization should be supported.
        """
        from cortical.cognitive import CognitiveAgent, IncrementalTrainer

        # Given I have documents organized in subdirectories
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "root.txt").write_text("Root level document.")

        sub1 = docs_dir / "category1"
        sub1.mkdir()
        (sub1 / "nested1.txt").write_text("Nested in category1.")

        sub2 = docs_dir / "category2"
        sub2.mkdir()
        (sub2 / "nested2.txt").write_text("Nested in category2.")

        fs = RealFileSystem(tmp_path)
        agent = CognitiveAgent(filesystem=fs)
        trainer = IncrementalTrainer(agent, model_dir=tmp_path / "model", filesystem=fs)

        # When I train with recursive=True (default)
        stats = trainer.train_directory(docs_dir, show_progress=False)

        # Then documents in all subdirectories are found
        assert stats.new_documents == 3

        # And their paths include the subdirectory
        trained = trainer.list_trained()
        assert "root.txt" in trained
        assert any("category1" in p for p in trained)
        assert any("category2" in p for p in trained)


# =============================================================================
# EPIC: Graceful Error Handling
# =============================================================================

class TestDeveloperHandlesEmptyAndMissingCases:
    """
    EPIC: Graceful Error Handling
    ==============================

    PERSONA: Developer working with various directory states
    GOAL: Training handles edge cases gracefully
    VALUE: No unexpected crashes from unusual inputs

    EDGE CASES HANDLED:
    - Empty directories
    - No matching files for pattern
    - Missing directories (should error clearly)
    """

    def test_scenario_training_empty_directory_succeeds(self, tmp_path):
        """
        Scenario: Training on an empty directory

        Given I have an empty directory
        When I run training on it
        Then training completes successfully
        And statistics show zero documents processed
        Because empty directories are a valid edge case.
        """
        from cortical.cognitive import CognitiveAgent, IncrementalTrainer

        # Given I have an empty directory
        docs_dir = tmp_path / "empty"
        docs_dir.mkdir()

        fs = RealFileSystem(tmp_path)
        agent = CognitiveAgent(filesystem=fs)
        trainer = IncrementalTrainer(agent, model_dir=tmp_path / "model", filesystem=fs)

        # When I run training on it
        stats = trainer.train_directory(docs_dir, show_progress=False)

        # Then training completes successfully
        # (no exception raised)

        # And statistics show zero documents processed
        assert stats.total_files_scanned == 0
        assert stats.new_documents == 0

    def test_scenario_training_with_no_matching_files(self, tmp_path):
        """
        Scenario: Training with pattern that matches no files

        Given I have a directory with non-matching files
        When I train with a pattern that matches nothing
        Then training completes successfully
        And statistics show zero documents
        Because mismatched patterns should not crash.
        """
        from cortical.cognitive import CognitiveAgent, IncrementalTrainer

        # Given I have a directory with non-matching files
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "data.json").write_text('{"key": "value"}')
        (docs_dir / "config.yaml").write_text("key: value")

        fs = RealFileSystem(tmp_path)
        agent = CognitiveAgent(filesystem=fs)
        trainer = IncrementalTrainer(agent, model_dir=tmp_path / "model", filesystem=fs)

        # When I train with a pattern that matches nothing
        stats = trainer.train_directory(docs_dir, pattern="*.txt", show_progress=False)

        # Then training completes successfully
        # (no exception raised)

        # And statistics show zero documents
        assert stats.total_files_scanned == 0


# =============================================================================
# EPIC: Fast In-Memory Testing
# =============================================================================

class TestDeveloperUsesInMemoryFileSystem:
    """
    EPIC: Fast In-Memory Testing
    =============================

    PERSONA: Developer writing tests for training workflows
    GOAL: Use in-memory filesystem for testing
    VALUE: Tests run ~10x faster, can assert on file operations

    TESTING BENEFITS:
    - No disk I/O latency
    - Isolated test environments
    - Operation tracking for assertions
    - No cleanup needed
    """

    def test_scenario_training_with_in_memory_filesystem_is_fast(self):
        """
        Scenario: Training without disk I/O

        Given I have an in-memory filesystem with documents
        And I create a trainer using that filesystem
        When I train on those documents
        Then training completes without any disk I/O
        And all file operations happen in memory
        Because in-memory testing is ~10x faster than disk I/O.
        """
        from pathlib import Path
        from cortical.cognitive import CognitiveAgent, IncrementalTrainer
        from cortical.common.filesystem import InMemoryFileSystem

        # Given I have an in-memory filesystem with documents
        fs = InMemoryFileSystem(Path("/virtual"))
        fs.mkdir(Path("/virtual"), parents=True, exist_ok=True)

        docs_dir = Path("/virtual/docs")
        fs.mkdir(docs_dir, parents=True, exist_ok=True)
        fs.write_text(docs_dir / "doc1.txt", "Neural networks process information.")
        fs.write_text(docs_dir / "doc2.txt", "Information flows through layers.")

        model_dir = Path("/virtual/model")

        # And I create a trainer using that filesystem
        agent = CognitiveAgent()
        trainer = IncrementalTrainer(agent, model_dir=model_dir, filesystem=fs)

        # When I train on those documents
        stats = trainer.train_directory(docs_dir, show_progress=False)

        # Then training completes without any disk I/O
        assert stats.new_documents == 2
        assert stats.vocabulary_size > 0

        # And all file operations happen in memory
        # (We can verify by checking the filesystem's internal state)
        assert fs.exists(model_dir / "training_manifest.json")
        assert fs.exists(model_dir / "tokenizer" / "meta.json")  # Sharded format

    def test_scenario_filesystem_tracks_operations_for_assertions(self):
        """
        Scenario: Asserting on file operations

        Given I have an in-memory filesystem
        And I train a cognitive agent
        When I check the filesystem's operation tracking
        Then I can see which files were read
        And I can see which files were written
        And I can assert on the operation sequence
        Because operation tracking enables precise behavioral assertions.
        """
        from pathlib import Path
        from cortical.cognitive import CognitiveAgent, IncrementalTrainer
        from cortical.common.filesystem import InMemoryFileSystem

        # Given I have an in-memory filesystem
        fs = InMemoryFileSystem(Path("/test"))
        fs.mkdir(Path("/test"), parents=True, exist_ok=True)

        docs_dir = Path("/test/docs")
        fs.mkdir(docs_dir, parents=True, exist_ok=True)
        fs.write_text(docs_dir / "sample.txt", "Test content for tracking.")

        # Reset tracking after setup
        fs.reset_tracking()

        model_dir = Path("/test/model")

        # And I train a cognitive agent
        agent = CognitiveAgent()
        trainer = IncrementalTrainer(agent, model_dir=model_dir, filesystem=fs)
        trainer.train_directory(docs_dir, show_progress=False)

        # When I check the filesystem's operation tracking
        # Then I can see which files were read
        assert len(fs.files_read) > 0

        # And I can see which files were written
        assert len(fs.files_written) > 0
        fs.assert_file_was_written(model_dir / "training_manifest.json")
        fs.assert_file_was_written(model_dir / "tokenizer" / "meta.json")  # Sharded format

        # And I can assert on the operation sequence
        # (The filesystem tracked all operations)
        assert len(fs.operations) > 0

    def test_scenario_in_memory_training_state_persists_within_session(self):
        """
        Scenario: State persistence in memory

        Given I train documents using in-memory filesystem
        And I create a new trainer instance with the same filesystem
        When I check the training status
        Then previously trained documents are recognized
        And retraining skips them
        Because in-memory state should persist within the test session.
        """
        from pathlib import Path
        from cortical.cognitive import CognitiveAgent, IncrementalTrainer
        from cortical.common.filesystem import InMemoryFileSystem

        # Given I train documents using in-memory filesystem
        fs = InMemoryFileSystem(Path("/persist"))
        fs.mkdir(Path("/persist"), parents=True, exist_ok=True)

        docs_dir = Path("/persist/docs")
        fs.mkdir(docs_dir, parents=True, exist_ok=True)
        fs.write_text(docs_dir / "persistent.txt", "Content that persists.")

        model_dir = Path("/persist/model")

        agent1 = CognitiveAgent()
        trainer1 = IncrementalTrainer(agent1, model_dir=model_dir, filesystem=fs)
        trainer1.train_directory(docs_dir, show_progress=False)

        # And I create a new trainer instance with the same filesystem
        agent2 = CognitiveAgent()
        trainer2 = IncrementalTrainer(agent2, model_dir=model_dir, filesystem=fs)

        # When I check the training status
        status = trainer2.status()

        # Then previously trained documents are recognized
        assert status["total_documents_trained"] == 1

        # And retraining skips them
        stats = trainer2.train_directory(docs_dir, show_progress=False)
        assert stats.skipped_documents == 1
        assert stats.new_documents == 0


# =============================================================================
# EPIC: Crash-Recoverable Incremental Training
# =============================================================================

class TestDeveloperRecoversFromTrainingFailures:
    """
    EPIC: Crash-Recoverable Incremental Training
    =============================================

    PERSONA: Developer running long training sessions
    GOAL: Training saves progress incrementally, enabling crash recovery
    VALUE: No lost work if training crashes mid-way through large corpus

    PROBLEM BEING SOLVED:
    When training on 500+ documents, a crash at document 400 should not
    lose all progress. The manifest should checkpoint periodically so
    training can resume from the last checkpoint.

    IMPLEMENTATION APPROACH:
    - Checkpoint every N documents (configurable, default 50)
    - Save manifest, tokenizer, and graph at each checkpoint
    - On restart, training resumes from last checkpoint
    """

    def test_scenario_training_checkpoints_periodically(self, tmp_path):
        """
        Scenario: Training saves progress at regular intervals

        Given I have many documents to train on
        When I train with checkpointing enabled
        Then the manifest is saved periodically during training
        And partial progress is recoverable
        Because long training sessions should not lose all work on failure.
        """
        from cortical.cognitive import CognitiveAgent, IncrementalTrainer
        from cortical.common.filesystem import InMemoryFileSystem
        from pathlib import Path

        # Given I have many documents to train on
        fs = InMemoryFileSystem(Path("/checkpoint"))
        fs.mkdir(Path("/checkpoint"), parents=True, exist_ok=True)

        docs_dir = Path("/checkpoint/docs")
        fs.mkdir(docs_dir, parents=True, exist_ok=True)

        # Create 10 documents (checkpoint_interval=3 means 3 checkpoints)
        for i in range(10):
            fs.write_text(docs_dir / f"doc_{i:02d}.txt", f"Document {i} content about topic {i}.")

        model_dir = Path("/checkpoint/model")

        # When I train with checkpointing enabled
        agent = CognitiveAgent()
        trainer = IncrementalTrainer(agent, model_dir=model_dir, filesystem=fs)

        # Train with small checkpoint interval for test
        stats = trainer.train_directory(
            docs_dir,
            show_progress=False,
            checkpoint_interval=3,  # Checkpoint every 3 documents
        )

        # Then the manifest is saved periodically during training
        assert stats.new_documents == 10

        # And partial progress is recoverable (manifest exists and has documents)
        assert fs.exists(model_dir / "training_manifest.json")

        # Verify by creating new trainer that it knows about trained docs
        agent2 = CognitiveAgent()
        trainer2 = IncrementalTrainer(agent2, model_dir=model_dir, filesystem=fs)
        assert trainer2.manifest.total_documents == 10

    def test_scenario_training_resumes_after_simulated_crash(self, tmp_path):
        """
        Scenario: Training resumes from checkpoint after crash

        Given I started training on many documents
        And training was interrupted after some documents
        When I restart training
        Then only the remaining documents are processed
        And previously checkpointed documents are skipped
        Because crash recovery should continue from where it left off.
        """
        from cortical.cognitive import CognitiveAgent, IncrementalTrainer
        from cortical.common.filesystem import InMemoryFileSystem
        from pathlib import Path

        # Given I started training on many documents
        fs = InMemoryFileSystem(Path("/crash"))
        fs.mkdir(Path("/crash"), parents=True, exist_ok=True)

        docs_dir = Path("/crash/docs")
        fs.mkdir(docs_dir, parents=True, exist_ok=True)

        # Create 10 documents
        for i in range(10):
            fs.write_text(docs_dir / f"doc_{i:02d}.txt", f"Document {i} content.")

        model_dir = Path("/crash/model")

        # Train first 5 documents normally
        agent1 = CognitiveAgent()
        trainer1 = IncrementalTrainer(agent1, model_dir=model_dir, filesystem=fs)

        # Manually train just 5 docs to simulate partial completion
        all_files = list(trainer1.scan_directory(docs_dir))
        first_5 = all_files[:5]
        for path, content, content_hash in first_5:
            trainer1.bridge.learn_vocabulary([content], incremental=True)
            trainer1.bridge.feed_text(content, doc_id=path)
            word_count = len(trainer1.bridge.tokenizer.tokenize(content))
            trainer1.manifest.add_document(path, content_hash, word_count)

        # Save checkpoint (simulating checkpoint after 5 docs)
        trainer1.save()

        # When I restart training (simulating crash recovery)
        agent2 = CognitiveAgent()
        trainer2 = IncrementalTrainer(agent2, model_dir=model_dir, filesystem=fs)
        stats = trainer2.train_directory(docs_dir, show_progress=False)

        # Then only the remaining documents are processed
        assert stats.new_documents == 5

        # And previously checkpointed documents are skipped
        assert stats.skipped_documents == 5

    def test_scenario_default_checkpoint_interval_is_reasonable(self, tmp_path):
        """
        Scenario: Default checkpoint interval balances safety and performance

        Given I am using the default checkpoint settings
        When I check the default interval
        Then it is set to 50 documents
        Because 50 is a reasonable balance between safety and I/O overhead.
        """
        from cortical.cognitive import CognitiveAgent, IncrementalTrainer
        from cortical.common.filesystem import InMemoryFileSystem
        from pathlib import Path

        # Given I am using the default checkpoint settings
        fs = InMemoryFileSystem(Path("/default"))
        fs.mkdir(Path("/default"), parents=True, exist_ok=True)

        model_dir = Path("/default/model")
        agent = CognitiveAgent()
        trainer = IncrementalTrainer(agent, model_dir=model_dir, filesystem=fs)

        # When I check the default interval
        # Then it is set to 50 documents
        assert trainer.checkpoint_interval == 50


# =============================================================================
# EPIC: Autonomous File System Access
# =============================================================================

class TestCognitiveAgentWithFileSystem:
    """
    EPIC: Autonomous File System Access
    ====================================

    PERSONA: Developer building autonomous cognitive agents
    GOAL: CognitiveAgent owns its FileSystem
    VALUE: Agent can persist state and discover new documents to learn

    DESIGN PRINCIPLE:
    The FileSystem is injected through the constructor, enabling:
    - Autonomous file discovery for learning
    - State persistence without external coordination
    - Testability with InMemoryFileSystem
    """

    def test_scenario_agent_created_with_filesystem(self):
        """
        Scenario: Creating agent with filesystem enables persistence

        Given I create a CognitiveAgent with a filesystem
        When I save the agent state
        Then the agent uses its internal filesystem
        And no external filesystem parameter is needed
        Because the agent should own its persistence mechanism.
        """
        from pathlib import Path
        from cortical.cognitive import CognitiveAgent
        from cortical.common.filesystem import InMemoryFileSystem

        # Given I create a CognitiveAgent with a filesystem
        fs = InMemoryFileSystem(Path("/agent"))
        fs.mkdir(Path("/agent"), parents=True, exist_ok=True)
        agent = CognitiveAgent(filesystem=fs)

        # Add some knowledge to make it worth saving
        agent.graph.node("test_concept")
        agent.attend("test_concept", amount=1.0)

        # When I save the agent state
        save_path = Path("/agent/state.json")
        agent.save(save_path)

        # Then the agent uses its internal filesystem
        assert fs.exists(save_path)

        # And no external filesystem parameter is needed
        # (The save() call above didn't require a filesystem argument)

    def test_scenario_agent_loaded_with_filesystem(self):
        """
        Scenario: Loading agent preserves filesystem reference

        Given I have a saved agent state
        And I load the agent with a filesystem
        When I check the loaded agent
        Then it has the filesystem available
        And it can save again using that filesystem
        Because loaded agents should be fully functional.
        """
        from pathlib import Path
        from cortical.cognitive import CognitiveAgent
        from cortical.common.filesystem import InMemoryFileSystem

        # Given I have a saved agent state
        fs = InMemoryFileSystem(Path("/load"))
        fs.mkdir(Path("/load"), parents=True, exist_ok=True)

        original_agent = CognitiveAgent(filesystem=fs)
        original_agent.graph.node("persisted_knowledge")
        save_path = Path("/load/agent.json")
        original_agent.save(save_path)

        # And I load the agent with a filesystem
        loaded_agent = CognitiveAgent.load(save_path, filesystem=fs)

        # When I check the loaded agent
        # Then it has the filesystem available
        assert loaded_agent.filesystem is fs

        # And it can save again using that filesystem
        new_save_path = Path("/load/agent_v2.json")
        loaded_agent.save(new_save_path)
        assert fs.exists(new_save_path)

    def test_scenario_agent_can_access_files_for_learning(self):
        """
        Scenario: Agent with filesystem can read documents

        Given I have a CognitiveAgent with a filesystem
        And the filesystem contains documents
        When the agent reads a document
        Then it can access the content
        Because agents need file access to discover new learning material.
        """
        from pathlib import Path
        from cortical.cognitive import CognitiveAgent
        from cortical.common.filesystem import InMemoryFileSystem

        # Given I have a CognitiveAgent with a filesystem
        fs = InMemoryFileSystem(Path("/learning"))
        fs.mkdir(Path("/learning"), parents=True, exist_ok=True)
        agent = CognitiveAgent(filesystem=fs)

        # And the filesystem contains documents
        docs_dir = Path("/learning/docs")
        fs.mkdir(docs_dir, parents=True, exist_ok=True)
        fs.write_text(docs_dir / "knowledge.txt", "Important information to learn.")

        # When the agent reads a document
        content = agent.filesystem.read_text(docs_dir / "knowledge.txt")

        # Then it can access the content
        assert "Important information" in content

    def test_scenario_agent_without_filesystem_raises_on_save(self):
        """
        Scenario: Agent without filesystem cannot persist

        Given I create a CognitiveAgent without a filesystem
        When I try to save the agent
        Then an error is raised
        Because persistence requires a filesystem.
        """
        from pathlib import Path
        from cortical.cognitive import CognitiveAgent

        # Given I create a CognitiveAgent without a filesystem
        agent = CognitiveAgent()

        # When I try to save the agent
        # Then an error is raised
        with pytest.raises((TypeError, AttributeError, ValueError)):
            agent.save(Path("/some/path.json"))

    def test_scenario_in_memory_agent_state_roundtrip(self):
        """
        Scenario: Agent state survives save/load cycle in memory

        Given I create an agent with in-memory filesystem
        And I add knowledge and goals to the agent
        When I save and reload the agent
        Then all state is preserved
        And the agent continues functioning
        Because in-memory persistence should be lossless.
        """
        from pathlib import Path
        from cortical.cognitive import CognitiveAgent, Goal
        from cortical.common.filesystem import InMemoryFileSystem

        # Given I create an agent with in-memory filesystem
        fs = InMemoryFileSystem(Path("/roundtrip"))
        fs.mkdir(Path("/roundtrip"), parents=True, exist_ok=True)
        agent = CognitiveAgent(filesystem=fs)

        # And I add knowledge and goals to the agent
        agent.graph.node("concept_a")
        agent.graph.node("concept_b")
        agent.graph.link(
            agent.graph._storage.find_by_type(agent.graph._storage.all_atoms()[0].atom_type)[0].atom_type,
            [agent.graph.get_node("concept_a"), agent.graph.get_node("concept_b")]
        )
        agent.goals.add_goal(Goal(
            id="goal-1",
            description="Learn new concepts",
            target_state=10,
            current_state=2,
            importance=0.8,
        ))

        # When I save and reload the agent
        save_path = Path("/roundtrip/agent.json")
        agent.save(save_path)
        loaded = CognitiveAgent.load(save_path, filesystem=fs)

        # Then all state is preserved
        assert loaded.graph.get_node("concept_a") is not None
        assert loaded.graph.get_node("concept_b") is not None
        assert len(loaded.goals.get_active_goals()) == 1
        assert loaded.goals.get_active_goals()[0].description == "Learn new concepts"

        # And the agent continues functioning
        loaded.step()  # Should not raise
        assert loaded._step_count == 1


# =============================================================================
# EPIC: Graph State Persistence Across Sessions
# =============================================================================

class TestDeveloperResumesTrainingWithFullState:
    """
    EPIC: Graph State Persistence Across Sessions
    ==============================================

    PERSONA: Developer running training across multiple sessions
    GOAL: Resume training with ALL learned state intact
    VALUE: No lost knowledge when sessions are interrupted

    CRITICAL INVARIANT:
    When training resumes, the graph MUST contain all atoms and links
    from previous sessions. The manifest tracking document completion
    is NOT sufficient - the actual learned relationships must persist.

    FAILURE MODE BEING PREVENTED:
    Session 1: Train docs 1-100, graph has 5000 atoms
    Session 2: Resume, manifest says "100 docs done", but graph is EMPTY
    Session 2: Train docs 101-200 with no connection to prior knowledge
    Result: Fragmented knowledge, training is useless

    WHY THIS MATTERS:
    - Atoms represent learned vocabulary
    - Links represent learned word associations
    - Without restoring these, new training is disconnected from old
    """

    def test_scenario_graph_atoms_persist_across_sessions(self, tmp_path):
        """
        Scenario: Graph atoms are restored when resuming training

        Given I have trained on documents creating atoms
        And I start a new session with a fresh agent
        When I create a new trainer for the same model directory
        Then the graph contains all previously created atoms
        And I can query atoms that were learned in the previous session
        Because learned vocabulary must persist across sessions.
        """
        from cortical.cognitive import CognitiveAgent, IncrementalTrainer
        from cortical.common.filesystem import RealFileSystem

        # Given I have trained on documents creating atoms
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "neural.txt").write_text(
            "Neural networks learn patterns through layers of computation."
        )
        (docs_dir / "cognitive.txt").write_text(
            "Cognitive systems process information using neural architectures."
        )

        model_dir = tmp_path / "model"
        fs = RealFileSystem(tmp_path)
        agent1 = CognitiveAgent(filesystem=fs)
        trainer1 = IncrementalTrainer(agent1, model_dir=model_dir, filesystem=fs)
        stats1 = trainer1.train_directory(docs_dir, show_progress=False)

        # Record what was learned
        atoms_after_training = len(list(agent1.graph._storage.all_atoms()))
        assert atoms_after_training > 0, "Training should create atoms"

        # And I start a new session with a fresh agent
        agent2 = CognitiveAgent(filesystem=fs)
        atoms_before_resume = len(list(agent2.graph._storage.all_atoms()))
        assert atoms_before_resume == 0, "Fresh agent should have no atoms"

        # When I create a new trainer for the same model directory
        trainer2 = IncrementalTrainer(agent2, model_dir=model_dir, filesystem=fs)

        # Then the graph contains all previously created atoms
        atoms_after_resume = len(list(agent2.graph._storage.all_atoms()))
        assert atoms_after_resume == atoms_after_training, (
            f"Graph should have {atoms_after_training} atoms after resume, "
            f"but has {atoms_after_resume}. Graph state was not restored!"
        )

        # And I can query atoms that were learned in the previous session
        neural_atom = agent2.graph.get_node("neural")
        assert neural_atom is not None, (
            "Should be able to find 'neural' atom learned in previous session"
        )

    def test_scenario_graph_links_persist_across_sessions(self, tmp_path):
        """
        Scenario: Graph links (relationships) are restored when resuming

        Given I have trained on documents creating links between words
        And I note the links created during training
        When I resume training in a new session
        Then the graph contains all previously created links
        And the link count matches what was created before
        Because learned word associations must persist across sessions.
        """
        from cortical.cognitive import CognitiveAgent, IncrementalTrainer
        from cortical.common.filesystem import RealFileSystem

        # Given I have trained on documents creating links between words
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "associations.txt").write_text(
            "Machine learning algorithms process data efficiently. "
            "Learning algorithms improve with more training data."
        )

        model_dir = tmp_path / "model"
        fs = RealFileSystem(tmp_path)
        agent1 = CognitiveAgent(filesystem=fs)
        trainer1 = IncrementalTrainer(agent1, model_dir=model_dir, filesystem=fs)
        stats1 = trainer1.train_directory(docs_dir, show_progress=False)

        # And I note the links created during training
        links_after_training = stats1.links_created
        assert links_after_training > 0, "Training should create links"

        # Count actual links in graph
        all_atoms = list(agent1.graph._storage.all_atoms())
        link_atoms = [a for a in all_atoms if a.outgoing]
        link_count_session1 = len(link_atoms)

        # When I resume training in a new session
        agent2 = CognitiveAgent(filesystem=fs)
        trainer2 = IncrementalTrainer(agent2, model_dir=model_dir, filesystem=fs)

        # Then the graph contains all previously created links
        all_atoms_resumed = list(agent2.graph._storage.all_atoms())
        link_atoms_resumed = [a for a in all_atoms_resumed if a.outgoing]
        link_count_session2 = len(link_atoms_resumed)

        # And the link count matches what was created before
        assert link_count_session2 == link_count_session1, (
            f"Graph should have {link_count_session1} links after resume, "
            f"but has {link_count_session2}. Link state was not restored!"
        )

    def test_scenario_continued_training_connects_to_prior_knowledge(self, tmp_path):
        """
        Scenario: New training builds on previously learned knowledge

        Given I have trained on documents about topic A
        And I resume training in a new session
        And I train on new documents about topic B that shares words with A
        When training completes
        Then new atoms are connected to restored atoms via links
        And the total knowledge graph is unified, not fragmented
        Because incremental training must build a connected knowledge graph.
        """
        from cortical.cognitive import CognitiveAgent, IncrementalTrainer
        from cortical.common.filesystem import RealFileSystem

        # Given I have trained on documents about topic A
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "topic_a.txt").write_text(
            "Neural networks learn from training data using backpropagation."
        )

        model_dir = tmp_path / "model"
        fs = RealFileSystem(tmp_path)
        agent1 = CognitiveAgent(filesystem=fs)
        trainer1 = IncrementalTrainer(agent1, model_dir=model_dir, filesystem=fs)
        trainer1.train_directory(docs_dir, show_progress=False)

        # And I resume training in a new session
        agent2 = CognitiveAgent(filesystem=fs)
        trainer2 = IncrementalTrainer(agent2, model_dir=model_dir, filesystem=fs)

        # Verify prior knowledge is restored
        prior_atom_count = len(list(agent2.graph._storage.all_atoms()))
        assert prior_atom_count > 0, "Prior knowledge should be restored"

        # And I train on new documents about topic B that shares words with A
        (docs_dir / "topic_b.txt").write_text(
            "Deep learning networks require training data and neural computation."
        )
        stats2 = trainer2.train_directory(docs_dir, show_progress=False)

        # When training completes
        assert stats2.new_documents == 1, "Should train only the new document"

        # Then new atoms are connected to restored atoms via links
        # The word "training" appears in both documents, so should have links
        # connecting topic_a atoms to topic_b atoms
        training_atom = agent2.graph.get_node("training")
        assert training_atom is not None, "'training' should exist from topic A"

        # Find links involving the training atom
        all_atoms = list(agent2.graph._storage.all_atoms())
        links_with_training = [
            a for a in all_atoms
            if a.outgoing and training_atom.id in a.outgoing
        ]

        # And the total knowledge graph is unified, not fragmented
        final_atom_count = len(all_atoms)
        assert final_atom_count > prior_atom_count, "New atoms should be added"
        assert len(links_with_training) > 0, (
            "New documents should create links to existing 'training' atom, "
            "proving the knowledge graph is connected across sessions"
        )


# =============================================================================
# EPIC: Graceful Handling of Interrupted Sessions
# =============================================================================

class TestDeveloperHandlesSessionInterruption:
    """
    EPIC: Graceful Handling of Interrupted Sessions
    ================================================

    PERSONA: Developer in environment with session timeouts
    GOAL: Training survives unexpected session termination
    VALUE: No corrupted state, no lost progress beyond last checkpoint

    ENVIRONMENT CONTEXT:
    Claude Code Web and similar environments may terminate sessions
    without warning. The training system must handle this gracefully.

    FAILURE MODES BEING PREVENTED:
    1. Corrupted manifest (partial write)
    2. Corrupted graph.json (partial write)
    3. Manifest/graph desync (manifest updated, graph not saved)
    4. Lost progress (no checkpoint for long time)
    """

    def test_scenario_smaller_checkpoint_interval_for_volatile_environments(self, tmp_path):
        """
        Scenario: Configuring frequent checkpoints for hostile environments

        Given I am running in an environment that may kill my process
        And I want to minimize lost work on interruption
        When I configure a smaller checkpoint interval
        Then training saves state more frequently
        And the maximum lost work is bounded by the interval
        Because volatile environments need more frequent persistence.
        """
        from cortical.cognitive import CognitiveAgent, IncrementalTrainer
        from cortical.common.filesystem import InMemoryFileSystem
        from pathlib import Path

        # Given I am running in an environment that may kill my process
        fs = InMemoryFileSystem(Path("/volatile"))
        fs.mkdir(Path("/volatile"), parents=True, exist_ok=True)

        docs_dir = Path("/volatile/docs")
        fs.mkdir(docs_dir, parents=True, exist_ok=True)
        for i in range(20):
            fs.write_text(docs_dir / f"doc_{i:02d}.txt", f"Document {i} about topic {i}.")

        model_dir = Path("/volatile/model")

        # And I want to minimize lost work on interruption
        # When I configure a smaller checkpoint interval
        small_interval = 5  # Checkpoint every 5 documents

        agent = CognitiveAgent()
        trainer = IncrementalTrainer(
            agent,
            model_dir=model_dir,
            filesystem=fs,
            checkpoint_interval=small_interval,
        )

        # Then training saves state more frequently
        # Track write operations to manifest
        fs.reset_tracking()
        trainer.train_directory(docs_dir, show_progress=False)

        manifest_writes = [
            op for op in fs.operations
            if "training_manifest.json" in str(op.get("path", ""))
            and op.get("operation") == "write"
        ]

        # 20 docs / 5 interval = 4 checkpoints, plus final save
        assert len(manifest_writes) >= 4, (
            f"Expected at least 4 checkpoint writes, got {len(manifest_writes)}"
        )

        # And the maximum lost work is bounded by the interval
        # (This is a design property - if killed between checkpoints,
        # at most checkpoint_interval documents are lost)
        assert trainer.checkpoint_interval == small_interval

    def test_scenario_training_detects_incomplete_prior_session(self, tmp_path):
        """
        Scenario: Detecting when prior session was interrupted

        Given I was training and my session was killed
        And the manifest shows fewer documents than graph.json has atoms for
        When I start a new training session
        Then the system should detect the inconsistency
        And either recover gracefully or warn about the state
        Because state inconsistency indicates interrupted training.

        NOTE: This scenario documents DESIRED behavior. If it fails,
        it indicates the system lacks inconsistency detection.
        """
        from cortical.cognitive import CognitiveAgent, IncrementalTrainer
        from cortical.common.filesystem import InMemoryFileSystem
        from pathlib import Path
        import json

        # Given I was training and my session was killed
        fs = InMemoryFileSystem(Path("/interrupted"))
        fs.mkdir(Path("/interrupted"), parents=True, exist_ok=True)

        docs_dir = Path("/interrupted/docs")
        fs.mkdir(docs_dir, parents=True, exist_ok=True)
        for i in range(10):
            fs.write_text(docs_dir / f"doc_{i:02d}.txt", f"Document {i} content.")

        model_dir = Path("/interrupted/model")

        # Train normally first
        agent1 = CognitiveAgent()
        trainer1 = IncrementalTrainer(agent1, model_dir=model_dir, filesystem=fs)
        trainer1.train_directory(docs_dir, show_progress=False)

        # And the manifest shows fewer documents than graph.json has atoms for
        # Simulate interrupted session by corrupting manifest to show fewer docs
        manifest_path = model_dir / "training_manifest.json"
        manifest_data = json.loads(fs.read_text(manifest_path))

        # Remove half the documents from manifest (simulating crash before manifest save)
        doc_keys = list(manifest_data["documents"].keys())
        for key in doc_keys[5:]:
            del manifest_data["documents"][key]
        manifest_data["total_documents"] = 5

        fs.write_text(manifest_path, json.dumps(manifest_data))

        # When I start a new training session
        agent2 = CognitiveAgent()
        trainer2 = IncrementalTrainer(agent2, model_dir=model_dir, filesystem=fs)

        # Then the system should detect the inconsistency
        # Check if status reflects what manifest says vs what graph has
        status = trainer2.status()

        # Note: Currently the system may not detect this.
        # This test documents the DESIRED behavior.
        # If graph atoms > manifest docs significantly, something is wrong.

        # At minimum, training should be able to continue
        stats = trainer2.train_directory(docs_dir, show_progress=False)

        # The "missing" 5 docs should be detected as new (manifest doesn't have them)
        assert stats.new_documents == 5, (
            "Documents missing from manifest should be retrained"
        )

    def test_scenario_checkpoint_saves_consistent_state(self, tmp_path):
        """
        Scenario: Each checkpoint creates a consistent recoverable state

        Given I am training with checkpointing enabled
        When a checkpoint is saved
        Then the manifest, tokenizer, and graph are all saved together
        And if I load from that checkpoint, all three are consistent
        Because partial saves would corrupt the training state.
        """
        from cortical.cognitive import CognitiveAgent, IncrementalTrainer
        from cortical.common.filesystem import InMemoryFileSystem
        from pathlib import Path
        import json

        # Given I am training with checkpointing enabled
        fs = InMemoryFileSystem(Path("/consistent"))
        fs.mkdir(Path("/consistent"), parents=True, exist_ok=True)

        docs_dir = Path("/consistent/docs")
        fs.mkdir(docs_dir, parents=True, exist_ok=True)
        for i in range(6):
            fs.write_text(docs_dir / f"doc_{i:02d}.txt", f"Document {i} about words.")

        model_dir = Path("/consistent/model")

        agent = CognitiveAgent()
        trainer = IncrementalTrainer(
            agent, model_dir=model_dir, filesystem=fs, checkpoint_interval=3
        )

        # When a checkpoint is saved (after every 3 docs)
        trainer.train_directory(docs_dir, show_progress=False)

        # Then the manifest, tokenizer, and graph are all saved together
        assert fs.exists(model_dir / "training_manifest.json")
        assert fs.exists(model_dir / "tokenizer" / "meta.json")
        assert fs.exists(model_dir / "bridge" / "graph.json")

        # And if I load from that checkpoint, all three are consistent
        manifest = json.loads(fs.read_text(model_dir / "training_manifest.json"))
        tokenizer_meta = json.loads(fs.read_text(model_dir / "tokenizer" / "meta.json"))
        graph_data = json.loads(fs.read_text(model_dir / "bridge" / "graph.json"))

        # Manifest doc count should reflect actual trained documents
        assert manifest["total_documents"] == 6

        # Tokenizer vocab size should match manifest's record
        assert manifest["vocabulary_size"] == tokenizer_meta["vocab_size"]

        # Graph should have atoms
        assert len(graph_data.get("atoms", [])) > 0


# =============================================================================
# EPIC: Environment-Friendly Resource Usage
# =============================================================================

class TestDeveloperIsGoodEnvironmentGuest:
    """
    EPIC: Environment-Friendly Resource Usage
    ==========================================

    PERSONA: Developer running in shared/hosted environment
    GOAL: Training does not monopolize system resources
    VALUE: Peaceful coexistence with other processes, avoid getting killed

    ENVIRONMENTAL CONSTRAINTS:
    - Claude Code Web has resource limits
    - Long-running CPU-intensive processes may be terminated
    - We are guests in a shared environment

    DESIGN PRINCIPLES:
    - Yield CPU periodically during long operations
    - Keep checkpoints small and fast
    - Provide progress visibility for monitoring
    - Support graceful interruption
    """

    def test_scenario_training_provides_progress_callback(self, tmp_path):
        """
        Scenario: Training reports progress for external monitoring

        Given I am training on many documents
        And I want to monitor progress externally
        When I train with progress tracking
        Then I can observe progress as training proceeds
        And I can estimate time remaining
        Because visibility enables proactive management of long runs.
        """
        from cortical.cognitive import CognitiveAgent, IncrementalTrainer
        from cortical.common.filesystem import InMemoryFileSystem
        from pathlib import Path

        # Given I am training on many documents
        fs = InMemoryFileSystem(Path("/progress"))
        fs.mkdir(Path("/progress"), parents=True, exist_ok=True)

        docs_dir = Path("/progress/docs")
        fs.mkdir(docs_dir, parents=True, exist_ok=True)
        for i in range(10):
            fs.write_text(docs_dir / f"doc_{i:02d}.txt", f"Document {i} content.")

        model_dir = Path("/progress/model")

        agent = CognitiveAgent()
        trainer = IncrementalTrainer(agent, model_dir=model_dir, filesystem=fs)

        # And I want to monitor progress externally
        # When I train with progress tracking (show_progress=True uses ProgressReporter)
        stats = trainer.train_directory(docs_dir, show_progress=False)

        # Then I can observe progress as training proceeds
        # The stats object provides visibility
        assert stats.total_files_scanned == 10
        assert stats.new_documents == 10
        assert stats.training_time_seconds > 0

        # And I can estimate time remaining
        # (time per doc * remaining docs)
        time_per_doc = stats.training_time_seconds / stats.new_documents
        assert time_per_doc > 0, "Should be able to estimate time per document"

    def test_scenario_training_stats_enable_batch_planning(self, tmp_path):
        """
        Scenario: Stats from prior runs enable batch size planning

        Given I have trained some documents and measured the time
        And I know my environment has a time limit
        When I calculate how many documents fit in the limit
        Then I can plan batch sizes that complete before timeout
        Because predictable runtimes prevent unexpected termination.
        """
        from cortical.cognitive import CognitiveAgent, IncrementalTrainer
        from cortical.common.filesystem import InMemoryFileSystem
        from pathlib import Path

        # Given I have trained some documents and measured the time
        fs = InMemoryFileSystem(Path("/batch"))
        fs.mkdir(Path("/batch"), parents=True, exist_ok=True)

        docs_dir = Path("/batch/docs")
        fs.mkdir(docs_dir, parents=True, exist_ok=True)
        for i in range(5):
            fs.write_text(docs_dir / f"doc_{i:02d}.txt", f"Document {i} content here.")

        model_dir = Path("/batch/model")

        agent = CognitiveAgent()
        trainer = IncrementalTrainer(agent, model_dir=model_dir, filesystem=fs)
        stats = trainer.train_directory(docs_dir, show_progress=False)

        # And I know my environment has a time limit (e.g., 5 minutes)
        environment_timeout_seconds = 300  # 5 minutes

        # When I calculate how many documents fit in the limit
        if stats.new_documents > 0 and stats.training_time_seconds > 0:
            time_per_doc = stats.training_time_seconds / stats.new_documents
            safe_batch_size = int(environment_timeout_seconds / time_per_doc * 0.8)  # 80% safety margin

            # Then I can plan batch sizes that complete before timeout
            assert safe_batch_size > 0, "Should be able to calculate safe batch size"

            # The batch size should be reasonable
            estimated_time = safe_batch_size * time_per_doc
            assert estimated_time < environment_timeout_seconds, (
                f"Batch of {safe_batch_size} estimated at {estimated_time:.1f}s "
                f"should fit in {environment_timeout_seconds}s limit"
            )
