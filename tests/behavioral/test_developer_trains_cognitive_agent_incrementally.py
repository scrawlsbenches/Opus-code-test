"""
Developer Trains Cognitive Agent Incrementally

Epic: Incremental Learning for Cognitive Systems

As a developer training a CognitiveAgent,
I want to incrementally train on new documents without reprocessing existing ones,
So that I can efficiently update the agent's knowledge as new content becomes available.
"""

import pytest
from pathlib import Path


class TestDeveloperTrainsOnNewDocuments:
    """
    Epic: Initial Training and Knowledge Acquisition

    As a developer with a corpus of training documents,
    I want to train my CognitiveAgent on text files,
    So that it learns vocabulary and word associations from the content.
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
        agent = CognitiveAgent()
        trainer = IncrementalTrainer(agent, model_dir=tmp_path / "model")

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
        agent = CognitiveAgent()
        trainer = IncrementalTrainer(agent, model_dir=tmp_path / "model")

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


class TestDeveloperSkipsAlreadyTrainedDocuments:
    """
    Epic: Efficient Incremental Updates

    As a developer updating an agent's knowledge,
    I want already-trained documents to be skipped,
    So that I don't waste time reprocessing unchanged content.
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

        agent = CognitiveAgent()
        trainer = IncrementalTrainer(agent, model_dir=tmp_path / "model")
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

        agent = CognitiveAgent()
        trainer = IncrementalTrainer(agent, model_dir=tmp_path / "model")
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


class TestDeveloperHandlesModifiedDocuments:
    """
    Epic: Change Detection for Updated Content

    As a developer maintaining training content,
    I want modified documents to be retrained,
    So that the agent learns from updated information.
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

        agent = CognitiveAgent()
        trainer = IncrementalTrainer(agent, model_dir=tmp_path / "model")
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

        agent = CognitiveAgent()
        trainer = IncrementalTrainer(agent, model_dir=tmp_path / "model")
        trainer.train_directory(docs_dir, show_progress=False)

        # And I touch the file without changing its content
        time.sleep(0.01)  # Ensure different mtime
        doc_path.write_text(original_content)  # Same content, new mtime

        # When I run training again
        stats = trainer.train_directory(docs_dir, show_progress=False)

        # Then the document is skipped
        assert stats.skipped_documents == 1
        assert stats.modified_documents == 0


class TestDeveloperPersistsTrainingState:
    """
    Epic: Training State Persistence

    As a developer with long-running training processes,
    I want training state to persist across sessions,
    So that I can resume training without starting over.
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
        agent1 = CognitiveAgent()
        trainer1 = IncrementalTrainer(agent1, model_dir=model_dir)
        trainer1.train_directory(docs_dir, show_progress=False)

        # And I start a new session with a fresh agent
        agent2 = CognitiveAgent()

        # When I load the trainer from the saved state
        trainer2 = IncrementalTrainer(agent2, model_dir=model_dir)

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

        agent = CognitiveAgent()
        trainer = IncrementalTrainer(agent, model_dir=tmp_path / "model")
        trainer.train_directory(docs_dir, show_progress=False)

        # When I check the training status
        status = trainer.status()

        # Then I can see how many documents were trained
        assert status["total_documents_trained"] == 2

        # And I can see the vocabulary size
        assert status["vocabulary_size"] > 0

        # And I can see when training last occurred
        assert status["last_training"] is not None


class TestDeveloperForcesRetraining:
    """
    Epic: Manual Training Control

    As a developer debugging training issues,
    I want to force retraining of all documents,
    So that I can rebuild knowledge from scratch when needed.
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

        agent = CognitiveAgent()
        trainer = IncrementalTrainer(agent, model_dir=tmp_path / "model")
        trainer.train_directory(docs_dir, show_progress=False)

        # When I run training with force_retrain=True
        stats = trainer.train_directory(docs_dir, show_progress=False, force_retrain=True)

        # Then all documents are processed again
        assert stats.new_documents == 2
        assert stats.skipped_documents == 0
        # Note: atoms_created may be 0 on retrain since atoms are content-addressed
        # and already exist, but links are still created
        assert stats.links_created > 0


class TestDeveloperListsTrainedDocuments:
    """
    Epic: Training Visibility and Auditing

    As a developer auditing training coverage,
    I want to list all trained documents,
    So that I can verify what knowledge the agent has acquired.
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

        agent = CognitiveAgent()
        trainer = IncrementalTrainer(agent, model_dir=tmp_path / "model")
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


class TestDeveloperHandlesSubdirectories:
    """
    Epic: Hierarchical Document Organization

    As a developer with organized document directories,
    I want training to handle subdirectories,
    So that I can organize training content hierarchically.
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

        agent = CognitiveAgent()
        trainer = IncrementalTrainer(agent, model_dir=tmp_path / "model")

        # When I train with recursive=True (default)
        stats = trainer.train_directory(docs_dir, show_progress=False)

        # Then documents in all subdirectories are found
        assert stats.new_documents == 3

        # And their paths include the subdirectory
        trained = trainer.list_trained()
        assert "root.txt" in trained
        assert any("category1" in p for p in trained)
        assert any("category2" in p for p in trained)


class TestDeveloperHandlesEmptyAndMissingCases:
    """
    Epic: Graceful Error Handling

    As a developer working with various directory states,
    I want training to handle edge cases gracefully,
    So that I don't encounter unexpected crashes.
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

        agent = CognitiveAgent()
        trainer = IncrementalTrainer(agent, model_dir=tmp_path / "model")

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

        agent = CognitiveAgent()
        trainer = IncrementalTrainer(agent, model_dir=tmp_path / "model")

        # When I train with a pattern that matches nothing
        stats = trainer.train_directory(docs_dir, pattern="*.txt", show_progress=False)

        # Then training completes successfully
        # (no exception raised)

        # And statistics show zero documents
        assert stats.total_files_scanned == 0


class TestDeveloperUsesInMemoryFileSystem:
    """
    Epic: Fast In-Memory Testing

    As a developer writing tests for training workflows,
    I want to use an in-memory filesystem,
    So that tests run faster and I can assert on file operations.
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
        assert fs.exists(model_dir / "tokenizer.json")

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
        fs.assert_file_was_written(model_dir / "tokenizer.json")

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
