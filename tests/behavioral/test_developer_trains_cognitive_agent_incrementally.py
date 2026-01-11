"""
Developer Trains Cognitive Agent Incrementally

Epic: Incremental Learning for Cognitive Systems

As a developer training a CognitiveAgent,
I want to incrementally train on new documents without reprocessing existing ones,
So that I can efficiently update the agent's knowledge as new content becomes available.
"""

import pytest
from pathlib import Path

from cortical.common.filesystem import RealFileSystem


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


class TestCognitiveAgentWithFileSystem:
    """
    Epic: Autonomous File System Access

    As a developer building autonomous cognitive agents,
    I want the CognitiveAgent to have its own FileSystem,
    So that it can persist state and discover new documents to learn from.
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
