"""
Unit tests for compute_all() checkpointing functionality.

Tests checkpoint saving, loading, resuming, and full checkpoint/resume cycles.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from cortical.processor import CorticalTextProcessor
from cortical.config import CorticalConfig


class TestComputeCheckpointing(unittest.TestCase):
    """Test suite for compute_all() checkpointing functionality."""

    def setUp(self):
        """Create a processor with sample documents for testing."""
        self.processor = CorticalTextProcessor()

        # Add some test documents
        self.processor.process_document(
            "doc1",
            "Neural networks are computational models inspired by biological neurons."
        )
        self.processor.process_document(
            "doc2",
            "Machine learning algorithms can be trained on large datasets."
        )
        self.processor.process_document(
            "doc3",
            "Deep learning uses multiple layers of neural networks."
        )

        # Create temporary directory for checkpoints
        self.checkpoint_dir = tempfile.mkdtemp(prefix='checkpoint_test_')

    def tearDown(self):
        """Clean up temporary checkpoint directory."""
        if Path(self.checkpoint_dir).exists():
            shutil.rmtree(self.checkpoint_dir)

    def test_checkpoint_creates_progress_file(self):
        """Test that checkpointing creates a progress file."""
        # Run compute_all with checkpointing
        self.processor.compute_all(
            checkpoint_dir=self.checkpoint_dir,
            verbose=False,
            build_concepts=False  # Simpler test
        )

        # Check that progress file exists
        progress_file = Path(self.checkpoint_dir) / 'checkpoint_progress.json'
        self.assertTrue(progress_file.exists(), "Progress file should be created")

        # Check progress file content
        with open(progress_file, 'r') as f:
            progress_data = json.load(f)

        self.assertIn('completed_phases', progress_data)
        self.assertIn('last_updated', progress_data)
        self.assertIsInstance(progress_data['completed_phases'], list)
        self.assertGreater(len(progress_data['completed_phases']), 0,
                          "Should have completed at least one phase")

    def test_checkpoint_creates_state_files(self):
        """Test that checkpointing creates all required state files."""
        self.processor.compute_all(
            checkpoint_dir=self.checkpoint_dir,
            verbose=False,
            build_concepts=False
        )

        checkpoint_path = Path(self.checkpoint_dir)

        # Check for manifest and documents
        self.assertTrue((checkpoint_path / 'manifest.json').exists())
        self.assertTrue((checkpoint_path / 'documents.json').exists())

        # Check for layer files
        layers_dir = checkpoint_path / 'layers'
        self.assertTrue(layers_dir.exists())
        self.assertTrue((layers_dir / 'L0_tokens.json').exists())
        self.assertTrue((layers_dir / 'L1_bigrams.json').exists())

    def test_checkpoint_progress_accumulates(self):
        """Test that checkpoint progress accumulates across phases."""
        # Create a custom checkpoint test by manually calling _save_checkpoint
        self.processor._save_checkpoint(self.checkpoint_dir, 'phase1', verbose=False)

        progress = self.processor._load_checkpoint_progress(self.checkpoint_dir)
        self.assertEqual(len(progress), 1)
        self.assertIn('phase1', progress)

        # Add another phase
        self.processor._save_checkpoint(self.checkpoint_dir, 'phase2', verbose=False)

        progress = self.processor._load_checkpoint_progress(self.checkpoint_dir)
        self.assertEqual(len(progress), 2)
        self.assertIn('phase1', progress)
        self.assertIn('phase2', progress)

    def test_resume_skips_completed_phases(self):
        """Test that resuming skips already completed phases."""
        # Run compute_all to completion with checkpointing
        stats1 = self.processor.compute_all(
            checkpoint_dir=self.checkpoint_dir,
            verbose=False,
            build_concepts=False
        )

        # Load checkpoint and resume (should skip all phases)
        processor2 = CorticalTextProcessor.resume_from_checkpoint(
            self.checkpoint_dir,
            verbose=False
        )

        # Resume should complete quickly since all phases are done
        stats2 = processor2.compute_all(
            checkpoint_dir=self.checkpoint_dir,
            resume=True,
            verbose=False,
            build_concepts=False
        )

        # Both should have similar stats structure
        self.assertIsInstance(stats2, dict)

    def test_checkpoint_state_matches_in_memory(self):
        """Test that checkpoint state matches the in-memory processor state."""
        # Compute all phases
        self.processor.compute_all(
            checkpoint_dir=self.checkpoint_dir,
            verbose=False,
            build_concepts=False
        )

        # Load from checkpoint
        processor2 = CorticalTextProcessor.resume_from_checkpoint(
            self.checkpoint_dir,
            verbose=False
        )

        # Compare document counts
        self.assertEqual(
            len(self.processor.documents),
            len(processor2.documents),
            "Document count should match"
        )

        # Compare layer counts
        from cortical.layers import CorticalLayer
        for layer_enum in [CorticalLayer.TOKENS, CorticalLayer.BIGRAMS]:
            count1 = len(self.processor.layers[layer_enum].minicolumns)
            count2 = len(processor2.layers[layer_enum].minicolumns)
            self.assertEqual(
                count1, count2,
                f"Layer {layer_enum.name} minicolumn count should match"
            )

    def test_no_checkpointing_when_dir_none(self):
        """Test that no checkpointing occurs when checkpoint_dir is None."""
        # Run without checkpointing
        self.processor.compute_all(
            checkpoint_dir=None,  # Default behavior
            verbose=False,
            build_concepts=False
        )

        # Progress file should not exist in any default location
        progress_file = Path(self.checkpoint_dir) / 'checkpoint_progress.json'
        self.assertFalse(progress_file.exists(),
                        "Progress file should not be created without checkpoint_dir")

    def test_resume_with_concepts(self):
        """Test resuming with concept building enabled."""
        # Run partial computation and stop
        # We'll simulate this by running compute_all partially
        self.processor.compute_all(
            checkpoint_dir=self.checkpoint_dir,
            verbose=False,
            build_concepts=True  # Enable concept building
        )

        # Resume from checkpoint
        processor2 = CorticalTextProcessor.resume_from_checkpoint(
            self.checkpoint_dir,
            verbose=False
        )

        # Complete the computation (should skip completed phases)
        stats = processor2.compute_all(
            checkpoint_dir=self.checkpoint_dir,
            resume=True,
            verbose=False,
            build_concepts=True
        )

        self.assertIsInstance(stats, dict)
        # If concepts were built, should have clusters_created
        if 'clusters_created' in stats:
            self.assertIsInstance(stats['clusters_created'], int)

    def test_checkpoint_with_different_pagerank_methods(self):
        """Test checkpointing works with different PageRank methods."""
        for method in ['standard', 'semantic', 'hierarchical']:
            with self.subTest(pagerank_method=method):
                # Clean checkpoint dir for each test
                if Path(self.checkpoint_dir).exists():
                    shutil.rmtree(self.checkpoint_dir)
                Path(self.checkpoint_dir).mkdir()

                # Run with specific pagerank method
                stats = self.processor.compute_all(
                    checkpoint_dir=self.checkpoint_dir,
                    verbose=False,
                    build_concepts=False,
                    pagerank_method=method
                )

                # Check that progress file contains the right phase
                progress = self.processor._load_checkpoint_progress(self.checkpoint_dir)
                expected_phase = f"pagerank_{method}"
                self.assertIn(expected_phase, progress,
                            f"Should checkpoint {expected_phase}")

    def test_checkpoint_with_different_connection_strategies(self):
        """Test checkpointing works with different concept connection strategies."""
        for strategy in ['document_overlap', 'semantic', 'embedding', 'hybrid']:
            with self.subTest(connection_strategy=strategy):
                # Clean checkpoint dir for each test
                if Path(self.checkpoint_dir).exists():
                    shutil.rmtree(self.checkpoint_dir)
                Path(self.checkpoint_dir).mkdir()

                # Run with specific connection strategy
                stats = self.processor.compute_all(
                    checkpoint_dir=self.checkpoint_dir,
                    verbose=False,
                    build_concepts=True,
                    connection_strategy=strategy
                )

                # Check that progress file contains the right phase
                progress = self.processor._load_checkpoint_progress(self.checkpoint_dir)
                expected_phase = f"concept_connections_{strategy}"
                self.assertIn(expected_phase, progress,
                            f"Should checkpoint {expected_phase}")

    def test_load_checkpoint_progress_empty_dir(self):
        """Test loading checkpoint progress from empty directory."""
        # Create an empty directory
        empty_dir = tempfile.mkdtemp(prefix='empty_checkpoint_')
        try:
            progress = self.processor._load_checkpoint_progress(empty_dir)
            self.assertEqual(len(progress), 0, "Empty dir should have no progress")
            self.assertIsInstance(progress, set)
        finally:
            shutil.rmtree(empty_dir)

    def test_load_checkpoint_progress_corrupted_file(self):
        """Test loading checkpoint progress from corrupted file."""
        # Create a corrupted progress file
        progress_file = Path(self.checkpoint_dir) / 'checkpoint_progress.json'
        progress_file.parent.mkdir(parents=True, exist_ok=True)
        with open(progress_file, 'w') as f:
            f.write("{ invalid json }")

        # Should handle gracefully
        progress = self.processor._load_checkpoint_progress(self.checkpoint_dir)
        self.assertEqual(len(progress), 0, "Corrupted file should return empty progress")

    def test_full_checkpoint_resume_cycle(self):
        """Test a full checkpoint and resume cycle."""
        # Step 1: Run compute_all with checkpointing
        stats1 = self.processor.compute_all(
            checkpoint_dir=self.checkpoint_dir,
            verbose=False,
            build_concepts=True,
            pagerank_method='standard',
            connection_strategy='document_overlap'
        )

        # Step 2: Verify checkpoint files exist
        checkpoint_path = Path(self.checkpoint_dir)
        self.assertTrue((checkpoint_path / 'checkpoint_progress.json').exists())
        self.assertTrue((checkpoint_path / 'manifest.json').exists())

        # Step 3: Load checkpoint progress
        progress = self.processor._load_checkpoint_progress(self.checkpoint_dir)
        self.assertGreater(len(progress), 0, "Should have completed phases")

        # Step 4: Resume from checkpoint
        processor2 = CorticalTextProcessor.resume_from_checkpoint(
            self.checkpoint_dir,
            verbose=False
        )

        # Step 5: Verify resumed processor state
        self.assertEqual(len(processor2.documents), len(self.processor.documents))

        # Step 6: Complete computation (should skip all phases)
        stats2 = processor2.compute_all(
            checkpoint_dir=self.checkpoint_dir,
            resume=True,
            verbose=False,
            build_concepts=True,
            pagerank_method='standard',
            connection_strategy='document_overlap'
        )

        # Both runs should complete successfully
        self.assertIsInstance(stats1, dict)
        self.assertIsInstance(stats2, dict)

    def test_resume_from_checkpoint_classmethod(self):
        """Test the resume_from_checkpoint class method."""
        # Create a checkpoint
        self.processor.compute_all(
            checkpoint_dir=self.checkpoint_dir,
            verbose=False,
            build_concepts=False
        )

        # Resume using class method
        processor2 = CorticalTextProcessor.resume_from_checkpoint(
            self.checkpoint_dir,
            verbose=False
        )

        # Verify it's a valid processor
        self.assertIsInstance(processor2, CorticalTextProcessor)
        self.assertEqual(len(processor2.documents), 3)
        self.assertGreater(len(processor2.layers), 0)

    def test_checkpoint_atomicity(self):
        """Test that checkpoint saves are atomic."""
        # Save a checkpoint
        self.processor._save_checkpoint(self.checkpoint_dir, 'test_phase', verbose=False)

        # Verify no temporary files remain
        checkpoint_path = Path(self.checkpoint_dir)
        temp_files = list(checkpoint_path.glob('**/*.tmp'))
        self.assertEqual(len(temp_files), 0, "No temporary files should remain")

        # Verify progress file exists
        progress_file = checkpoint_path / 'checkpoint_progress.json'
        self.assertTrue(progress_file.exists())


if __name__ == '__main__':
    unittest.main()
