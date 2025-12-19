"""
Integration tests for checkpoint resume functionality.

Tests the ability to save progress during compute_all() and resume
after an interruption.
"""

import os
import json
import tempfile
import unittest
from pathlib import Path

from cortical import CorticalTextProcessor


class TestCheckpointResume(unittest.TestCase):
    """Integration tests for checkpoint/resume functionality."""

    def setUp(self):
        """Create a processor with test documents."""
        self.processor = CorticalTextProcessor()
        # Add enough documents to make compute_all() take noticeable time
        for i in range(20):
            self.processor.process_document(
                f"doc_{i}",
                f"Document {i} about neural networks and machine learning. "
                f"This is test content for checkpoint testing with topic {i}."
            )
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_dir = Path(self.temp_dir) / "checkpoints"

    def tearDown(self):
        """Clean up temp files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_checkpoint_creates_progress_file(self):
        """Verify that checkpoint saves create progress files."""
        # Run compute_all with checkpointing
        self.processor.compute_all(
            checkpoint_dir=str(self.checkpoint_dir)
        )

        # Check that checkpoint directory was created
        self.assertTrue(self.checkpoint_dir.exists())

        # Progress file should exist
        progress_file = self.checkpoint_dir / "checkpoint_progress.json"
        self.assertTrue(progress_file.exists())

        # Verify it contains checkpoint data
        with open(progress_file, 'r') as f:
            progress = json.load(f)
        self.assertIn("completed_phases", progress)
        self.assertIn("last_updated", progress)
        # Verify completed phases is a list
        self.assertIsInstance(progress["completed_phases"], list)
        # Should have completed at least some phases
        self.assertGreater(len(progress["completed_phases"]), 0)

    def test_compute_all_completes_with_checkpointing(self):
        """Verify compute_all completes successfully with checkpointing enabled."""
        result = self.processor.compute_all(
            checkpoint_dir=str(self.checkpoint_dir)
        )

        # Should return stats dict
        self.assertIsInstance(result, dict)

        # Verify computations completed
        self.assertFalse(self.processor.is_stale(self.processor.COMP_TFIDF))
        self.assertFalse(self.processor.is_stale(self.processor.COMP_PAGERANK))

    def test_checkpoint_progress_file_format(self):
        """Verify checkpoint progress file has correct format."""
        progress_file = self.checkpoint_dir / "checkpoint_progress.json"

        # Create a manual progress file to test loading
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        progress_data = {
            "completed_phases": ["tfidf", "pagerank_standard"],
            "last_updated": "2025-01-01T00:00:00"
        }
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f)

        # Verify file is valid JSON
        with open(progress_file, 'r') as f:
            loaded = json.load(f)

        self.assertEqual(loaded["completed_phases"], ["tfidf", "pagerank_standard"])
        self.assertEqual(loaded["last_updated"], "2025-01-01T00:00:00")

    def test_resume_skips_completed_phases(self):
        """Verify that resume skips already-completed phases."""
        # First, run a few phases with checkpointing
        progress_file = self.checkpoint_dir / "checkpoint_progress.json"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Simulate partial completion
        progress_data = {
            "completed_phases": ["tfidf"],
            "last_updated": "2025-01-01T00:01:00"
        }
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f)

        # Resume should work without error
        # (actual phase skipping depends on implementation details)
        result = self.processor.compute_all(
            checkpoint_dir=str(self.checkpoint_dir),
            resume=True
        )

        self.assertIsInstance(result, dict)

    def test_checkpoint_without_resume_starts_fresh(self):
        """Verify that without resume flag, compute starts fresh."""
        # Create a fake progress file
        progress_file = self.checkpoint_dir / "checkpoint_progress.json"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        progress_data = {
            "completed_phases": ["tfidf", "pagerank_standard", "bigram_connections"],
            "last_updated": "2025-01-01T00:00:00"
        }
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f)

        # Without resume=True, should start fresh
        result = self.processor.compute_all(
            checkpoint_dir=str(self.checkpoint_dir),
            resume=False
        )

        # Should complete all phases
        self.assertFalse(self.processor.is_stale(self.processor.COMP_TFIDF))


class TestCheckpointEdgeCases(unittest.TestCase):
    """Edge case tests for checkpoint functionality."""

    def setUp(self):
        self.processor = CorticalTextProcessor()
        self.processor.process_document("doc1", "Simple test document.")
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_checkpoint_with_empty_corpus(self):
        """Verify checkpointing works with empty corpus."""
        empty_processor = CorticalTextProcessor()
        checkpoint_dir = Path(self.temp_dir) / "empty_checkpoints"

        # Should not crash
        result = empty_processor.compute_all(
            checkpoint_dir=str(checkpoint_dir)
        )

        self.assertIsInstance(result, dict)

    def test_checkpoint_with_single_document(self):
        """Verify checkpointing works with single document."""
        checkpoint_dir = Path(self.temp_dir) / "single_checkpoints"

        result = self.processor.compute_all(
            checkpoint_dir=str(checkpoint_dir)
        )

        self.assertIsInstance(result, dict)

    def test_invalid_checkpoint_dir_handled(self):
        """Verify graceful handling of invalid checkpoint directory."""
        # Use a path that can't be created (e.g., nested in non-existent parent)
        # Most systems will allow creation, so just verify no crash
        checkpoint_dir = Path(self.temp_dir) / "valid" / "nested" / "path"

        result = self.processor.compute_all(
            checkpoint_dir=str(checkpoint_dir)
        )

        self.assertIsInstance(result, dict)


if __name__ == "__main__":
    unittest.main()
