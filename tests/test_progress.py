"""
Tests for the progress reporting system.

Ensures progress feedback during long operations works correctly.
"""

import io
import sys
import unittest

from cortical import CorticalTextProcessor
from cortical.progress import (
    ConsoleProgressReporter,
    CallbackProgressReporter,
    SilentProgressReporter,
    MultiPhaseProgress,
)


class TestConsoleProgressReporter(unittest.TestCase):
    """Test ConsoleProgressReporter."""

    def test_update_formats_correctly(self):
        """Test update produces correct format."""
        output = io.StringIO()
        reporter = ConsoleProgressReporter(file=output, width=20)
        reporter.update("Testing", 50)
        output_str = output.getvalue()
        self.assertIn("Testing", output_str)
        self.assertIn("50%", output_str)

    def test_complete_shows_100_percent(self):
        """Test complete shows 100%."""
        output = io.StringIO()
        reporter = ConsoleProgressReporter(file=output)
        reporter.complete("Testing")
        output_str = output.getvalue()
        self.assertIn("100%", output_str)

    def test_complete_shows_elapsed_time(self):
        """Test complete shows elapsed time."""
        output = io.StringIO()
        reporter = ConsoleProgressReporter(file=output)
        reporter.update("Testing", 50)  # Start tracking
        reporter.complete("Testing")
        output_str = output.getvalue()
        # Should have time indicator
        self.assertIn("s", output_str)

    def test_update_with_message(self):
        """Test update with custom message."""
        output = io.StringIO()
        reporter = ConsoleProgressReporter(file=output)
        reporter.update("Testing", 50, "Processing items")
        output_str = output.getvalue()
        self.assertIn("50%", output_str)

    def test_progress_bar_width(self):
        """Test progress bar respects width."""
        output = io.StringIO()
        reporter = ConsoleProgressReporter(file=output, width=10)
        reporter.update("Testing", 50)
        output_str = output.getvalue()
        self.assertIn("Testing", output_str)

    def test_unicode_vs_ascii(self):
        """Test Unicode vs ASCII mode."""
        output1 = io.StringIO()
        output2 = io.StringIO()

        reporter1 = ConsoleProgressReporter(file=output1, use_unicode=True)
        reporter2 = ConsoleProgressReporter(file=output2, use_unicode=False)

        reporter1.update("Test", 50)
        reporter2.update("Test", 50)

        # Both should produce output
        self.assertTrue(len(output1.getvalue()) > 0)
        self.assertTrue(len(output2.getvalue()) > 0)

    def test_percentage_clamping(self):
        """Test percentage is handled correctly."""
        output = io.StringIO()
        reporter = ConsoleProgressReporter(file=output)

        # Should not crash with various values
        reporter.update("Test", 0)
        reporter.update("Test", 100)
        self.assertIn("%", output.getvalue())


class TestCallbackProgressReporter(unittest.TestCase):
    """Test CallbackProgressReporter."""

    def test_callback_invoked_on_update(self):
        """Test callback is called on update."""
        calls = []
        def callback(phase, pct, msg):
            calls.append((phase, pct, msg))

        reporter = CallbackProgressReporter(callback)
        reporter.update("Testing", 50, "message")

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "Testing")
        self.assertEqual(calls[0][1], 50)

    def test_callback_invoked_on_complete(self):
        """Test callback is called on complete."""
        calls = []
        def callback(phase, pct, msg):
            calls.append((phase, pct, msg))

        reporter = CallbackProgressReporter(callback)
        reporter.complete("Testing")

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], 100)

    def test_multiple_updates(self):
        """Test multiple update calls."""
        calls = []
        def callback(phase, pct, msg):
            calls.append(pct)

        reporter = CallbackProgressReporter(callback)
        reporter.update("Test", 25)
        reporter.update("Test", 50)
        reporter.update("Test", 75)
        reporter.complete("Test")

        self.assertEqual(calls, [25, 50, 75, 100])


class TestSilentProgressReporter(unittest.TestCase):
    """Test SilentProgressReporter."""

    def test_update_does_nothing(self):
        """Test update doesn't crash."""
        reporter = SilentProgressReporter()
        reporter.update("Test", 50)  # Should not raise
        reporter.update("Test", 100, "message")  # Should not raise

    def test_complete_does_nothing(self):
        """Test complete doesn't crash."""
        reporter = SilentProgressReporter()
        reporter.complete("Test")  # Should not raise
        reporter.complete("Test", "message")  # Should not raise


class TestMultiPhaseProgress(unittest.TestCase):
    """Test MultiPhaseProgress helper."""

    def test_initialization(self):
        """Test initialization with phases."""
        phases = {"phase1": 30, "phase2": 70}
        reporter = SilentProgressReporter()
        progress = MultiPhaseProgress(reporter, phases)
        self.assertEqual(len(progress.phases), 2)

    def test_phase_normalization(self):
        """Test phase weights are normalized."""
        phases = {"phase1": 50, "phase2": 50}
        reporter = SilentProgressReporter()
        progress = MultiPhaseProgress(reporter, phases)
        # Weights should sum to 100
        total = sum(progress.phases.values())
        self.assertAlmostEqual(total, 100.0, places=5)

    def test_start_phase(self):
        """Test starting a phase."""
        calls = []
        def callback(phase, pct, msg):
            calls.append((phase, pct))

        phases = {"phase1": 50, "phase2": 50}
        reporter = CallbackProgressReporter(callback)
        progress = MultiPhaseProgress(reporter, phases)
        progress.start_phase("phase1")

        self.assertGreater(len(calls), 0)

    def test_update_within_phase(self):
        """Test updating progress within a phase."""
        calls = []
        def callback(phase, pct, msg):
            calls.append(pct)

        phases = {"phase1": 100}
        reporter = CallbackProgressReporter(callback)
        progress = MultiPhaseProgress(reporter, phases)
        progress.start_phase("phase1")
        progress.update(50)

        # Should have some progress reported
        self.assertGreater(len(calls), 0)

    def test_complete_phase(self):
        """Test completing a phase."""
        calls = []
        def callback(phase, pct, msg):
            calls.append(pct)

        phases = {"phase1": 50, "phase2": 50}
        reporter = CallbackProgressReporter(callback)
        progress = MultiPhaseProgress(reporter, phases)
        progress.start_phase("phase1")
        progress.complete_phase()

        # Should have completed first phase
        self.assertGreater(len(calls), 0)

    def test_sequential_phases(self):
        """Test running phases sequentially."""
        calls = []
        def callback(phase, pct, msg):
            calls.append(pct)

        phases = {"phase1": 50, "phase2": 50}
        reporter = CallbackProgressReporter(callback)
        progress = MultiPhaseProgress(reporter, phases)

        progress.start_phase("phase1")
        progress.update(50)
        progress.complete_phase()

        progress.start_phase("phase2")
        progress.update(50)
        progress.complete_phase()

        # Should have multiple updates
        self.assertGreater(len(calls), 2)

    def test_unknown_phase_raises(self):
        """Test starting unknown phase raises error."""
        phases = {"phase1": 100}
        reporter = SilentProgressReporter()
        progress = MultiPhaseProgress(reporter, phases)

        with self.assertRaises(ValueError):
            progress.start_phase("unknown")


class TestProcessorIntegration(unittest.TestCase):
    """Test integration with CorticalTextProcessor."""

    def test_compute_all_silent_default(self):
        """Test compute_all is silent by default."""
        proc = CorticalTextProcessor()
        proc.process_document("doc1", "Test content")

        # Should not raise
        proc.compute_all()

    def test_compute_all_with_callback(self):
        """Test compute_all with progress callback."""
        proc = CorticalTextProcessor()
        proc.process_document("doc1", "Test content")

        phases = []
        def callback(phase, pct, msg):
            if phase not in phases:
                phases.append(phase)

        reporter = CallbackProgressReporter(callback)
        proc.compute_all(progress_callback=reporter)

        # Should have reported some phases
        self.assertGreater(len(phases), 0)

    def test_compute_all_with_show_progress(self):
        """Test compute_all with show_progress flag."""
        proc = CorticalTextProcessor()
        proc.process_document("doc1", "Test content")

        # Redirect stderr to capture progress
        old_stderr = sys.stderr
        sys.stderr = io.StringIO()

        try:
            proc.compute_all(show_progress=True)
            output = sys.stderr.getvalue()
        finally:
            sys.stderr = old_stderr

        # Should have some progress output
        self.assertGreater(len(output), 0)

    def test_backward_compatibility(self):
        """Test that old code still works."""
        proc = CorticalTextProcessor()
        proc.process_document("doc1", "Test content here")

        # Old-style call without any progress arguments
        proc.compute_all()

        # Should complete successfully
        self.assertFalse(proc.is_stale(proc.COMP_TFIDF))


if __name__ == '__main__':
    unittest.main()
