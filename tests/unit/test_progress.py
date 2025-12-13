"""
Unit tests for progress reporting infrastructure.

Tests the progress reporting system including:
- ConsoleProgressReporter formatting
- CallbackProgressReporter callback invocation
- SilentProgressReporter no-op behavior
- MultiPhaseProgress phase tracking
- Integration with CorticalTextProcessor.compute_all()
"""

import unittest
import io
import time
from unittest.mock import Mock, call

from cortical.progress import (
    ConsoleProgressReporter,
    CallbackProgressReporter,
    SilentProgressReporter,
    MultiPhaseProgress,
)
from cortical import CorticalTextProcessor


class TestConsoleProgressReporter(unittest.TestCase):
    """Test console-based progress reporting."""

    def test_update_formats_correctly(self):
        """Test that update() formats output correctly."""
        buffer = io.StringIO()
        reporter = ConsoleProgressReporter(file=buffer, width=20, show_eta=False)

        reporter.update("Test phase", 50.0)

        output = buffer.getvalue()
        self.assertIn("Test phase", output)
        self.assertIn("50%", output)
        self.assertIn("[", output)
        self.assertIn("]", output)

    def test_update_with_message(self):
        """Test that custom messages are included."""
        buffer = io.StringIO()
        reporter = ConsoleProgressReporter(file=buffer, width=20, show_eta=False)

        reporter.update("Test phase", 75.0, "custom message")

        output = buffer.getvalue()
        self.assertIn("custom message", output)

    def test_complete_shows_100_percent(self):
        """Test that complete() shows 100% and newline."""
        buffer = io.StringIO()
        reporter = ConsoleProgressReporter(file=buffer, width=20, show_eta=False)

        reporter.update("Test phase", 50.0)
        reporter.complete("Test phase")

        output = buffer.getvalue()
        self.assertIn("100%", output)
        self.assertTrue(output.endswith("\n"))

    def test_complete_shows_elapsed_time(self):
        """Test that complete() shows elapsed time."""
        buffer = io.StringIO()
        reporter = ConsoleProgressReporter(file=buffer, width=20, show_eta=False)

        reporter.update("Test phase", 50.0)
        time.sleep(0.1)
        reporter.complete("Test phase")

        output = buffer.getvalue()
        # Should contain time in seconds
        self.assertRegex(output, r"\(\d+\.\d+s\)")

    def test_complete_with_message(self):
        """Test that completion messages are included."""
        buffer = io.StringIO()
        reporter = ConsoleProgressReporter(file=buffer, width=20, show_eta=False)

        reporter.complete("Test phase", "All done!")

        output = buffer.getvalue()
        self.assertIn("All done!", output)

    def test_progress_bar_width(self):
        """Test that progress bar respects width parameter."""
        buffer = io.StringIO()
        reporter = ConsoleProgressReporter(file=buffer, width=10, show_eta=False)

        reporter.update("Test", 50.0)

        output = buffer.getvalue()
        # Count filled and empty characters (should be 10 total)
        filled = output.count(reporter.fill_char)
        empty = output.count(reporter.empty_char)
        self.assertEqual(filled + empty, 10)

    def test_unicode_vs_ascii_mode(self):
        """Test that Unicode and ASCII modes use different characters."""
        buffer_unicode = io.StringIO()
        buffer_ascii = io.StringIO()

        unicode_reporter = ConsoleProgressReporter(
            file=buffer_unicode, width=10, show_eta=False, use_unicode=True
        )
        ascii_reporter = ConsoleProgressReporter(
            file=buffer_ascii, width=10, show_eta=False, use_unicode=False
        )

        unicode_reporter.update("Test", 50.0)
        ascii_reporter.update("Test", 50.0)

        unicode_output = buffer_unicode.getvalue()
        ascii_output = buffer_ascii.getvalue()

        # Unicode uses █ and ░, ASCII uses # and -
        self.assertIn('█', unicode_output)
        self.assertIn('#', ascii_output)
        self.assertNotIn('█', ascii_output)
        self.assertNotIn('#', unicode_output)

    def test_percentage_clamping(self):
        """Test that percentages are clamped to 0-100 range."""
        buffer = io.StringIO()
        reporter = ConsoleProgressReporter(file=buffer, width=20, show_eta=False)

        # Test negative percentage
        reporter.update("Test", -10.0)
        output = buffer.getvalue()
        self.assertIn("0%", output)

        # Test over 100%
        buffer = io.StringIO()
        reporter = ConsoleProgressReporter(file=buffer, width=20, show_eta=False)
        reporter.update("Test", 150.0)
        output = buffer.getvalue()
        self.assertIn("100%", output)

    def test_eta_estimation(self):
        """Test that ETA is calculated and displayed."""
        buffer = io.StringIO()
        reporter = ConsoleProgressReporter(file=buffer, width=20, show_eta=True)

        reporter.update("Test", 10.0)
        time.sleep(0.2)
        reporter.update("Test", 20.0)

        output = buffer.getvalue()
        # Should contain ETA after sufficient progress
        # Note: ETA may not appear on first update
        if "ETA:" in output:
            self.assertRegex(output, r"ETA:\s*\d+s")


class TestCallbackProgressReporter(unittest.TestCase):
    """Test callback-based progress reporting."""

    def test_callback_invoked_on_update(self):
        """Test that callback is called with correct arguments on update."""
        callback = Mock()
        reporter = CallbackProgressReporter(callback)

        reporter.update("Test phase", 50.0, "message")

        callback.assert_called_once_with("Test phase", 50.0, "message")

    def test_callback_invoked_on_complete(self):
        """Test that callback is called on completion."""
        callback = Mock()
        reporter = CallbackProgressReporter(callback)

        reporter.complete("Test phase", "Done")

        callback.assert_called_once_with("Test phase", 100.0, "Done")

    def test_callback_with_none_message(self):
        """Test that None message is handled correctly."""
        callback = Mock()
        reporter = CallbackProgressReporter(callback)

        reporter.update("Test phase", 50.0, None)
        reporter.complete("Test phase", None)

        self.assertEqual(callback.call_count, 2)
        # Update call
        callback.assert_any_call("Test phase", 50.0, None)
        # Complete call with default message
        callback.assert_any_call("Test phase", 100.0, "Complete")

    def test_multiple_updates(self):
        """Test that callback is invoked for multiple updates."""
        callback = Mock()
        reporter = CallbackProgressReporter(callback)

        reporter.update("Phase 1", 25.0)
        reporter.update("Phase 1", 50.0)
        reporter.update("Phase 1", 75.0)
        reporter.complete("Phase 1")

        self.assertEqual(callback.call_count, 4)


class TestSilentProgressReporter(unittest.TestCase):
    """Test silent (no-op) progress reporter."""

    def test_update_does_nothing(self):
        """Test that update() is a no-op."""
        reporter = SilentProgressReporter()

        # Should not raise any exceptions
        reporter.update("Test", 50.0)
        reporter.update("Test", 100.0, "message")

    def test_complete_does_nothing(self):
        """Test that complete() is a no-op."""
        reporter = SilentProgressReporter()

        # Should not raise any exceptions
        reporter.complete("Test")
        reporter.complete("Test", "message")


class TestMultiPhaseProgress(unittest.TestCase):
    """Test multi-phase progress tracking."""

    def test_initialization(self):
        """Test that MultiPhaseProgress initializes correctly."""
        callback = Mock()
        reporter = CallbackProgressReporter(callback)
        phases = {"Phase 1": 30, "Phase 2": 70}

        progress = MultiPhaseProgress(reporter, phases)

        self.assertEqual(progress.overall_progress, 0.0)

    def test_phase_normalization(self):
        """Test that phase weights are normalized."""
        callback = Mock()
        reporter = CallbackProgressReporter(callback)
        phases = {"Phase 1": 1, "Phase 2": 2, "Phase 3": 1}

        progress = MultiPhaseProgress(reporter, phases, normalize=True)

        # Should normalize to 25%, 50%, 25%
        self.assertAlmostEqual(progress.phases["Phase 1"], 25.0)
        self.assertAlmostEqual(progress.phases["Phase 2"], 50.0)
        self.assertAlmostEqual(progress.phases["Phase 3"], 25.0)

    def test_phase_no_normalization(self):
        """Test that normalization can be disabled."""
        callback = Mock()
        reporter = CallbackProgressReporter(callback)
        phases = {"Phase 1": 10, "Phase 2": 20}

        progress = MultiPhaseProgress(reporter, phases, normalize=False)

        # Should keep original values
        self.assertEqual(progress.phases["Phase 1"], 10)
        self.assertEqual(progress.phases["Phase 2"], 20)

    def test_start_phase(self):
        """Test starting a new phase."""
        callback = Mock()
        reporter = CallbackProgressReporter(callback)
        phases = {"Phase 1": 30, "Phase 2": 70}

        progress = MultiPhaseProgress(reporter, phases)
        progress.start_phase("Phase 1")

        # Should call reporter.update with 0%
        callback.assert_called_with("Phase 1", 0.0, None)

    def test_start_unknown_phase_raises(self):
        """Test that starting an unknown phase raises ValueError."""
        callback = Mock()
        reporter = CallbackProgressReporter(callback)
        phases = {"Phase 1": 30, "Phase 2": 70}

        progress = MultiPhaseProgress(reporter, phases)

        with self.assertRaises(ValueError):
            progress.start_phase("Unknown Phase")

    def test_update_within_phase(self):
        """Test updating progress within a phase."""
        callback = Mock()
        reporter = CallbackProgressReporter(callback)
        phases = {"Phase 1": 30, "Phase 2": 70}

        progress = MultiPhaseProgress(reporter, phases)
        progress.start_phase("Phase 1")
        progress.update(50.0)

        # 50% of Phase 1 (30% weight) = 15% overall
        self.assertAlmostEqual(progress.overall_progress, 15.0)

    def test_complete_phase(self):
        """Test completing a phase."""
        callback = Mock()
        reporter = CallbackProgressReporter(callback)
        phases = {"Phase 1": 30, "Phase 2": 70}

        progress = MultiPhaseProgress(reporter, phases)
        progress.start_phase("Phase 1")
        progress.update(100.0)
        progress.complete_phase("Done")

        # Should call callback with 100.0 and completion message
        # Last call should be the completion
        last_call = callback.call_args_list[-1]
        self.assertEqual(last_call[0][0], "Phase 1")
        self.assertEqual(last_call[0][1], 100.0)
        self.assertEqual(last_call[0][2], "Done")

    def test_sequential_phases(self):
        """Test progressing through multiple phases."""
        callback = Mock()
        reporter = CallbackProgressReporter(callback)
        phases = {"Phase 1": 25, "Phase 2": 50, "Phase 3": 25}

        progress = MultiPhaseProgress(reporter, phases)

        # Phase 1: 0% to 25%
        progress.start_phase("Phase 1")
        self.assertAlmostEqual(progress.overall_progress, 0.0)
        progress.update(100.0)
        self.assertAlmostEqual(progress.overall_progress, 25.0)
        progress.complete_phase()

        # Phase 2: 25% to 75%
        progress.start_phase("Phase 2")
        progress.update(50.0)
        self.assertAlmostEqual(progress.overall_progress, 50.0)
        progress.update(100.0)
        self.assertAlmostEqual(progress.overall_progress, 75.0)
        progress.complete_phase()

        # Phase 3: 75% to 100%
        progress.start_phase("Phase 3")
        progress.update(100.0)
        self.assertAlmostEqual(progress.overall_progress, 100.0)
        progress.complete_phase()

    def test_update_with_message(self):
        """Test that messages are passed through to reporter."""
        callback = Mock()
        reporter = CallbackProgressReporter(callback)
        phases = {"Phase 1": 100}

        progress = MultiPhaseProgress(reporter, phases)
        progress.start_phase("Phase 1")
        progress.update(50.0, "Processing...")

        # Should call reporter.update with message
        callback.assert_called_with("Phase 1", 50.0, "Processing...")


class TestProcessorIntegration(unittest.TestCase):
    """Test integration with CorticalTextProcessor."""

    def test_compute_all_with_callback(self):
        """Test compute_all() with custom callback."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Neural networks process information.")
        processor.process_document("doc2", "Machine learning algorithms analyze data.")

        callback = Mock()
        reporter = CallbackProgressReporter(callback)

        processor.compute_all(progress_callback=reporter, verbose=False)

        # Callback should have been invoked multiple times
        self.assertGreater(callback.call_count, 0)

        # Check that phases were reported
        phase_names = [call[0][0] for call in callback.call_args_list]
        self.assertIn("TF-IDF computation", phase_names)
        self.assertIn("PageRank computation", phase_names)

    def test_compute_all_with_show_progress(self):
        """Test compute_all() with show_progress flag."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Neural networks process information.")

        # Should not raise exceptions
        processor.compute_all(show_progress=True, verbose=False)

    def test_compute_all_silent_by_default(self):
        """Test that compute_all() is silent by default."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Neural networks process information.")

        # Capture stderr to ensure nothing is written
        import sys
        old_stderr = sys.stderr
        sys.stderr = io.StringIO()

        try:
            processor.compute_all(verbose=False)
            output = sys.stderr.getvalue()
            # Should be empty (no progress output)
            self.assertEqual(output, "")
        finally:
            sys.stderr = old_stderr

    def test_compute_all_phases_reported(self):
        """Test that all expected phases are reported."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Neural networks process information.")
        processor.process_document("doc2", "Machine learning algorithms analyze data.")

        callback = Mock()
        reporter = CallbackProgressReporter(callback)

        processor.compute_all(
            progress_callback=reporter,
            verbose=False,
            build_concepts=True
        )

        # Extract phase names from callback calls
        phase_names = set()
        for call_args in callback.call_args_list:
            if len(call_args[0]) > 0:
                phase_names.add(call_args[0][0])

        # Check expected phases
        expected_phases = {
            "Activation propagation",
            "PageRank computation",
            "TF-IDF computation",
            "Document connections",
            "Bigram connections",
            "Concept clustering",
            "Concept connections",
        }

        for phase in expected_phases:
            self.assertIn(phase, phase_names, f"Missing phase: {phase}")

    def test_compute_all_completion_calls(self):
        """Test that completion is called for each phase."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Neural networks process information.")

        phases_completed = []

        def track_completion(phase, percent, message):
            if percent == 100.0:
                phases_completed.append(phase)

        reporter = CallbackProgressReporter(track_completion)

        processor.compute_all(
            progress_callback=reporter,
            verbose=False,
            build_concepts=True
        )

        # Should have completed multiple phases
        self.assertGreater(len(phases_completed), 0)
        self.assertIn("TF-IDF computation", phases_completed)

    def test_backward_compatibility(self):
        """Test that existing code without progress parameters still works."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Neural networks process information.")

        # Old-style call should still work
        stats = processor.compute_all(verbose=False)

        # Should return stats
        self.assertIsInstance(stats, dict)


if __name__ == '__main__':
    unittest.main()
