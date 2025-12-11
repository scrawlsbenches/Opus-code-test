"""
Tests for showcase.py - Timer class and utility functions.
"""

import unittest
import time
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from showcase import Timer, print_header, print_subheader, render_bar


class TestTimer(unittest.TestCase):
    """Tests for the Timer class."""

    def test_timer_start_stop(self):
        """Test basic start/stop timing."""
        timer = Timer()
        timer.start('test_op')
        time.sleep(0.01)  # Small delay
        elapsed = timer.stop()

        self.assertGreater(elapsed, 0.005)
        self.assertLess(elapsed, 0.1)

    def test_timer_records_time(self):
        """Test that timer records time in times dict."""
        timer = Timer()
        timer.start('operation')
        time.sleep(0.01)
        timer.stop()

        self.assertIn('operation', timer.times)
        self.assertGreater(timer.times['operation'], 0)

    def test_timer_get(self):
        """Test get method returns recorded time."""
        timer = Timer()
        timer.start('op1')
        time.sleep(0.01)
        timer.stop()

        recorded = timer.get('op1')
        self.assertGreater(recorded, 0)
        self.assertEqual(recorded, timer.times['op1'])

    def test_timer_get_missing(self):
        """Test get returns 0 for unrecorded operation."""
        timer = Timer()
        self.assertEqual(timer.get('nonexistent'), 0)

    def test_timer_multiple_operations(self):
        """Test timing multiple operations."""
        timer = Timer()

        timer.start('op1')
        time.sleep(0.01)
        timer.stop()

        timer.start('op2')
        time.sleep(0.02)
        timer.stop()

        self.assertIn('op1', timer.times)
        self.assertIn('op2', timer.times)
        self.assertGreater(timer.get('op2'), timer.get('op1'))

    def test_timer_overwrite(self):
        """Test that timing same operation overwrites previous."""
        timer = Timer()

        timer.start('op')
        time.sleep(0.01)
        timer.stop()
        first_time = timer.get('op')

        timer.start('op')
        time.sleep(0.02)
        timer.stop()
        second_time = timer.get('op')

        # Second time should overwrite and be longer
        self.assertNotEqual(first_time, second_time)


class TestRenderBar(unittest.TestCase):
    """Tests for the render_bar function."""

    def test_render_bar_full(self):
        """Test render_bar at 100%."""
        bar = render_bar(100, 100, width=10)
        self.assertEqual(bar, "█" * 10)

    def test_render_bar_empty(self):
        """Test render_bar at 0%."""
        bar = render_bar(0, 100, width=10)
        self.assertEqual(bar, "░" * 10)

    def test_render_bar_half(self):
        """Test render_bar at 50%."""
        bar = render_bar(50, 100, width=10)
        self.assertEqual(bar, "█" * 5 + "░" * 5)

    def test_render_bar_zero_max(self):
        """Test render_bar with zero max value."""
        bar = render_bar(50, 0, width=10)
        self.assertEqual(bar, " " * 10)

    def test_render_bar_custom_width(self):
        """Test render_bar with custom width."""
        bar = render_bar(75, 100, width=20)
        self.assertEqual(len(bar), 20)
        self.assertEqual(bar.count("█"), 15)


class TestPrintFunctions(unittest.TestCase):
    """Tests for print helper functions."""

    def test_print_header_returns_none(self):
        """Test print_header doesn't raise."""
        # Just verify it doesn't raise
        result = print_header("Test Header")
        self.assertIsNone(result)

    def test_print_subheader_returns_none(self):
        """Test print_subheader doesn't raise."""
        result = print_subheader("Test Subheader")
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
