"""
Tests for GoT configuration module.

Tests GoT-specific configuration like GoTConfig dataclass and
DurabilityMode enum values.

Note: CDG durability implementation tests (fsync behavior in WAL/Store)
are in tests/unit/cdg/test_cdg_durability.py
"""

import tempfile
import unittest
from pathlib import Path

from cortical.got import (
    DurabilityMode,
    GoTConfig,
    GoTManager,
)
from cortical.core.bootstrap import create_container


class TestDurabilityMode(unittest.TestCase):
    """Test DurabilityMode enum."""

    def test_durability_mode_enum_values(self):
        """Test that DurabilityMode has correct values."""
        self.assertEqual(DurabilityMode.PARANOID.value, "paranoid")
        self.assertEqual(DurabilityMode.BALANCED.value, "balanced")
        self.assertEqual(DurabilityMode.RELAXED.value, "relaxed")

    def test_durability_mode_has_three_values(self):
        """Test that DurabilityMode has exactly 3 modes."""
        modes = list(DurabilityMode)
        self.assertEqual(len(modes), 3)
        self.assertIn(DurabilityMode.PARANOID, modes)
        self.assertIn(DurabilityMode.BALANCED, modes)
        self.assertIn(DurabilityMode.RELAXED, modes)


class TestGoTConfig(unittest.TestCase):
    """Test GoTConfig dataclass."""

    def test_default_config_is_balanced(self):
        """Test that default durability mode is BALANCED."""
        config = GoTConfig()
        self.assertEqual(config.durability, DurabilityMode.BALANCED)

    def test_config_accepts_durability_param(self):
        """Test that GoTConfig accepts durability parameter."""
        config = GoTConfig(durability=DurabilityMode.PARANOID)
        self.assertEqual(config.durability, DurabilityMode.PARANOID)

        config = GoTConfig(durability=DurabilityMode.RELAXED)
        self.assertEqual(config.durability, DurabilityMode.RELAXED)


class TestGoTManagerDurability(unittest.TestCase):
    """Test that GoTManager accepts and uses durability parameter."""

    def setUp(self):
        """Set up temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.got_dir = Path(self.temp_dir) / ".got"

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @unittest.skip("TODO: create_container doesn't support durability parameter yet")
    def test_manager_accepts_durability_param(self):
        """Test that GoTManager accepts durability parameter."""
        # Test PARANOID
        container = create_container(got_dir=self.got_dir, durability=DurabilityMode.PARANOID)

        manager = container.resolve(GoTManager)
        self.assertEqual(manager.durability, DurabilityMode.PARANOID)
        self.assertEqual(manager.tx_manager.durability, DurabilityMode.PARANOID)

        # Clean up
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

        # Test BALANCED (default)
        container = create_container(got_dir=self.got_dir)

        manager = container.resolve(GoTManager)
        self.assertEqual(manager.durability, DurabilityMode.BALANCED)
        self.assertEqual(manager.tx_manager.durability, DurabilityMode.BALANCED)

        # Clean up
        shutil.rmtree(self.temp_dir, ignore_errors=True)

        # Test RELAXED
        container = create_container(got_dir=self.got_dir, durability=DurabilityMode.RELAXED)

        manager = container.resolve(GoTManager)
        self.assertEqual(manager.durability, DurabilityMode.RELAXED)
        self.assertEqual(manager.tx_manager.durability, DurabilityMode.RELAXED)

    def test_manager_default_is_balanced(self):
        """Test that GoTManager defaults to BALANCED mode."""
        container = create_container(got_dir=self.got_dir)

        manager = container.resolve(GoTManager)
        self.assertEqual(manager.durability, DurabilityMode.BALANCED)


if __name__ == '__main__':
    unittest.main()
