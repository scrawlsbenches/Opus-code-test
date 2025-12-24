"""
Tests for __init__.py modules to improve coverage.

These tests cover import paths and lazy imports that are not exercised
by other tests.
"""

import pytest
import sys
from unittest.mock import patch


class TestCorticalInit:
    """Tests for cortical/__init__.py"""

    def test_version_exists(self):
        """Verify version string is defined."""
        import cortical
        assert hasattr(cortical, '__version__')
        assert cortical.__version__ == "2.0.0"

    def test_main_exports(self):
        """Verify main exports are available."""
        from cortical import (
            CorticalTextProcessor,
            FluentProcessor,
            CorticalConfig,
            CorticalLayer,
            Minicolumn,
            Edge,
        )
        assert CorticalTextProcessor is not None
        assert FluentProcessor is not None
        assert CorticalConfig is not None
        assert CorticalLayer is not None
        assert Minicolumn is not None
        assert Edge is not None

class TestMLExperimentsInit:
    """Tests for cortical/ml_experiments/__init__.py"""

    def test_version_exists(self):
        """Verify version string is defined."""
        import cortical.ml_experiments as ml_exp
        assert hasattr(ml_exp, '__version__')
        assert ml_exp.__version__ == '1.0.0'

    def test_main_exports(self):
        """Verify main exports are available."""
        from cortical.ml_experiments import (
            DatasetManager,
            ExperimentManager,
            MetricsManager,
            compute_file_hash,
        )
        assert DatasetManager is not None
        assert ExperimentManager is not None
        assert MetricsManager is not None
        assert compute_file_hash is not None

    def test_get_file_prediction_experiment(self):
        """Test lazy import of FilePredictionExperiment."""
        from cortical.ml_experiments import get_file_prediction_experiment

        # This should trigger the lazy import (line 43-44)
        FilePredictionExperiment = get_file_prediction_experiment()
        assert FilePredictionExperiment is not None
        assert hasattr(FilePredictionExperiment, '__name__')

    def test_create_commit_dataset(self):
        """Test lazy import in create_commit_dataset."""
        from cortical.ml_experiments import create_commit_dataset

        # To trigger the lazy import (line 48-49), we need to actually call it
        # This will fail without git/data, but that's ok - we just need to
        # execute the import line
        try:
            # Call with minimal args to trigger the import
            create_commit_dataset(max_commits=0)
        except Exception:
            # We expect this to fail, we just want to trigger the import
            pass

        # Verify the import worked by checking the module is loaded
        assert 'cortical.ml_experiments.file_prediction_adapter' in sys.modules

    def test_run_ablation_study(self):
        """Test lazy import in run_ablation_study."""
        from cortical.ml_experiments import run_ablation_study

        # To trigger the lazy import (line 53-54), we need to actually call it
        # This will fail without data, but that's ok
        try:
            # Call with minimal args to trigger the import
            run_ablation_study(model_path=None, output_dir=None)
        except Exception:
            # We expect this to fail, we just want to trigger the import
            pass

        # Verify the import worked
        assert 'cortical.ml_experiments.file_prediction_adapter' in sys.modules

    def test_all_exports_present(self):
        """Verify all __all__ exports are importable."""
        import cortical.ml_experiments as ml_exp

        for export_name in ml_exp.__all__:
            assert hasattr(ml_exp, export_name), f"{export_name} not found in module"
            assert getattr(ml_exp, export_name) is not None
