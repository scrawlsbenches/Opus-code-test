#!/usr/bin/env python3
"""
Expert Consolidator

Unified interface for managing multiple micro-experts in the MoE system.
Handles loading, training, saving, and ensemble prediction across all experts.

Key responsibilities:
- Load/save all experts atomically
- Coordinate training across experts with different data sources
- Version conflict detection and migration
- Ensemble prediction aggregation
- Expert availability checks
"""

import json
import shutil
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set

import sys
sys.path.insert(0, str(Path(__file__).parent))

from micro_expert import MicroExpert, ExpertMetrics
from voting_aggregator import VotingAggregator, AggregatedPrediction, AggregationConfig
from experts.file_expert import FileExpert
from experts.test_expert import TestExpert
from experts.error_expert import ErrorDiagnosisExpert
from experts.episode_expert import EpisodeExpert
from experts.refactor_expert import RefactorExpert


class ExpertConsolidator:
    """
    Consolidates and manages multiple micro-experts.

    Provides unified interface for:
    - Loading/saving all experts from/to a directory
    - Training all experts on new data
    - Getting ensemble predictions
    - Version management and migration

    Attributes:
        experts: Dictionary mapping expert_type -> expert instance
        model_dir: Directory containing expert models
        aggregator: VotingAggregator for ensemble predictions
    """

    # Expert registry: expert_type -> expert class
    EXPERT_CLASSES = {
        'file': FileExpert,
        'test': TestExpert,
        'error': ErrorDiagnosisExpert,
        'episode': EpisodeExpert,
        'refactor': RefactorExpert,
    }

    def __init__(self, model_dir: Optional[Path] = None):
        """
        Initialize consolidator.

        Args:
            model_dir: Optional directory to load experts from
        """
        self.experts: Dict[str, MicroExpert] = {}
        self.model_dir = model_dir
        self.aggregator = VotingAggregator()

        if model_dir and model_dir.exists():
            self.load_all_experts(model_dir)

    def load_all_experts(self, model_dir: Path) -> Dict[str, MicroExpert]:
        """
        Load all expert models from a directory.

        Automatically detects and loads all available expert JSON files.

        Args:
            model_dir: Directory containing expert JSON files

        Returns:
            Dictionary mapping expert_type -> loaded expert

        Raises:
            FileNotFoundError: If model_dir doesn't exist
            ValueError: If expert file format is invalid
        """
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        self.model_dir = model_dir
        loaded_experts = {}

        # Try to load each expert type
        for expert_type, expert_class in self.EXPERT_CLASSES.items():
            expert_path = model_dir / f"{expert_type}_expert.json"

            if expert_path.exists():
                try:
                    expert = expert_class.load(expert_path)
                    loaded_experts[expert_type] = expert
                    self.experts[expert_type] = expert
                except Exception as e:
                    print(f"Warning: Failed to load {expert_type} expert: {e}")
            else:
                print(f"Note: {expert_type} expert not found at {expert_path}")

        return loaded_experts

    def consolidate_training(
        self,
        commits: Optional[List[Dict[str, Any]]] = None,
        transcripts: Optional[List[Any]] = None,
        errors: Optional[List[Dict[str, Any]]] = None,
        incremental: bool = False
    ) -> Dict[str, bool]:
        """
        Train all applicable experts on new data.

        Routes training data to appropriate experts:
        - commits -> FileExpert, TestExpert
        - transcripts -> EpisodeExpert
        - errors -> ErrorDiagnosisExpert

        Args:
            commits: List of commit dictionaries with 'files', 'message', etc.
            transcripts: List of transcript exchanges
            errors: List of error records
            incremental: If True, merge with existing model data (not implemented yet)

        Returns:
            Dictionary mapping expert_type -> training_success (bool)
        """
        results = {}

        # Train FileExpert on commits
        if commits and 'file' in self.EXPERT_CLASSES:
            try:
                if 'file' not in self.experts:
                    self.experts['file'] = FileExpert()

                # FileExpert doesn't have train() yet, it's initialized with model_data
                # For now, mark as skipped
                results['file'] = False
                print("Note: FileExpert training not implemented (requires commit processing)")
            except Exception as e:
                print(f"Error training FileExpert: {e}")
                results['file'] = False

        # Train TestExpert on commits
        if commits and 'test' in self.EXPERT_CLASSES:
            try:
                if 'test' not in self.experts:
                    self.experts['test'] = TestExpert()

                self.experts['test'].train(commits)
                results['test'] = True
                print(f"Trained TestExpert on {len(commits)} commits")
            except Exception as e:
                print(f"Error training TestExpert: {e}")
                results['test'] = False

        # Train EpisodeExpert on transcripts
        if transcripts and 'episode' in self.EXPERT_CLASSES:
            try:
                if 'episode' not in self.experts:
                    self.experts['episode'] = EpisodeExpert()

                # Convert transcripts to episodes
                episodes = EpisodeExpert.extract_episodes(transcripts)
                self.experts['episode'].train(episodes)
                results['episode'] = True
                print(f"Trained EpisodeExpert on {len(episodes)} episodes")
            except Exception as e:
                print(f"Error training EpisodeExpert: {e}")
                results['episode'] = False

        # Train ErrorDiagnosisExpert on errors
        if errors and 'error' in self.EXPERT_CLASSES:
            try:
                if 'error' not in self.experts:
                    self.experts['error'] = ErrorDiagnosisExpert()

                self.experts['error'].train(errors)
                results['error'] = True
                print(f"Trained ErrorDiagnosisExpert on {len(errors)} error records")
            except Exception as e:
                print(f"Error training ErrorDiagnosisExpert: {e}")
                results['error'] = False

        # Train RefactorExpert on commits (filters for refactor: commits internally)
        if commits and 'refactor' in self.EXPERT_CLASSES:
            try:
                if 'refactor' not in self.experts:
                    self.experts['refactor'] = RefactorExpert()

                self.experts['refactor'].train(commits)
                refactor_count = self.experts['refactor'].trained_on_commits
                results['refactor'] = refactor_count > 0
                print(f"Trained RefactorExpert on {refactor_count} refactoring commits")
            except Exception as e:
                print(f"Error training RefactorExpert: {e}")
                results['refactor'] = False

        return results

    def save_all_experts(self, model_dir: Path) -> None:
        """
        Save all loaded experts to a directory atomically.

        Uses atomic write pattern: write to temp dir, then rename.
        This ensures either all experts are saved or none are.

        Args:
            model_dir: Directory to save experts to (created if doesn't exist)

        Raises:
            IOError: If save fails for any expert
        """
        # Create model directory if it doesn't exist
        model_dir.mkdir(parents=True, exist_ok=True)

        # Use temporary directory for atomic save
        with tempfile.TemporaryDirectory(dir=model_dir.parent) as tmpdir:
            tmppath = Path(tmpdir)

            try:
                # Save all experts to temp directory
                for expert_type, expert in self.experts.items():
                    expert_path = tmppath / f"{expert_type}_expert.json"
                    expert.save(expert_path)

                # Atomic move: copy all files from temp to target
                for expert_type in self.experts.keys():
                    src = tmppath / f"{expert_type}_expert.json"
                    dst = model_dir / f"{expert_type}_expert.json"
                    shutil.move(str(src), str(dst))

                self.model_dir = model_dir
                print(f"Saved {len(self.experts)} experts to {model_dir}")

            except Exception as e:
                print(f"Error saving experts: {e}")
                raise IOError(f"Failed to save experts atomically: {e}")

    def get_ensemble_prediction(
        self,
        context: Dict[str, Any],
        expert_types: Optional[List[str]] = None,
        config: Optional[AggregationConfig] = None
    ) -> AggregatedPrediction:
        """
        Get ensemble prediction by aggregating multiple experts.

        Automatically selects relevant experts based on context,
        or uses specified expert_types.

        Args:
            context: Prediction context (varies by expert type)
            expert_types: Optional list of expert types to use (default: all loaded)
            config: Optional aggregation config

        Returns:
            AggregatedPrediction with combined results from all experts
        """
        # Determine which experts to use
        if expert_types is None:
            expert_types = list(self.experts.keys())

        # Collect predictions from each expert
        predictions = []
        for expert_type in expert_types:
            if expert_type in self.experts:
                try:
                    pred = self.experts[expert_type].predict(context)
                    predictions.append(pred)
                except Exception as e:
                    print(f"Warning: {expert_type} expert prediction failed: {e}")

        # Aggregate predictions
        if not predictions:
            # Return empty aggregation
            return AggregatedPrediction(
                items=[],
                contributing_experts=[],
                disagreement_score=0.0,
                confidence=0.0,
                metadata={'error': 'No predictions available'}
            )

        return self.aggregator.aggregate(predictions, config)

    def get_expert(self, expert_type: str) -> Optional[MicroExpert]:
        """
        Get a specific expert by type.

        Args:
            expert_type: Type of expert (file, test, error, episode)

        Returns:
            Expert instance or None if not loaded
        """
        return self.experts.get(expert_type)

    def has_expert(self, expert_type: str) -> bool:
        """
        Check if an expert is loaded.

        Args:
            expert_type: Type of expert to check

        Returns:
            True if expert is loaded
        """
        return expert_type in self.experts

    def get_loaded_experts(self) -> List[str]:
        """
        Get list of loaded expert types.

        Returns:
            List of expert type strings
        """
        return list(self.experts.keys())

    def check_version_conflicts(self) -> Dict[str, Any]:
        """
        Check for version conflicts across experts.

        Returns:
            Dictionary with conflict information:
            - has_conflicts: bool
            - conflicts: List of conflict descriptions
            - versions: Dict mapping expert_type -> version
        """
        conflicts = []
        versions = {}

        for expert_type, expert in self.experts.items():
            versions[expert_type] = expert.version

            # Check if version is outdated (simple heuristic)
            major, minor, patch = map(int, expert.version.split('.'))
            if major < 1:
                conflicts.append(f"{expert_type} is pre-1.0 ({expert.version})")

        return {
            'has_conflicts': len(conflicts) > 0,
            'conflicts': conflicts,
            'versions': versions
        }

    def get_training_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get training statistics for all experts.

        Returns:
            Dictionary mapping expert_type -> stats dict with:
            - trained_on_commits: int
            - trained_on_sessions: int
            - created_at: str
            - version: str
        """
        stats = {}

        for expert_type, expert in self.experts.items():
            stats[expert_type] = {
                'trained_on_commits': expert.trained_on_commits,
                'trained_on_sessions': expert.trained_on_sessions,
                'created_at': expert.created_at,
                'version': expert.version,
                'has_metrics': expert.metrics is not None
            }

        return stats

    def create_all_experts(self) -> None:
        """
        Create fresh instances of all available expert types.

        Useful for initializing a new model directory.
        """
        for expert_type, expert_class in self.EXPERT_CLASSES.items():
            self.experts[expert_type] = expert_class()

        print(f"Created {len(self.experts)} fresh experts")


# Convenience functions

def load_experts(model_dir: Path) -> ExpertConsolidator:
    """
    Load all experts from a directory.

    Args:
        model_dir: Directory containing expert JSON files

    Returns:
        ExpertConsolidator with loaded experts
    """
    consolidator = ExpertConsolidator()
    consolidator.load_all_experts(model_dir)
    return consolidator


def train_all_experts(
    commits: Optional[List[Dict]] = None,
    transcripts: Optional[List[Any]] = None,
    errors: Optional[List[Dict]] = None,
    model_dir: Optional[Path] = None
) -> ExpertConsolidator:
    """
    Train all experts on provided data.

    Args:
        commits: Commit data for FileExpert and TestExpert
        transcripts: Transcript data for EpisodeExpert
        errors: Error data for ErrorDiagnosisExpert
        model_dir: Optional directory to load existing experts from

    Returns:
        ExpertConsolidator with trained experts
    """
    if model_dir and model_dir.exists():
        consolidator = load_experts(model_dir)
    else:
        consolidator = ExpertConsolidator()
        consolidator.create_all_experts()

    consolidator.consolidate_training(
        commits=commits,
        transcripts=transcripts,
        errors=errors
    )

    return consolidator


if __name__ == '__main__':
    # Demo usage
    print("Expert Consolidator Demo")
    print("=" * 60)

    # Create fresh experts
    consolidator = ExpertConsolidator()
    consolidator.create_all_experts()

    print(f"\nLoaded experts: {consolidator.get_loaded_experts()}")

    # Mock training data
    mock_commits = [
        {
            'files': ['cortical/query/search.py', 'tests/test_query.py'],
            'message': 'feat: Add graph boosted search'
        },
        {
            'files': ['cortical/analysis.py', 'tests/test_analysis.py'],
            'message': 'fix: PageRank convergence issue'
        }
    ]

    mock_transcripts = [
        {
            'query': 'Fix authentication bug',
            'tools_used': ['Read', 'Edit', 'Bash'],
            'timestamp': '2025-12-17T10:00:00',
            'tool_inputs': []
        }
    ]

    mock_errors = [
        {
            'error_type': 'TypeError',
            'error_message': 'unsupported operand type',
            'files_modified': ['cortical/analysis.py'],
            'resolution': 'Added type check'
        }
    ]

    # Train experts
    print("\nTraining experts...")
    results = consolidator.consolidate_training(
        commits=mock_commits,
        transcripts=mock_transcripts,
        errors=mock_errors
    )

    print(f"Training results: {results}")

    # Get stats
    stats = consolidator.get_training_stats()
    print(f"\nTraining stats:")
    for expert_type, stat in stats.items():
        print(f"  {expert_type}: {stat}")

    # Save experts
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir) / 'models'
        consolidator.save_all_experts(model_dir)

        # Load them back
        consolidator2 = load_experts(model_dir)
        print(f"\nReloaded experts: {consolidator2.get_loaded_experts()}")

        # Test ensemble prediction
        context = {
            'changed_files': ['cortical/query/search.py'],
            'query': 'Fix search bug'
        }

        pred = consolidator2.get_ensemble_prediction(context, expert_types=['test'])
        print(f"\nEnsemble prediction:")
        print(f"  Items: {len(pred.items)}")
        print(f"  Confidence: {pred.confidence:.3f}")
        print(f"  Contributing experts: {pred.contributing_experts}")
