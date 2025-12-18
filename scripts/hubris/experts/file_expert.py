#!/usr/bin/env python3
"""
File Expert

Micro-expert specialized in predicting which files need modification
for a given task description.

Migrated from ml_file_prediction_v1.FilePredictionModel to use
the MicroExpert interface.
"""

import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from micro_expert import MicroExpert, ExpertPrediction, ExpertMetrics


class FileExpert(MicroExpert):
    """
    Expert for predicting files to modify for a task.

    Uses patterns learned from commit history:
    - Commit type prefix patterns (feat:, fix:, docs:, etc.)
    - File co-occurrence patterns
    - Module keyword associations
    - Semantic similarity (optional)

    Model Data Structure:
        file_cooccurrence: Dict[str, Dict[str, int]] - File co-occurrence matrix
        type_to_files: Dict[str, Dict[str, int]] - Commit type -> files mapping
        keyword_to_files: Dict[str, Dict[str, int]] - Keyword -> files mapping
        file_frequency: Dict[str, int] - File change frequency
        file_to_commits: Dict[str, List[str]] - File -> recent commit messages
        total_commits: int - Total commits in training data
    """

    def __init__(
        self,
        expert_id: str = "file_expert",
        version: str = "1.1.0",
        **kwargs
    ):
        """
        Initialize FileExpert.

        Args:
            expert_id: Unique identifier (default: "file_expert")
            version: Expert version (default: "1.1.0")
            **kwargs: Additional arguments passed to MicroExpert base class
        """
        # Remove expert_type from kwargs if present (avoids conflict when loading)
        kwargs.pop('expert_type', None)
        super().__init__(
            expert_id=expert_id,
            expert_type="file",
            version=version,
            **kwargs
        )

        # Ensure model_data has required keys
        if not self.model_data:
            self.model_data = {
                'file_cooccurrence': {},
                'type_to_files': {},
                'keyword_to_files': {},
                'file_frequency': {},
                'file_to_commits': {},
                'total_commits': 0
            }

    def predict(self, context: Dict[str, Any]) -> ExpertPrediction:
        """
        Predict files for a given task.

        Args:
            context: Dictionary with:
                - query (str): Task description or commit message
                - top_n (int, optional): Number of predictions (default: 10)
                - seed_files (List[str], optional): Known files for co-occurrence boost
                - use_semantic (bool, optional): Use semantic similarity (default: False)
                - min_confidence (float, optional): Minimum confidence threshold (default: 0.1)

        Returns:
            ExpertPrediction with ranked (file_path, confidence) pairs
        """
        query = context.get('query', '')
        top_n = context.get('top_n', 10)
        seed_files = context.get('seed_files', [])
        use_semantic = context.get('use_semantic', False)
        min_confidence = context.get('min_confidence', 0.1)

        if not query:
            return ExpertPrediction(
                expert_id=self.expert_id,
                expert_type=self.expert_type,
                items=[],
                metadata={'error': 'Empty query'}
            )

        # Score files
        file_scores = self._score_files(
            query=query,
            seed_files=seed_files,
            use_semantic=use_semantic
        )

        # Sort and filter
        sorted_files = sorted(file_scores.items(), key=lambda x: -x[1])

        # Filter by minimum confidence
        if min_confidence > 0:
            sorted_files = [(f, s) for f, s in sorted_files if s >= min_confidence]

        # Limit to top_n
        items = sorted_files[:top_n]

        # Extract keywords for metadata
        keywords = self._extract_keywords(query)

        return ExpertPrediction(
            expert_id=self.expert_id,
            expert_type=self.expert_type,
            items=items,
            metadata={
                'query': query,
                'keywords': list(keywords),
                'num_seed_files': len(seed_files),
                'total_candidates': len(file_scores)
            }
        )

    def _score_files(
        self,
        query: str,
        seed_files: List[str],
        use_semantic: bool
    ) -> Dict[str, float]:
        """
        Score files based on query and model patterns.

        Args:
            query: Task description
            seed_files: Known files for co-occurrence boost
            use_semantic: Whether to use semantic similarity

        Returns:
            Dict of file_path -> score
        """
        file_scores: Dict[str, float] = defaultdict(float)

        # Extract features
        commit_type = self._extract_commit_type(query)
        keywords = self._extract_keywords(query)

        # Score based on commit type
        if commit_type and commit_type in self.model_data['type_to_files']:
            type_files = self.model_data['type_to_files'][commit_type]
            type_total = sum(type_files.values())

            for f, count in type_files.items():
                # TF-IDF-like scoring
                tf = count / type_total if type_total > 0 else 0
                idf = math.log(self.model_data['total_commits'] /
                              (self.model_data['file_frequency'].get(f, 1) + 1))
                file_scores[f] += tf * idf * 2.0  # Weight for type match

        # Score based on keywords
        for keyword in keywords:
            if keyword in self.model_data['keyword_to_files']:
                kw_files = self.model_data['keyword_to_files'][keyword]
                kw_total = sum(kw_files.values())

                for f, count in kw_files.items():
                    tf = count / kw_total if kw_total > 0 else 0
                    idf = math.log(self.model_data['total_commits'] /
                                  (self.model_data['file_frequency'].get(f, 1) + 1))
                    file_scores[f] += tf * idf * 1.5  # Weight for keyword match

        # Boost based on co-occurrence with seed files
        if seed_files:
            for seed in seed_files:
                if seed in self.model_data['file_cooccurrence']:
                    cooc = self.model_data['file_cooccurrence'][seed]

                    for f, count in cooc.items():
                        # Jaccard-like similarity
                        seed_freq = self.model_data['file_frequency'].get(seed, 1)
                        file_freq = self.model_data['file_frequency'].get(f, 1)
                        union = seed_freq + file_freq - count
                        similarity = count / union if union > 0 else 0
                        file_scores[f] += similarity * 3.0  # Strong weight

        # Semantic similarity boost (optional)
        if use_semantic and self.model_data.get('file_to_commits'):
            for filepath, commit_msgs in self.model_data['file_to_commits'].items():
                for commit_msg in commit_msgs[:5]:  # Top 5 recent
                    sim = self._compute_semantic_similarity(query, commit_msg)
                    if sim > 0.2:
                        file_scores[filepath] += sim * 0.5

        return file_scores

    def _extract_commit_type(self, message: str) -> Optional[str]:
        """
        Extract commit type from message (feat, fix, docs, etc.).

        Args:
            message: Commit message or task description

        Returns:
            Commit type string or None
        """
        import re

        patterns = {
            'feat': r'^feat(?:\(.+?\))?:\s*',
            'fix': r'^fix(?:\(.+?\))?:\s*',
            'docs': r'^docs(?:\(.+?\))?:\s*',
            'refactor': r'^refactor(?:\(.+?\))?:\s*',
            'test': r'^test(?:\(.+?\))?:\s*',
            'chore': r'^chore(?:\(.+?\))?:\s*',
        }

        msg_lower = message.lower()
        for commit_type, pattern in patterns.items():
            if re.search(pattern, msg_lower):
                return commit_type

        return None

    def _extract_keywords(self, text: str) -> set:
        """
        Extract meaningful keywords from text.

        Args:
            text: Text to extract keywords from

        Returns:
            Set of keywords
        """
        # Simple word extraction (lowercase, alphanumeric)
        import re
        words = re.findall(r'\b[a-z_][a-z0-9_]*\b', text.lower())

        # Filter stop words
        stop_words = {
            'add', 'change', 'fix', 'update', 'implement', 'the', 'a', 'an',
            'is', 'are', 'to', 'of', 'in', 'for', 'on', 'with', 'test'
        }

        return set(w for w in words if w not in stop_words and len(w) > 2)

    def _compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Compute simple semantic similarity using word overlap.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0-1)
        """
        words1 = self._extract_keywords(text1)
        words2 = self._extract_keywords(text2)

        if not words1 or not words2:
            return 0.0

        # Jaccard similarity
        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union) if union else 0.0

    @classmethod
    def from_v1_model(cls, v1_model: Any) -> 'FileExpert':
        """
        Create FileExpert from legacy FilePredictionModel.

        Args:
            v1_model: Legacy FilePredictionModel instance

        Returns:
            FileExpert instance
        """
        # Convert v1 model data to expert format
        model_data = {
            'file_cooccurrence': v1_model.file_cooccurrence,
            'type_to_files': v1_model.type_to_files,
            'keyword_to_files': v1_model.keyword_to_files,
            'file_frequency': v1_model.file_frequency,
            'file_to_commits': v1_model.file_to_commits,
            'total_commits': v1_model.total_commits
        }

        # Convert metrics if available
        metrics = None
        if v1_model.metrics:
            metrics = ExpertMetrics(
                mrr=v1_model.metrics.get('mrr', 0.0),
                recall_at_k={
                    1: v1_model.metrics.get('recall_at_1', 0.0),
                    5: v1_model.metrics.get('recall_at_5', 0.0),
                    10: v1_model.metrics.get('recall_at_10', 0.0)
                },
                precision_at_k={
                    1: v1_model.metrics.get('precision_at_1', 0.0)
                },
                test_examples=v1_model.metrics.get('test_examples', 0)
            )

        return cls(
            expert_id="file_expert",
            version=v1_model.version,
            created_at=v1_model.trained_at,
            trained_on_commits=v1_model.total_commits,
            git_hash=v1_model.git_commit_hash,
            model_data=model_data,
            metrics=metrics
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FileExpert':
        """
        Load FileExpert from dict.

        Args:
            data: Dictionary representation

        Returns:
            FileExpert instance
        """
        metrics = None
        if data.get('metrics'):
            metrics = ExpertMetrics.from_dict(data['metrics'])

        return cls(
            expert_id=data.get('expert_id', 'file_expert'),
            version=data['version'],
            created_at=data['created_at'],
            trained_on_commits=data['trained_on_commits'],
            trained_on_sessions=data['trained_on_sessions'],
            git_hash=data['git_hash'],
            model_data=data['model_data'],
            metrics=metrics,
            calibration_curve=data.get('calibration_curve')
        )
