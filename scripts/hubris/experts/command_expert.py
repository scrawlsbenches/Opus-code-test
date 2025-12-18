#!/usr/bin/env python3
"""
Command Expert

Micro-expert specialized in predicting useful shell commands
(especially python -c one-liners) for a given task description.

Learns patterns from Bash tool call history in .git-ml/actions/.
"""

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from micro_expert import MicroExpert, ExpertPrediction, ExpertMetrics


class CommandExpert(MicroExpert):
    """
    Expert for predicting useful shell commands for a task.

    Uses patterns learned from Bash tool call history:
    - Command templates (python -c patterns, git commands, etc.)
    - Keyword to command associations
    - Command success/failure patterns
    - Context sequences (what commands follow what)

    Model Data Structure:
        command_templates: Dict[str, Dict] - Normalized command -> {count, examples, keywords}
        keyword_to_commands: Dict[str, Dict[str, int]] - Keyword -> command counts
        command_frequency: Dict[str, int] - Command usage frequency
        context_patterns: Dict[str, Dict[str, int]] - Previous action -> next command
        total_commands: int - Total commands in training data
    """

    def __init__(
        self,
        expert_id: str = "command_expert",
        version: str = "1.0.0",
        **kwargs
    ):
        """
        Initialize CommandExpert.

        Args:
            expert_id: Unique identifier (default: "command_expert")
            version: Expert version (default: "1.0.0")
            **kwargs: Additional arguments passed to MicroExpert base class
        """
        kwargs.pop('expert_type', None)
        super().__init__(
            expert_id=expert_id,
            expert_type="command",
            version=version,
            **kwargs
        )

        if not self.model_data:
            self.model_data = {
                'command_templates': {},
                'keyword_to_commands': {},
                'command_frequency': {},
                'context_patterns': {},
                'python_c_patterns': {},  # Specific python -c patterns
                'total_commands': 0
            }

    def predict(self, context: Dict[str, Any]) -> ExpertPrediction:
        """
        Predict commands for a given task.

        Args:
            context: Dictionary with:
                - query (str): Task description
                - top_n (int, optional): Number of predictions (default: 5)
                - previous_action (str, optional): Last action for context
                - command_type (str, optional): Filter to specific type (python, git, etc.)

        Returns:
            ExpertPrediction with ranked (command, confidence) pairs
        """
        query = context.get('query', '')
        top_n = context.get('top_n', 5)
        previous_action = context.get('previous_action', '')
        command_type = context.get('command_type', None)

        if not query:
            return ExpertPrediction(
                expert_id=self.expert_id,
                expert_type=self.expert_type,
                items=[],
                metadata={'error': 'Empty query'}
            )

        # Score commands
        command_scores = self._score_commands(
            query=query,
            previous_action=previous_action,
            command_type=command_type
        )

        # Sort and limit
        sorted_commands = sorted(command_scores.items(), key=lambda x: -x[1])
        items = sorted_commands[:top_n]

        # Normalize scores to 0-1
        if items:
            max_score = items[0][1] if items[0][1] > 0 else 1.0
            items = [(cmd, score / max_score) for cmd, score in items]

        keywords = self._extract_keywords(query)

        return ExpertPrediction(
            expert_id=self.expert_id,
            expert_type=self.expert_type,
            items=items,
            metadata={
                'query': query,
                'keywords': list(keywords),
                'total_candidates': len(command_scores),
                'command_type_filter': command_type
            }
        )

    def _score_commands(
        self,
        query: str,
        previous_action: str,
        command_type: Optional[str]
    ) -> Dict[str, float]:
        """
        Score commands based on query and patterns.

        Args:
            query: Task description
            previous_action: Last action for context boost
            command_type: Optional filter (python, git, etc.)

        Returns:
            Dict of command -> score
        """
        command_scores: Dict[str, float] = defaultdict(float)
        keywords = self._extract_keywords(query)
        total = max(self.model_data.get('total_commands', 1), 1)

        # Score based on keyword matches
        for keyword in keywords:
            if keyword in self.model_data.get('keyword_to_commands', {}):
                kw_commands = self.model_data['keyword_to_commands'][keyword]
                kw_total = sum(kw_commands.values())

                for cmd, count in kw_commands.items():
                    if command_type and not cmd.startswith(command_type):
                        continue
                    # TF-IDF scoring
                    tf = count / kw_total if kw_total > 0 else 0
                    cmd_freq = self.model_data.get('command_frequency', {}).get(cmd, 1)
                    idf = math.log(total / (cmd_freq + 1))
                    command_scores[cmd] += tf * idf * 2.0

        # Boost based on context patterns
        if previous_action and previous_action in self.model_data.get('context_patterns', {}):
            ctx_commands = self.model_data['context_patterns'][previous_action]
            for cmd, count in ctx_commands.items():
                if command_type and not cmd.startswith(command_type):
                    continue
                command_scores[cmd] += count * 1.5

        # Boost python -c patterns if query mentions python/script/run
        python_keywords = {'python', 'script', 'run', 'execute', 'eval', 'one-liner'}
        if keywords & python_keywords:
            for cmd, data in self.model_data.get('python_c_patterns', {}).items():
                pattern_keywords = set(data.get('keywords', []))
                overlap = len(keywords & pattern_keywords)
                if overlap > 0:
                    command_scores[cmd] += overlap * 3.0

        return command_scores

    def _extract_keywords(self, text: str) -> set:
        """Extract meaningful keywords from text."""
        words = re.findall(r'\b[a-z_][a-z0-9_]*\b', text.lower())
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'to', 'of', 'in', 'for',
            'on', 'with', 'and', 'or', 'this', 'that', 'it', 'be'
        }
        return set(w for w in words if w not in stop_words and len(w) > 2)

    def train(self, actions_dir: Path) -> Dict[str, Any]:
        """
        Train on Bash action history.

        Args:
            actions_dir: Path to .git-ml/actions/ directory

        Returns:
            Training statistics
        """
        stats = {
            'total_actions': 0,
            'bash_actions': 0,
            'python_c_commands': 0,
            'unique_commands': 0,
            'sessions': set()
        }

        command_templates: Dict[str, Dict] = defaultdict(lambda: {
            'count': 0, 'examples': [], 'keywords': set()
        })
        keyword_to_commands: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        command_frequency: Dict[str, int] = defaultdict(int)
        context_patterns: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        python_c_patterns: Dict[str, Dict] = {}

        # Process all action files
        if not actions_dir.exists():
            return {'error': f'Actions directory not found: {actions_dir}'}

        for date_dir in sorted(actions_dir.iterdir()):
            if not date_dir.is_dir():
                continue

            for action_file in sorted(date_dir.glob('A-*.json')):
                try:
                    with open(action_file) as f:
                        action = json.load(f)

                    stats['total_actions'] += 1
                    stats['sessions'].add(action.get('session_id', ''))

                    # Only process Bash actions
                    ctx = action.get('context', {})
                    if ctx.get('tool') != 'Bash':
                        continue

                    stats['bash_actions'] += 1

                    # Extract command
                    input_data = ctx.get('input', {})
                    if isinstance(input_data, dict):
                        input_data = input_data.get('input', input_data)

                    command = input_data.get('command', '') if isinstance(input_data, dict) else ''
                    if not command:
                        continue

                    # Normalize command for template matching
                    normalized = self._normalize_command(command)
                    command_frequency[normalized] += 1

                    # Track python -c patterns specifically
                    if 'python -c' in command or 'python3 -c' in command:
                        stats['python_c_commands'] += 1
                        python_c_patterns[normalized] = {
                            'count': python_c_patterns.get(normalized, {}).get('count', 0) + 1,
                            'example': command[:500],
                            'keywords': list(self._extract_keywords(command))
                        }

                    # Extract keywords from command and description
                    description = input_data.get('description', '') if isinstance(input_data, dict) else ''
                    keywords = self._extract_keywords(f"{command} {description}")

                    for kw in keywords:
                        keyword_to_commands[kw][normalized] += 1

                    command_templates[normalized]['count'] += 1
                    command_templates[normalized]['keywords'].update(keywords)
                    if len(command_templates[normalized]['examples']) < 3:
                        command_templates[normalized]['examples'].append(command[:200])

                except (json.JSONDecodeError, KeyError) as e:
                    continue

        # Convert to serializable format
        self.model_data = {
            'command_templates': {
                k: {
                    'count': v['count'],
                    'examples': v['examples'],
                    'keywords': list(v['keywords'])
                }
                for k, v in command_templates.items()
            },
            'keyword_to_commands': {k: dict(v) for k, v in keyword_to_commands.items()},
            'command_frequency': dict(command_frequency),
            'context_patterns': {k: dict(v) for k, v in context_patterns.items()},
            'python_c_patterns': python_c_patterns,
            'total_commands': stats['bash_actions']
        }

        self.trained_on_commits = stats['bash_actions']
        self.trained_on_sessions = len(stats['sessions'])

        stats['unique_commands'] = len(command_frequency)
        stats['sessions'] = len(stats['sessions'])

        return stats

    def _normalize_command(self, command: str) -> str:
        """
        Normalize command for template matching.

        Replaces specific paths/values with placeholders.
        """
        # Truncate very long commands
        if len(command) > 200:
            command = command[:200] + '...'

        # Replace common variable parts
        normalized = command
        normalized = re.sub(r'/home/\w+/', '/home/USER/', normalized)
        normalized = re.sub(r'--\w+=\S+', '--ARG=VALUE', normalized)

        return normalized

    def evaluate(self, test_actions: List[Dict]) -> ExpertMetrics:
        """
        Evaluate model on test actions.

        Args:
            test_actions: List of action dicts with 'query' and 'expected_command'

        Returns:
            ExpertMetrics with evaluation results
        """
        if not test_actions:
            return ExpertMetrics()

        reciprocal_ranks = []
        recall_at = {1: 0, 5: 0, 10: 0}
        precision_at = {1: 0}

        for action in test_actions:
            query = action.get('query', action.get('description', ''))
            expected = self._normalize_command(action.get('expected_command', action.get('command', '')))

            prediction = self.predict({'query': query, 'top_n': 10})
            predicted_commands = [self._normalize_command(cmd) for cmd, _ in prediction.items]

            # Find rank of expected command
            try:
                rank = predicted_commands.index(expected) + 1
                reciprocal_ranks.append(1.0 / rank)

                for k in recall_at.keys():
                    if rank <= k:
                        recall_at[k] += 1

                if rank == 1:
                    precision_at[1] += 1
            except ValueError:
                reciprocal_ranks.append(0.0)

        n = len(test_actions)
        metrics = ExpertMetrics(
            mrr=sum(reciprocal_ranks) / n if n > 0 else 0.0,
            recall_at_k={k: v / n for k, v in recall_at.items()},
            precision_at_k={k: v / n for k, v in precision_at.items()},
            test_examples=n,
            last_evaluated=self._now_iso()
        )

        self.metrics = metrics
        return metrics

    def _now_iso(self) -> str:
        """Get current time as ISO string."""
        from datetime import datetime
        return datetime.utcnow().isoformat() + 'Z'

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CommandExpert':
        """Load CommandExpert from dict."""
        metrics = None
        if data.get('metrics'):
            metrics = ExpertMetrics.from_dict(data['metrics'])

        return cls(
            expert_id=data.get('expert_id', 'command_expert'),
            version=data['version'],
            created_at=data['created_at'],
            trained_on_commits=data['trained_on_commits'],
            trained_on_sessions=data['trained_on_sessions'],
            git_hash=data['git_hash'],
            model_data=data['model_data'],
            metrics=metrics,
            calibration_curve=data.get('calibration_curve')
        )
