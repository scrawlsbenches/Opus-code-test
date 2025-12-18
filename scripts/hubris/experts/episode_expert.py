#!/usr/bin/env python3
"""
Episode Expert

Micro-expert specialized in learning from session transcripts to predict
next likely actions based on current context.

Learns episode patterns (state, action, outcome) from transcript history
to suggest what actions to take next in a similar context.
"""

import re
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from micro_expert import MicroExpert, ExpertPrediction, ExpertMetrics


class EpisodeExpert(MicroExpert):
    """
    Expert for predicting next actions from session transcript patterns.

    Learns from session transcripts:
    - Action sequences (tool1 → tool2 → tool3)
    - Context to action mappings (keywords → recommended tools)
    - Success patterns (context + action → positive outcome)
    - Failure patterns (context + action → negative outcome)

    Model Data Structure:
        action_sequences: Dict[str, Dict[str, int]] - Action -> follow-up action counts
        context_to_actions: Dict[str, Dict[str, int]] - Keyword -> action counts
        success_patterns: List[Dict] - Successful (context, action, outcome) episodes
        failure_patterns: List[Dict] - Failed episodes to avoid
        action_frequency: Dict[str, int] - Overall action frequency
        total_episodes: int - Total episodes in training data
    """

    # Common tool/action categories
    ACTION_CATEGORIES = {
        'read': ['Read', 'Grep', 'Glob'],
        'write': ['Write', 'Edit', 'MultiEdit', 'NotebookEdit'],
        'execute': ['Bash', 'BashOutput'],
        'search': ['Grep', 'Glob', 'WebSearch', 'WebFetch'],
        'organize': ['TodoWrite', 'Task', 'SlashCommand', 'Skill'],
    }

    def __init__(
        self,
        expert_id: str = "episode_expert",
        version: str = "1.0.0",
        **kwargs
    ):
        """
        Initialize EpisodeExpert.

        Args:
            expert_id: Unique identifier (default: "episode_expert")
            version: Expert version (default: "1.0.0")
            **kwargs: Additional arguments passed to MicroExpert base class
        """
        # Remove expert_type from kwargs if present (avoids conflict when loading)
        kwargs.pop('expert_type', None)
        super().__init__(
            expert_id=expert_id,
            expert_type="episode",
            version=version,
            **kwargs
        )

        # Ensure model_data has required keys
        if not self.model_data:
            self.model_data = {
                'action_sequences': {},
                'context_to_actions': {},
                'success_patterns': [],
                'failure_patterns': [],
                'action_frequency': {},
                'total_episodes': 0
            }

    def predict(self, context: Dict[str, Any]) -> ExpertPrediction:
        """
        Predict next likely actions based on current context.

        Args:
            context: Dictionary with:
                - query (str, optional): Current query/task description
                - last_actions (List[str], optional): Recently used actions/tools
                - files_touched (List[str], optional): Files recently accessed
                - current_state (str, optional): Current workflow state
                - top_n (int, optional): Number of predictions (default: 5)

        Returns:
            ExpertPrediction with ranked (action, confidence) pairs
        """
        query = context.get('query', '')
        last_actions = context.get('last_actions', [])
        files_touched = context.get('files_touched', [])
        current_state = context.get('current_state', '')
        top_n = context.get('top_n', 5)

        action_scores: Dict[str, float] = defaultdict(float)

        # Signal 1: Action sequence patterns
        if last_actions:
            seq_suggestions = self._predict_by_sequence(last_actions)
            for action, score in seq_suggestions.items():
                action_scores[action] += score * 2.5  # Strong weight

        # Signal 2: Context keyword matching
        keywords = self._extract_keywords(query)
        context_suggestions = self._predict_by_context(keywords)
        for action, score in context_suggestions.items():
            action_scores[action] += score * 2.0

        # Signal 3: File type patterns
        if files_touched:
            file_suggestions = self._predict_by_files(files_touched)
            for action, score in file_suggestions.items():
                action_scores[action] += score * 1.5

        # Signal 4: Success pattern matching
        success_suggestions = self._predict_from_success_patterns(query, last_actions)
        for action, score in success_suggestions.items():
            action_scores[action] += score * 1.8

        # Signal 5: Failure pattern avoidance (negative weight)
        failure_actions = self._get_failure_actions(query, last_actions)
        for action in failure_actions:
            action_scores[action] -= 0.5  # Penalize actions that previously failed

        # Normalize scores
        if action_scores:
            max_score = max(action_scores.values())
            if max_score > 0:
                action_scores = {k: v / max_score for k, v in action_scores.items()}

        # Sort and filter
        sorted_actions = sorted(action_scores.items(), key=lambda x: -x[1])
        items = [(action, max(0.0, score)) for action, score in sorted_actions[:top_n] if score > 0]

        metadata = {
            'keywords': list(keywords),
            'sequence_matched': len(last_actions) > 0,
            'success_patterns_used': len(self.model_data.get('success_patterns', [])),
            'total_episodes': self.model_data.get('total_episodes', 0)
        }

        return ExpertPrediction(
            expert_id=self.expert_id,
            expert_type=self.expert_type,
            items=items,
            metadata=metadata
        )

    def _predict_by_sequence(self, last_actions: List[str]) -> Dict[str, float]:
        """Predict actions based on recent action sequence."""
        predictions: Dict[str, float] = defaultdict(float)
        action_sequences = self.model_data.get('action_sequences', {})

        # Use last 1-3 actions for prediction
        for i in range(min(3, len(last_actions))):
            prev_action = last_actions[-(i+1)]
            if prev_action in action_sequences:
                follow_ups = action_sequences[prev_action]
                total = sum(follow_ups.values())

                for action, count in follow_ups.items():
                    # Weight decreases with distance (recent actions matter more)
                    weight = 1.0 / (i + 1)
                    predictions[action] += (count / total) * weight

        return predictions

    def _predict_by_context(self, keywords: Set[str]) -> Dict[str, float]:
        """Predict actions based on context keywords."""
        predictions: Dict[str, float] = defaultdict(float)
        context_to_actions = self.model_data.get('context_to_actions', {})

        for keyword in keywords:
            if keyword in context_to_actions:
                actions = context_to_actions[keyword]
                total = sum(actions.values())

                for action, count in actions.items():
                    predictions[action] += count / total if total > 0 else 0

        # Normalize by number of matching keywords
        if keywords:
            predictions = {k: v / len(keywords) for k, v in predictions.items()}

        return predictions

    def _predict_by_files(self, files_touched: List[str]) -> Dict[str, float]:
        """Predict actions based on file types recently touched."""
        predictions: Dict[str, float] = defaultdict(float)

        for filepath in files_touched:
            # Predict based on file extension
            if filepath.endswith('.py'):
                predictions['Read'] += 0.3
                predictions['Edit'] += 0.2
                predictions['Bash'] += 0.1  # Might run tests
            elif filepath.endswith('.md'):
                predictions['Edit'] += 0.3
                predictions['Read'] += 0.2
            elif filepath.endswith(('.json', '.yaml', '.yml', '.toml')):
                predictions['Read'] += 0.3
                predictions['Edit'] += 0.2
            elif filepath.endswith(('.sh', '.bash')):
                predictions['Bash'] += 0.4
                predictions['Read'] += 0.2

            # Predict based on file location
            if 'test' in filepath.lower():
                predictions['Bash'] += 0.2  # Likely to run tests
            if 'scripts/' in filepath:
                predictions['Bash'] += 0.2

        return predictions

    def _predict_from_success_patterns(
        self,
        query: str,
        last_actions: List[str]
    ) -> Dict[str, float]:
        """Predict based on successful pattern matches."""
        predictions: Dict[str, float] = defaultdict(float)
        success_patterns = self.model_data.get('success_patterns', [])

        query_words = set(query.lower().split()) if query else set()

        for pattern in success_patterns[-100:]:  # Check last 100 patterns
            pattern_context = pattern.get('context', '')
            pattern_actions = pattern.get('actions', [])
            pattern_words = set(pattern_context.lower().split())

            # Calculate context similarity
            if query_words and pattern_words:
                similarity = len(query_words & pattern_words) / len(query_words | pattern_words)

                if similarity > 0.2:  # Minimum similarity threshold
                    for action in pattern_actions:
                        predictions[action] += similarity

        return predictions

    def _get_failure_actions(
        self,
        query: str,
        last_actions: List[str]
    ) -> Set[str]:
        """Get actions that have failed in similar contexts."""
        failure_actions = set()
        failure_patterns = self.model_data.get('failure_patterns', [])

        query_words = set(query.lower().split()) if query else set()

        for pattern in failure_patterns[-50:]:  # Check last 50 failures
            pattern_context = pattern.get('context', '')
            pattern_actions = pattern.get('actions', [])
            pattern_words = set(pattern_context.lower().split())

            # Check if context is similar
            if query_words and pattern_words:
                similarity = len(query_words & pattern_words) / len(query_words | pattern_words)

                if similarity > 0.3:  # Higher threshold for failures
                    failure_actions.update(pattern_actions)

        return failure_actions

    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract meaningful keywords from text."""
        if not text:
            return set()

        # Lowercase and extract words
        words = re.findall(r'\b[a-z_][a-z0-9_]*\b', text.lower())

        # Filter stop words
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them',
            'and', 'or', 'but', 'not', 'if', 'then', 'else', 'when', 'where',
            'what', 'how', 'why', 'who', 'which'
        }

        return set(w for w in words if w not in stop_words and len(w) > 2)

    def train(self, episodes: List[Dict[str, Any]]) -> None:
        """
        Train the expert on episode data.

        Args:
            episodes: List of episode dictionaries with:
                - context (str): The context/query for this episode
                - actions (List[str]): Actions taken in sequence
                - outcome (str): 'success', 'failure', or 'partial'
                - files (List[str], optional): Files accessed/modified
        """
        action_sequences: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        context_to_actions: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        action_frequency: Dict[str, int] = defaultdict(int)
        success_patterns: List[Dict] = []
        failure_patterns: List[Dict] = []

        for episode in episodes:
            context = episode.get('context', '')
            actions = episode.get('actions', [])
            outcome = episode.get('outcome', 'unknown')
            files = episode.get('files', [])

            # Extract keywords from context
            keywords = self._extract_keywords(context)

            # Build action sequence patterns
            for i in range(len(actions)):
                action = actions[i]
                action_frequency[action] += 1

                # Map keywords to actions
                for keyword in keywords:
                    context_to_actions[keyword][action] += 1

                # Map action to next action
                if i < len(actions) - 1:
                    next_action = actions[i + 1]
                    action_sequences[action][next_action] += 1

            # Store success/failure patterns
            pattern = {
                'context': context[:500],  # Limit length
                'actions': actions[:10],  # Limit actions
                'files': files[:10],  # Limit files
                'timestamp': episode.get('timestamp', '')
            }

            if outcome == 'success':
                success_patterns.append(pattern)
            elif outcome == 'failure':
                failure_patterns.append(pattern)

        # Store in model_data
        self.model_data = {
            'action_sequences': {k: dict(v) for k, v in action_sequences.items()},
            'context_to_actions': {k: dict(v) for k, v in context_to_actions.items()},
            'success_patterns': success_patterns[-1000:],  # Keep last 1000
            'failure_patterns': failure_patterns[-500:],  # Keep last 500
            'action_frequency': dict(action_frequency),
            'total_episodes': len(episodes)
        }

    @staticmethod
    def extract_episodes(transcript_exchanges: List[Any]) -> List[Dict[str, Any]]:
        """
        Extract episodes from transcript exchanges.

        Args:
            transcript_exchanges: List of TranscriptExchange objects or dicts

        Returns:
            List of episode dicts ready for training
        """
        episodes = []

        for exchange in transcript_exchanges:
            # Handle both object and dict formats
            if hasattr(exchange, 'query'):
                query = exchange.query
                actions = exchange.tools_used
                timestamp = exchange.timestamp
                tool_inputs = getattr(exchange, 'tool_inputs', [])
            else:
                query = exchange.get('query', '')
                actions = exchange.get('tools_used', [])
                timestamp = exchange.get('timestamp', '')
                tool_inputs = exchange.get('tool_inputs', [])

            # Extract files from tool inputs
            files = []
            for ti in tool_inputs:
                if isinstance(ti, dict):
                    inp = ti.get('input', {})
                    if 'file_path' in inp:
                        files.append(inp['file_path'])
                    elif 'notebook_path' in inp:
                        files.append(inp['notebook_path'])

            # Determine outcome heuristically
            # If actions were taken, assume partial success at minimum
            outcome = 'success' if actions else 'unknown'

            # Check for error indicators in query
            error_keywords = {'error', 'fail', 'bug', 'issue', 'problem', 'wrong', 'broken'}
            if any(kw in query.lower() for kw in error_keywords):
                # Debugging episode - outcome depends on resolution
                outcome = 'partial'

            episodes.append({
                'context': query,
                'actions': actions,
                'outcome': outcome,
                'files': files,
                'timestamp': timestamp
            })

        return episodes

    def get_action_stats(self) -> Dict[str, Any]:
        """
        Get statistics about learned action patterns.

        Returns:
            Dictionary with action statistics
        """
        action_sequences = self.model_data.get('action_sequences', {})
        action_frequency = self.model_data.get('action_frequency', {})

        # Find most common sequences
        common_sequences = []
        for action, follow_ups in action_sequences.items():
            for next_action, count in follow_ups.items():
                common_sequences.append((action, next_action, count))

        common_sequences.sort(key=lambda x: -x[2])

        # Find most frequent actions
        frequent_actions = sorted(
            action_frequency.items(),
            key=lambda x: -x[1]
        )

        return {
            'total_episodes': self.model_data.get('total_episodes', 0),
            'unique_actions': len(action_frequency),
            'success_patterns': len(self.model_data.get('success_patterns', [])),
            'failure_patterns': len(self.model_data.get('failure_patterns', [])),
            'top_sequences': common_sequences[:10],
            'top_actions': frequent_actions[:10]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EpisodeExpert':
        """
        Load EpisodeExpert from dict.

        Args:
            data: Dictionary representation

        Returns:
            EpisodeExpert instance
        """
        metrics = None
        if data.get('metrics'):
            metrics = ExpertMetrics.from_dict(data['metrics'])

        return cls(
            expert_id=data.get('expert_id', 'episode_expert'),
            version=data['version'],
            created_at=data['created_at'],
            trained_on_commits=data['trained_on_commits'],
            trained_on_sessions=data['trained_on_sessions'],
            git_hash=data['git_hash'],
            model_data=data['model_data'],
            metrics=metrics,
            calibration_curve=data.get('calibration_curve')
        )


if __name__ == '__main__':
    # Demo usage
    expert = EpisodeExpert()

    # Example episodes for training
    episodes = [
        {
            'context': 'Fix bug in authentication module',
            'actions': ['Read', 'Grep', 'Edit', 'Bash'],
            'outcome': 'success',
            'files': ['auth.py', 'tests/test_auth.py']
        },
        {
            'context': 'Add new feature for user registration',
            'actions': ['Read', 'Write', 'Edit', 'TodoWrite', 'Bash'],
            'outcome': 'success',
            'files': ['registration.py', 'tests/test_registration.py']
        },
        {
            'context': 'Debug test failure',
            'actions': ['Read', 'Bash', 'Read', 'Edit', 'Bash'],
            'outcome': 'success',
            'files': ['tests/test_foo.py']
        }
    ]

    # Train
    expert.train(episodes)

    # Predict next action
    prediction = expert.predict({
        'query': 'Fix authentication bug',
        'last_actions': ['Read', 'Grep'],
        'files_touched': ['auth.py']
    })

    print("\nNext action predictions:")
    for action, confidence in prediction.items[:5]:
        print(f"  {action}: {confidence:.3f}")

    print("\nAction statistics:")
    stats = expert.get_action_stats()
    print(f"  Total episodes: {stats['total_episodes']}")
    print(f"  Unique actions: {stats['unique_actions']}")
    print(f"  Success patterns: {stats['success_patterns']}")
    print(f"  Top sequences:")
    for a1, a2, count in stats['top_sequences'][:5]:
        print(f"    {a1} → {a2} ({count}x)")
