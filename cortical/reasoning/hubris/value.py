"""
ValueSignal: Reward-Based Learning for Expert Behavior.

ValueSignals guide expert behavior through reinforcement learning principles.
When an action or prediction leads to a good outcome, its value increases;
when it leads to a bad outcome, its value decreases.

Design Philosophy:
    Experts should learn from outcomes. A value signal encodes which
    approaches tend to work and which tend to fail, enabling data-driven
    improvement over time.

Key Concepts:
    - Value: Expected utility of an action/approach
    - Reward: Immediate feedback from an outcome
    - Temporal Difference: Learning from prediction errors
    - Exploration vs Exploitation: Balancing known-good vs new approaches

Example:
    >>> signal = ValueSignal(learning_rate=0.1)
    >>> signal.reward("approach_A", 1.0)  # Positive outcome
    >>> signal.reward("approach_B", -0.5)  # Negative outcome
    >>> print(signal.get_value("approach_A"))  # Higher than B
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import math
import random


@dataclass
class ValueRecord:
    """
    Record of a value signal update.

    Tracks the history of value changes for analysis and debugging.
    """
    action: str
    reward: float
    old_value: float
    new_value: float
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)


class ValueSignal:
    """
    Value signal system for learning from outcomes.

    Maintains value estimates for actions/approaches and updates them
    based on reward signals. Uses temporal difference learning principles.

    Learning Rule:
        V(a) <- V(a) + alpha * (reward - V(a))

    Where:
        - V(a) is the value of action a
        - alpha is the learning rate
        - reward is the immediate feedback

    Example:
        >>> signal = ValueSignal(learning_rate=0.1)
        >>> # Approach A works well
        >>> signal.reward("approach_A", reward=1.0)
        >>> signal.reward("approach_A", reward=0.8)
        >>> # Approach B doesn't work as well
        >>> signal.reward("approach_B", reward=0.2)
        >>> signal.reward("approach_B", reward=-0.3)
        >>> # Check values
        >>> print(signal.get_value("approach_A"))  # ~0.18
        >>> print(signal.get_value("approach_B"))  # ~-0.01
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        initial_value: float = 0.0,
        decay_rate: float = 0.01,
        enable_decay: bool = False,
    ):
        """
        Initialize the value signal system.

        Args:
            learning_rate: How fast values update (0.0 to 1.0)
            initial_value: Default value for new actions
            decay_rate: Rate at which values decay toward initial (if enabled)
            enable_decay: Whether to decay values over time
        """
        self.learning_rate = learning_rate
        self.initial_value = initial_value
        self.decay_rate = decay_rate
        self.enable_decay = enable_decay

        self._values: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}  # Number of updates per action
        self._history: List[ValueRecord] = []
        self._last_update: Dict[str, datetime] = {}

    def get_value(self, action: str) -> float:
        """
        Get the current value for an action.

        If decay is enabled, applies time-based decay before returning.

        Args:
            action: The action/approach to get value for

        Returns:
            Current value estimate
        """
        if action not in self._values:
            return self.initial_value

        value = self._values[action]

        # Apply decay if enabled
        if self.enable_decay and action in self._last_update:
            elapsed = (datetime.now() - self._last_update[action]).total_seconds()
            decay_factor = math.exp(-self.decay_rate * elapsed / 3600)  # Hourly decay
            value = self.initial_value + (value - self.initial_value) * decay_factor

        return value

    def reward(
        self,
        action: str,
        reward: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Apply a reward signal to update an action's value.

        Uses the TD(0) update rule: V(a) <- V(a) + alpha * (reward - V(a))

        Args:
            action: The action that received the reward
            reward: The reward value (positive or negative)
            context: Optional context for logging

        Returns:
            New value after update
        """
        old_value = self.get_value(action)

        # TD(0) update
        new_value = old_value + self.learning_rate * (reward - old_value)

        # Store updated value
        self._values[action] = new_value
        self._counts[action] = self._counts.get(action, 0) + 1
        self._last_update[action] = datetime.now()

        # Record history
        self._history.append(ValueRecord(
            action=action,
            reward=reward,
            old_value=old_value,
            new_value=new_value,
            context=context or {},
        ))

        return new_value

    def batch_reward(
        self,
        updates: List[Tuple[str, float]],
    ) -> Dict[str, float]:
        """
        Apply multiple reward signals at once.

        Args:
            updates: List of (action, reward) tuples

        Returns:
            Dictionary of action -> new_value
        """
        results = {}
        for action, reward in updates:
            results[action] = self.reward(action, reward)
        return results

    def get_best_actions(self, n: int = 5) -> List[Tuple[str, float]]:
        """
        Get the top n actions by value.

        Args:
            n: Number of actions to return

        Returns:
            List of (action, value) tuples, sorted by value descending
        """
        all_values = [(a, self.get_value(a)) for a in self._values]
        return sorted(all_values, key=lambda x: x[1], reverse=True)[:n]

    def get_worst_actions(self, n: int = 5) -> List[Tuple[str, float]]:
        """Get the bottom n actions by value."""
        all_values = [(a, self.get_value(a)) for a in self._values]
        return sorted(all_values, key=lambda x: x[1])[:n]

    def select_action(
        self,
        candidates: List[str],
        exploration_rate: float = 0.1,
    ) -> str:
        """
        Select an action using epsilon-greedy exploration.

        With probability (1 - exploration_rate), selects the highest-value action.
        With probability exploration_rate, selects randomly for exploration.

        Args:
            candidates: List of candidate actions
            exploration_rate: Probability of exploring (0.0 to 1.0)

        Returns:
            Selected action
        """
        if not candidates:
            raise ValueError("No candidate actions provided")

        # Explore
        if random.random() < exploration_rate:
            return random.choice(candidates)

        # Exploit
        values = [(a, self.get_value(a)) for a in candidates]
        return max(values, key=lambda x: x[1])[0]

    def select_action_softmax(
        self,
        candidates: List[str],
        temperature: float = 1.0,
    ) -> str:
        """
        Select an action using softmax (Boltzmann) exploration.

        Higher temperature = more exploration, lower = more exploitation.

        Args:
            candidates: List of candidate actions
            temperature: Softmax temperature

        Returns:
            Selected action
        """
        if not candidates:
            raise ValueError("No candidate actions provided")

        # Get values
        values = [self.get_value(a) for a in candidates]

        # Compute softmax probabilities
        max_val = max(values)  # Numerical stability
        exp_values = [math.exp((v - max_val) / temperature) for v in values]
        total = sum(exp_values)
        probs = [e / total for e in exp_values]

        # Sample
        r = random.random()
        cumsum = 0.0
        for action, prob in zip(candidates, probs):
            cumsum += prob
            if r < cumsum:
                return action

        return candidates[-1]  # Fallback

    def get_update_count(self, action: str) -> int:
        """Get number of times an action has been updated."""
        return self._counts.get(action, 0)

    def get_confidence(self, action: str) -> float:
        """
        Estimate confidence in value based on update count.

        More updates = higher confidence.

        Args:
            action: Action to check

        Returns:
            Confidence score (0.0 to 1.0)
        """
        count = self.get_update_count(action)
        # Asymptotic approach to 1.0
        return 1.0 - math.exp(-count / 10.0)

    def get_history(
        self,
        action: Optional[str] = None,
        limit: int = 100,
    ) -> List[ValueRecord]:
        """
        Get value update history.

        Args:
            action: Filter by action (optional)
            limit: Maximum records to return

        Returns:
            List of ValueRecords
        """
        if action:
            filtered = [r for r in self._history if r.action == action]
        else:
            filtered = self._history

        return filtered[-limit:]

    def reset(self, action: Optional[str] = None) -> None:
        """
        Reset value signal(s).

        Args:
            action: Specific action to reset (or None for all)
        """
        if action:
            if action in self._values:
                del self._values[action]
            if action in self._counts:
                del self._counts[action]
            if action in self._last_update:
                del self._last_update[action]
        else:
            self._values.clear()
            self._counts.clear()
            self._last_update.clear()
            self._history.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get value signal statistics."""
        if not self._values:
            return {
                'total_actions': 0,
                'total_updates': 0,
                'mean_value': self.initial_value,
            }

        values = list(self._values.values())
        return {
            'total_actions': len(self._values),
            'total_updates': sum(self._counts.values()),
            'mean_value': sum(values) / len(values),
            'max_value': max(values),
            'min_value': min(values),
            'std_value': (
                (sum((v - sum(values)/len(values))**2 for v in values) / len(values)) ** 0.5
                if len(values) > 1 else 0.0
            ),
        }


class HierarchicalValueSignal:
    """
    Hierarchical value signal for multi-level action spaces.

    Supports actions organized in a hierarchy (e.g., category -> subcategory -> action).
    Values propagate up and down the hierarchy.

    Example:
        >>> hvs = HierarchicalValueSignal()
        >>> hvs.reward("code/python/parsing", 1.0)
        >>> # Affects "code", "code/python", and "code/python/parsing"
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        propagation_factor: float = 0.5,
        separator: str = "/",
    ):
        """
        Initialize hierarchical value signal.

        Args:
            learning_rate: Learning rate for direct updates
            propagation_factor: How much updates propagate to parent/child
            separator: Character separating hierarchy levels
        """
        self.learning_rate = learning_rate
        self.propagation_factor = propagation_factor
        self.separator = separator
        self._signal = ValueSignal(learning_rate=learning_rate)

    def _get_ancestors(self, action: str) -> List[str]:
        """Get all ancestor actions in hierarchy."""
        parts = action.split(self.separator)
        ancestors = []
        for i in range(1, len(parts)):
            ancestors.append(self.separator.join(parts[:i]))
        return ancestors

    def reward(
        self,
        action: str,
        reward: float,
        propagate: bool = True,
    ) -> float:
        """
        Apply reward with optional hierarchy propagation.

        Args:
            action: The action that received reward
            reward: The reward value
            propagate: Whether to propagate to ancestors

        Returns:
            New value for the action
        """
        # Update the action directly
        new_value = self._signal.reward(action, reward)

        # Propagate to ancestors with decay
        if propagate:
            ancestors = self._get_ancestors(action)
            for i, ancestor in enumerate(reversed(ancestors)):
                # Reward decays as we go up the hierarchy
                propagated_reward = reward * (self.propagation_factor ** (i + 1))
                self._signal.reward(ancestor, propagated_reward)

        return new_value

    def get_value(self, action: str) -> float:
        """Get value for an action."""
        return self._signal.get_value(action)

    def select_action(
        self,
        candidates: List[str],
        exploration_rate: float = 0.1,
    ) -> str:
        """Select action using epsilon-greedy."""
        return self._signal.select_action(candidates, exploration_rate)

    def get_subtree_values(self, prefix: str) -> Dict[str, float]:
        """Get all values under a hierarchy prefix."""
        return {
            action: value
            for action, value in self._signal._values.items()
            if action.startswith(prefix)
        }
