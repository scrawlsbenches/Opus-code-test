"""
PRISM-SLM: Statistical Language Model with Synaptic Learning.

A biologically-inspired language model that treats word transitions as
synaptic connections. Connections strengthen through use (Hebbian learning)
and decay when unused.

Key concepts:
- SynapticTransition: Connection between tokens that learns from usage
- ContextWindow: Sliding window of recent tokens for context
- TransitionGraph: Network of all token transitions
- PRISMLanguageModel: Main model for training and generation

Example:
    model = PRISMLanguageModel(context_size=3)
    model.train("The quick brown fox jumps over the lazy dog.")
    generated = model.generate("The quick", max_tokens=10)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Iterator, Any
from collections import defaultdict
import json
import math
import random
import re


@dataclass
class SynapticTransition:
    """
    A transition between tokens that learns from usage.

    Inspired by synaptic plasticity - connections strengthen with use
    and decay when unused.
    """
    from_token: str
    to_token: str
    weight: float = 1.0
    count: int = 0
    decay_rate: float = 0.99

    def observe(self, amount: float = 0.1) -> None:
        """Record an observation of this transition (Hebbian strengthening)."""
        self.count += 1
        self.weight += amount

    def apply_decay(self, factor: Optional[float] = None) -> None:
        """Apply decay to this transition."""
        decay = factor if factor is not None else self.decay_rate
        self.weight *= decay

    def probability(self, total_weight: float) -> float:
        """Compute probability given total outgoing weight."""
        if total_weight <= 0:
            return 0.0
        return self.weight / total_weight

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "from_token": self.from_token,
            "to_token": self.to_token,
            "weight": self.weight,
            "count": self.count,
            "decay_rate": self.decay_rate,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SynapticTransition":
        """Deserialize from dictionary."""
        return cls(
            from_token=data["from_token"],
            to_token=data["to_token"],
            weight=data.get("weight", 1.0),
            count=data.get("count", 0),
            decay_rate=data.get("decay_rate", 0.99),
        )


class ContextWindow:
    """
    A sliding window of recent tokens for context tracking.

    Maintains a fixed-size window that slides as new tokens are added.
    """

    def __init__(self, size: int = 3):
        self.size = size
        self._tokens: List[str] = []

    def add(self, token: str) -> None:
        """Add a token to the context."""
        self._tokens.append(token)
        if len(self._tokens) > self.size:
            self._tokens.pop(0)

    def clear(self) -> None:
        """Clear the context."""
        self._tokens = []

    def as_key(self) -> Tuple[str, ...]:
        """Get context as hashable key."""
        return tuple(self._tokens)

    def __len__(self) -> int:
        return len(self._tokens)

    def __iter__(self) -> Iterator[str]:
        return iter(self._tokens)

    def __repr__(self) -> str:
        return f"ContextWindow({list(self._tokens)})"


class TransitionGraph:
    """
    Graph of token transitions with synaptic learning.

    Stores all observed transitions and provides lookup by context.
    """

    def __init__(self, context_size: int = 2):
        self.context_size = context_size
        # Map from context tuple to list of transitions
        self._transitions: Dict[Tuple[str, ...], List[SynapticTransition]] = defaultdict(list)
        self._vocab: set = set()
        self._total_tokens: int = 0

    @property
    def token_count(self) -> int:
        """Total number of tokens seen."""
        return self._total_tokens

    @property
    def transition_count(self) -> int:
        """Total number of unique transitions."""
        return sum(len(t) for t in self._transitions.values())

    @property
    def vocab_size(self) -> int:
        """Number of unique tokens."""
        return len(self._vocab)

    def learn_sequence(self, tokens: List[str]) -> None:
        """Learn transitions from a sequence of tokens."""
        if len(tokens) < 2:
            return

        # Add all tokens to vocab
        for token in tokens:
            self._vocab.add(token)
            self._total_tokens += 1

        # Learn transitions for each context size from 1 to context_size
        for ctx_len in range(1, self.context_size + 1):
            for i in range(len(tokens) - 1):
                # Get context (previous tokens)
                start = max(0, i - ctx_len + 1)
                context = tuple(tokens[start:i + 1])
                next_token = tokens[i + 1]

                # Find or create transition
                self._observe_transition(context, next_token)

    def _observe_transition(self, context: Tuple[str, ...], next_token: str) -> None:
        """Record a transition observation."""
        transitions = self._transitions[context]

        # Look for existing transition
        for trans in transitions:
            if trans.to_token == next_token:
                trans.observe()
                return

        # Create new transition
        trans = SynapticTransition(
            from_token=context[-1] if context else "",
            to_token=next_token,
        )
        trans.observe()
        transitions.append(trans)

    def get_transitions(self, context: Tuple[str, ...]) -> List[SynapticTransition]:
        """Get all transitions from a context."""
        return self._transitions.get(context, [])

    def apply_decay(self, factor: float = 0.99) -> None:
        """Apply decay to all transitions."""
        for transitions in self._transitions.values():
            for trans in transitions:
                trans.apply_decay(factor)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        transitions_data = {}
        for context, trans_list in self._transitions.items():
            key = "|".join(context)
            transitions_data[key] = [t.to_dict() for t in trans_list]

        return {
            "context_size": self.context_size,
            "vocab": list(self._vocab),
            "total_tokens": self._total_tokens,
            "transitions": transitions_data,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TransitionGraph":
        """Deserialize from dictionary."""
        graph = cls(context_size=data.get("context_size", 2))
        graph._vocab = set(data.get("vocab", []))
        graph._total_tokens = data.get("total_tokens", 0)

        for key, trans_list in data.get("transitions", {}).items():
            context = tuple(key.split("|")) if key else ()
            graph._transitions[context] = [
                SynapticTransition.from_dict(t) for t in trans_list
            ]

        return graph


class PRISMLanguageModel:
    """
    Statistical Language Model with Synaptic Learning.

    Inspired by PRISM-GoT, this model treats word transitions as
    synaptic connections that strengthen through use and decay when unused.

    Features:
    - N-gram-like context modeling with variable context size
    - Hebbian learning for transition strengthening
    - Temperature-controlled sampling for generation
    - Reward-based path strengthening
    - Perplexity computation for evaluation
    """

    def __init__(self, context_size: int = 3):
        self.context_size = context_size
        self.graph = TransitionGraph(context_size=context_size)
        self._generation_path: List[str] = []

    @property
    def vocab_size(self) -> int:
        """Number of unique tokens in vocabulary."""
        return self.graph.vocab_size

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Simple word tokenization
        text = text.lower()
        # Keep alphanumeric and basic punctuation
        tokens = re.findall(r"[a-z]+(?:'[a-z]+)?|[.,!?;:]", text)
        return tokens

    def train(self, text: str) -> None:
        """Train the model on a text."""
        tokens = self._tokenize(text)
        if tokens:
            self.graph.learn_sequence(tokens)

    def generate_next(
        self,
        context: List[str],
        temperature: float = 1.0,
    ) -> Optional[str]:
        """
        Generate the next token given context.

        Args:
            context: List of previous tokens
            temperature: Sampling temperature (lower = more deterministic)

        Returns:
            Next token or None if no transitions available
        """
        # Normalize context
        context = [t.lower() for t in context]

        # Try progressively shorter contexts until we find transitions
        for ctx_len in range(min(len(context), self.context_size), 0, -1):
            ctx_tuple = tuple(context[-ctx_len:])
            transitions = self.graph.get_transitions(ctx_tuple)

            if transitions:
                return self._sample_transition(transitions, temperature)

        # Fallback: random word from vocabulary
        if self.graph._vocab:
            return random.choice(list(self.graph._vocab))

        return None

    def _sample_transition(
        self,
        transitions: List[SynapticTransition],
        temperature: float = 1.0,
    ) -> str:
        """Sample a token from transitions using temperature scaling."""
        if not transitions:
            return ""

        if temperature <= 0.01:
            # Greedy: pick highest weight
            return max(transitions, key=lambda t: t.weight).to_token

        # Compute softmax with temperature
        weights = [t.weight for t in transitions]
        max_weight = max(weights)

        # Apply temperature scaling
        scaled = [(w / max_weight) ** (1.0 / temperature) for w in weights]
        total = sum(scaled)
        probs = [s / total for s in scaled]

        # Sample
        r = random.random()
        cumsum = 0.0
        for i, p in enumerate(probs):
            cumsum += p
            if r <= cumsum:
                return transitions[i].to_token

        return transitions[-1].to_token

    def generate(
        self,
        prompt: str = "",
        max_tokens: int = 50,
        temperature: float = 1.0,
        stop_tokens: Optional[List[str]] = None,
        return_path: bool = False,
    ) -> Any:
        """
        Generate text from a prompt.

        Args:
            prompt: Starting text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop_tokens: Tokens that end generation
            return_path: If True, return dict with text and generation path

        Returns:
            Generated text string, or dict if return_path=True
        """
        stop_tokens = stop_tokens or [".", "!", "?"]

        # Tokenize prompt
        context = self._tokenize(prompt) if prompt else []
        generated = list(context)
        self._generation_path = list(context)

        for _ in range(max_tokens):
            next_token = self.generate_next(generated, temperature)

            if next_token is None:
                break

            generated.append(next_token)
            self._generation_path.append(next_token)

            if next_token in stop_tokens:
                break

        # Reconstruct text
        text = self._tokens_to_text(generated)

        if return_path:
            return {
                "text": text,
                "path": self._generation_path.copy(),
                "tokens": generated,
            }

        return text

    def _tokens_to_text(self, tokens: List[str]) -> str:
        """Convert tokens back to text."""
        if not tokens:
            return ""

        result = tokens[0].capitalize()
        for token in tokens[1:]:
            if token in ".,!?;:":
                result += token
            else:
                result += " " + token

        return result

    def apply_decay(self, factor: float = 0.99) -> None:
        """Apply decay to all transitions."""
        self.graph.apply_decay(factor)

    def reward_path(self, path: List[str], reward: float = 1.0) -> None:
        """
        Reward a generation path (reinforcement learning).

        Positive reward strengthens transitions, negative weakens them.
        """
        if len(path) < 2:
            return

        path = [t.lower() for t in path]

        for i in range(len(path) - 1):
            for ctx_len in range(1, min(i + 2, self.context_size + 1)):
                start = max(0, i - ctx_len + 1)
                context = tuple(path[start:i + 1])
                next_token = path[i + 1]

                transitions = self.graph.get_transitions(context)
                for trans in transitions:
                    if trans.to_token == next_token:
                        if reward > 0:
                            trans.weight += reward * 0.1
                        else:
                            trans.weight = max(0.1, trans.weight + reward * 0.1)

    def perplexity(self, text: str) -> float:
        """
        Compute perplexity of text under the model.

        Lower perplexity = better fit to the model.
        """
        tokens = self._tokenize(text)
        if len(tokens) < 2:
            return float("inf")

        log_prob_sum = 0.0
        count = 0

        for i in range(1, len(tokens)):
            # Try progressively shorter contexts
            prob = 0.0
            for ctx_len in range(min(i, self.context_size), 0, -1):
                context = tuple(tokens[i - ctx_len:i])
                transitions = self.graph.get_transitions(context)

                if transitions:
                    total_weight = sum(t.weight for t in transitions)
                    for trans in transitions:
                        if trans.to_token == tokens[i]:
                            prob = trans.probability(total_weight)
                            break
                    if prob > 0:
                        break

            if prob > 0:
                log_prob_sum += math.log(prob)
            else:
                # Smoothing for unseen transitions
                log_prob_sum += math.log(1e-10)

            count += 1

        if count == 0:
            return float("inf")

        avg_log_prob = log_prob_sum / count
        return math.exp(-avg_log_prob)

    def save(self, path: str) -> None:
        """Save model to file."""
        data = {
            "context_size": self.context_size,
            "graph": self.graph.to_dict(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "PRISMLanguageModel":
        """Load model from file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        model = cls(context_size=data.get("context_size", 3))
        model.graph = TransitionGraph.from_dict(data.get("graph", {}))

        return model

    def get_stats(self) -> Dict[str, Any]:
        """Get model statistics."""
        return {
            "vocab_size": self.vocab_size,
            "context_size": self.context_size,
            "transition_count": self.graph.transition_count,
            "token_count": self.graph.token_count,
        }
