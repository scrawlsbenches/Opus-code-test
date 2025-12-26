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


@dataclass
class HiveNode:
    """
    A node in the Hebbian Hive with activation traces.

    Tracks activation history for homeostatic regulation and
    maintains excitability for modulating node responsiveness.

    Part of Sprint 2: Hebbian Hive Enhancement (Woven Mind + PRISM Marriage)
    """
    id: str
    activation: float = 0.0

    # Eligibility trace for temporal credit assignment
    trace: float = 0.0
    trace_decay: float = 0.95

    # Homeostatic regulation (integrates with HomeostasisRegulator)
    target_activation: float = 0.05
    excitability: float = 1.0

    # Statistics
    activation_count: int = 0
    last_activation_step: int = -1

    def activate(self, amount: float = 1.0, step: int = 0) -> float:
        """
        Activate this node with given amount.

        Args:
            amount: Activation strength (0.0-1.0).
            step: Current time step for temporal tracking.

        Returns:
            Actual activation after excitability modulation.
        """
        # Apply excitability modulation
        modulated = amount * self.excitability

        # Update state
        self.activation = modulated
        self.trace = 1.0  # Reset trace on activation
        self.activation_count += 1
        self.last_activation_step = step

        return modulated

    def decay_trace(self) -> None:
        """Apply decay to the eligibility trace."""
        self.trace *= self.trace_decay

    def reset(self) -> None:
        """Reset activation state."""
        self.activation = 0.0
        self.trace = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "activation": self.activation,
            "trace": self.trace,
            "trace_decay": self.trace_decay,
            "target_activation": self.target_activation,
            "excitability": self.excitability,
            "activation_count": self.activation_count,
            "last_activation_step": self.last_activation_step,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HiveNode":
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            activation=data.get("activation", 0.0),
            trace=data.get("trace", 0.0),
            trace_decay=data.get("trace_decay", 0.95),
            target_activation=data.get("target_activation", 0.05),
            excitability=data.get("excitability", 1.0),
            activation_count=data.get("activation_count", 0),
            last_activation_step=data.get("last_activation_step", -1),
        )


@dataclass
class HiveEdge:
    """
    An edge in the Hebbian Hive with learning traces.

    Implements STDP-inspired (Spike-Timing-Dependent Plasticity) learning
    through pre- and post-synaptic traces.

    Part of Sprint 2: Hebbian Hive Enhancement (Woven Mind + PRISM Marriage)
    """
    source_id: str
    target_id: str
    weight: float = 0.0

    # Hebbian traces for learning
    pre_trace: float = 0.0   # Trace of source activation
    post_trace: float = 0.0  # Trace of target activation

    # Statistics
    co_activations: int = 0
    total_observations: int = 0

    # Learning parameters
    trace_decay: float = 0.95
    learning_rate: float = 0.01

    @property
    def correlation(self) -> float:
        """How often do source and target fire together?"""
        if self.total_observations == 0:
            return 0.0
        return self.co_activations / self.total_observations

    def observe_pre(self) -> None:
        """Record pre-synaptic activation."""
        self.pre_trace = 1.0
        self.total_observations += 1

    def observe_post(self) -> None:
        """Record post-synaptic activation."""
        self.post_trace = 1.0

    def observe_co_activation(self) -> None:
        """Record that both source and target are active."""
        self.co_activations += 1
        self.total_observations += 1

    def learn(self) -> float:
        """
        Apply Hebbian learning based on traces.

        Returns:
            Weight change amount.
        """
        # STDP-style: weight change proportional to trace product
        delta = self.learning_rate * self.pre_trace * self.post_trace
        self.weight += delta
        return delta

    def decay_traces(self) -> None:
        """Apply decay to both traces."""
        self.pre_trace *= self.trace_decay
        self.post_trace *= self.trace_decay

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "weight": self.weight,
            "pre_trace": self.pre_trace,
            "post_trace": self.post_trace,
            "co_activations": self.co_activations,
            "total_observations": self.total_observations,
            "trace_decay": self.trace_decay,
            "learning_rate": self.learning_rate,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HiveEdge":
        """Deserialize from dictionary."""
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            weight=data.get("weight", 0.0),
            pre_trace=data.get("pre_trace", 0.0),
            post_trace=data.get("post_trace", 0.0),
            co_activations=data.get("co_activations", 0),
            total_observations=data.get("total_observations", 0),
            trace_decay=data.get("trace_decay", 0.95),
            learning_rate=data.get("learning_rate", 0.01),
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

    # ==========================================================================
    # WOVEN MIND ENHANCEMENTS (Sprint 2: Hebbian Hive)
    # ==========================================================================

    def lateral_inhibition(
        self,
        activations: Dict[str, float],
        inhibition_radius: int = 2,
        inhibition_strength: float = 0.5,
    ) -> Dict[str, float]:
        """
        Apply lateral inhibition to activations for sparse representations.

        Nearby tokens inhibit each other based on activation strength.
        Produces sparse activation patterns (target: 5-10% active).

        Args:
            activations: Token -> activation level mapping
            inhibition_radius: How many "neighbors" to inhibit
            inhibition_strength: How much neighbors reduce activation (0-1)

        Returns:
            Inhibited activation levels (sparse)

        Example:
            >>> graph = TransitionGraph()
            >>> activations = {"cat": 0.9, "dog": 0.7, "bird": 0.5}
            >>> sparse = graph.lateral_inhibition(activations)
            >>> # "cat" inhibits "dog" and "bird"
        """
        if not activations:
            return {}

        # Sort tokens by activation strength (strongest first)
        sorted_tokens = sorted(
            activations.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        # Track inhibition each token receives
        inhibition = {token: 0.0 for token in activations}

        # Stronger activations inhibit weaker neighbors
        for i, (token, activation) in enumerate(sorted_tokens):
            # Inhibit neighbors within radius
            for j in range(max(0, i - inhibition_radius), min(len(sorted_tokens), i + inhibition_radius + 1)):
                if i != j:
                    neighbor_token = sorted_tokens[j][0]
                    # Inhibition proportional to the difference in rank
                    distance = abs(i - j)
                    local_inhibition = inhibition_strength * activation * (1.0 / (distance + 1))
                    inhibition[neighbor_token] += local_inhibition

        # Apply inhibition
        result = {}
        for token, activation in activations.items():
            inhibited = max(0.0, activation - inhibition[token])
            result[token] = inhibited

        return result

    def k_winners_take_all(
        self,
        activations: Dict[str, float],
        k: int = 5,
        min_activation: float = 0.1,
    ) -> Dict[str, float]:
        """
        Apply k-winners-take-all competition.

        Only the top-k most active tokens remain active (nonzero).
        All others are set to zero. This enforces sparsity.

        Args:
            activations: Token -> activation level mapping
            k: Number of winners to keep
            min_activation: Minimum activation to qualify as winner

        Returns:
            Sparse activation with only k active tokens

        Example:
            >>> graph = TransitionGraph()
            >>> activations = {"a": 0.9, "b": 0.7, "c": 0.5, "d": 0.3, "e": 0.1}
            >>> sparse = graph.k_winners_take_all(activations, k=2)
            >>> # Only "a" and "b" remain active
        """
        if not activations:
            return {}

        # Sort by activation (strongest first)
        sorted_tokens = sorted(
            activations.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        result = {}
        winners = 0

        for token, activation in sorted_tokens:
            if winners < k and activation >= min_activation:
                result[token] = activation
                winners += 1
            else:
                result[token] = 0.0

        return result

    def spreading_activation(
        self,
        seed_tokens: Dict[str, float],
        spread_factor: float = 0.5,
        decay_per_step: float = 0.7,
        max_steps: int = 3,
        threshold: float = 0.01,
    ) -> Dict[str, float]:
        """
        Spread activation through the transition graph.

        Starting from seed tokens, activation flows along transitions
        to associated tokens. This enables associative retrieval.

        Args:
            seed_tokens: Initial token -> activation mapping
            spread_factor: How much activation spreads (0-1)
            decay_per_step: How much activation decays each step (0-1)
            max_steps: Maximum propagation steps
            threshold: Minimum activation to propagate

        Returns:
            All activated tokens with their activation levels

        Example:
            >>> graph = TransitionGraph()
            >>> graph.learn_sequence(["the", "quick", "brown", "fox"])
            >>> activations = graph.spreading_activation({"quick": 1.0})
            >>> # "brown" and "fox" should have some activation
        """
        if not seed_tokens:
            return {}

        # Current activations
        current = dict(seed_tokens)

        # Track all activations across steps
        all_activations = dict(seed_tokens)

        for step in range(max_steps):
            next_activations: Dict[str, float] = {}

            for token, activation in current.items():
                if activation < threshold:
                    continue

                # Find contexts containing this token
                for context, transitions in self._transitions.items():
                    if token in context:
                        # Spread to connected tokens
                        for trans in transitions:
                            spread_amount = (
                                activation
                                * spread_factor
                                * (trans.weight / (1.0 + trans.weight))
                                * (decay_per_step ** step)
                            )

                            if spread_amount >= threshold:
                                target = trans.to_token
                                next_activations[target] = max(
                                    next_activations.get(target, 0.0),
                                    spread_amount,
                                )

            # Update all_activations with new values
            for token, activation in next_activations.items():
                all_activations[token] = max(
                    all_activations.get(token, 0.0),
                    activation,
                )

            current = next_activations

            if not current:
                break

        return all_activations

    def sparse_activate(
        self,
        query_tokens: List[str],
        k: int = 5,
        use_inhibition: bool = True,
        use_spreading: bool = True,
    ) -> Dict[str, float]:
        """
        Combine spreading activation with lateral inhibition for sparse retrieval.

        This is the main entry point for Hebbian Hive pattern matching:
        1. Seed activation from query tokens
        2. Spread activation through graph
        3. Apply lateral inhibition
        4. Select k winners

        Args:
            query_tokens: Tokens to start activation from
            k: Number of final active tokens
            use_inhibition: Whether to apply lateral inhibition
            use_spreading: Whether to spread activation

        Returns:
            Sparse activation pattern (k active tokens)

        Example:
            >>> graph = TransitionGraph()
            >>> graph.learn_sequence(["neural", "network", "learning"])
            >>> result = graph.sparse_activate(["neural"], k=3)
        """
        # Initialize seed activations
        seed = {token.lower(): 1.0 for token in query_tokens if token}

        if not seed:
            return {}

        # Spread if requested
        if use_spreading:
            activations = self.spreading_activation(seed)
        else:
            activations = seed

        # Apply inhibition if requested
        if use_inhibition:
            activations = self.lateral_inhibition(activations)

        # Select k winners
        return self.k_winners_take_all(activations, k=k)


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
