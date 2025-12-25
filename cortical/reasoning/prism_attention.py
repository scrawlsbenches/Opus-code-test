"""
PRISM Attention Mechanisms.

Attention enables selective focus on relevant parts of the thought graph,
combining:
- Query-Key-Value attention from transformers
- Synaptic gating from neuroscience
- Relevance weighting from information retrieval

"The Caterpillar sat on a mushroom, paying attention to only what mattered."

Key components:
- AttentionLayer: Basic query-based attention over graph nodes
- MultiHeadAttention: Multiple heads for different relation types
- SynapticAttention: Attention that respects synaptic edge weights
- LearnableAttention: Attention that learns from reinforcement
- TemporalAttention: Attention over sequences of thoughts
- UnifiedAttention: Cross-system integration (GoT + SLM + PLN)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple

if TYPE_CHECKING:
    from .prism_pln import PLNReasoner
    from .prism_slm import PRISMLanguageModel
    from .thought_graph import PRISMGraph


@dataclass
class AttentionResult:
    """Result of unified attention across systems."""

    top_nodes: List[str]
    weights: Dict[str, float]
    pln_support: float = 0.0
    slm_fluency: float = 0.0
    rules_explored: int = 0


class AttentionLayer:
    """
    Basic query-based attention over graph nodes.

    Uses TF-IDF-like scoring to weight nodes by relevance to a query.
    The Caterpillar's focused gaze - only what matters gets attention.
    """

    def __init__(self, graph: "PRISMGraph") -> None:
        """Initialize attention layer with a graph."""
        self.graph = graph
        self._term_idf: Dict[str, float] = {}
        self._build_index()

    def _build_index(self) -> None:
        """Build inverted index for attention scoring."""
        # Count document frequency for each term
        term_doc_count: Dict[str, int] = {}
        total_docs = len(self.graph.nodes)

        for node_id, node in self.graph.nodes.items():
            terms = self._tokenize(node.content)
            seen: Set[str] = set()
            for term in terms:
                if term not in seen:
                    term_doc_count[term] = term_doc_count.get(term, 0) + 1
                    seen.add(term)

        # Compute IDF
        for term, count in term_doc_count.items():
            self._term_idf[term] = math.log((total_docs + 1) / (count + 1)) + 1

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return re.findall(r"\w+", text.lower())

    def attend(self, query: str) -> Dict[str, float]:
        """
        Compute attention weights for all nodes given a query.

        Args:
            query: The query string

        Returns:
            Dictionary mapping node_id to attention weight
        """
        query_terms = set(self._tokenize(query))
        weights: Dict[str, float] = {}

        for node_id, node in self.graph.nodes.items():
            node_terms = set(self._tokenize(node.content))

            # Score based on term overlap weighted by IDF
            score = 0.0
            for term in query_terms & node_terms:
                score += self._term_idf.get(term, 1.0)

            # Boost for exact content match
            if any(term in node.content.lower() for term in query_terms):
                score *= 1.5

            weights[node_id] = score

        # Normalize to sum to 1 (softmax-like)
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights


class MultiHeadAttention:
    """
    Multi-head attention capturing different relation types.

    One head for "who" (entities), another for "where" (locations),
    another for "what" (objects/actions). Each head specializes.
    """

    # Keywords that activate different attention heads
    HEAD_PATTERNS = {
        "who": ["who", "person", "character", "someone", "they", "he", "she"],
        "where": ["where", "place", "location", "at", "in", "setting"],
        "what": ["what", "thing", "object", "tool", "item", "using"],
        "when": ["when", "time", "before", "after", "during"],
        "why": ["why", "because", "reason", "cause"],
        "how": ["how", "way", "method", "process"],
    }

    # Node types preferred by each head
    HEAD_TYPE_PREFERENCE = {
        "who": ["ENTITY", "CONCEPT"],
        "where": ["LOCATION", "CONTEXT"],
        "what": ["OBJECT", "ARTIFACT", "ACTION"],
        "when": ["FACT", "OBSERVATION"],
        "why": ["INSIGHT", "DECISION"],
        "how": ["ACTION", "TASK"],
    }

    def __init__(self, graph: "PRISMGraph", num_heads: int = 3) -> None:
        """Initialize multi-head attention."""
        self.graph = graph
        self.num_heads = num_heads
        self.base_attention = AttentionLayer(graph)

    def _detect_query_type(self, query: str) -> str:
        """Detect which head should be primary for this query."""
        query_lower = query.lower()

        scores = {}
        for head_name, keywords in self.HEAD_PATTERNS.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            scores[head_name] = score

        if max(scores.values()) > 0:
            return max(scores, key=lambda k: scores[k])
        return "what"  # Default

    def attend(self, query: str) -> Dict[str, float]:
        """
        Compute attention using multiple heads.

        Args:
            query: The query string

        Returns:
            Dictionary mapping node_id to attention weight
        """
        # Get base attention
        base_weights = self.base_attention.attend(query)

        # Determine query type
        query_type = self._detect_query_type(query)
        preferred_types = self.HEAD_TYPE_PREFERENCE.get(query_type, [])

        # Boost nodes of preferred types
        weights: Dict[str, float] = {}
        for node_id, node in self.graph.nodes.items():
            weight = base_weights.get(node_id, 0.0)

            # Type-based boost
            node_type_name = node.node_type.name
            if node_type_name in preferred_types:
                weight *= 2.0

            # Edge-based boost: follow relevant edges
            if query_type == "where":
                # Boost nodes connected via LOCATED_IN
                for edge in self.graph.edges:
                    if edge.edge_type.name == "LOCATED_IN":
                        if edge.target_id == node_id:
                            weight *= 1.5
            elif query_type == "what":
                # Boost nodes connected via USES
                for edge in self.graph.edges:
                    if edge.edge_type.name == "USES":
                        if edge.target_id == node_id:
                            weight *= 1.5

            weights[node_id] = weight

        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights


class SynapticAttention:
    """
    Attention that respects synaptic edge weights.

    Strong paths through the graph get more attention.
    Well-worn paths through Wonderland are easier to traverse.
    """

    def __init__(self, graph: "PRISMGraph") -> None:
        """Initialize synaptic attention."""
        self.graph = graph

    def attend_from(self, source_id: str) -> Dict[str, float]:
        """
        Compute attention weights for nodes reachable from source.

        Weights are based on synaptic edge strengths.

        Args:
            source_id: Starting node ID

        Returns:
            Dictionary mapping node_id to attention weight
        """
        weights: Dict[str, float] = {}

        # Use synaptic edges if available (PRISMGraph/SynapticMemoryGraph)
        if hasattr(self.graph, "get_synaptic_edges_from"):
            # Direct synaptic connections from source
            for edge in self.graph.get_synaptic_edges_from(source_id):
                target = edge.target_id
                weights[target] = max(weights.get(target, 0.0), edge.weight)

            # Secondary connections (one hop further) with decay
            decay = 0.5
            secondary: Dict[str, float] = {}
            for first_hop, first_weight in list(weights.items()):
                for edge in self.graph.get_synaptic_edges_from(first_hop):
                    target = edge.target_id
                    if target != source_id:
                        score = first_weight * edge.weight * decay
                        secondary[target] = max(secondary.get(target, 0.0), score)

            # Merge secondary into weights
            for node_id, score in secondary.items():
                if node_id not in weights:
                    weights[node_id] = score
        else:
            # Fallback to regular edges
            for edge in self.graph.edges:
                if edge.source_id == source_id:
                    weight = getattr(edge, "weight", 1.0)
                    target = edge.target_id
                    weights[target] = max(weights.get(target, 0.0), weight)

        return weights


class LearnableAttention:
    """
    Attention that learns from reinforcement.

    Alice learns to pay attention to the right clues.
    Positive feedback strengthens attention, negative weakens it.
    """

    def __init__(self, graph: "PRISMGraph", learning_rate: float = 0.1) -> None:
        """Initialize learnable attention."""
        self.graph = graph
        self.learning_rate = learning_rate
        self.base_attention = AttentionLayer(graph)

        # Learned attention biases (start at 1.0 = neutral)
        self._biases: Dict[str, float] = {
            node_id: 1.0 for node_id in graph.nodes
        }

    def attend(self, query: str) -> Dict[str, float]:
        """
        Compute attention with learned biases.

        Args:
            query: The query string

        Returns:
            Dictionary mapping node_id to attention weight
        """
        base_weights = self.base_attention.attend(query)

        # Apply learned biases
        weights: Dict[str, float] = {}
        for node_id, base_weight in base_weights.items():
            bias = self._biases.get(node_id, 1.0)
            weights[node_id] = base_weight * bias

        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights

    def reinforce(self, node_ids: List[str], reward: float) -> None:
        """
        Reinforce attention to specific nodes.

        Args:
            node_ids: Nodes to reinforce
            reward: Positive to strengthen, negative to weaken
        """
        for node_id in node_ids:
            if node_id in self._biases:
                # Update bias with bounded learning
                delta = self.learning_rate * reward
                self._biases[node_id] = max(0.1, self._biases[node_id] + delta)


class TemporalAttention:
    """
    Attention over sequences of thoughts.

    "Begin at the beginning and go on till you come to the end."
    Recent thoughts and relevant context get higher attention.
    """

    def __init__(self, decay: float = 0.9) -> None:
        """Initialize temporal attention."""
        self.decay = decay
        self.sequence: List[str] = []
        self._term_positions: Dict[str, List[int]] = {}

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return re.findall(r"\w+", text.lower())

    def process_sequence(self, thoughts: List[str]) -> None:
        """
        Process a sequence of thoughts.

        Args:
            thoughts: List of thought strings in order
        """
        self.sequence = thoughts
        self._term_positions = {}

        for i, thought in enumerate(thoughts):
            terms = self._tokenize(thought)
            for term in terms:
                if term not in self._term_positions:
                    self._term_positions[term] = []
                self._term_positions[term].append(i)

    def attend(self, query: str) -> Dict[int, float]:
        """
        Compute attention over the sequence.

        Args:
            query: The query string

        Returns:
            Dictionary mapping position index to attention weight
        """
        if not self.sequence:
            return {}

        query_terms = set(self._tokenize(query))
        weights: Dict[int, float] = {}
        seq_len = len(self.sequence)

        for i, thought in enumerate(self.sequence):
            thought_terms = set(self._tokenize(thought))

            # Base score: recency (later = higher)
            recency = (i + 1) / seq_len

            # Relevance score: term overlap
            overlap = len(query_terms & thought_terms)
            relevance = overlap / max(len(query_terms), 1)

            # Combined score
            weights[i] = recency * (1 + relevance * 2)

        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights


class UnifiedAttention:
    """
    Cross-system attention integrating GoT, SLM, and PLN.

    The Grand Unified Theory of PRISM attention.
    Each system contributes its unique perspective.
    """

    def __init__(
        self,
        graph: "PRISMGraph",
        slm: Optional["PRISMLanguageModel"] = None,
        pln: Optional["PLNReasoner"] = None,
    ) -> None:
        """Initialize unified attention."""
        self.graph = graph
        self.slm = slm
        self.pln = pln
        self.base_attention = AttentionLayer(graph)

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return re.findall(r"\w+", text.lower())

    def attend(self, query: str) -> AttentionResult:
        """
        Compute unified attention across all systems.

        Args:
            query: The query string

        Returns:
            AttentionResult with weights and system support scores
        """
        # Base graph attention
        weights = self.base_attention.attend(query)

        # PLN support: check if query concepts have logical backing
        pln_support = 0.0
        rules_explored = 0
        if self.pln:
            query_terms = self._tokenize(query)
            for term in query_terms:
                result = self.pln.query(term)
                rules_explored += 1
                if result is not None:
                    pln_support = max(pln_support, result.strength)

        # SLM fluency: check if query patterns exist in language model
        slm_fluency = 0.0
        if self.slm:
            # Use perplexity as inverse fluency measure
            try:
                perplexity = self.slm.perplexity(query)
                # Convert perplexity to fluency (lower perplexity = higher fluency)
                slm_fluency = 1.0 / (1.0 + math.log(perplexity + 1))
            except (ZeroDivisionError, ValueError):
                slm_fluency = 0.5  # Default

        # Boost nodes mentioned in PLN rules
        if self.pln:
            for node_id in weights:
                if self.pln.query(node_id) is not None:
                    weights[node_id] *= 1.5

        # Sort by weight
        sorted_nodes = sorted(weights.items(), key=lambda x: -x[1])
        top_nodes = [node_id for node_id, _ in sorted_nodes[:5]]

        return AttentionResult(
            top_nodes=top_nodes,
            weights=weights,
            pln_support=pln_support,
            slm_fluency=slm_fluency,
            rules_explored=rules_explored,
        )


class AttentionGuidedReasoner:
    """
    PLN reasoning guided by attention.

    Don't search the whole garden - follow the Cheshire Cat's gaze.
    Attention focuses the search to relevant rules.
    """

    def __init__(self, pln: "PLNReasoner") -> None:
        """Initialize attention-guided reasoner."""
        self.pln = pln
        self._rules_explored = 0

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return re.findall(r"\w+", text.lower())

    def query_with_attention(self, query: str) -> AttentionResult:
        """
        Query PLN with attention-guided focus.

        Args:
            query: Natural language query

        Returns:
            AttentionResult with findings
        """
        query_terms = set(self._tokenize(query))
        self._rules_explored = 0

        weights: Dict[str, float] = {}
        findings: List[str] = []

        # Focus on rules relevant to query terms
        for term in query_terms:
            # Try to find relevant rules
            result = self.pln.query(term)
            self._rules_explored += 1

            if result is not None:
                weights[term] = result.strength
                findings.append(term)

            # Check for related predicates (e.g., cat -> can_disappear)
            # This is a simplified version - real implementation would
            # traverse the PLN graph more thoroughly
            for predicate in ["can_disappear", "can_grin", "has_fur"]:
                if term in predicate or predicate.split("_")[-1] in query_terms:
                    result = self.pln.query(predicate)
                    self._rules_explored += 1
                    if result is not None:
                        weights[predicate] = result.strength
                        findings.append(predicate)

        return AttentionResult(
            top_nodes=findings,
            weights=weights,
            rules_explored=self._rules_explored,
        )


class AttentionVisualizer:
    """
    Generate attention visualizations for interpretability.

    "Curiouser and curiouser!" - see where the model looks.
    """

    def __init__(self, graph: "PRISMGraph") -> None:
        """Initialize visualizer."""
        self.graph = graph
        self.attention = AttentionLayer(graph)

    def generate_heatmap(
        self, query: str, nodes: List[str]
    ) -> "AttentionHeatmap":
        """
        Generate attention heatmap matrix.

        Args:
            query: The query string
            nodes: List of node IDs to include

        Returns:
            2D heatmap of attention weights
        """
        # Get attention weights for query
        weights = self.attention.attend(query)

        # Build attention matrix (node x node)
        size = len(nodes)
        matrix = [[0.0] * size for _ in range(size)]

        for i, node_i in enumerate(nodes):
            for j, node_j in enumerate(nodes):
                if i == j:
                    # Self-attention based on query relevance
                    matrix[i][j] = weights.get(node_i, 0.0)
                else:
                    # Cross-attention based on edge presence
                    for edge in self.graph.edges:
                        if edge.source_id == node_i and edge.target_id == node_j:
                            matrix[i][j] = weights.get(node_i, 0.0) * 0.5
                            break

        return AttentionHeatmap(matrix, nodes)


@dataclass
class AttentionHeatmap:
    """A 2D attention heatmap with numpy-like interface."""

    data: List[List[float]]
    nodes: List[str]

    @property
    def shape(self) -> Tuple[int, int]:
        """Return shape like numpy array."""
        return (len(self.data), len(self.data[0]) if self.data else 0)

    def sum(self) -> float:
        """Sum all values."""
        return sum(sum(row) for row in self.data)

    def max(self) -> float:
        """Maximum value."""
        if not self.data:
            return 0.0
        return max(max(row) for row in self.data)

    def __getitem__(self, key: Tuple[int, int]) -> float:
        """Index into heatmap."""
        i, j = key
        return self.data[i][j]
