"""
AttentionGraph: A graph neural network using self-attention instead of message passing.

=============================================================================
THE STORY: WHY ATTENTION FOR GRAPHS?
=============================================================================

Imagine you're learning to spell words. With message passing (TrainableGraph),
each letter asks its neighbors "what are you?" and averages their answers.
The problem: information flows in all directions equally, losing the crucial
left-to-right order that makes "the" different from "eth" or "het".

With attention, each letter asks a smarter question: "Given who I am (query),
which of my predecessors (keys) should I pay attention to, and what should
I learn from them (values)?" This preserves sequential structure because:

1. Each position has a unique query based on its content and position
2. The attention weights are computed dynamically, not fixed
3. Causal masking ensures we only look backward, never forward

The graph structure here serves as a "potential attention mask" - edges define
which positions CAN attend to which, and attention weights determine how much.

=============================================================================
DESIGN DECISIONS AND THEIR RATIONALE
=============================================================================

Decision 1: Inherit from BaseGraph, not TrainableGraph
-------------------------------------------------------
WHY: TrainableGraph's forward() is fundamentally about message passing.
     AttentionGraph's forward() is about query-key-value attention.
     Sharing a base would force awkward abstractions.

INSTEAD: Both inherit from BaseGraph (for graph operations) and both
         satisfy TrainableGraphProtocol (for the experiment kernel).

Decision 2: Graph edges define attention mask, not attention weights
--------------------------------------------------------------------
WHY: In pure transformers, every position can attend to every other.
     In our graph setting, edges encode structural priors:
     - Causal edges (i -> i+1) enforce sequential order
     - Skip edges (i -> i+2) allow longer-range attention
     - Missing edges mean "cannot attend" (implicit masking)

BENEFIT: Experimenters can try different graph topologies while
         keeping the same attention mechanism.

Decision 3: Single-head attention first, designed for multi-head
----------------------------------------------------------------
WHY: Multi-head adds complexity (concatenation, head dimension math).
     Single-head lets us validate the core idea first.

HOW: The implementation uses clean abstractions that extend to multi-head.

Decision 4: Store attention weights for backward pass
-----------------------------------------------------
WHY: The gradient through softmax requires knowing the forward weights.
     We could recompute, but storing is clearer and matches PyTorch style.

TRADEOFF: Memory vs compute. For small experiments, memory is fine.

=============================================================================
USAGE EXAMPLE
=============================================================================

    from cortical.graph.attention import AttentionGraph, AttentionNode

    # Create graph with attention mechanism
    graph = AttentionGraph(embedding_dim=64, num_heads=1)

    # Add nodes (positions in a sequence)
    for i in range(context_size):
        graph.add_node(f"pos_{i}")

    # Add causal edges (each position can attend to previous positions)
    for i in range(1, context_size):
        for j in range(i):  # Position i can attend to all j < i
            graph.add_edge(f"pos_{j}", f"pos_{i}")

    # Forward pass with input embeddings
    inputs = {"pos_0": embed_0, "pos_1": embed_1, ...}
    outputs = graph.forward(inputs)

    # The output at each position incorporates attended information
    # from previous positions, weighted by learned attention

See also:
- cortical/graph/trainable.py for the message-passing alternative
- examples/trainable_graph_benchmark.py for the experiment harness
- samples/session_handoff_trainable_graph.txt for context on why we built this
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    Union,
    runtime_checkable,
)

import numpy as np
from numpy.typing import NDArray

from .base import BaseGraph
from .protocols import NodeBase, EdgeBase
from .storage import InMemoryGraphStorage


# Type alias for clarity
Array = NDArray[np.float64]


# =============================================================================
# TRAINABLE GRAPH PROTOCOL
# =============================================================================
#
# This protocol defines what the ExperimentKernel expects from any trainable
# graph implementation. Both TrainableGraph and AttentionGraph satisfy this,
# allowing the benchmark harness to work with either.
#
# Think of this as a "contract": any graph that signs this contract can
# participate in experiments, regardless of its internal mechanism.
# =============================================================================


@runtime_checkable
class TrainableGraphProtocol(Protocol):
    """
    Contract for graphs that can be trained via gradient descent.

    Any graph satisfying this protocol can be used with the ExperimentKernel.
    This enables swapping TrainableGraph for AttentionGraph (or HybridGraph)
    without changing the experiment harness.

    The Story:
        The ExperimentKernel is like a gym trainer. It doesn't care if you're
        a weightlifter (TrainableGraph) or a yoga practitioner (AttentionGraph).
        It just needs you to:
        - Tell it what muscles you're training (parameters)
        - Do the exercise (forward)
        - Feel the burn and adapt (backward)
        - Remember your progress (save/load state)
    """

    @property
    def embedding_dim(self) -> int:
        """Dimension of node embeddings."""
        ...

    def parameters(self) -> List["Parameter"]:
        """
        Return all learnable parameters.

        The trainer needs to know what to optimize. This returns every
        Parameter object that should receive gradient updates.
        """
        ...

    def forward(
        self,
        num_layers: int = 1,
        input_nodes: Optional[Dict[str, Array]] = None,
    ) -> Dict[str, Array]:
        """
        Compute forward pass, returning output for each node.

        Args:
            num_layers: How many processing layers to apply
            input_nodes: Optional override for node input values

        Returns:
            Dict mapping node_id to output embedding
        """
        ...

    def backward(
        self,
        output_gradients: Dict[str, Array],
        num_layers: int = 1,
    ) -> None:
        """
        Compute gradients via backpropagation.

        Args:
            output_gradients: Gradient of loss w.r.t. each output node
            num_layers: Must match the forward pass
        """
        ...

    def zero_grad(self) -> None:
        """Reset all parameter gradients to zero."""
        ...

    def save_state(self) -> Dict[str, Any]:
        """Serialize learnable parameters for checkpointing."""
        ...

    def load_state(self, state: Dict[str, Any]) -> None:
        """Restore learnable parameters from checkpoint."""
        ...


# =============================================================================
# PARAMETER CLASS
# =============================================================================
#
# Reusing the Parameter class from trainable.py would be ideal, but to keep
# this module self-contained for now, we define a compatible version.
# TODO: Extract Parameter to a shared module.
# =============================================================================


@dataclass
class Parameter:
    """
    A learnable parameter with gradient tracking.

    The Story:
        A Parameter is like a muscle that remembers how to improve.
        - data: The current muscle strength (the actual values)
        - grad: The feedback from the last workout (accumulated gradients)
        - requires_grad: Whether this muscle is being trained

    During training:
        1. Forward pass uses data to compute outputs
        2. Backward pass accumulates feedback in grad
        3. Optimizer reads grad and updates data
        4. zero_grad() clears the feedback for next iteration
    """

    data: Array
    grad: Optional[Array] = None
    requires_grad: bool = True
    name: str = ""

    def zero_grad(self) -> None:
        """Clear accumulated gradients. Called before each training step."""
        self.grad = None

    def add_grad(self, grad: Array) -> None:
        """
        Accumulate gradient (for when multiple paths contribute).

        Why accumulate instead of replace? In a graph, a parameter might
        influence multiple outputs. Each output's gradient contribution
        should sum up, not overwrite.
        """
        if self.grad is None:
            self.grad = grad.copy()
        else:
            self.grad += grad

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the parameter tensor."""
        return self.data.shape


# =============================================================================
# ATTENTION NODE
# =============================================================================
#
# Each node in an AttentionGraph represents a position that can attend to
# other positions. Unlike TrainableNode which stores messages from neighbors,
# AttentionNode stores attention weights computed during forward pass.
# =============================================================================


@dataclass
class AttentionNode(NodeBase):
    """
    A node that participates in attention-based processing.

    The Story:
        Imagine each node as a student in a classroom. During the lesson
        (forward pass), each student:
        1. Formulates a question based on their current understanding (query)
        2. Looks at what other students know (keys)
        3. Decides who to listen to (attention weights)
        4. Learns from those students (weighted sum of values)

    The attention_weights field remembers "who did I listen to?" so we can
    trace back during the backward pass and update accordingly.

    Attributes:
        embedding: The learnable representation for this position
        output: The result after attention processing (for this layer)
        attention_weights: Who this node attended to and how much
                          Dict mapping source_node_id -> weight
        pre_softmax: The raw attention scores before softmax (for gradients)
    """

    embedding: Optional[Parameter] = None
    output: Optional[Array] = None
    attention_weights: Dict[str, float] = field(default_factory=dict)
    pre_softmax: Optional[Array] = None

    # For multi-layer processing, we need to track intermediate values
    layer_inputs: List[Array] = field(default_factory=list)
    layer_outputs: List[Array] = field(default_factory=list)

    def __hash__(self) -> int:
        return hash(self.id)


@dataclass
class AttentionEdge(EdgeBase):
    """
    An edge representing a potential attention path.

    The Story:
        In message-passing graphs, edges carry messages.
        In attention graphs, edges define "who can attend to whom."

        Think of edges as one-way windows: if there's an edge from A to B,
        then B can look through the window and see A. Without the edge,
        B cannot see A at all (infinite attention mask).

        The weight field here is not the attention weight (that's computed
        dynamically). Instead, it can serve as a prior or bias:
        - weight=1.0: Normal attention possible
        - weight=0.5: Attention is dampened (multiplied by 0.5)
        - weight=0.0: Edge exists but attention is blocked

    For causal language modeling:
        - Add edge from pos_j to pos_i for all j < i
        - This creates the lower-triangular attention mask
    """

    # We keep weight unconstrained for trainable edges
    def __post_init__(self) -> None:
        """Skip weight validation for attention edges."""
        pass


# =============================================================================
# PROCESSING LAYER PROTOCOL
# =============================================================================
#
# This abstraction allows composing different processing strategies.
# An AttentionLayer, MessagePassingLayer, or MixedLayer all satisfy this.
# =============================================================================


class ProcessingLayer(ABC):
    """
    Abstract base for composable graph processing layers.

    The Story:
        Think of layers as different exercise machines in a gym.
        Each machine works your muscles (node embeddings) differently:
        - AttentionLayer: "Look at others and learn selectively"
        - MessagePassingLayer: "Average what your neighbors say"
        - FeedForwardLayer: "Transform yourself independently"

        A workout (forward pass) can combine machines in sequence.
        The pain (gradients) flows backward through each machine.

    This protocol enables the HybridGraph to stack layers arbitrarily:
        layers = [AttentionLayer(), AttentionLayer(), FeedForwardLayer()]
    """

    @abstractmethod
    def forward(
        self,
        node_values: Dict[str, Array],
        graph: "AttentionGraph",
    ) -> Dict[str, Array]:
        """
        Process node values through this layer.

        Args:
            node_values: Current embedding for each node
            graph: The graph structure (for edges, node lookup)

        Returns:
            Updated embeddings for each node
        """
        ...

    @abstractmethod
    def backward(
        self,
        output_gradients: Dict[str, Array],
        graph: "AttentionGraph",
    ) -> Dict[str, Array]:
        """
        Backpropagate gradients through this layer.

        Args:
            output_gradients: Gradient w.r.t. this layer's output
            graph: The graph structure

        Returns:
            Gradient w.r.t. this layer's input (for previous layer)
        """
        ...

    @abstractmethod
    def parameters(self) -> List[Parameter]:
        """Return learnable parameters in this layer."""
        ...


# =============================================================================
# SCALED DOT-PRODUCT ATTENTION
# =============================================================================
#
# The core attention mechanism, separated for clarity and reuse.
# This is the "how much should I listen to each source?" computation.
# =============================================================================


def scaled_dot_product_attention(
    query: Array,           # Shape: (d_k,) - what am I looking for?
    keys: Array,            # Shape: (n_sources, d_k) - what do sources offer?
    values: Array,          # Shape: (n_sources, d_v) - what can I learn?
    mask: Optional[Array] = None,  # Shape: (n_sources,) - who can I see?
) -> Tuple[Array, Array]:
    """
    Compute scaled dot-product attention.

    The Story:
        You're at a party (the sequence) trying to learn something.
        - Your query is your current interest/question
        - Each person (source) has a key (what they're about) and value (what they know)
        - You compute compatibility: how relevant is each person to my question?
        - You listen proportionally to compatibility (softmax)
        - You learn a weighted combination of everyone's knowledge

    The scaling (/ sqrt(d_k)) prevents dot products from getting too large
    when dimensionality is high, which would make softmax too peaky.

    Args:
        query: The query vector for the attending position
        keys: Key vectors for all source positions
        values: Value vectors for all source positions
        mask: Optional mask (0 = can attend, -inf = cannot attend)

    Returns:
        Tuple of (output, attention_weights):
        - output: Weighted sum of values
        - attention_weights: The softmax weights (for backward pass)

    Math:
        scores = Q @ K.T / sqrt(d_k)
        weights = softmax(scores + mask)
        output = weights @ V
    """
    d_k = query.shape[0]

    # Compute attention scores: how compatible is query with each key?
    # Shape: (n_sources,)
    scores = keys @ query / math.sqrt(d_k)

    # Apply mask if provided (e.g., causal mask for future positions)
    if mask is not None:
        scores = scores + mask

    # Softmax to get attention weights (sum to 1)
    # Numerical stability: subtract max before exp
    scores_stable = scores - np.max(scores)
    exp_scores = np.exp(scores_stable)
    attention_weights = exp_scores / (np.sum(exp_scores) + 1e-10)

    # Weighted sum of values
    # Shape: (d_v,)
    output = values.T @ attention_weights

    return output, attention_weights


def attention_backward(
    grad_output: Array,          # Gradient w.r.t. attention output
    query: Array,                # Query from forward pass
    keys: Array,                 # Keys from forward pass
    values: Array,               # Values from forward pass
    attention_weights: Array,    # Softmax weights from forward pass
) -> Tuple[Array, Array, Array]:
    """
    Backpropagate through scaled dot-product attention.

    The Story:
        During the backward pass, we ask: "If the output should change,
        how should query, keys, and values change to make that happen?"

        This is chain rule through three operations:
        1. output = weights @ values  ->  grad_values, grad_weights
        2. weights = softmax(scores)  ->  grad_scores (tricky!)
        3. scores = Q @ K.T / sqrt(d)  ->  grad_query, grad_keys

    The softmax gradient is the subtle part:
        d_softmax/d_input = diag(softmax) - outer(softmax, softmax)

    Args:
        grad_output: Gradient of loss w.r.t. attention output
        query, keys, values: Saved from forward pass
        attention_weights: Softmax output from forward pass

    Returns:
        Tuple of (grad_query, grad_keys, grad_values)
    """
    d_k = query.shape[0]
    n_sources = keys.shape[0]

    # Gradient through: output = weights @ values
    # grad_values: how should values change?
    grad_values = np.outer(attention_weights, grad_output)  # (n_sources, d_v)

    # grad_weights: how should attention weights change?
    grad_weights = values @ grad_output  # (n_sources,)

    # Gradient through softmax: the tricky part
    # d(softmax)/d(input) = softmax * (I - softmax.T)
    # For vector: grad_scores = weights * (grad_weights - sum(weights * grad_weights))
    weighted_sum = np.sum(attention_weights * grad_weights)
    grad_scores = attention_weights * (grad_weights - weighted_sum)

    # Gradient through: scores = Q @ K.T / sqrt(d_k)
    scale = 1.0 / math.sqrt(d_k)
    grad_query = (keys.T @ grad_scores) * scale  # (d_k,)
    grad_keys = np.outer(grad_scores, query) * scale  # (n_sources, d_k)

    return grad_query, grad_keys, grad_values


# =============================================================================
# ATTENTION LAYER
# =============================================================================
#
# A ProcessingLayer that applies self-attention across the graph.
# This is the core building block for AttentionGraph.
# =============================================================================


class AttentionLayer(ProcessingLayer):
    """
    Self-attention layer for graph processing.

    The Story:
        This layer transforms node embeddings by letting each node
        selectively attend to other nodes it's connected to.

        For each node:
        1. Compute query = embedding @ W_q ("What am I looking for?")
        2. For each source with incoming edge:
           - Compute key = source_embedding @ W_k ("What do I offer?")
           - Compute value = source_embedding @ W_v ("What can you learn?")
        3. Attention weight = softmax(query @ keys.T / sqrt(d))
        4. Output = sum(weights * values)

    The learned projections (W_q, W_k, W_v) allow the network to learn
    WHAT to attend to, not just whether to attend.

    Unlike message passing which uses fixed/learned edge weights,
    attention computes weights dynamically based on content.
    """

    def __init__(
        self,
        embedding_dim: int,
        use_bias: bool = False,
        dropout: float = 0.0,
    ):
        """
        Initialize attention layer.

        Args:
            embedding_dim: Dimension of node embeddings
            use_bias: Whether to use bias in projections
            dropout: Attention dropout rate (0 = no dropout)
        """
        self.embedding_dim = embedding_dim
        self.use_bias = use_bias
        self.dropout = dropout
        self._training = True

        # Initialize projection matrices with Xavier initialization
        # Why Xavier? It keeps variance stable across layers, preventing
        # vanishing/exploding gradients in deep networks.
        scale = math.sqrt(2.0 / (embedding_dim + embedding_dim))

        # Query projection: "What am I looking for?"
        self.W_q = Parameter(
            data=np.random.randn(embedding_dim, embedding_dim) * scale,
            name="attention_W_q",
        )

        # Key projection: "What do I offer?"
        self.W_k = Parameter(
            data=np.random.randn(embedding_dim, embedding_dim) * scale,
            name="attention_W_k",
        )

        # Value projection: "What can you learn from me?"
        self.W_v = Parameter(
            data=np.random.randn(embedding_dim, embedding_dim) * scale,
            name="attention_W_v",
        )

        # Output projection: "How do I integrate what I learned?"
        self.W_o = Parameter(
            data=np.random.randn(embedding_dim, embedding_dim) * scale,
            name="attention_W_o",
        )

        # Optional biases
        if use_bias:
            self.b_q = Parameter(data=np.zeros(embedding_dim), name="attention_b_q")
            self.b_k = Parameter(data=np.zeros(embedding_dim), name="attention_b_k")
            self.b_v = Parameter(data=np.zeros(embedding_dim), name="attention_b_v")
            self.b_o = Parameter(data=np.zeros(embedding_dim), name="attention_b_o")

        # Cache for backward pass
        self._cache: Dict[str, Any] = {}

    def train(self, mode: bool = True) -> "AttentionLayer":
        """Set training mode (affects dropout)."""
        self._training = mode
        return self

    def eval(self) -> "AttentionLayer":
        """Set evaluation mode."""
        return self.train(False)

    def parameters(self) -> List[Parameter]:
        """Return all learnable parameters."""
        params = [self.W_q, self.W_k, self.W_v, self.W_o]
        if self.use_bias:
            params.extend([self.b_q, self.b_k, self.b_v, self.b_o])
        return params

    def forward(
        self,
        node_values: Dict[str, Array],
        graph: "AttentionGraph",
    ) -> Dict[str, Array]:
        """
        Apply self-attention to all nodes.

        The Story:
            We visit each node and ask: "Based on my current state (query),
            what should I learn from the nodes I can see (via edges)?"

            The graph structure determines visibility:
            - If there's an edge from A to B, then B can attend to A
            - No edge means no attention (masked out)

            This naturally handles causal masking for sequences:
            just don't add edges from future to past positions.
        """
        self._cache = {
            "input_values": {},
            "queries": {},
            "keys": {},
            "values": {},
            "attention_weights": {},
            "pre_output": {},
        }

        outputs: Dict[str, Array] = {}

        for node in graph.nodes:
            node_id = node.id

            # Get current value for this node
            if node_id in node_values:
                current = node_values[node_id]
            elif node.embedding is not None:
                current = node.embedding.data
            else:
                current = np.zeros(self.embedding_dim)

            self._cache["input_values"][node_id] = current.copy()

            # Compute query for this node: "What am I looking for?"
            query = current @ self.W_q.data
            if self.use_bias:
                query = query + self.b_q.data
            self._cache["queries"][node_id] = query.copy()

            # Find all nodes this node can attend to (incoming edges)
            incoming_edges = graph.edges_to(node_id)

            if not incoming_edges:
                # No one to attend to - output is just transformed self
                # This handles the first position in a causal sequence
                output = current @ self.W_o.data
                if self.use_bias:
                    output = output + self.b_o.data
                outputs[node_id] = output
                node.attention_weights = {}
                self._cache["attention_weights"][node_id] = {}
                continue

            # Gather keys and values from sources
            source_ids = []
            keys_list = []
            values_list = []

            for edge in incoming_edges:
                source_id = edge.source_id
                source_ids.append(source_id)

                # Get source value
                if source_id in node_values:
                    source_val = node_values[source_id]
                else:
                    source_node = graph.get_node(source_id)
                    if source_node and source_node.embedding:
                        source_val = source_node.embedding.data
                    else:
                        source_val = np.zeros(self.embedding_dim)

                # Compute key and value for this source
                key = source_val @ self.W_k.data
                value = source_val @ self.W_v.data

                if self.use_bias:
                    key = key + self.b_k.data
                    value = value + self.b_v.data

                keys_list.append(key)
                values_list.append(value)

            # Stack into matrices
            keys = np.array(keys_list)      # (n_sources, d)
            values = np.array(values_list)  # (n_sources, d)

            self._cache["keys"][node_id] = keys.copy()
            self._cache["values"][node_id] = values.copy()

            # Compute attention (the magic happens here)
            attended, attn_weights = scaled_dot_product_attention(
                query=query,
                keys=keys,
                values=values,
                mask=None,  # Masking is implicit in graph structure
            )

            # Store attention weights for interpretability and backward pass
            attn_dict = {sid: float(w) for sid, w in zip(source_ids, attn_weights)}
            node.attention_weights = attn_dict
            self._cache["attention_weights"][node_id] = {
                "source_ids": source_ids,
                "weights": attn_weights.copy(),
            }

            # Apply dropout during training
            if self._training and self.dropout > 0:
                mask = np.random.binomial(1, 1 - self.dropout, attended.shape)
                attended = attended * mask / (1 - self.dropout)

            self._cache["pre_output"][node_id] = attended.copy()

            # Output projection
            output = attended @ self.W_o.data
            if self.use_bias:
                output = output + self.b_o.data

            outputs[node_id] = output
            node.output = output.copy()

        return outputs

    def backward(
        self,
        output_gradients: Dict[str, Array],
        graph: "AttentionGraph",
    ) -> Dict[str, Array]:
        """
        Backpropagate through attention layer.

        The Story:
            The loss told us "the outputs should change like this" (output_gradients).
            Now we figure out how each parameter contributed to the output,
            and accumulate gradients accordingly.

            Chain rule through:
            1. Output projection (W_o)
            2. Attention aggregation (the subtle softmax gradient)
            3. Query/Key/Value projections (W_q, W_k, W_v)
            4. Input values (for previous layer)
        """
        input_gradients: Dict[str, Array] = {}

        for node_id, grad_output in output_gradients.items():
            if node_id not in self._cache["input_values"]:
                continue

            # Gradient through output projection
            pre_output = self._cache.get("pre_output", {}).get(node_id)

            if pre_output is not None:
                # Has attention (not first position)
                self.W_o.add_grad(np.outer(pre_output, grad_output))
                if self.use_bias:
                    self.b_o.add_grad(grad_output)

                grad_attended = grad_output @ self.W_o.data.T

                # Get cached values for attention backward
                attn_cache = self._cache["attention_weights"].get(node_id, {})
                source_ids = attn_cache.get("source_ids", [])
                attn_weights = attn_cache.get("weights", np.array([]))

                if len(source_ids) > 0:
                    query = self._cache["queries"][node_id]
                    keys = self._cache["keys"][node_id]
                    values = self._cache["values"][node_id]

                    # Backward through attention
                    grad_query, grad_keys, grad_values = attention_backward(
                        grad_output=grad_attended,
                        query=query,
                        keys=keys,
                        values=values,
                        attention_weights=attn_weights,
                    )

                    # Gradient through query projection
                    input_val = self._cache["input_values"][node_id]
                    self.W_q.add_grad(np.outer(input_val, grad_query))
                    if self.use_bias:
                        self.b_q.add_grad(grad_query)

                    # Accumulate gradient for this node's input
                    grad_input = grad_query @ self.W_q.data.T
                    if node_id not in input_gradients:
                        input_gradients[node_id] = np.zeros(self.embedding_dim)
                    input_gradients[node_id] += grad_input

                    # Gradient through key/value projections for each source
                    for i, source_id in enumerate(source_ids):
                        source_node = graph.get_node(source_id)
                        if source_node and source_node.embedding:
                            source_val = source_node.embedding.data
                        else:
                            source_val = self._cache["input_values"].get(
                                source_id, np.zeros(self.embedding_dim)
                            )

                        # Gradient for key projection
                        self.W_k.add_grad(np.outer(source_val, grad_keys[i]))
                        if self.use_bias:
                            self.b_k.add_grad(grad_keys[i])

                        # Gradient for value projection
                        self.W_v.add_grad(np.outer(source_val, grad_values[i]))
                        if self.use_bias:
                            self.b_v.add_grad(grad_values[i])

                        # Gradient to source input
                        grad_source = (
                            grad_keys[i] @ self.W_k.data.T +
                            grad_values[i] @ self.W_v.data.T
                        )
                        if source_id not in input_gradients:
                            input_gradients[source_id] = np.zeros(self.embedding_dim)
                        input_gradients[source_id] += grad_source
            else:
                # No attention (first position) - just output projection
                input_val = self._cache["input_values"][node_id]
                self.W_o.add_grad(np.outer(input_val, grad_output))
                if self.use_bias:
                    self.b_o.add_grad(grad_output)

                grad_input = grad_output @ self.W_o.data.T
                if node_id not in input_gradients:
                    input_gradients[node_id] = np.zeros(self.embedding_dim)
                input_gradients[node_id] += grad_input

        return input_gradients


# =============================================================================
# ATTENTION GRAPH
# =============================================================================
#
# The main class that brings everything together. Inherits from BaseGraph
# for graph operations, and satisfies TrainableGraphProtocol for experiments.
# =============================================================================


class AttentionGraph(BaseGraph[AttentionNode, AttentionEdge]):
    """
    A graph neural network using self-attention instead of message passing.

    The Story:
        Imagine a classroom where students (nodes) learn from each other.
        In a TrainableGraph (message passing), each student simply averages
        what their neighbors say - everyone's voice has equal weight.

        In an AttentionGraph, students are smarter. Each one thinks:
        "Given what I'm trying to learn (query), which classmates (keys)
        have relevant knowledge? Let me pay more attention to those and
        learn their insights (values) proportionally."

        This is powerful for sequences (like text) because:
        - The graph structure encodes who can talk to whom (causal mask)
        - Attention weights adapt to the actual content
        - Sequential order is preserved through careful edge design

    Key Differences from TrainableGraph:
        1. No fixed edge weights - attention is computed dynamically
        2. Each node has Q/K/V projections, not just embeddings
        3. Causal structure comes from edges, not explicit masks
        4. Better gradient flow through softmax vs. aggregation

    Usage:
        # Create attention graph
        graph = AttentionGraph(embedding_dim=64)

        # Build causal sequence structure
        for i in range(seq_len):
            graph.add_node(f"pos_{i}")
        for i in range(1, seq_len):
            for j in range(i):  # Causal: i can only see j < i
                graph.add_edge(f"pos_{j}", f"pos_{i}")

        # Forward pass
        outputs = graph.forward(num_layers=2)

        # Backward pass
        graph.backward({"pos_last": loss_grad}, num_layers=2)
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        num_heads: int = 1,  # Reserved for future multi-head attention
        use_bias: bool = False,
        dropout: float = 0.0,
        seed: Optional[int] = None,
    ):
        """
        Initialize attention graph.

        Args:
            embedding_dim: Dimension of node embeddings
            num_heads: Number of attention heads (1 for now, multi-head TODO)
            use_bias: Whether to use bias in projections
            dropout: Dropout rate for attention weights
            seed: Random seed for reproducibility
        """
        super().__init__(InMemoryGraphStorage())

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.use_bias = use_bias
        self.dropout = dropout

        if seed is not None:
            np.random.seed(seed)

        # Attention layers (one per num_layers in forward)
        self._attention_layers: List[AttentionLayer] = []

        # Training state
        self._training = True

    def _create_node(self, id: str, **kwargs: Any) -> AttentionNode:
        """
        Create a node for the attention graph.

        Each node gets a learnable embedding that serves as the base
        representation. During forward pass, this embedding is transformed
        via Q/K/V projections for attention computation.
        """
        embedding_data = kwargs.get("embedding")

        if embedding_data is None:
            # Xavier initialization for stable training
            scale = math.sqrt(2.0 / self.embedding_dim)
            embedding_data = np.random.randn(self.embedding_dim) * scale
        elif isinstance(embedding_data, (list, tuple)):
            embedding_data = np.array(embedding_data, dtype=np.float64)

        embedding = Parameter(
            data=embedding_data,
            requires_grad=kwargs.get("requires_grad", True),
            name=f"embedding_{id}",
        )

        return AttentionNode(
            id=id,
            node_type=kwargs.get("node_type", ""),
            content=kwargs.get("content", ""),
            properties=kwargs.get("properties", {}),
            metadata=kwargs.get("metadata", {}),
            embedding=embedding,
        )

    def _create_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str = "",
        **kwargs: Any,
    ) -> AttentionEdge:
        """
        Create an edge defining attention visibility.

        In AttentionGraph, an edge from A to B means "B can attend to A".
        The edge weight is not the attention weight (that's computed),
        but can serve as a bias or prior.
        """
        return AttentionEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            weight=kwargs.get("weight", 1.0),
            bidirectional=kwargs.get("bidirectional", False),
            properties=kwargs.get("properties", {}),
        )

    def _ensure_layers(self, num_layers: int) -> None:
        """Ensure we have enough attention layers."""
        while len(self._attention_layers) < num_layers:
            layer = AttentionLayer(
                embedding_dim=self.embedding_dim,
                use_bias=self.use_bias,
                dropout=self.dropout,
            )
            if self._training:
                layer.train()
            else:
                layer.eval()
            self._attention_layers.append(layer)

    def train(self, mode: bool = True) -> "AttentionGraph":
        """Set training mode (affects dropout)."""
        self._training = mode
        for layer in self._attention_layers:
            layer.train(mode)
        return self

    def eval(self) -> "AttentionGraph":
        """Set evaluation mode."""
        return self.train(False)

    def parameters(self) -> List[Parameter]:
        """
        Return all learnable parameters.

        Includes:
        - Node embeddings
        - Attention layer projections (Q, K, V, O for each layer)
        """
        params: List[Parameter] = []

        # Node embeddings
        for node in self.nodes:
            if node.embedding is not None and node.embedding.requires_grad:
                params.append(node.embedding)

        # Attention layer parameters
        for layer in self._attention_layers:
            params.extend(layer.parameters())

        return params

    def zero_grad(self) -> None:
        """Reset all gradients to zero."""
        for param in self.parameters():
            param.zero_grad()

    def forward(
        self,
        num_layers: int = 1,
        input_nodes: Optional[Dict[str, Array]] = None,
    ) -> Dict[str, Array]:
        """
        Forward pass through attention layers.

        The Story:
            Information flows through layers of attention:

            Layer 1: Each node attends to its sources based on raw embeddings
            Layer 2: Each node attends based on Layer 1 outputs
            ...and so on.

            With each layer, nodes incorporate information from further away
            in the graph (assuming edges allow it). For a causal sequence,
            later positions gradually learn about earlier context.
        """
        self._ensure_layers(num_layers)

        # Initialize node values from embeddings or inputs
        values: Dict[str, Array] = {}
        for node in self.nodes:
            if input_nodes and node.id in input_nodes:
                values[node.id] = input_nodes[node.id].copy()
            elif node.embedding is not None:
                values[node.id] = node.embedding.data.copy()
            else:
                values[node.id] = np.zeros(self.embedding_dim)

            # Store for backward pass
            node.layer_inputs = [values[node.id].copy()]

        # Apply attention layers
        for layer_idx in range(num_layers):
            layer = self._attention_layers[layer_idx]
            values = layer.forward(values, self)

            # Store intermediate outputs for backward pass
            for node in self.nodes:
                if node.id in values:
                    node.layer_outputs.append(values[node.id].copy())
                    if layer_idx < num_layers - 1:
                        node.layer_inputs.append(values[node.id].copy())

        # Store final outputs
        for node in self.nodes:
            if node.id in values:
                node.output = values[node.id].copy()

        return values

    def backward(
        self,
        output_gradients: Dict[str, Array],
        num_layers: int = 1,
    ) -> None:
        """
        Backward pass through attention layers.

        The Story:
            Gradients flow backward through the layers, updating:
            - Attention projections (W_q, W_k, W_v, W_o)
            - Node embeddings

            Each layer receives gradients from the layer above (or loss),
            computes its contribution, and passes gradients down.
        """
        # Backprop through layers in reverse order
        gradients = output_gradients.copy()

        for layer_idx in range(num_layers - 1, -1, -1):
            layer = self._attention_layers[layer_idx]
            gradients = layer.backward(gradients, self)

        # Final gradients go to node embeddings
        for node in self.nodes:
            if node.id in gradients and node.embedding is not None:
                node.embedding.add_grad(gradients[node.id])

    def save_state(self) -> Dict[str, Any]:
        """
        Save graph state for checkpointing.

        Saves:
        - Node embeddings
        - Attention layer parameters
        """
        state = {
            "embeddings": {},
            "layers": [],
        }

        for node in self.nodes:
            if node.embedding is not None:
                state["embeddings"][node.id] = node.embedding.data.copy()

        for layer in self._attention_layers:
            layer_state = {
                "W_q": layer.W_q.data.copy(),
                "W_k": layer.W_k.data.copy(),
                "W_v": layer.W_v.data.copy(),
                "W_o": layer.W_o.data.copy(),
            }
            if layer.use_bias:
                layer_state.update({
                    "b_q": layer.b_q.data.copy(),
                    "b_k": layer.b_k.data.copy(),
                    "b_v": layer.b_v.data.copy(),
                    "b_o": layer.b_o.data.copy(),
                })
            state["layers"].append(layer_state)

        return state

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load graph state from checkpoint."""
        for node_id, embedding in state.get("embeddings", {}).items():
            node = self.get_node(node_id)
            if node is not None and node.embedding is not None:
                node.embedding.data = embedding.copy()

        for i, layer_state in enumerate(state.get("layers", [])):
            if i < len(self._attention_layers):
                layer = self._attention_layers[i]
                layer.W_q.data = layer_state["W_q"].copy()
                layer.W_k.data = layer_state["W_k"].copy()
                layer.W_v.data = layer_state["W_v"].copy()
                layer.W_o.data = layer_state["W_o"].copy()

                if layer.use_bias and "b_q" in layer_state:
                    layer.b_q.data = layer_state["b_q"].copy()
                    layer.b_k.data = layer_state["b_k"].copy()
                    layer.b_v.data = layer_state["b_v"].copy()
                    layer.b_o.data = layer_state["b_o"].copy()

    def get_attention_weights(self) -> Dict[str, Dict[str, float]]:
        """
        Get attention weights from last forward pass.

        Returns dict: node_id -> {source_id -> weight}

        Useful for interpretability: "What did this position attend to?"
        """
        return {
            node.id: node.attention_weights.copy()
            for node in self.nodes
        }

    def visualize_attention(self, node_id: str) -> str:
        """
        Generate ASCII visualization of attention for a node.

        The Story:
            Sometimes you want to peek inside the model's head and see
            what it's paying attention to. This gives a quick text-based
            visualization of attention weights.
        """
        node = self.get_node(node_id)
        if node is None:
            return f"Node {node_id} not found"

        weights = node.attention_weights
        if not weights:
            return f"Node {node_id} has no attention weights (first position?)"

        lines = [f"Attention from {node_id}:"]
        max_source_len = max(len(s) for s in weights.keys())

        for source_id, weight in sorted(weights.items(), key=lambda x: -x[1]):
            bar_len = int(weight * 40)
            bar = "#" * bar_len
            lines.append(f"  {source_id:<{max_source_len}} [{weight:.3f}] {bar}")

        return "\n".join(lines)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def create_causal_attention_graph(
    seq_len: int,
    embedding_dim: int = 64,
    **kwargs,
) -> AttentionGraph:
    """
    Create an AttentionGraph with causal (autoregressive) structure.

    The Story:
        For language modeling, we need each position to only see previous
        positions. This function builds that structure automatically:

        Position 0: Can't see anyone (no incoming edges)
        Position 1: Can see position 0
        Position 2: Can see positions 0, 1
        ...and so on.

    Args:
        seq_len: Number of positions in sequence
        embedding_dim: Dimension of node embeddings
        **kwargs: Additional args for AttentionGraph

    Returns:
        AttentionGraph with causal edges
    """
    graph = AttentionGraph(embedding_dim=embedding_dim, **kwargs)

    # Add position nodes
    for i in range(seq_len):
        graph.add_node(f"pos_{i}")

    # Add causal edges: each position can attend to all previous positions
    for i in range(1, seq_len):
        for j in range(i):
            graph.add_edge(f"pos_{j}", f"pos_{i}")

    return graph
