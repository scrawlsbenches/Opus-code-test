"""
Activation propagation algorithm.

Contains:
- propagate_activation: Spread activation through the network layers
"""

from typing import Dict

from ..layers import CorticalLayer, HierarchicalLayer


def propagate_activation(
    layers: Dict[CorticalLayer, HierarchicalLayer],
    iterations: int = 3,
    decay: float = 0.8,
    lateral_weight: float = 0.3
) -> None:
    """
    Propagate activation through the network.

    This simulates how information flows through cortical layers:
    - Activation spreads to connected columns (lateral)
    - Activation flows up the hierarchy (feedforward)
    - Activation decays over time

    Args:
        layers: Dictionary of all layers
        iterations: Number of propagation iterations
        decay: How much activation decays per iteration
        lateral_weight: Weight for lateral spreading
    """
    for _ in range(iterations):
        # Store new activations
        new_activations: Dict[str, float] = {}

        # Process each layer
        for layer_enum in CorticalLayer:
            if layer_enum not in layers:
                continue
            layer = layers[layer_enum]

            for col in layer.minicolumns.values():
                # Start with decayed current activation
                new_act = col.activation * decay

                # Add lateral input using O(1) ID lookup
                for neighbor_id, weight in col.lateral_connections.items():
                    neighbor = layer.get_by_id(neighbor_id)
                    if neighbor:
                        new_act += neighbor.activation * weight * lateral_weight

                # Add feedforward input using O(1) ID lookup
                for source_id in col.feedforward_sources:
                    # Find source in lower layers
                    for lower_enum in CorticalLayer:
                        if lower_enum >= layer_enum:
                            break
                        if lower_enum not in layers:
                            continue
                        lower_layer = layers[lower_enum]
                        source = lower_layer.get_by_id(source_id)
                        if source:
                            new_act += source.activation * 0.5
                            break

                new_activations[col.id] = new_act

        # Apply new activations
        for layer_enum in CorticalLayer:
            if layer_enum not in layers:
                continue
            layer = layers[layer_enum]
            for col in layer.minicolumns.values():
                if col.id in new_activations:
                    col.activation = new_activations[col.id]
