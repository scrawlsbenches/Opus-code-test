---
title: "Data Structures"
generated: "2025-12-16T17:26:23.833745Z"
generator: "architecture"
source_files:
  - "/home/user/Opus-code-test/cortical/minicolumn.py"
  - "/home/user/Opus-code-test/cortical/layers.py"
  - "/home/user/Opus-code-test/cortical/types.py"
tags:
  - architecture
  - modules
  - data-structures
---

# Data Structures

Fundamental data structures used throughout the system.

## Modules

- **layers.py**: Layers Module
- **minicolumn.py**: Minicolumn Module
- **types.py**: Type Aliases for the Cortical Text Processor.


## layers.py

Layers Module
=============

Defines the hierarchical layer structure inspired by the visual cortex.

The neocortex processes information through a hierarchy of layers,
each extracting progressively m...


### Classes

#### CorticalLayer

Enumeration of cortical processing layers.

**Methods:**

- `description`
- `analogy`

#### HierarchicalLayer

A layer in the cortical hierarchy containing minicolumns.

**Methods:**

- `get_or_create_minicolumn`
- `get_minicolumn`
- `get_by_id`
- `remove_minicolumn`
- `column_count`
- `total_connections`
- `average_activation`
- `activation_range`
- `sparsity`
- `top_by_pagerank`
- `top_by_tfidf`
- `top_by_activation`
- `to_dict`
- `from_dict`

### Functions

#### CorticalLayer.description

```python
CorticalLayer.description(self) -> str
```

Human-readable description of this layer.

#### CorticalLayer.analogy

```python
CorticalLayer.analogy(self) -> str
```

Visual cortex analogy for this layer.

#### HierarchicalLayer.get_or_create_minicolumn

```python
HierarchicalLayer.get_or_create_minicolumn(self, content: str) -> Minicolumn
```

Get existing minicolumn or create new one.

#### HierarchicalLayer.get_minicolumn

```python
HierarchicalLayer.get_minicolumn(self, content: str) -> Optional[Minicolumn]
```

Get a minicolumn by content, or None if not found.

#### HierarchicalLayer.get_by_id

```python
HierarchicalLayer.get_by_id(self, col_id: str) -> Optional[Minicolumn]
```

Get a minicolumn by its ID in O(1) time.

#### HierarchicalLayer.remove_minicolumn

```python
HierarchicalLayer.remove_minicolumn(self, content: str) -> bool
```

Remove a minicolumn from this layer.

#### HierarchicalLayer.column_count

```python
HierarchicalLayer.column_count(self) -> int
```

Return the number of minicolumns in this layer.

#### HierarchicalLayer.total_connections

```python
HierarchicalLayer.total_connections(self) -> int
```

Return total number of lateral connections in this layer.

#### HierarchicalLayer.average_activation

```python
HierarchicalLayer.average_activation(self) -> float
```

Calculate average activation across all minicolumns.

#### HierarchicalLayer.activation_range

```python
HierarchicalLayer.activation_range(self) -> tuple
```

Return (min, max) activation values.

#### HierarchicalLayer.sparsity

```python
HierarchicalLayer.sparsity(self, threshold_fraction: float = 0.5) -> float
```

Calculate sparsity (fraction of columns with below-average activation).

#### HierarchicalLayer.top_by_pagerank

```python
HierarchicalLayer.top_by_pagerank(self, n: int = 10) -> list
```

Get top minicolumns by PageRank score.

#### HierarchicalLayer.top_by_tfidf

```python
HierarchicalLayer.top_by_tfidf(self, n: int = 10) -> list
```

Get top minicolumns by TF-IDF score.

#### HierarchicalLayer.top_by_activation

```python
HierarchicalLayer.top_by_activation(self, n: int = 10) -> list
```

Get top minicolumns by activation level.

#### HierarchicalLayer.to_dict

```python
HierarchicalLayer.to_dict(self) -> Dict
```

Convert layer to dictionary for serialization.

#### HierarchicalLayer.from_dict

```python
HierarchicalLayer.from_dict(cls, data: Dict) -> 'HierarchicalLayer'
```

Create a layer from dictionary representation.

### Dependencies

**Standard Library:**

- `enum.IntEnum`
- `minicolumn.Minicolumn`
- `typing.Dict`
- `typing.Iterator`
- `typing.Optional`



## minicolumn.py

Minicolumn Module
=================

Core data structure representing a cortical minicolumn.

In the neocortex, minicolumns are vertical structures containing
~80-100 neurons that respond to similar f...


### Classes

#### Edge

Typed edge with metadata for ConceptNet-style graph representation.

**Methods:**

- `to_dict`
- `from_dict`

#### Minicolumn

A minicolumn represents a single concept/feature at a given hierarchy level.

**Methods:**

- `lateral_connections`
- `lateral_connections`
- `add_lateral_connection`
- `add_lateral_connections_batch`
- `set_lateral_connection_weight`
- `add_typed_connection`
- `get_typed_connection`
- `get_connections_by_type`
- `get_connections_by_source`
- `add_feedforward_connection`
- `add_feedback_connection`
- `connection_count`
- `top_connections`
- `to_dict`
- `from_dict`

### Functions

#### Edge.to_dict

```python
Edge.to_dict(self) -> Dict
```

Convert to dictionary for serialization.

#### Edge.from_dict

```python
Edge.from_dict(cls, data: Dict) -> 'Edge'
```

Create an Edge from dictionary representation.

#### Minicolumn.lateral_connections

```python
Minicolumn.lateral_connections(self, value: Dict[str, float]) -> None
```

Set lateral connections from a dictionary (for deserialization).

#### Minicolumn.add_lateral_connection

```python
Minicolumn.add_lateral_connection(self, target_id: str, weight: float = 1.0) -> None
```

Add or strengthen a lateral connection to another column.

#### Minicolumn.add_lateral_connections_batch

```python
Minicolumn.add_lateral_connections_batch(self, connections: Dict[str, float]) -> None
```

Add or strengthen multiple lateral connections at once.

#### Minicolumn.set_lateral_connection_weight

```python
Minicolumn.set_lateral_connection_weight(self, target_id: str, weight: float) -> None
```

Set the weight of a lateral connection directly (not additive).

#### Minicolumn.add_typed_connection

```python
Minicolumn.add_typed_connection(self, target_id: str, weight: float = 1.0, relation_type: str = 'co_occurrence', confidence: float = 1.0, source: str = 'corpus') -> None
```

Add or update a typed connection with metadata.

#### Minicolumn.get_typed_connection

```python
Minicolumn.get_typed_connection(self, target_id: str) -> Optional[Edge]
```

Get a typed connection by target ID.

#### Minicolumn.get_connections_by_type

```python
Minicolumn.get_connections_by_type(self, relation_type: str) -> List[Edge]
```

Get all typed connections with a specific relation type.

#### Minicolumn.get_connections_by_source

```python
Minicolumn.get_connections_by_source(self, source: str) -> List[Edge]
```

Get all typed connections from a specific source.

#### Minicolumn.add_feedforward_connection

```python
Minicolumn.add_feedforward_connection(self, target_id: str, weight: float = 1.0) -> None
```

Add or strengthen a feedforward connection to a lower layer column.

#### Minicolumn.add_feedback_connection

```python
Minicolumn.add_feedback_connection(self, target_id: str, weight: float = 1.0) -> None
```

Add or strengthen a feedback connection to a higher layer column.

#### Minicolumn.connection_count

```python
Minicolumn.connection_count(self) -> int
```

Return the number of lateral connections.

#### Minicolumn.top_connections

```python
Minicolumn.top_connections(self, n: int = 5) -> list
```

Get the strongest lateral connections.

#### Minicolumn.to_dict

```python
Minicolumn.to_dict(self) -> Dict
```

Convert to dictionary for serialization.

#### Minicolumn.from_dict

```python
Minicolumn.from_dict(cls, data: Dict) -> 'Minicolumn'
```

Create a minicolumn from dictionary representation.

### Dependencies

**Standard Library:**

- `dataclasses.asdict`
- `dataclasses.dataclass`
- `dataclasses.field`
- `typing.Dict`
- `typing.List`
- ... and 2 more



## types.py

Type Aliases for the Cortical Text Processor.

This module provides type aliases for complex return types used throughout
the library, making function signatures more readable and maintainable.

Task ...


### Dependencies

**Standard Library:**

- `typing.Any`
- `typing.Dict`
- `typing.List`
- `typing.Optional`
- `typing.Tuple`



---

*This chapter is part of [The Cortical Chronicles](../README.md), a self-documenting book generated by the Cortical Text Processor.*
