# Architecture Overview

## System Structure

The Cognitive Agent is part of the larger Cortical system. Understanding how pieces fit together helps navigate the codebase.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI Interface                            │
│              python -m cortical.cognitive [command]             │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                     IncrementalTrainer                          │
│  - Manages training lifecycle                                    │
│  - Tracks document manifest                                      │
│  - Coordinates save/load                                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  BPETokenizer   │ │TextToAtomsBridge│ │TrainingManifest │
│                 │ │                 │ │                 │
│ - Vocabulary    │ │ - feed_text()   │ │ - Documents     │
│ - IDF values    │ │ - Creates atoms │ │ - Staleness     │
│ - Doc frequency │ │ - Creates links │ │ - IDF epoch     │
└─────────────────┘ └────────┬────────┘ └─────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      CognitiveAgent                              │
│                                                                  │
│  Core API:                                                       │
│  - get_associations(word, top_k) → List[Association]            │
│  - predict_next(word, top_k) → Prediction                       │
│  - get_incoming/get_outgoing → Link traversal                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      CognitiveGraph                              │
│                                                                  │
│  Hypergraph operations:                                         │
│  - node(name) → Create/retrieve atom                            │
│  - link(type, targets) → Create relationship                    │
│  - get_incoming/get_outgoing → Indexed traversal                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     InMemoryStorage                              │
│                                                                  │
│  Indexes:                                                        │
│  - _atoms: Dict[id, Atom]                                       │
│  - _by_name: Dict[name, id]                                     │
│  - _incoming: Dict[atom_id, Set[link_ids]]                      │
│  - _outgoing: Dict[atom_id, Set[link_ids]]                      │
│  - _dirty_atoms: Set[id] (for incremental saves)                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ShardedGraphStorage                           │
│                                                                  │
│  Persistence:                                                    │
│  - atoms_word.json                                               │
│  - atoms_similarity_00..03.json                                  │
│  - atoms_follows_00..03.json                                     │
│  - meta.json                                                     │
└─────────────────────────────────────────────────────────────────┘
```

## Key Files

| File | Purpose |
|------|---------|
| `cortical/cognitive/__main__.py` | CLI entry point |
| `cortical/cognitive/graph.py` | CognitiveGraph, CognitiveAgent, InMemoryStorage |
| `cortical/cognitive/text_bridge.py` | TextToAtomsBridge, BPETokenizer |
| `cortical/cognitive/training.py` | IncrementalTrainer, TrainingManifest |
| `cortical/cognitive/graph_storage.py` | ShardedGraphStorage |

## Data Flow: Training

```
Document Text
     │
     ▼
BPETokenizer.tokenize()
     │ Updates:
     │ - vocab
     │ - _doc_frequency
     │ - _total_docs
     ▼
TextToAtomsBridge.feed_text()
     │ Creates:
     │ - WORD atoms (for each token)
     │ - SIMILARITY links (co-occurrence)
     │ - FOLLOWS links (sequences)
     ▼
CognitiveGraph.link()
     │ Computes:
     │ - raw_strength
     │ - idf_strength
     ▼
InMemoryStorage.save()
     │ Updates:
     │ - _atoms
     │ - _incoming index
     │ - _outgoing index
     │ - _dirty_atoms
     ▼
ShardedGraphStorage.save()
     │ Writes:
     │ - Only dirty shards
     │ - Updates meta.json
     ▼
Disk (JSON files)
```

## Data Flow: Query

```
User Query: "storage"
     │
     ▼
CognitiveAgent.get_associations("storage")
     │
     ▼
CognitiveGraph.get_node("storage")
     │ Returns: WORD atom
     ▼
InMemoryStorage.get_incoming(atom.id)
     │ O(1) lookup in _incoming index
     │ Returns: SIMILARITY links
     ▼
Filter and sort by idf_strength
     │
     ▼
Extract target words from links
     │
     ▼
Return: List[Association]
     │ Each has: word, weight
```

## Atom Types

| Type | Category | Purpose |
|------|----------|---------|
| WORD | Node | Vocabulary term |
| CONCEPT | Node | Abstract idea |
| FILE | Node | Source file path |
| CLASS | Node | Class definition |
| FUNCTION | Node | Function definition |
| SIMILARITY | Link | Co-occurrence (bidirectional) |
| FOLLOWS | Link | Sequence (directional) |
| DEFINES | Link | FILE defines CLASS/FUNCTION |
| CONTAINS | Link | CLASS contains METHOD |
| CALLS | Link | FUNCTION calls FUNCTION |

## Link Structure

**SIMILARITY** links:
```
outgoing: [word_a_id, word_b_id]  # Order doesn't matter
metadata: {
  raw_strength: 0.85,
  idf_strength: 0.42,
  idf_epoch: 1
}
```

**FOLLOWS** links:
```
outgoing: [from_word_id, to_word_id]  # Order matters!
metadata: {
  raw_strength: 0.65,
  idf_strength: 0.31,
  idf_epoch: 1
}
```

## Index Invariants

These must always be true:

1. Every atom in `_atoms` with a name is in `_by_name`
2. Every link in `_atoms` has entries in `_incoming` for all targets
3. Every link in `_atoms` has an entry in `_outgoing` for its first target
4. `_dirty_atoms` contains only IDs that exist in `_atoms`
5. After `clear_dirty()`, `_dirty_atoms` is empty and `_all_dirty` is False

## Extension Points

To add new functionality:

1. **New atom type**: Add to `AtomType` enum in `graph.py`
2. **New link type**: Add to `AtomType` enum, update `is_link()` check
3. **New query method**: Add to `CognitiveAgent` class
4. **New CLI command**: Add to `__main__.py`
5. **New storage backend**: Implement `StorageBackend` protocol

## Performance Characteristics

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Node lookup by name | O(1) | _by_name index |
| Node lookup by ID | O(1) | _atoms dict |
| Find incoming links | O(1) | _incoming index |
| Find outgoing links | O(1) | _outgoing index |
| Find by type | O(n) | Scans all atoms |
| Save (no changes) | O(1) | Dirty check only |
| Save (incremental) | O(k) | k = atoms in dirty shards |
| Load | O(n) | Reads all atoms |
