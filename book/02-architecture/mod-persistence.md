---
title: "Persistence Layer"
generated: "2025-12-16T17:26:23.831119Z"
generator: "architecture"
source_files:
  - "/home/user/Opus-code-test/cortical/persistence.py"
  - "/home/user/Opus-code-test/cortical/chunk_index.py"
  - "/home/user/Opus-code-test/cortical/state_storage.py"
tags:
  - architecture
  - modules
  - persistence
---

# Persistence Layer

Save and load functionality for maintaining processor state.

## Modules

- **chunk_index.py**: Chunk-based indexing for git-compatible corpus storage.
- **persistence.py**: Persistence Module
- **state_storage.py**: Git-friendly State Storage Module


## chunk_index.py

Chunk-based indexing for git-compatible corpus storage.

This module provides append-only, time-stamped JSON chunks that can be
safely committed to git without merge conflicts. Each indexing session
c...


### Classes

#### ChunkOperation

A single operation in a chunk (add, modify, or delete).

**Methods:**

- `to_dict`
- `from_dict`

#### Chunk

A chunk containing operations from a single indexing session.

**Methods:**

- `to_dict`
- `from_dict`
- `get_filename`

#### ChunkWriter

Writes indexing session changes to timestamped JSON chunks.

**Methods:**

- `add_document`
- `modify_document`
- `delete_document`
- `has_operations`
- `save`

#### ChunkLoader

Loads and combines chunks to rebuild document state.

**Methods:**

- `get_chunk_files`
- `load_chunk`
- `load_all`
- `get_documents`
- `get_mtimes`
- `get_metadata`
- `get_chunks`
- `compute_hash`
- `is_cache_valid`
- `save_cache_hash`
- `get_stats`

#### ChunkCompactor

Compacts multiple chunk files into a single file.

**Methods:**

- `compact`

### Functions

#### get_changes_from_manifest

```python
get_changes_from_manifest(current_files: Dict[str, float], manifest: Dict[str, float]) -> Tuple[List[str], List[str], List[str]]
```

Compare current files to manifest to find changes.

#### ChunkOperation.to_dict

```python
ChunkOperation.to_dict(self) -> Dict[str, Any]
```

Convert to dictionary for JSON serialization.

#### ChunkOperation.from_dict

```python
ChunkOperation.from_dict(cls, d: Dict[str, Any]) -> 'ChunkOperation'
```

Create from dictionary.

#### Chunk.to_dict

```python
Chunk.to_dict(self) -> Dict[str, Any]
```

Convert to dictionary for JSON serialization.

#### Chunk.from_dict

```python
Chunk.from_dict(cls, d: Dict[str, Any]) -> 'Chunk'
```

Create from dictionary.

#### Chunk.get_filename

```python
Chunk.get_filename(self) -> str
```

Generate filename for this chunk.

#### ChunkWriter.add_document

```python
ChunkWriter.add_document(self, doc_id: str, content: str, mtime: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None)
```

Record an add operation.

#### ChunkWriter.modify_document

```python
ChunkWriter.modify_document(self, doc_id: str, content: str, mtime: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None)
```

Record a modify operation.

#### ChunkWriter.delete_document

```python
ChunkWriter.delete_document(self, doc_id: str)
```

Record a delete operation.

#### ChunkWriter.has_operations

```python
ChunkWriter.has_operations(self) -> bool
```

Check if any operations were recorded.

#### ChunkWriter.save

```python
ChunkWriter.save(self, warn_size_kb: int = DEFAULT_WARN_SIZE_KB) -> Optional[Path]
```

Save chunk to file.

#### ChunkLoader.get_chunk_files

```python
ChunkLoader.get_chunk_files(self) -> List[Path]
```

Get all chunk files sorted by timestamp.

#### ChunkLoader.load_chunk

```python
ChunkLoader.load_chunk(self, filepath: Path) -> Chunk
```

Load a single chunk file.

#### ChunkLoader.load_all

```python
ChunkLoader.load_all(self) -> Dict[str, str]
```

Load all chunks and replay operations to get current document state.

#### ChunkLoader.get_documents

```python
ChunkLoader.get_documents(self) -> Dict[str, str]
```

Get loaded documents (calls load_all if needed).

#### ChunkLoader.get_mtimes

```python
ChunkLoader.get_mtimes(self) -> Dict[str, float]
```

Get document modification times.

#### ChunkLoader.get_metadata

```python
ChunkLoader.get_metadata(self) -> Dict[str, Dict[str, Any]]
```

Get document metadata (doc_type, headings, etc.).

#### ChunkLoader.get_chunks

```python
ChunkLoader.get_chunks(self) -> List[Chunk]
```

Get loaded chunks.

#### ChunkLoader.compute_hash

```python
ChunkLoader.compute_hash(self) -> str
```

Compute hash of current document state.

#### ChunkLoader.is_cache_valid

```python
ChunkLoader.is_cache_valid(self, cache_path: str, cache_hash_path: Optional[str] = None) -> bool
```

Check if pkl cache is valid for current chunk state.

#### ChunkLoader.save_cache_hash

```python
ChunkLoader.save_cache_hash(self, cache_path: str, cache_hash_path: Optional[str] = None)
```

Save current document hash for cache validation.

#### ChunkLoader.get_stats

```python
ChunkLoader.get_stats(self) -> Dict[str, Any]
```

Get statistics about loaded chunks.

#### ChunkCompactor.compact

```python
ChunkCompactor.compact(self, before: Optional[str] = None, keep_recent: int = 0, dry_run: bool = False) -> Dict[str, Any]
```

Compact chunks into a single chunk.

### Dependencies

**Standard Library:**

- `dataclasses.asdict`
- `dataclasses.dataclass`
- `dataclasses.field`
- `datetime.datetime`
- `hashlib`
- ... and 11 more



## persistence.py

Persistence Module
==================

Save and load functionality for the cortical processor.

Supports:
- Pickle serialization for full state
- JSON export for graph visualization
- Incremental upda...


### Classes

#### SignatureVerificationError

Raised when HMAC signature verification fails.

### Functions

#### save_processor

```python
save_processor(filepath: str, layers: Dict[CorticalLayer, HierarchicalLayer], documents: Dict[str, str], document_metadata: Optional[Dict[str, Dict[str, Any]]] = None, embeddings: Optional[Dict[str, list]] = None, semantic_relations: Optional[list] = None, metadata: Optional[Dict] = None, verbose: bool = True, format: str = 'pickle', signing_key: Optional[bytes] = None) -> None
```

Save processor state to a file.

#### load_processor

```python
load_processor(filepath: str, verbose: bool = True, format: Optional[str] = None, verify_key: Optional[bytes] = None) -> tuple
```

Load processor state from a file.

#### export_graph_json

```python
export_graph_json(filepath: str, layers: Dict[CorticalLayer, HierarchicalLayer], layer_filter: Optional[CorticalLayer] = None, min_weight: float = 0.0, max_nodes: int = 500, verbose: bool = True) -> Dict
```

Export graph structure as JSON for visualization.

#### export_embeddings_json

```python
export_embeddings_json(filepath: str, embeddings: Dict[str, list], metadata: Optional[Dict] = None) -> None
```

Export embeddings as JSON.

#### load_embeddings_json

```python
load_embeddings_json(filepath: str) -> Dict[str, list]
```

Load embeddings from JSON.

#### export_semantic_relations_json

```python
export_semantic_relations_json(filepath: str, relations: list) -> None
```

Export semantic relations as JSON.

#### load_semantic_relations_json

```python
load_semantic_relations_json(filepath: str) -> list
```

Load semantic relations from JSON.

#### get_state_summary

```python
get_state_summary(layers: Dict[CorticalLayer, HierarchicalLayer], documents: Dict[str, str]) -> Dict
```

Get a summary of the current processor state.

#### export_conceptnet_json

```python
export_conceptnet_json(filepath: str, layers: Dict[CorticalLayer, HierarchicalLayer], semantic_relations: Optional[list] = None, include_cross_layer: bool = True, include_typed_edges: bool = True, min_weight: float = 0.0, min_confidence: float = 0.0, max_nodes_per_layer: int = 100, verbose: bool = True) -> Dict[str, Any]
```

Export ConceptNet-style graph for visualization.

### Dependencies

**Standard Library:**

- `hashlib`
- `hmac`
- `json`
- `layers.CorticalLayer`
- `layers.HierarchicalLayer`
- ... and 9 more



## state_storage.py

Git-friendly State Storage Module
=================================

Replaces pickle-based persistence with JSON files that:
- Can be diff'd and reviewed in git
- Won't cause merge conflicts
- Support...


### Classes

#### StateManifest

Manifest file tracking state version and component checksums.

**Methods:**

- `to_dict`
- `from_dict`
- `update_checksum`

#### StateWriter

Writes processor state to git-friendly JSON files.

**Methods:**

- `save_layer`
- `save_documents`
- `save_semantic_relations`
- `save_embeddings`
- `save_manifest`
- `save_all`

#### StateLoader

Loads processor state from git-friendly JSON files.

**Methods:**

- `exists`
- `load_manifest`
- `validate_checksum`
- `load_layer`
- `load_documents`
- `load_semantic_relations`
- `load_embeddings`
- `load_all`
- `get_stats`

### Functions

#### migrate_pkl_to_json

```python
migrate_pkl_to_json(pkl_path: str, json_dir: str, verbose: bool = True) -> bool
```

Migrate a pickle file to git-friendly JSON format.

#### StateManifest.to_dict

```python
StateManifest.to_dict(self) -> Dict[str, Any]
```

Convert to dictionary for JSON serialization.

#### StateManifest.from_dict

```python
StateManifest.from_dict(cls, data: Dict[str, Any]) -> 'StateManifest'
```

Create manifest from dictionary.

#### StateManifest.update_checksum

```python
StateManifest.update_checksum(self, component: str, content: str) -> bool
```

Update checksum for a component.

#### StateWriter.save_layer

```python
StateWriter.save_layer(self, layer: HierarchicalLayer, force: bool = False) -> bool
```

Save a single layer to its JSON file.

#### StateWriter.save_documents

```python
StateWriter.save_documents(self, documents: Dict[str, str], document_metadata: Optional[Dict[str, Dict[str, Any]]] = None, force: bool = False) -> bool
```

Save documents and metadata.

#### StateWriter.save_semantic_relations

```python
StateWriter.save_semantic_relations(self, relations: List[Tuple], force: bool = False) -> bool
```

Save semantic relations.

#### StateWriter.save_embeddings

```python
StateWriter.save_embeddings(self, embeddings: Dict[str, List[float]], force: bool = False) -> bool
```

Save graph embeddings.

#### StateWriter.save_manifest

```python
StateWriter.save_manifest(self) -> None
```

Save the manifest file.

#### StateWriter.save_all

```python
StateWriter.save_all(self, layers: Dict[CorticalLayer, HierarchicalLayer], documents: Dict[str, str], document_metadata: Optional[Dict[str, Dict[str, Any]]] = None, embeddings: Optional[Dict[str, List[float]]] = None, semantic_relations: Optional[List[Tuple]] = None, stale_computations: Optional[Set[str]] = None, force: bool = False, verbose: bool = True) -> Dict[str, bool]
```

Save all processor state.

#### StateLoader.exists

```python
StateLoader.exists(self) -> bool
```

Check if state directory exists and has manifest.

#### StateLoader.load_manifest

```python
StateLoader.load_manifest(self) -> StateManifest
```

Load the manifest file.

#### StateLoader.validate_checksum

```python
StateLoader.validate_checksum(self, component: str, filepath: Path) -> bool
```

Validate a component's checksum.

#### StateLoader.load_layer

```python
StateLoader.load_layer(self, level: int) -> HierarchicalLayer
```

Load a single layer.

#### StateLoader.load_documents

```python
StateLoader.load_documents(self) -> Tuple[Dict[str, str], Dict[str, Dict[str, Any]]]
```

Load documents and metadata.

#### StateLoader.load_semantic_relations

```python
StateLoader.load_semantic_relations(self) -> List[Tuple]
```

Load semantic relations.

#### StateLoader.load_embeddings

```python
StateLoader.load_embeddings(self) -> Dict[str, List[float]]
```

Load graph embeddings.

#### StateLoader.load_all

```python
StateLoader.load_all(self, validate: bool = True, verbose: bool = True) -> Tuple[Dict[CorticalLayer, HierarchicalLayer], Dict[str, str], Dict[str, Dict[str, Any]], Dict[str, List[float]], List[Tuple], Dict[str, Any]]
```

Load all processor state.

#### StateLoader.get_stats

```python
StateLoader.get_stats(self) -> Dict[str, Any]
```

Get statistics about stored state without loading everything.

### Dependencies

**Standard Library:**

- `dataclasses.asdict`
- `dataclasses.dataclass`
- `dataclasses.field`
- `datetime.datetime`
- `hashlib`
- ... and 13 more



---

*This chapter is part of [The Cortical Chronicles](../README.md), a self-documenting book generated by the Cortical Text Processor.*
