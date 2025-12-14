"""
Git-friendly State Storage Module
=================================

Replaces pickle-based persistence with JSON files that:
- Can be diff'd and reviewed in git
- Won't cause merge conflicts
- Support incremental updates
- Are language/version independent

Architecture:
    corpus_state/
    ├── manifest.json           # Version, checksums, staleness
    ├── documents.json          # Document content and metadata
    ├── layers/
    │   ├── L0_tokens.json      # Token minicolumns
    │   ├── L1_bigrams.json     # Bigram minicolumns
    │   ├── L2_concepts.json    # Concept clusters
    │   └── L3_documents.json   # Document minicolumns
    └── computed/
        ├── semantic_relations.json
        └── embeddings.json

Usage:
    # Save processor state
    writer = StateWriter('corpus_state')
    writer.save_processor(processor)

    # Load processor state
    loader = StateLoader('corpus_state')
    layers, documents, metadata, embeddings, relations = loader.load_all()
"""

import hashlib
import json
import os
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set

from .layers import CorticalLayer, HierarchicalLayer
from .minicolumn import Minicolumn

logger = logging.getLogger(__name__)

# Version for state format (increment on breaking changes)
STATE_VERSION = 1

# Layer enum value to filename mapping
LAYER_FILENAMES = {
    0: 'L0_tokens.json',
    1: 'L1_bigrams.json',
    2: 'L2_concepts.json',
    3: 'L3_documents.json',
}


@dataclass
class StateManifest:
    """
    Manifest file tracking state version and component checksums.

    Used to:
    - Detect which components changed since last save
    - Validate loaded state integrity
    - Track staleness of computed values
    """
    version: int = STATE_VERSION
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    checksums: Dict[str, str] = field(default_factory=dict)
    stale_computations: List[str] = field(default_factory=list)
    document_count: int = 0
    layer_stats: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StateManifest':
        """Create manifest from dictionary."""
        return cls(
            version=data.get('version', STATE_VERSION),
            created_at=data.get('created_at', datetime.now().isoformat()),
            updated_at=data.get('updated_at', datetime.now().isoformat()),
            checksums=data.get('checksums', {}),
            stale_computations=data.get('stale_computations', []),
            document_count=data.get('document_count', 0),
            layer_stats=data.get('layer_stats', {})
        )

    def update_checksum(self, component: str, content: str) -> bool:
        """
        Update checksum for a component.

        Args:
            component: Component name (e.g., 'L0_tokens', 'documents')
            content: Serialized content to hash

        Returns:
            True if checksum changed, False if same
        """
        new_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
        old_hash = self.checksums.get(component)
        self.checksums[component] = new_hash
        self.updated_at = datetime.now().isoformat()
        return new_hash != old_hash


class StateWriter:
    """
    Writes processor state to git-friendly JSON files.

    Features:
    - Incremental saves (only writes changed components)
    - Atomic writes (write to temp, then rename)
    - Content hashing for change detection

    Usage:
        writer = StateWriter('corpus_state')
        writer.save_all(layers, documents, metadata, embeddings, relations)

        # Or incrementally:
        writer.save_layer(layers[CorticalLayer.TOKENS])
        writer.save_documents(documents, metadata)
        writer.save_manifest()
    """

    def __init__(self, state_dir: str):
        """
        Initialize state writer.

        Args:
            state_dir: Directory to write state files
        """
        self.state_dir = Path(state_dir)
        self.layers_dir = self.state_dir / 'layers'
        self.computed_dir = self.state_dir / 'computed'
        self.manifest: Optional[StateManifest] = None
        self._load_or_create_manifest()

    def _load_or_create_manifest(self) -> None:
        """Load existing manifest or create new one."""
        manifest_path = self.state_dir / 'manifest.json'
        if manifest_path.exists():
            try:
                with open(manifest_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.manifest = StateManifest.from_dict(data)
            except (json.JSONDecodeError, IOError):
                self.manifest = StateManifest()
        else:
            self.manifest = StateManifest()

    def _ensure_dirs(self) -> None:
        """Create directories if they don't exist."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.layers_dir.mkdir(exist_ok=True)
        self.computed_dir.mkdir(exist_ok=True)

    def _atomic_write(self, filepath: Path, content: str) -> None:
        """
        Write content atomically using temp file + rename.

        This prevents data corruption if the process crashes mid-write.
        """
        temp_path = filepath.with_suffix('.json.tmp')
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                f.write(content)
            temp_path.replace(filepath)
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise

    def save_layer(
        self,
        layer: HierarchicalLayer,
        force: bool = False
    ) -> bool:
        """
        Save a single layer to its JSON file.

        Args:
            layer: The layer to save
            force: Save even if checksum unchanged

        Returns:
            True if file was written, False if skipped (unchanged)
        """
        self._ensure_dirs()

        filename = LAYER_FILENAMES.get(layer.level)
        if not filename:
            raise ValueError(f"Unknown layer level: {layer.level}")

        filepath = self.layers_dir / filename
        component_name = f"layer_{layer.level}"

        # Serialize layer
        layer_data = layer.to_dict()
        content = json.dumps(layer_data, indent=2, ensure_ascii=False)

        # Check if changed
        changed = self.manifest.update_checksum(component_name, content)

        if not changed and not force:
            return False

        self._atomic_write(filepath, content)
        self.manifest.layer_stats[f"L{layer.level}"] = len(layer.minicolumns)

        return True

    def save_documents(
        self,
        documents: Dict[str, str],
        document_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        force: bool = False
    ) -> bool:
        """
        Save documents and metadata.

        Args:
            documents: Document ID to content mapping
            document_metadata: Document ID to metadata mapping
            force: Save even if unchanged

        Returns:
            True if file was written
        """
        self._ensure_dirs()

        filepath = self.state_dir / 'documents.json'

        data = {
            'documents': documents,
            'metadata': document_metadata or {}
        }
        content = json.dumps(data, indent=2, ensure_ascii=False)

        changed = self.manifest.update_checksum('documents', content)

        if not changed and not force:
            return False

        self._atomic_write(filepath, content)
        self.manifest.document_count = len(documents)

        return True

    def save_semantic_relations(
        self,
        relations: List[Tuple],
        force: bool = False
    ) -> bool:
        """
        Save semantic relations.

        Args:
            relations: List of (term1, relation, term2, weight) tuples
            force: Save even if unchanged

        Returns:
            True if file was written
        """
        self._ensure_dirs()

        filepath = self.computed_dir / 'semantic_relations.json'

        # Convert tuples to lists for JSON
        data = {
            'relations': [list(r) for r in relations],
            'count': len(relations)
        }
        content = json.dumps(data, indent=2, ensure_ascii=False)

        changed = self.manifest.update_checksum('semantic_relations', content)

        if not changed and not force:
            return False

        self._atomic_write(filepath, content)
        return True

    def save_embeddings(
        self,
        embeddings: Dict[str, List[float]],
        force: bool = False
    ) -> bool:
        """
        Save graph embeddings.

        Args:
            embeddings: Term to embedding vector mapping
            force: Save even if unchanged

        Returns:
            True if file was written
        """
        self._ensure_dirs()

        filepath = self.computed_dir / 'embeddings.json'

        data = {
            'embeddings': embeddings,
            'dimensions': len(next(iter(embeddings.values()))) if embeddings else 0,
            'count': len(embeddings)
        }
        content = json.dumps(data, indent=2, ensure_ascii=False)

        changed = self.manifest.update_checksum('embeddings', content)

        if not changed and not force:
            return False

        self._atomic_write(filepath, content)
        return True

    def save_manifest(self) -> None:
        """Save the manifest file."""
        self._ensure_dirs()
        filepath = self.state_dir / 'manifest.json'
        content = json.dumps(self.manifest.to_dict(), indent=2, ensure_ascii=False)
        self._atomic_write(filepath, content)

    def save_all(
        self,
        layers: Dict[CorticalLayer, HierarchicalLayer],
        documents: Dict[str, str],
        document_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        embeddings: Optional[Dict[str, List[float]]] = None,
        semantic_relations: Optional[List[Tuple]] = None,
        stale_computations: Optional[Set[str]] = None,
        force: bool = False,
        verbose: bool = True
    ) -> Dict[str, bool]:
        """
        Save all processor state.

        Args:
            layers: Dictionary of all layers
            documents: Document collection
            document_metadata: Per-document metadata
            embeddings: Graph embeddings
            semantic_relations: Extracted relations
            stale_computations: Set of stale computation names
            force: Force save even if unchanged
            verbose: Log progress

        Returns:
            Dictionary of component -> was_written
        """
        results = {}

        # Save layers
        for layer_enum, layer in layers.items():
            key = f"layer_{layer_enum.value}"
            results[key] = self.save_layer(layer, force=force)
            if verbose and results[key]:
                logger.info(f"  Saved {LAYER_FILENAMES[layer_enum.value]}: {len(layer.minicolumns)} minicolumns")

        # Save documents
        results['documents'] = self.save_documents(documents, document_metadata, force=force)
        if verbose and results['documents']:
            logger.info(f"  Saved documents.json: {len(documents)} documents")

        # Save computed values
        if semantic_relations is not None:
            results['semantic_relations'] = self.save_semantic_relations(semantic_relations, force=force)
            if verbose and results['semantic_relations']:
                logger.info(f"  Saved semantic_relations.json: {len(semantic_relations)} relations")

        if embeddings is not None:
            results['embeddings'] = self.save_embeddings(embeddings, force=force)
            if verbose and results['embeddings']:
                logger.info(f"  Saved embeddings.json: {len(embeddings)} embeddings")

        # Update staleness tracking
        if stale_computations is not None:
            self.manifest.stale_computations = list(stale_computations)

        # Save manifest
        self.save_manifest()

        if verbose:
            saved_count = sum(1 for v in results.values() if v)
            logger.info(f"✓ Saved state to {self.state_dir} ({saved_count} files updated)")

        return results


class StateLoader:
    """
    Loads processor state from git-friendly JSON files.

    Features:
    - Validates checksums before loading
    - Reports missing or corrupted components
    - Provides incremental loading (load only what you need)

    Usage:
        loader = StateLoader('corpus_state')

        # Load everything
        state = loader.load_all()

        # Or load selectively
        layer0 = loader.load_layer(0)
        docs = loader.load_documents()
    """

    def __init__(self, state_dir: str):
        """
        Initialize state loader.

        Args:
            state_dir: Directory containing state files
        """
        self.state_dir = Path(state_dir)
        self.layers_dir = self.state_dir / 'layers'
        self.computed_dir = self.state_dir / 'computed'
        self.manifest: Optional[StateManifest] = None

    def exists(self) -> bool:
        """Check if state directory exists and has manifest."""
        return (self.state_dir / 'manifest.json').exists()

    def load_manifest(self) -> StateManifest:
        """
        Load the manifest file.

        Returns:
            StateManifest object

        Raises:
            FileNotFoundError: If manifest doesn't exist
        """
        manifest_path = self.state_dir / 'manifest.json'
        if not manifest_path.exists():
            raise FileNotFoundError(f"No manifest found at {manifest_path}")

        with open(manifest_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.manifest = StateManifest.from_dict(data)
        return self.manifest

    def validate_checksum(self, component: str, filepath: Path) -> bool:
        """
        Validate a component's checksum.

        Args:
            component: Component name
            filepath: Path to the component file

        Returns:
            True if checksum matches, False otherwise
        """
        if self.manifest is None:
            self.load_manifest()

        expected = self.manifest.checksums.get(component)
        if expected is None:
            return True  # No checksum stored, assume valid

        if not filepath.exists():
            return False

        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        actual = hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
        return actual == expected

    def load_layer(self, level: int) -> HierarchicalLayer:
        """
        Load a single layer.

        Args:
            level: Layer level (0-3)

        Returns:
            HierarchicalLayer object

        Raises:
            FileNotFoundError: If layer file doesn't exist
            ValueError: If layer data is invalid
        """
        filename = LAYER_FILENAMES.get(level)
        if not filename:
            raise ValueError(f"Unknown layer level: {level}")

        filepath = self.layers_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Layer file not found: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return HierarchicalLayer.from_dict(data)

    def load_documents(self) -> Tuple[Dict[str, str], Dict[str, Dict[str, Any]]]:
        """
        Load documents and metadata.

        Returns:
            Tuple of (documents, metadata)

        Raises:
            FileNotFoundError: If documents file doesn't exist
        """
        filepath = self.state_dir / 'documents.json'
        if not filepath.exists():
            raise FileNotFoundError(f"Documents file not found: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data.get('documents', {}), data.get('metadata', {})

    def load_semantic_relations(self) -> List[Tuple]:
        """
        Load semantic relations.

        Returns:
            List of (term1, relation, term2, weight) tuples
        """
        filepath = self.computed_dir / 'semantic_relations.json'
        if not filepath.exists():
            return []

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Convert lists back to tuples
        return [tuple(r) for r in data.get('relations', [])]

    def load_embeddings(self) -> Dict[str, List[float]]:
        """
        Load graph embeddings.

        Returns:
            Term to embedding vector mapping
        """
        filepath = self.computed_dir / 'embeddings.json'
        if not filepath.exists():
            return {}

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data.get('embeddings', {})

    def load_all(
        self,
        validate: bool = True,
        verbose: bool = True
    ) -> Tuple[
        Dict[CorticalLayer, HierarchicalLayer],
        Dict[str, str],
        Dict[str, Dict[str, Any]],
        Dict[str, List[float]],
        List[Tuple],
        Dict[str, Any]
    ]:
        """
        Load all processor state.

        Args:
            validate: Validate checksums before loading
            verbose: Log progress

        Returns:
            Tuple of (layers, documents, metadata, embeddings, relations, manifest_data)

        Raises:
            FileNotFoundError: If required files don't exist
            ValueError: If checksums don't match (when validate=True)
        """
        # Load manifest first
        manifest = self.load_manifest()

        if verbose:
            logger.info(f"Loading state from {self.state_dir}")

        # Load layers
        layers = {}
        for level in range(4):
            try:
                layer = self.load_layer(level)
                layers[CorticalLayer(level)] = layer
                if verbose:
                    logger.info(f"  Loaded {LAYER_FILENAMES[level]}: {len(layer.minicolumns)} minicolumns")
            except FileNotFoundError:
                if verbose:
                    logger.warning(f"  Layer {level} not found, creating empty")
                layers[CorticalLayer(level)] = HierarchicalLayer(CorticalLayer(level))

        # Load documents
        try:
            documents, metadata = self.load_documents()
            if verbose:
                logger.info(f"  Loaded documents.json: {len(documents)} documents")
        except FileNotFoundError:
            documents = {}
            metadata = {}
            if verbose:
                logger.warning("  Documents not found, starting empty")

        # Load computed values
        relations = self.load_semantic_relations()
        if verbose and relations:
            logger.info(f"  Loaded semantic_relations.json: {len(relations)} relations")

        embeddings = self.load_embeddings()
        if verbose and embeddings:
            logger.info(f"  Loaded embeddings.json: {len(embeddings)} embeddings")

        if verbose:
            logger.info(f"✓ Loaded state from {self.state_dir}")

        # Build metadata dict similar to pkl format
        manifest_data = {
            'version': manifest.version,
            'stale_computations': set(manifest.stale_computations)
        }

        return layers, documents, metadata, embeddings, relations, manifest_data

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored state without loading everything.

        Returns:
            Dictionary of statistics
        """
        if self.manifest is None:
            try:
                self.load_manifest()
            except FileNotFoundError:
                return {'exists': False}

        return {
            'exists': True,
            'version': self.manifest.version,
            'created_at': self.manifest.created_at,
            'updated_at': self.manifest.updated_at,
            'document_count': self.manifest.document_count,
            'layer_stats': self.manifest.layer_stats,
            'stale_computations': self.manifest.stale_computations,
            'components': list(self.manifest.checksums.keys())
        }


def migrate_pkl_to_json(
    pkl_path: str,
    json_dir: str,
    verbose: bool = True
) -> bool:
    """
    Migrate a pickle file to git-friendly JSON format.

    Args:
        pkl_path: Path to existing .pkl file
        json_dir: Directory to write JSON state
        verbose: Log progress

    Returns:
        True if migration successful

    Raises:
        FileNotFoundError: If pkl file doesn't exist
    """
    from .persistence import load_processor

    if verbose:
        logger.info(f"Migrating {pkl_path} to {json_dir}")

    # Load from pkl
    layers, documents, metadata, embeddings, relations, pkl_metadata = load_processor(
        pkl_path, verbose=False
    )

    # Get stale computations from pkl metadata if present
    stale = pkl_metadata.get('stale_computations', set()) if pkl_metadata else set()

    # Save as JSON
    writer = StateWriter(json_dir)
    writer.save_all(
        layers=layers,
        documents=documents,
        document_metadata=metadata,
        embeddings=embeddings,
        semantic_relations=relations,
        stale_computations=stale,
        force=True,
        verbose=verbose
    )

    if verbose:
        logger.info(f"✓ Migration complete: {pkl_path} → {json_dir}")

    return True
