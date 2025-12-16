"""
Persistence API: save, load, export, and migration methods.

This module contains all methods related to saving and loading processor state.
"""

import logging
from typing import Dict, Optional, Any, TYPE_CHECKING

from ..layers import CorticalLayer
from ..config import CorticalConfig
from .. import persistence
from .. import state_storage
from ..observability import timed

if TYPE_CHECKING:
    from . import CorticalTextProcessor

logger = logging.getLogger(__name__)


class PersistenceMixin:
    """
    Mixin providing persistence functionality.

    Requires CoreMixin to be present (provides layers, documents, document_metadata,
    embeddings, semantic_relations, config, _stale_computations).
    """

    @timed("save")
    def save(
        self,
        filepath: str,
        verbose: bool = True,
        signing_key: Optional[bytes] = None
    ) -> None:
        """
        Save processor state to a file.

        Saves all computed state including embeddings, semantic relations,
        and configuration, so they don't need to be recomputed when loading.

        Args:
            filepath: Path to save file
            verbose: Print progress
            signing_key: Optional HMAC key for signing pickle files (SEC-003).
                If provided, creates a .sig file alongside the pickle file.
        """
        metadata = {
            'has_embeddings': bool(self.embeddings),
            'has_relations': bool(self.semantic_relations),
            'config': self.config.to_dict(),
            'doc_lengths': self.doc_lengths,
            'avg_doc_length': self.avg_doc_length
        }
        persistence.save_processor(
            filepath,
            self.layers,
            self.documents,
            self.document_metadata,
            self.embeddings,
            self.semantic_relations,
            metadata,
            verbose,
            signing_key=signing_key
        )

    @classmethod
    def load(
        cls,
        filepath: str,
        verbose: bool = True,
        verify_key: Optional[bytes] = None
    ) -> 'CorticalTextProcessor':
        """
        Load processor state from a file.

        Restores all computed state including embeddings, semantic relations,
        and configuration.

        Args:
            filepath: Path to saved file
            verbose: Print progress
            verify_key: Optional HMAC key for verifying pickle file signatures (SEC-003).

        Raises:
            SignatureVerificationError: If verify_key is provided and verification fails
            FileNotFoundError: If verify_key is provided but no .sig file exists
        """
        result = persistence.load_processor(filepath, verbose, verify_key=verify_key)
        layers, documents, document_metadata, embeddings, semantic_relations, metadata = result

        # Restore config if available
        config = None
        if metadata and 'config' in metadata:
            try:
                config = CorticalConfig.from_dict(metadata['config'])
            except (KeyError, TypeError):
                config = None

        processor = cls(config=config)
        processor.layers = layers
        processor.documents = documents
        processor.document_metadata = document_metadata
        processor.embeddings = embeddings
        processor.semantic_relations = semantic_relations

        # Restore BM25 document length data if available
        if metadata:
            processor.doc_lengths = metadata.get('doc_lengths', {})
            processor.avg_doc_length = metadata.get('avg_doc_length', 0.0)

        # Recompute doc_lengths if not in metadata (backward compatibility)
        if not processor.doc_lengths and processor.documents:
            from ..tokenizer import Tokenizer
            tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else Tokenizer()
            for doc_id, content in processor.documents.items():
                tokens = tokenizer.tokenize(content)
                processor.doc_lengths[doc_id] = len(tokens)
            if processor.doc_lengths:
                processor.avg_doc_length = sum(processor.doc_lengths.values()) / len(processor.doc_lengths)

        return processor

    def save_json(self, state_dir: str, force: bool = False, verbose: bool = True) -> Dict[str, bool]:
        """
        Save processor state to git-friendly JSON format.

        Instead of a single monolithic pickle file, creates a directory with:
        - manifest.json: Version, checksums, staleness tracking
        - documents.json: Document content and metadata
        - layers/*.json: One file per layer
        - computed/*.json: Semantic relations and embeddings

        Args:
            state_dir: Directory to write JSON state files
            force: Force save even if unchanged (default: False)
            verbose: Print progress messages (default: True)

        Returns:
            Dictionary mapping component names to whether they were written
        """
        writer = state_storage.StateWriter(state_dir)

        stale = self._stale_computations if hasattr(self, '_stale_computations') else set()

        # Save config and BM25 metadata
        writer.save_config(
            self.config.to_dict(),
            self.doc_lengths,
            self.avg_doc_length
        )

        return writer.save_all(
            layers=self.layers,
            documents=self.documents,
            document_metadata=self.document_metadata,
            embeddings=self.embeddings,
            semantic_relations=self.semantic_relations,
            stale_computations=stale,
            force=force,
            verbose=verbose
        )

    @classmethod
    def load_json(
        cls,
        state_dir: str,
        config: Optional[CorticalConfig] = None,
        verbose: bool = True
    ) -> 'CorticalTextProcessor':
        """
        Load processor from git-friendly JSON format.

        Args:
            state_dir: Directory containing JSON state files
            config: Optional configuration (default: uses CorticalConfig defaults)
            verbose: Print progress messages (default: True)

        Returns:
            Reconstructed CorticalTextProcessor instance

        Raises:
            FileNotFoundError: If state directory or required files don't exist
            ValueError: If state format is invalid
        """
        loader = state_storage.StateLoader(state_dir)

        layers, documents, metadata, embeddings, relations, manifest_data = loader.load_all(
            validate=True,
            verbose=verbose
        )

        # Load config and BM25 metadata
        config_dict, doc_lengths, avg_doc_length = loader.load_config()

        # Use saved config if available and no override provided
        if config_dict and config is None:
            config = CorticalConfig.from_dict(config_dict)

        processor = cls(config=config)
        processor.layers = layers
        processor.documents = documents
        processor.document_metadata = metadata
        processor.embeddings = embeddings
        processor.semantic_relations = relations

        # Restore BM25 document length data
        processor.doc_lengths = doc_lengths
        processor.avg_doc_length = avg_doc_length

        if 'stale_computations' in manifest_data:
            processor._stale_computations = manifest_data['stale_computations']

        return processor

    def migrate_to_json(self, pkl_path: str, json_dir: str, verbose: bool = True) -> bool:
        """
        Migrate existing pickle file to git-friendly JSON format.

        Args:
            pkl_path: Path to existing .pkl file
            json_dir: Directory to write JSON state
            verbose: Print progress messages (default: True)

        Returns:
            True if migration successful

        Raises:
            FileNotFoundError: If pkl file doesn't exist
        """
        return state_storage.migrate_pkl_to_json(pkl_path, json_dir, verbose=verbose)

    def export_graph(
        self,
        filepath: str,
        layer: Optional[CorticalLayer] = None,
        max_nodes: int = 500
    ) -> Dict:
        """Export graph to JSON for visualization."""
        return persistence.export_graph_json(filepath, self.layers, layer, max_nodes=max_nodes)

    def export_conceptnet_json(
        self,
        filepath: str,
        include_cross_layer: bool = True,
        include_typed_edges: bool = True,
        min_weight: float = 0.0,
        min_confidence: float = 0.0,
        max_nodes_per_layer: int = 100,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Export ConceptNet-style graph for visualization.

        Creates a rich graph format with color-coded nodes by layer
        and typed edges with relation types and confidence scores.

        Args:
            filepath: Output file path (JSON)
            include_cross_layer: Include feedforward/feedback edges
            include_typed_edges: Include typed_connections with relation types
            min_weight: Minimum edge weight to include
            min_confidence: Minimum confidence for typed edges
            max_nodes_per_layer: Maximum nodes per layer (by PageRank)
            verbose: Print progress messages

        Returns:
            The exported graph data
        """
        return persistence.export_conceptnet_json(
            filepath,
            self.layers,
            semantic_relations=self.semantic_relations,
            include_cross_layer=include_cross_layer,
            include_typed_edges=include_typed_edges,
            min_weight=min_weight,
            min_confidence=min_confidence,
            max_nodes_per_layer=max_nodes_per_layer,
            verbose=verbose
        )
