"""
Compute methods: analysis, clustering, embeddings, semantic extraction.

This module contains all methods that perform computational analysis on the corpus,
including PageRank, TF-IDF, clustering, and checkpointing.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set

from ..layers import CorticalLayer
from .. import analysis
from .. import semantics
from .. import embeddings as emb_module
from ..progress import (
    ProgressReporter,
    ConsoleProgressReporter,
    SilentProgressReporter,
    MultiPhaseProgress
)
from ..observability import timed

logger = logging.getLogger(__name__)


class ComputeMixin:
    """
    Mixin providing computation functionality.

    Requires CoreMixin to be present (provides layers, documents, tokenizer,
    config, COMP_*, _mark_all_stale, _mark_fresh, _stale_computations).
    """

    def recompute(
        self,
        level: str = 'stale',
        verbose: bool = True
    ) -> Dict[str, bool]:
        """
        Recompute specified analysis levels.

        Use this after adding documents with recompute='none' to batch
        the recomputation step.

        Args:
            level: What to recompute:
                - 'stale': Only recompute what's marked as stale
                - 'tfidf': Only TF-IDF (marks others stale)
                - 'full': Run complete compute_all()
            verbose: Print progress messages

        Returns:
            Dict indicating what was recomputed

        Example:
            >>> # Add documents without recomputation
            >>> processor.add_document_incremental("doc1", "content", recompute='none')
            >>> processor.add_document_incremental("doc2", "content", recompute='none')
            >>> # Batch recompute
            >>> processor.recompute(level='full')
        """
        recomputed = {}

        if level == 'full':
            self.compute_all(verbose=verbose)
            self._stale_computations.clear()
            recomputed = {
                self.COMP_ACTIVATION: True,
                self.COMP_PAGERANK: True,
                self.COMP_TFIDF: True,
                self.COMP_DOC_CONNECTIONS: True,
                self.COMP_BIGRAM_CONNECTIONS: True,
                self.COMP_CONCEPTS: True,
            }
        elif level == 'tfidf':
            self.compute_tfidf(verbose=verbose)
            self._mark_fresh(self.COMP_TFIDF)
            recomputed[self.COMP_TFIDF] = True
        elif level == 'stale':
            # Recompute only what's stale, in dependency order
            if self.COMP_ACTIVATION in self._stale_computations:
                self.propagate_activation(verbose=verbose)
                self._mark_fresh(self.COMP_ACTIVATION)
                recomputed[self.COMP_ACTIVATION] = True

            if self.COMP_PAGERANK in self._stale_computations:
                self.compute_importance(verbose=verbose)
                self._mark_fresh(self.COMP_PAGERANK)
                recomputed[self.COMP_PAGERANK] = True

            if self.COMP_TFIDF in self._stale_computations:
                self.compute_tfidf(verbose=verbose)
                self._mark_fresh(self.COMP_TFIDF)
                recomputed[self.COMP_TFIDF] = True

            if self.COMP_DOC_CONNECTIONS in self._stale_computations:
                self.compute_document_connections(verbose=verbose)
                self._mark_fresh(self.COMP_DOC_CONNECTIONS)
                recomputed[self.COMP_DOC_CONNECTIONS] = True

            if self.COMP_BIGRAM_CONNECTIONS in self._stale_computations:
                self.compute_bigram_connections(verbose=verbose)
                self._mark_fresh(self.COMP_BIGRAM_CONNECTIONS)
                recomputed[self.COMP_BIGRAM_CONNECTIONS] = True

            if self.COMP_CONCEPTS in self._stale_computations:
                self.build_concept_clusters(verbose=verbose)
                self._mark_fresh(self.COMP_CONCEPTS)
                recomputed[self.COMP_CONCEPTS] = True

            if self.COMP_EMBEDDINGS in self._stale_computations:
                self.compute_graph_embeddings(verbose=verbose)
                self._mark_fresh(self.COMP_EMBEDDINGS)
                recomputed[self.COMP_EMBEDDINGS] = True

            if self.COMP_SEMANTICS in self._stale_computations:
                self.extract_corpus_semantics(verbose=verbose)
                self._mark_fresh(self.COMP_SEMANTICS)
                recomputed[self.COMP_SEMANTICS] = True

        return recomputed

    @timed("compute_all")
    def compute_all(
        self,
        verbose: bool = True,
        build_concepts: bool = True,
        pagerank_method: str = 'standard',
        connection_strategy: str = 'document_overlap',
        cluster_strictness: float = 1.0,
        bridge_weight: float = 0.0,
        progress_callback: Optional[ProgressReporter] = None,
        show_progress: bool = False,
        checkpoint_dir: Optional[str] = None,
        resume: bool = False,
        parallel: bool = False,
        parallel_num_workers: Optional[int] = None,
        parallel_chunk_size: int = 1000
    ) -> Dict[str, Any]:
        """
        Run all computation steps.

        Args:
            verbose: Print debug messages via Python logging (complementary to show_progress)
            build_concepts: Build concept clusters in Layer 2 (default True)
                           This enables topic-based filtering and hierarchical search.
            pagerank_method: PageRank algorithm to use:
                - 'standard': Traditional PageRank using connection weights
                - 'semantic': ConceptNet-style PageRank with relation type weighting.
                              Requires semantic relations (extracts automatically if needed).
                - 'hierarchical': Cross-layer PageRank with importance propagation
                                  between layers (tokens <-> bigrams <-> concepts <-> documents).
            connection_strategy: Strategy for connecting Layer 2 concepts:
                - 'document_overlap': Traditional Jaccard similarity (default)
                - 'semantic': Connect via semantic relations between members
                - 'embedding': Connect via embedding centroid similarity
                - 'hybrid': Combine all three strategies for maximum connectivity
            cluster_strictness: Controls clustering aggressiveness (0.0-1.0).
                Lower values create fewer, larger clusters with more connections.
            bridge_weight: Weight for inter-document token bridging (0.0-1.0).
                Higher values help bridge topic-isolated clusters.
            progress_callback: Optional ProgressReporter for custom progress tracking
            show_progress: Show progress bar on console (uses stderr)
            checkpoint_dir: Directory to save checkpoints after each phase (enables checkpointing).
                If None (default), no checkpointing is performed.
            resume: If True, resume from last checkpoint in checkpoint_dir.
                Requires checkpoint_dir to be set.
            parallel: Use parallel processing for TF-IDF/BM25 (default False).
                Provides ~2-3x speedup on large corpora (5000+ terms).
            parallel_num_workers: Number of worker processes (None = CPU count)
            parallel_chunk_size: Terms per chunk for parallel processing (default 1000)

        Returns:
            Dict with computation statistics (concept_stats, etc.)

        Example:
            >>> # Default behavior (silent)
            >>> processor.compute_all()
            >>>
            >>> # With console progress bar
            >>> processor.compute_all(show_progress=True)
            >>>
            >>> # With parallel processing for large corpora
            >>> processor.compute_all(parallel=True)
            >>>
            >>> # With checkpointing for long-running operations
            >>> processor.compute_all(checkpoint_dir='checkpoints')
            >>>
            >>> # Resume from checkpoint after crash/timeout
            >>> processor = CorticalTextProcessor.resume_from_checkpoint('checkpoints')
            >>> processor.compute_all(checkpoint_dir='checkpoints', resume=True)
        """
        stats: Dict[str, Any] = {}

        # Load checkpoint progress if resuming
        completed_phases: Set[str] = set()
        if resume and checkpoint_dir:
            completed_phases = self._load_checkpoint_progress(checkpoint_dir)
            if verbose and completed_phases:
                logger.info(f"Resuming from checkpoint with {len(completed_phases)} completed phases")

        # Set up progress reporter
        if progress_callback:
            reporter = progress_callback
        elif show_progress:
            reporter = ConsoleProgressReporter()
        else:
            reporter = SilentProgressReporter()

        # Define phase weights based on typical execution times
        phase_weights = {
            "Activation propagation": 5,
            "PageRank computation": 10,
            "TF-IDF computation": 15,
            "Document connections": 10,
            "Bigram connections": 30,
        }

        # Add concept-related phases if building concepts
        if build_concepts:
            phase_weights["Concept clustering"] = 15
            if connection_strategy in ('semantic', 'hybrid'):
                phase_weights["Semantic extraction"] = 10
            if connection_strategy in ('embedding', 'hybrid'):
                phase_weights["Graph embeddings"] = 10
            phase_weights["Concept connections"] = 15

        # Create multi-phase progress tracker
        progress = MultiPhaseProgress(reporter, phase_weights)

        # Phase 1: Activation propagation
        phase_name = "activation_propagation"
        if phase_name in completed_phases:
            if verbose:
                logger.info("  Skipping activation propagation (already checkpointed)")
        else:
            progress.start_phase("Activation propagation")
            if verbose:
                logger.info("Computing activation propagation...")
            self.propagate_activation(verbose=False)
            progress.update(100)
            progress.complete_phase()

            if checkpoint_dir:
                self._save_checkpoint(checkpoint_dir, phase_name, verbose=verbose)

        # Phase 2: PageRank (varies by method)
        phase_name = f"pagerank_{pagerank_method}"
        if phase_name in completed_phases:
            if verbose:
                logger.info(f"  Skipping PageRank computation ({pagerank_method}) (already checkpointed)")
        else:
            progress.start_phase("PageRank computation")
            if pagerank_method == 'semantic':
                if not self.semantic_relations:
                    if verbose:
                        logger.info("Extracting semantic relations...")
                    progress.update(30, "Extracting semantic relations")
                    self.extract_corpus_semantics(verbose=False)
                if verbose:
                    logger.info("Computing importance (Semantic PageRank)...")
                progress.update(70, "Computing semantic PageRank")
                self.compute_semantic_importance(verbose=False)
            elif pagerank_method == 'hierarchical':
                if verbose:
                    logger.info("Computing importance (Hierarchical PageRank)...")
                progress.update(50, "Computing hierarchical PageRank")
                self.compute_hierarchical_importance(verbose=False)
            else:
                if verbose:
                    logger.info("Computing importance (PageRank)...")
                progress.update(50, "Computing PageRank")
                self.compute_importance(verbose=False)
            progress.update(100)
            progress.complete_phase()

            if checkpoint_dir:
                self._save_checkpoint(checkpoint_dir, phase_name, verbose=verbose)

        # Phase 3: TF-IDF/BM25
        phase_name = "tfidf"
        if phase_name in completed_phases:
            if verbose:
                logger.info("  Skipping TF-IDF computation (already checkpointed)")
        else:
            progress.start_phase("TF-IDF computation")
            if verbose:
                scoring = self.config.scoring_algorithm.upper()
                mode = "parallel" if parallel else "sequential"
                logger.info(f"Computing {scoring} ({mode})...")

            if parallel:
                # Use parallel processing
                if self.config.scoring_algorithm == 'bm25':
                    self.compute_bm25_parallel(
                        num_workers=parallel_num_workers,
                        chunk_size=parallel_chunk_size,
                        verbose=False
                    )
                else:
                    self.compute_tfidf_parallel(
                        num_workers=parallel_num_workers,
                        chunk_size=parallel_chunk_size,
                        verbose=False
                    )
            else:
                # Use sequential processing (default)
                self.compute_tfidf(verbose=False)

            progress.update(100)
            progress.complete_phase()

            if checkpoint_dir:
                self._save_checkpoint(checkpoint_dir, phase_name, verbose=verbose)

        # Phase 4: Document connections
        phase_name = "document_connections"
        if phase_name in completed_phases:
            if verbose:
                logger.info("  Skipping document connections (already checkpointed)")
        else:
            progress.start_phase("Document connections")
            if verbose:
                logger.info("Computing document connections...")
            self.compute_document_connections(verbose=False)
            progress.update(100)
            progress.complete_phase()

            if checkpoint_dir:
                self._save_checkpoint(checkpoint_dir, phase_name, verbose=verbose)

        # Phase 5: Bigram connections
        phase_name = "bigram_connections"
        if phase_name in completed_phases:
            if verbose:
                logger.info("  Skipping bigram connections (already checkpointed)")
        else:
            progress.start_phase("Bigram connections")
            if verbose:
                logger.info("Computing bigram connections...")
            self.compute_bigram_connections(verbose=False)
            progress.update(100)
            progress.complete_phase()

            if checkpoint_dir:
                self._save_checkpoint(checkpoint_dir, phase_name, verbose=verbose)

        if build_concepts:
            # Phase 6: Concept clustering
            phase_name = "concept_clustering"
            if phase_name in completed_phases:
                if verbose:
                    logger.info("  Skipping concept clustering (already checkpointed)")
                stats['clusters_created'] = len([c for c in self.layers[CorticalLayer.CONCEPTS].minicolumns.values()])
            else:
                progress.start_phase("Concept clustering")
                if verbose:
                    logger.info("Building concept clusters...")
                clusters = self.build_concept_clusters(
                    cluster_strictness=cluster_strictness,
                    bridge_weight=bridge_weight,
                    verbose=False
                )
                stats['clusters_created'] = len(clusters)
                progress.update(100)
                progress.complete_phase()

                if checkpoint_dir:
                    self._save_checkpoint(checkpoint_dir, phase_name, verbose=verbose)

            # Determine connection parameters based on strategy
            use_member_semantics = connection_strategy in ('semantic', 'hybrid')
            use_embedding_similarity = connection_strategy in ('embedding', 'hybrid')

            # Phase 7: Semantic extraction (if needed)
            if use_member_semantics and not self.semantic_relations:
                phase_name = "semantic_extraction"
                if phase_name in completed_phases:
                    if verbose:
                        logger.info("  Skipping semantic extraction (already checkpointed)")
                else:
                    progress.start_phase("Semantic extraction")
                    if verbose:
                        logger.info("Extracting semantic relations...")
                    self.extract_corpus_semantics(verbose=False)
                    progress.update(100)
                    progress.complete_phase()

                    if checkpoint_dir:
                        self._save_checkpoint(checkpoint_dir, phase_name, verbose=verbose)

            # Phase 8: Graph embeddings (if needed)
            if use_embedding_similarity and not self.embeddings:
                phase_name = "graph_embeddings"
                if phase_name in completed_phases:
                    if verbose:
                        logger.info("  Skipping graph embeddings (already checkpointed)")
                else:
                    progress.start_phase("Graph embeddings")
                    if verbose:
                        logger.info("Computing graph embeddings...")
                    self.compute_graph_embeddings(verbose=False)
                    progress.update(100)
                    progress.complete_phase()

                    if checkpoint_dir:
                        self._save_checkpoint(checkpoint_dir, phase_name, verbose=verbose)

            # Set thresholds based on strategy
            if connection_strategy == 'hybrid':
                min_shared_docs = 0
                min_jaccard = 0.0
            elif connection_strategy in ('semantic', 'embedding'):
                min_shared_docs = 0
                min_jaccard = 0.0
            else:  # document_overlap
                min_shared_docs = 1
                min_jaccard = 0.1

            # Phase 9: Concept connections
            phase_name = f"concept_connections_{connection_strategy}"
            if phase_name in completed_phases:
                if verbose:
                    logger.info(f"  Skipping concept connections ({connection_strategy}) (already checkpointed)")
                stats['concept_connections'] = {'strategy': connection_strategy}
            else:
                progress.start_phase("Concept connections")
                if verbose:
                    logger.info(f"Computing concept connections ({connection_strategy})...")
                concept_stats = self.compute_concept_connections(
                    use_member_semantics=use_member_semantics,
                    use_embedding_similarity=use_embedding_similarity,
                    min_shared_docs=min_shared_docs,
                    min_jaccard=min_jaccard,
                    verbose=False
                )
                stats['concept_connections'] = concept_stats
                progress.update(100)
                progress.complete_phase()

                if checkpoint_dir:
                    self._save_checkpoint(checkpoint_dir, phase_name, verbose=verbose)

        # Mark core computations as fresh
        fresh_comps = [
            self.COMP_ACTIVATION,
            self.COMP_PAGERANK,
            self.COMP_TFIDF,
            self.COMP_DOC_CONNECTIONS,
            self.COMP_BIGRAM_CONNECTIONS,
        ]
        if build_concepts:
            fresh_comps.append(self.COMP_CONCEPTS)
        self._mark_fresh(*fresh_comps)

        # Invalidate query cache since corpus state changed
        self._query_expansion_cache.clear()

        if verbose:
            logger.info("Done.")

        return stats

    def _save_checkpoint(self, checkpoint_dir: str, completed_phase: str, verbose: bool = True) -> None:
        """
        Save checkpoint after completing a phase.

        Args:
            checkpoint_dir: Directory to save checkpoint files
            completed_phase: Name of the phase that was just completed
            verbose: Print progress messages
        """
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Save current state using save_json
        self.save_json(checkpoint_dir, force=True, verbose=False)

        # Track completed phases in a separate progress file
        progress_file = checkpoint_path / 'checkpoint_progress.json'
        progress_data = {
            'completed_phases': [],
            'last_updated': datetime.now().isoformat()
        }

        # Load existing progress if it exists
        if progress_file.exists():
            try:
                with open(progress_file, 'r', encoding='utf-8') as f:
                    progress_data = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass  # Use fresh progress data

        # Add newly completed phase
        if completed_phase not in progress_data['completed_phases']:
            progress_data['completed_phases'].append(completed_phase)
            progress_data['last_updated'] = datetime.now().isoformat()

        # Write progress file atomically
        temp_progress_file = progress_file.with_suffix('.json.tmp')
        try:
            with open(temp_progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, indent=2, ensure_ascii=False)
            temp_progress_file.replace(progress_file)
        except Exception:
            if temp_progress_file.exists():
                temp_progress_file.unlink()
            raise

        if verbose:
            logger.info(f"  Checkpoint saved: {completed_phase}")

    def _load_checkpoint_progress(self, checkpoint_dir: str) -> Set[str]:
        """
        Load completed phases from checkpoint directory.

        Args:
            checkpoint_dir: Directory containing checkpoint files

        Returns:
            Set of completed phase names
        """
        progress_file = Path(checkpoint_dir) / 'checkpoint_progress.json'
        if not progress_file.exists():
            return set()

        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return set(data.get('completed_phases', []))
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load checkpoint progress: {e}")
            return set()

    @classmethod
    def resume_from_checkpoint(
        cls,
        checkpoint_dir: str,
        config: Optional['CorticalConfig'] = None,
        verbose: bool = True
    ) -> 'CorticalTextProcessor':
        """
        Resume processing from a checkpoint directory.

        This is a convenience method that loads the processor state from
        a checkpoint directory created by compute_all() with checkpointing enabled.

        Args:
            checkpoint_dir: Directory containing checkpoint files
            config: Optional configuration (default: uses CorticalConfig defaults)
            verbose: Print progress messages

        Returns:
            Reconstructed CorticalTextProcessor instance ready to resume computation

        Raises:
            FileNotFoundError: If checkpoint directory doesn't exist

        Example:
            >>> # After a crash during compute_all()
            >>> processor = CorticalTextProcessor.resume_from_checkpoint('checkpoints')
            >>> # Continue from where it left off
            >>> processor.compute_all(checkpoint_dir='checkpoints', resume=True)
        """
        if verbose:
            logger.info(f"Resuming from checkpoint: {checkpoint_dir}")

        # Load the processor state from JSON
        processor = cls.load_json(checkpoint_dir, config=config, verbose=verbose)

        # Load and display progress
        progress = processor._load_checkpoint_progress(checkpoint_dir)
        if verbose and progress:
            logger.info(f"Found {len(progress)} completed phases: {', '.join(sorted(progress))}")

        return processor

    @timed("propagate_activation")
    def propagate_activation(self, iterations: int = 3, decay: float = 0.8, verbose: bool = True) -> None:
        analysis.propagate_activation(self.layers, iterations, decay)
        if verbose:
            logger.info(f"Propagated activation ({iterations} iterations)")

    @timed("compute_importance")
    def compute_importance(self, verbose: bool = True) -> None:
        for layer_enum in [CorticalLayer.TOKENS, CorticalLayer.BIGRAMS]:
            analysis.compute_pagerank(self.layers[layer_enum])
        if verbose:
            logger.info("Computed PageRank importance")

    def compute_semantic_importance(
        self,
        relation_weights: Optional[Dict[str, float]] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Compute PageRank with semantic relation weighting.

        Uses semantic relations to weight edges in the PageRank graph.
        Edges with stronger semantic relationships (e.g., IsA, PartOf) receive
        higher weights, affecting importance propagation.

        Args:
            relation_weights: Optional custom relation type weights dict.
                Defaults to built-in weights (IsA: 1.5, PartOf: 1.3, etc.)
            verbose: Print progress messages

        Returns:
            Dict with statistics:
            - total_edges_with_relations: Sum across layers
            - token_layer: Stats for token layer
            - bigram_layer: Stats for bigram layer
        """
        if not self.semantic_relations:
            self.compute_importance(verbose=verbose)
            return {
                'total_edges_with_relations': 0,
                'token_layer': {'edges_with_relations': 0},
                'bigram_layer': {'edges_with_relations': 0}
            }

        total_edges = 0
        layer_stats = {}

        for layer_enum in [CorticalLayer.TOKENS, CorticalLayer.BIGRAMS]:
            result = analysis.compute_semantic_pagerank(
                self.layers[layer_enum],
                self.semantic_relations,
                relation_weights=relation_weights
            )
            layer_name = 'token_layer' if layer_enum == CorticalLayer.TOKENS else 'bigram_layer'
            layer_stats[layer_name] = {
                'iterations_run': result['iterations_run'],
                'edges_with_relations': result['edges_with_relations']
            }
            total_edges += result['edges_with_relations']

        if verbose:
            logger.info(f"Computed semantic PageRank ({total_edges} relation-weighted edges)")

        return {
            'total_edges_with_relations': total_edges,
            **layer_stats
        }

    def compute_hierarchical_importance(
        self,
        layer_iterations: int = 10,
        global_iterations: int = 5,
        cross_layer_damping: Optional[float] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Compute PageRank with cross-layer propagation.

        This hierarchical PageRank allows importance to flow between layers:
        - Upward: tokens -> bigrams -> concepts -> documents
        - Downward: documents -> concepts -> bigrams -> tokens

        Args:
            layer_iterations: Max iterations for intra-layer PageRank (default 10)
            global_iterations: Max iterations for cross-layer propagation (default 5)
            cross_layer_damping: Damping factor at layer boundaries (default from config)
            verbose: Print progress messages

        Returns:
            Dict with statistics:
            - iterations_run: Number of global iterations
            - converged: Whether the algorithm converged
            - layer_stats: Per-layer statistics
        """
        if cross_layer_damping is None:
            cross_layer_damping = self.config.cross_layer_damping

        result = analysis.compute_hierarchical_pagerank(
            self.layers,
            layer_iterations=layer_iterations,
            global_iterations=global_iterations,
            cross_layer_damping=cross_layer_damping
        )

        if verbose:
            status = "converged" if result['converged'] else "did not converge"
            logger.info(f"Computed hierarchical PageRank ({result['iterations_run']} iterations, {status})")

        return result

    @timed("compute_tfidf")
    def compute_tfidf(self, verbose: bool = True) -> None:
        """
        Compute document relevance scores using the configured algorithm.

        Uses the scoring_algorithm from config ('tfidf' or 'bm25').
        BM25 provides improved relevance through term frequency saturation
        and document length normalization.

        Args:
            verbose: Print progress messages
        """
        if self.config.scoring_algorithm == 'bm25':
            analysis.compute_bm25(
                self.layers,
                self.documents,
                self.doc_lengths,
                self.avg_doc_length,
                k1=self.config.bm25_k1,
                b=self.config.bm25_b
            )
            if verbose:
                logger.info(f"Computed BM25 scores (k1={self.config.bm25_k1}, b={self.config.bm25_b})")
        else:
            analysis.compute_tfidf(self.layers, self.documents)
            if verbose:
                logger.info("Computed TF-IDF scores")

    @timed("compute_tfidf_parallel")
    def compute_tfidf_parallel(
        self,
        num_workers: Optional[int] = None,
        chunk_size: int = 1000,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Compute TF-IDF scores using parallel processing.

        This method provides ~2-3x speedup on large corpora (5000+ terms) by
        distributing computation across multiple CPU cores. Falls back to
        sequential for small corpora to avoid multiprocessing overhead.

        Args:
            num_workers: Number of worker processes (None = CPU count)
            chunk_size: Terms per chunk (default 1000)
            verbose: Print progress messages

        Returns:
            Statistics dict with terms_processed, method (parallel/sequential)

        Example:
            >>> # For large corpora
            >>> processor.compute_tfidf_parallel(num_workers=4)
            >>>
            >>> # Automatically falls back to sequential for small corpora
            >>> small_processor.compute_tfidf_parallel()  # Uses sequential
        """
        from ..layers import CorticalLayer

        layer0 = self.layers[CorticalLayer.TOKENS]
        num_terms = layer0.column_count()

        # Extract term stats (must be picklable for multiprocessing)
        term_stats = analysis.extract_term_stats(layer0)

        # Configure parallel processing
        config = analysis.ParallelConfig(
            num_workers=num_workers,
            chunk_size=chunk_size,
            min_items_for_parallel=2000
        )

        # Compute in parallel (or sequential if corpus is small)
        used_parallel = num_terms >= config.min_items_for_parallel
        results = analysis.parallel_tfidf(
            term_stats,
            len(self.documents),
            config=config
        )

        # Apply results back to minicolumns
        for term, (global_tfidf, per_doc_tfidf) in results.items():
            col = layer0.get_minicolumn(term)
            if col:
                col.tfidf = global_tfidf
                col.tfidf_per_doc = per_doc_tfidf

        if verbose:
            method = "parallel" if used_parallel else "sequential (small corpus)"
            logger.info(f"Computed TF-IDF scores for {num_terms} terms ({method})")

        return {
            'terms_processed': num_terms,
            'method': 'parallel' if used_parallel else 'sequential'
        }

    @timed("compute_bm25_parallel")
    def compute_bm25_parallel(
        self,
        k1: Optional[float] = None,
        b: Optional[float] = None,
        num_workers: Optional[int] = None,
        chunk_size: int = 1000,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Compute BM25 scores using parallel processing.

        This method provides ~2-3x speedup on large corpora (5000+ terms) by
        distributing computation across multiple CPU cores. Falls back to
        sequential for small corpora to avoid multiprocessing overhead.

        Args:
            k1: Term frequency saturation (0-3). Default from config (1.2)
            b: Length normalization (0-1). Default from config (0.75)
            num_workers: Number of worker processes (None = CPU count)
            chunk_size: Terms per chunk (default 1000)
            verbose: Print progress messages

        Returns:
            Statistics dict with terms_processed, method (parallel/sequential)

        Example:
            >>> # For large corpora
            >>> processor.compute_bm25_parallel(num_workers=4)
            >>>
            >>> # Automatically falls back to sequential for small corpora
            >>> small_processor.compute_bm25_parallel()  # Uses sequential
        """
        from ..layers import CorticalLayer

        k1 = k1 if k1 is not None else self.config.bm25_k1
        b = b if b is not None else self.config.bm25_b

        layer0 = self.layers[CorticalLayer.TOKENS]
        num_terms = layer0.column_count()

        # Extract term stats (must be picklable for multiprocessing)
        term_stats = analysis.extract_term_stats(layer0)

        # Configure parallel processing
        config = analysis.ParallelConfig(
            num_workers=num_workers,
            chunk_size=chunk_size,
            min_items_for_parallel=2000
        )

        # Compute in parallel (or sequential if corpus is small)
        used_parallel = num_terms >= config.min_items_for_parallel
        results = analysis.parallel_bm25(
            term_stats,
            len(self.documents),
            self.doc_lengths,
            self.avg_doc_length,
            k1=k1,
            b=b,
            config=config
        )

        # Apply results back to minicolumns
        for term, (global_bm25, per_doc_bm25) in results.items():
            col = layer0.get_minicolumn(term)
            if col:
                col.tfidf = global_bm25
                col.tfidf_per_doc = per_doc_bm25

        if verbose:
            method = "parallel" if used_parallel else "sequential (small corpus)"
            logger.info(f"Computed BM25 scores for {num_terms} terms (k1={k1}, b={b}, {method})")

        return {
            'terms_processed': num_terms,
            'method': 'parallel' if used_parallel else 'sequential',
            'k1': k1,
            'b': b
        }

    @timed("compute_bm25")
    def compute_bm25(
        self,
        k1: float = None,
        b: float = None,
        verbose: bool = True
    ) -> None:
        """
        Compute BM25 scores for document relevance ranking.

        BM25 (Best Match 25) improves on TF-IDF by:
        - Term frequency saturation: diminishing returns for repeated terms
        - Document length normalization: fair comparison across lengths

        Args:
            k1: Term frequency saturation (0-3). Default from config (1.2)
            b: Length normalization (0-1). Default from config (0.75)
            verbose: Print progress messages
        """
        k1 = k1 if k1 is not None else self.config.bm25_k1
        b = b if b is not None else self.config.bm25_b

        analysis.compute_bm25(
            self.layers,
            self.documents,
            self.doc_lengths,
            self.avg_doc_length,
            k1=k1,
            b=b
        )
        if verbose:
            logger.info(f"Computed BM25 scores (k1={k1}, b={b})")

    @timed("compute_document_connections")
    def compute_document_connections(self, min_shared_terms: int = 3, verbose: bool = True) -> None:
        analysis.compute_document_connections(self.layers, self.documents, min_shared_terms)
        if verbose:
            logger.info("Computed document connections")

    @timed("compute_bigram_connections")
    def compute_bigram_connections(
        self,
        min_shared_docs: int = 1,
        component_weight: float = 0.5,
        chain_weight: float = 0.7,
        cooccurrence_weight: float = 0.3,
        max_bigrams_per_term: int = 100,
        max_bigrams_per_doc: int = 500,
        max_connections_per_bigram: int = 50,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Build lateral connections between bigrams based on shared components and co-occurrence.

        Args:
            min_shared_docs: Minimum shared documents for co-occurrence connection
            component_weight: Weight for shared component connections (default 0.5)
            chain_weight: Weight for chain connections (default 0.7)
            cooccurrence_weight: Weight for document co-occurrence (default 0.3)
            max_bigrams_per_term: Skip terms appearing in more than this many bigrams
            max_bigrams_per_doc: Skip documents with more than this many bigrams
            max_connections_per_bigram: Maximum lateral connections per bigram
            verbose: Print progress messages

        Returns:
            Statistics about connections created
        """
        stats = analysis.compute_bigram_connections(
            self.layers,
            min_shared_docs=min_shared_docs,
            component_weight=component_weight,
            chain_weight=chain_weight,
            cooccurrence_weight=cooccurrence_weight,
            max_bigrams_per_term=max_bigrams_per_term,
            max_bigrams_per_doc=max_bigrams_per_doc,
            max_connections_per_bigram=max_connections_per_bigram
        )
        if verbose:
            skipped_terms = stats.get('skipped_common_terms', 0)
            skipped_docs = stats.get('skipped_large_docs', 0)
            skipped_conns = stats.get('skipped_max_connections', 0)
            skip_parts = []
            if skipped_terms:
                skip_parts.append(f"{skipped_terms} common terms")
            if skipped_docs:
                skip_parts.append(f"{skipped_docs} large docs")
            if skipped_conns:
                skip_parts.append(f"{skipped_conns} over-limit")
            skip_msg = f", skipped {', '.join(skip_parts)}" if skip_parts else ""
            logger.info(f"Created {stats['connections_created']} bigram connections "
                        f"(component: {stats['component_connections']}, "
                        f"chain: {stats['chain_connections']}, "
                        f"cooccur: {stats['cooccurrence_connections']}{skip_msg})")
        return stats

    @timed("build_concept_clusters")
    def build_concept_clusters(
        self,
        min_cluster_size: Optional[int] = None,
        clustering_method: str = 'louvain',
        cluster_strictness: Optional[float] = None,
        bridge_weight: float = 0.0,
        resolution: Optional[float] = None,
        verbose: bool = True
    ) -> Dict[int, List[str]]:
        """
        Build concept clusters from token layer.

        Args:
            min_cluster_size: Minimum tokens per cluster (default from config)
            clustering_method: Algorithm to use ('louvain' or 'label_propagation')
            cluster_strictness: For label_propagation only (0.0-1.0)
            bridge_weight: For label_propagation only (0.0-1.0)
            resolution: For louvain only (default from config)
            verbose: Print progress messages

        Returns:
            Dictionary mapping cluster_id to list of token contents
        """
        if min_cluster_size is None:
            min_cluster_size = self.config.min_cluster_size
        if cluster_strictness is None:
            cluster_strictness = self.config.cluster_strictness
        if resolution is None:
            resolution = self.config.louvain_resolution

        if clustering_method == 'louvain':
            clusters = analysis.cluster_by_louvain(
                self.layers[CorticalLayer.TOKENS],
                min_cluster_size=min_cluster_size,
                resolution=resolution
            )
        elif clustering_method == 'label_propagation':
            clusters = analysis.cluster_by_label_propagation(
                self.layers[CorticalLayer.TOKENS],
                min_cluster_size=min_cluster_size,
                cluster_strictness=cluster_strictness,
                bridge_weight=bridge_weight
            )
        else:
            raise ValueError(
                f"Unknown clustering_method: {clustering_method}. "
                f"Use 'louvain' or 'label_propagation'."
            )

        analysis.build_concept_clusters(self.layers, clusters)
        if verbose:
            logger.info(f"Built {len(clusters)} concept clusters using {clustering_method}")
        return clusters

    def compute_clustering_quality(self, sample_size: int = 500) -> Dict[str, Any]:
        """
        Compute clustering quality metrics for the concept layer.

        Args:
            sample_size: Max tokens to sample for silhouette calculation

        Returns:
            Dictionary with modularity, silhouette, balance, num_clusters, quality_assessment
        """
        return analysis.compute_clustering_quality(self.layers, sample_size)

    def compute_concept_connections(
        self,
        use_semantics: bool = True,
        min_shared_docs: int = 1,
        min_jaccard: float = 0.1,
        use_member_semantics: bool = False,
        use_embedding_similarity: bool = False,
        embedding_threshold: float = 0.3,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Build lateral connections between concepts based on document overlap and semantics.

        Args:
            use_semantics: Use semantic relations to boost connection weights
            min_shared_docs: Minimum shared documents for connection
            min_jaccard: Minimum Jaccard similarity threshold
            use_member_semantics: Connect concepts via member token semantic relations
            use_embedding_similarity: Connect concepts via embedding centroid similarity
            embedding_threshold: Minimum cosine similarity for embedding connections
            verbose: Print progress messages

        Returns:
            Statistics about connections created
        """
        semantic_rels = self.semantic_relations if use_semantics else None
        emb = self.embeddings if use_embedding_similarity else None
        stats = analysis.compute_concept_connections(
            self.layers,
            semantic_relations=semantic_rels,
            min_shared_docs=min_shared_docs,
            min_jaccard=min_jaccard,
            use_member_semantics=use_member_semantics,
            use_embedding_similarity=use_embedding_similarity,
            embedding_threshold=embedding_threshold,
            embeddings=emb
        )
        if verbose:
            parts = [f"Created {stats['connections_created']} concept connections"]
            if stats.get('doc_overlap_connections', 0) > 0:
                parts.append(f"doc_overlap: {stats['doc_overlap_connections']}")
            if stats.get('semantic_connections', 0) > 0:
                parts.append(f"semantic: {stats['semantic_connections']}")
            if stats.get('embedding_connections', 0) > 0:
                parts.append(f"embedding: {stats['embedding_connections']}")
            logger.info(", ".join(parts) if len(parts) > 1 else parts[0])
        return stats

    def extract_corpus_semantics(
        self,
        use_pattern_extraction: bool = True,
        min_pattern_confidence: float = 0.6,
        max_similarity_pairs: int = 100000,
        min_context_keys: int = 3,
        verbose: bool = True
    ) -> int:
        """
        Extract semantic relations from the corpus.

        Args:
            use_pattern_extraction: Extract relations from text patterns
            min_pattern_confidence: Minimum confidence for pattern-based relations
            max_similarity_pairs: Maximum pairs to check for SimilarTo relations
            min_context_keys: Minimum context keys for SimilarTo consideration
            verbose: Print progress messages

        Returns:
            Number of relations extracted
        """
        self.semantic_relations = semantics.extract_corpus_semantics(
            self.layers,
            self.documents,
            self.tokenizer,
            use_pattern_extraction=use_pattern_extraction,
            min_pattern_confidence=min_pattern_confidence,
            max_similarity_pairs=max_similarity_pairs,
            min_context_keys=min_context_keys
        )
        if verbose:
            logger.info(f"Extracted {len(self.semantic_relations)} semantic relations")
        return len(self.semantic_relations)

    def extract_pattern_relations(
        self,
        min_confidence: float = 0.6,
        verbose: bool = True
    ) -> List[Tuple[str, str, str, float]]:
        """
        Extract semantic relations using pattern matching only.

        Args:
            min_confidence: Minimum confidence for extracted relations
            verbose: Print progress messages

        Returns:
            List of (term1, relation_type, term2, confidence) tuples
        """
        layer0 = self.get_layer(CorticalLayer.TOKENS)
        valid_terms = set(layer0.minicolumns.keys())

        relations = semantics.extract_pattern_relations(
            self.documents,
            valid_terms,
            min_confidence=min_confidence
        )

        if verbose:
            stats = semantics.get_pattern_statistics(relations)
            logger.info(f"Extracted {stats['total_relations']} pattern-based relations")
            logger.info(f"  Types: {stats['relation_type_counts']}")

        return relations

    def retrofit_connections(self, iterations: int = 10, alpha: float = 0.3, verbose: bool = True) -> Dict:
        if not self.semantic_relations:
            self.extract_corpus_semantics(verbose=False)
        stats = semantics.retrofit_connections(self.layers, self.semantic_relations, iterations, alpha)
        if verbose:
            logger.info(f"Retrofitted {stats['tokens_affected']} tokens")
        return stats

    def compute_property_inheritance(
        self,
        decay_factor: float = 0.7,
        max_depth: int = 5,
        apply_to_connections: bool = True,
        boost_factor: float = 0.3,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Compute property inheritance based on IsA hierarchy.

        Args:
            decay_factor: Weight multiplier per inheritance level (default 0.7)
            max_depth: Maximum inheritance depth (default 5)
            apply_to_connections: Boost lateral connections for shared properties
            boost_factor: Weight boost for shared inherited properties
            verbose: Print progress messages

        Returns:
            Dict with statistics about inheritance computation
        """
        if not self.semantic_relations:
            self.extract_corpus_semantics(verbose=False)

        inherited = semantics.inherit_properties(
            self.semantic_relations,
            decay_factor=decay_factor,
            max_depth=max_depth
        )

        total_props = sum(len(props) for props in inherited.values())

        result = {
            'terms_with_inheritance': len(inherited),
            'total_properties_inherited': total_props,
            'inherited': inherited
        }

        if apply_to_connections and inherited:
            conn_stats = semantics.apply_inheritance_to_connections(
                self.layers,
                inherited,
                boost_factor=boost_factor
            )
            result['connections_boosted'] = conn_stats['connections_boosted']
            result['total_boost'] = conn_stats['total_boost']
        else:
            result['connections_boosted'] = 0
            result['total_boost'] = 0.0

        if verbose:
            logger.info(f"Computed property inheritance: {result['terms_with_inheritance']} terms, "
                        f"{total_props} properties, {result['connections_boosted']} connections boosted")

        return result

    def compute_property_similarity(self, term1: str, term2: str) -> float:
        """
        Compute similarity between terms based on shared properties.

        Args:
            term1: First term
            term2: Second term

        Returns:
            Similarity score (0.0-1.0) based on property overlap
        """
        if not self.semantic_relations:
            return 0.0

        inherited = semantics.inherit_properties(self.semantic_relations)
        return semantics.compute_property_similarity(term1, term2, inherited)

    def compute_graph_embeddings(
        self,
        dimensions: int = 64,
        method: str = 'fast',
        max_terms: Optional[int] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Compute graph embeddings for tokens.

        Args:
            dimensions: Number of embedding dimensions (default 64)
            method: Embedding method ('tfidf', 'fast', 'adjacency', 'random_walk', 'spectral')
            max_terms: Maximum number of terms to embed (by PageRank)
            verbose: Print progress messages

        Returns:
            Statistics dict with method, dimensions, terms_embedded
        """
        token_count = self.layers[CorticalLayer.TOKENS].column_count()
        if max_terms is None:
            if token_count < 2000:
                max_terms = None
            elif token_count < 5000:
                max_terms = 1500
            else:
                max_terms = 1000

        self.embeddings, stats = emb_module.compute_graph_embeddings(
            self.layers, dimensions, method, max_terms
        )
        if verbose:
            sampled = stats.get('sampled', False)
            sample_info = f", sampled top {max_terms}" if sampled else ""
            logger.info(f"Computed {stats['terms_embedded']} embeddings ({method}{sample_info})")
        return stats

    def retrofit_embeddings(self, iterations: int = 10, alpha: float = 0.4, verbose: bool = True) -> Dict:
        if not self.embeddings:
            self.compute_graph_embeddings(verbose=False)
        if not self.semantic_relations:
            self.extract_corpus_semantics(verbose=False)
        stats = semantics.retrofit_embeddings(self.embeddings, self.semantic_relations, iterations, alpha)
        if verbose:
            logger.info(f"Retrofitted embeddings (moved {stats['total_movement']:.2f} total)")
        return stats

    def embedding_similarity(self, term1: str, term2: str) -> float:
        return emb_module.embedding_similarity(self.embeddings, term1, term2)

    def find_similar_by_embedding(self, term: str, top_n: int = 10) -> List[Tuple[str, float]]:
        return emb_module.find_similar_by_embedding(self.embeddings, term, top_n)
