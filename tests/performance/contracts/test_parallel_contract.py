"""
╔══════════════════════════════════════════════════════════════════════╗
║                    PARALLEL PERFORMANCE CONTRACT                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  Ratified:     2024-12-31                                            ║
║  Guardian:     CI Pipeline                                            ║
║  Renegotiation: Requires team review + documented justification      ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  We solemnly contract the following guarantees:                      ║
║                                                                       ║
║  • Parallel TF-IDF produces identical results to sequential         ║
║  • Parallel BM25 produces identical results to sequential           ║
║  • Sequential fallback for small corpora (< 2,000 items)            ║
║  • No external dependencies beyond Python stdlib                    ║
║  • Chunk processing is deterministic                                ║
║  • Results are correctly merged across chunks                       ║
║                                                                       ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import pytest
import math


@pytest.mark.contract
class TestParallelTFIDFCorrectnessContract:
    """
    Parallel TF-IDF Correctness Contract

    As a developer using parallel processing,
    I expect identical results to sequential execution,
    So that parallelization is transparent.
    """

    def test_parallel_tfidf_matches_sequential(self):
        """
        CONTRACT: Parallel TF-IDF produces identical results to sequential.

        Parallelization must be transparent - same input, same output.
        """
        from cortical.analysis.tfidf import _tfidf_core
        from cortical.analysis.parallel import parallel_tfidf, ParallelConfig

        # Create test data
        term_stats = {
            f"term{i}": (i * 10, i % 5 + 1, {f"doc{j}": i + j for j in range(i % 5 + 1)})
            for i in range(100)
        }
        num_docs = 10

        # Sequential computation
        sequential = _tfidf_core(term_stats, num_docs)

        # Parallel computation (force parallel even for small data)
        config = ParallelConfig(min_items_for_parallel=1, chunk_size=20)
        parallel = parallel_tfidf(term_stats, num_docs, config)

        # Results must match exactly
        assert set(sequential.keys()) == set(parallel.keys()), (
            "CONTRACT VIOLATION: Parallel and sequential produce different term sets"
        )

        for term in sequential:
            seq_global, seq_per_doc = sequential[term]
            par_global, par_per_doc = parallel[term]

            # Global TF-IDF must match
            assert abs(seq_global - par_global) < 1e-9, (
                f"CONTRACT VIOLATION: Term '{term}' global TF-IDF mismatch. "
                f"Sequential: {seq_global}, Parallel: {par_global}"
            )

            # Per-doc TF-IDF must match
            assert seq_per_doc.keys() == par_per_doc.keys(), (
                f"CONTRACT VIOLATION: Term '{term}' per-doc keys mismatch"
            )

            for doc_id in seq_per_doc:
                assert abs(seq_per_doc[doc_id] - par_per_doc[doc_id]) < 1e-9, (
                    f"CONTRACT VIOLATION: Term '{term}' doc '{doc_id}' TF-IDF mismatch"
                )

    def test_parallel_tfidf_falls_back_for_small_data(self):
        """
        CONTRACT: Parallel TF-IDF uses sequential for small datasets.

        Multiprocessing overhead isn't worth it for small data.
        """
        from cortical.analysis.parallel import parallel_tfidf, ParallelConfig

        small_data = {
            "term1": (10, 2, {"doc1": 5, "doc2": 5}),
            "term2": (20, 3, {"doc1": 10, "doc2": 5, "doc3": 5})
        }

        # With default config, should fall back to sequential
        config = ParallelConfig()  # min_items_for_parallel = 2000
        result = parallel_tfidf(small_data, num_docs=10, config=config)

        # Should succeed (falls back internally)
        assert len(result) == 2


@pytest.mark.contract
class TestParallelBM25CorrectnessContract:
    """
    Parallel BM25 Correctness Contract

    As a developer using parallel BM25,
    I expect identical results to sequential execution,
    So that ranking is consistent.
    """

    def test_parallel_bm25_matches_sequential(self):
        """
        CONTRACT: Parallel BM25 produces identical results to sequential.

        BM25 parallelization must be transparent.
        """
        from cortical.analysis.tfidf import _bm25_core
        from cortical.analysis.parallel import parallel_bm25, ParallelConfig

        # Create test data
        term_stats = {
            f"term{i}": (i * 10, i % 5 + 1, {f"doc{j}": i + j for j in range(i % 5 + 1)})
            for i in range(100)
        }
        num_docs = 10
        doc_lengths = {f"doc{i}": 100 + i * 10 for i in range(10)}
        avg_length = sum(doc_lengths.values()) / len(doc_lengths)

        # Sequential computation
        sequential = _bm25_core(
            term_stats, num_docs, doc_lengths, avg_length, k1=1.2, b=0.75
        )

        # Parallel computation
        config = ParallelConfig(min_items_for_parallel=1, chunk_size=20)
        parallel = parallel_bm25(
            term_stats, num_docs, doc_lengths, avg_length, k1=1.2, b=0.75, config=config
        )

        # Results must match exactly
        assert set(sequential.keys()) == set(parallel.keys())

        for term in sequential:
            seq_global, seq_per_doc = sequential[term]
            par_global, par_per_doc = parallel[term]

            # Global BM25 must match
            assert abs(seq_global - par_global) < 1e-9, (
                f"CONTRACT VIOLATION: Term '{term}' global BM25 mismatch. "
                f"Sequential: {seq_global}, Parallel: {par_global}"
            )

            # Per-doc BM25 must match
            assert seq_per_doc.keys() == par_per_doc.keys()

            for doc_id in seq_per_doc:
                assert abs(seq_per_doc[doc_id] - par_per_doc[doc_id]) < 1e-9, (
                    f"CONTRACT VIOLATION: Term '{term}' doc '{doc_id}' BM25 mismatch"
                )


@pytest.mark.contract
class TestParallelInfrastructureContract:
    """
    Parallel Infrastructure Contract

    As a developer relying on parallel processing,
    I expect robust chunking and merging,
    So that parallelization is reliable.
    """

    def test_chunk_dict_preserves_all_items(self):
        """
        CONTRACT: chunk_dict preserves all dictionary items.

        No data should be lost during chunking.
        """
        from cortical.analysis.parallel import chunk_dict

        data = {f"key{i}": i for i in range(100)}

        chunks = chunk_dict(data, chunk_size=25)

        # All items should be preserved
        reconstructed = {}
        for chunk in chunks:
            reconstructed.update(chunk)

        assert reconstructed == data, (
            "CONTRACT VIOLATION: chunk_dict lost or modified items"
        )

    def test_chunk_dict_respects_chunk_size(self):
        """
        CONTRACT: chunk_dict respects maximum chunk size.

        No chunk should exceed the specified size.
        """
        from cortical.analysis.parallel import chunk_dict

        data = {f"key{i}": i for i in range(100)}
        chunk_size = 30

        chunks = chunk_dict(data, chunk_size)

        for i, chunk in enumerate(chunks):
            assert len(chunk) <= chunk_size, (
                f"CONTRACT VIOLATION: Chunk {i} has {len(chunk)} items, "
                f"max is {chunk_size}"
            )

    def test_extract_term_stats_produces_primitives(self):
        """
        CONTRACT: extract_term_stats produces picklable primitives.

        All data must be serializable for multiprocessing.
        """
        from cortical import CorticalTextProcessor
        from cortical.layers import CorticalLayer
        from cortical.analysis.parallel import extract_term_stats
        import pickle

        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Custom neural network implementation.")
        processor.process_document("doc2", "Hand-built search algorithm.")
        processor.compute_all(verbose=False)

        layer = processor.layers[CorticalLayer.TOKENS]
        stats = extract_term_stats(layer)

        # Should be picklable (required for multiprocessing)
        try:
            serialized = pickle.dumps(stats)
            deserialized = pickle.loads(serialized)
            assert deserialized == stats
        except Exception as e:
            pytest.fail(
                f"CONTRACT VIOLATION: extract_term_stats produced non-picklable data: {e}"
            )

    def test_parallel_config_has_reasonable_defaults(self):
        """
        CONTRACT: ParallelConfig has sensible defaults.

        Default configuration should be production-ready.
        """
        from cortical.analysis.parallel import ParallelConfig

        config = ParallelConfig()

        # Defaults should be reasonable
        assert config.chunk_size > 0, "Chunk size must be positive"
        assert config.min_items_for_parallel > config.chunk_size, (
            "Minimum items should be larger than chunk size"
        )

        # num_workers=None means CPU count, which is valid


@pytest.mark.contract
class TestParallelDependenciesContract:
    """
    Parallel Dependencies Contract

    As a developer building sovereign systems,
    I expect zero external dependencies,
    So that we control our own infrastructure.
    """

    def test_parallel_uses_only_stdlib(self):
        """
        CONTRACT: Parallel module uses only Python stdlib.

        No external dependencies - we build it ourselves.
        """
        import cortical.analysis.parallel as parallel_module

        # Get all imported modules
        imports = []
        for name, val in vars(parallel_module).items():
            if hasattr(val, '__module__'):
                module = val.__module__
                if module and not module.startswith('cortical'):
                    imports.append(module.split('.')[0])

        # Known stdlib modules
        stdlib_modules = {
            'concurrent', 'dataclasses', 'typing', 'math',
            '__builtin__', 'builtins',
        }

        # Filter out Python internals (start with _) and known stdlib
        external_deps = {m for m in imports if not m.startswith('_')} - stdlib_modules

        assert not external_deps, (
            f"CONTRACT VIOLATION: Parallel module has external dependencies: {external_deps}"
        )

    def test_parallel_imports_only_from_stdlib_and_cortical(self):
        """
        CONTRACT: All imports are from stdlib or cortical package.

        We build everything ourselves - no third-party code.
        """
        # Read the parallel.py source to check imports
        import cortical.analysis.parallel
        import inspect

        source = inspect.getsource(cortical.analysis.parallel)

        # Extract import lines
        import_lines = [
            line.strip() for line in source.split('\n')
            if line.strip().startswith('import ') or line.strip().startswith('from ')
        ]

        # Check no third-party imports
        forbidden_patterns = ['numpy', 'scipy', 'pandas', 'sklearn', 'torch', 'tensorflow']

        for line in import_lines:
            for pattern in forbidden_patterns:
                assert pattern not in line.lower(), (
                    f"CONTRACT VIOLATION: Found third-party import: {line}"
                )
