"""
Tests for processor/ package mixin boundaries and edge cases.

This module tests critical interactions between the processor mixins:
- CoreMixin <-> DocumentsMixin (staleness propagation)
- DocumentsMixin <-> QueryMixin (cache invalidation)
- ComputeMixin <-> PersistenceMixin (checkpoint/recovery)
- QueryMixin (RAG passage edge cases)
- IntrospectionMixin (semantic diff)

These tests ensure the modular processor/ package maintains correct
state across mixin boundaries.
"""

import json
import tempfile
import unittest
from pathlib import Path

from cortical import CorticalTextProcessor
from cortical.config import CorticalConfig


class TestMixinBoundaryInteractions(unittest.TestCase):
    """Test critical mixin boundary interactions."""

    def test_compute_all_invalidates_query_cache(self):
        """
        Verify that compute_all clears query expansion cache.

        Tests boundary: ComputeMixin -> QueryMixin
        When corpus state is recomputed, cached query expansions become stale.

        Note: process_document does NOT clear cache (allows batch adds),
        but compute_all DOES clear it (finalizes corpus state).
        """
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural networks process data")
        processor.compute_all(verbose=False)

        # Prime the cache with a query
        expanded = processor.expand_query_cached("neural")
        self.assertGreater(len(processor._query_expansion_cache), 0)

        # Add new document (cache still exists - allows batch adds)
        processor.process_document("doc2", "artificial intelligence systems")
        self.assertGreater(len(processor._query_expansion_cache), 0)

        # compute_all should clear cache
        processor.compute_all(verbose=False)
        self.assertEqual(len(processor._query_expansion_cache), 0)

    def test_document_removal_invalidates_query_cache(self):
        """
        Verify that removing documents clears query expansion cache.
        """
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural networks process data")
        processor.process_document("doc2", "machine learning algorithms")
        processor.compute_all(verbose=False)

        # Prime cache
        _ = processor.expand_query_cached("neural")
        self.assertGreater(len(processor._query_expansion_cache), 0)

        # Remove document
        processor.remove_document("doc1")

        # Cache should be cleared
        self.assertEqual(len(processor._query_expansion_cache), 0)

    def test_staleness_tracking_across_compute_phases(self):
        """
        Verify staleness is correctly tracked across compute operations.

        Tests boundary: CoreMixin staleness <-> ComputeMixin operations
        """
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content here")

        # All should be stale after adding document
        self.assertTrue(processor.is_stale(processor.COMP_TFIDF))
        self.assertTrue(processor.is_stale(processor.COMP_PAGERANK))

        # Compute only TF-IDF
        processor.compute_tfidf(verbose=False)
        processor._mark_fresh(processor.COMP_TFIDF)

        # TF-IDF fresh, PageRank still stale
        self.assertFalse(processor.is_stale(processor.COMP_TFIDF))
        self.assertTrue(processor.is_stale(processor.COMP_PAGERANK))

    def test_compute_all_marks_all_fresh(self):
        """
        Verify compute_all marks all computations as fresh.
        """
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural networks")

        # All stale initially
        stale_before = processor.get_stale_computations()
        self.assertGreater(len(stale_before), 0)

        processor.compute_all(verbose=False)

        # Core computations should be fresh
        self.assertFalse(processor.is_stale(processor.COMP_TFIDF))
        self.assertFalse(processor.is_stale(processor.COMP_PAGERANK))
        self.assertFalse(processor.is_stale(processor.COMP_ACTIVATION))

    def test_recompute_stale_only(self):
        """
        Verify recompute(level='stale') only recomputes stale items.
        """
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural networks")
        processor.compute_all(verbose=False)

        # Manually mark only TFIDF stale
        processor._stale_computations = {processor.COMP_TFIDF}

        # Recompute stale only
        recomputed = processor.recompute(level='stale', verbose=False)

        # Only TFIDF should have been recomputed
        self.assertTrue(recomputed.get(processor.COMP_TFIDF, False))
        self.assertFalse(recomputed.get(processor.COMP_PAGERANK, False))


class TestCheckpointRecovery(unittest.TestCase):
    """Test checkpoint and recovery edge cases."""

    def test_checkpoint_creates_progress_file(self):
        """
        Verify checkpoint creates progress tracking file.
        """
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        with tempfile.TemporaryDirectory() as tmpdir:
            processor.compute_all(checkpoint_dir=tmpdir, verbose=False)

            progress_file = Path(tmpdir) / 'checkpoint_progress.json'
            self.assertTrue(progress_file.exists())

            with open(progress_file) as f:
                data = json.load(f)

            self.assertIn('completed_phases', data)
            self.assertGreater(len(data['completed_phases']), 0)

    def test_resume_from_checkpoint(self):
        """
        Verify resuming from checkpoint restores state correctly.
        """
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural networks")
        processor.process_document("doc2", "machine learning")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save with checkpoint
            processor.compute_all(checkpoint_dir=tmpdir, verbose=False)

            # Resume from checkpoint
            resumed = CorticalTextProcessor.resume_from_checkpoint(
                tmpdir, verbose=False
            )

            # Verify state restored
            self.assertEqual(len(resumed.documents), 2)
            self.assertIn("doc1", resumed.documents)
            self.assertIn("doc2", resumed.documents)

    def test_resume_with_partial_checkpoint(self):
        """
        Verify resume handles partial checkpoint gracefully.
        """
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save initial state
            processor.save_json(tmpdir, verbose=False)

            # Create progress file with some phases marked complete
            progress_file = Path(tmpdir) / 'checkpoint_progress.json'
            progress_data = {
                'completed_phases': ['activation_propagation'],
                'last_updated': '2025-12-14T00:00:00'
            }
            with open(progress_file, 'w') as f:
                json.dump(progress_data, f)

            # Resume should work
            resumed = CorticalTextProcessor.resume_from_checkpoint(
                tmpdir, verbose=False
            )

            self.assertIsNotNone(resumed)
            self.assertEqual(len(resumed.documents), 1)

    def test_checkpoint_dir_not_found(self):
        """
        Verify resume raises FileNotFoundError for missing directory.
        """
        with self.assertRaises(FileNotFoundError):
            CorticalTextProcessor.resume_from_checkpoint('/nonexistent/path/xyz')

    def test_save_load_preserves_state(self):
        """
        Verify save/load roundtrip preserves all state.
        """
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural networks process data")
        processor.compute_all(verbose=False)

        # Get state before save
        docs_before = len(processor.documents)
        layer0_count = processor.layers[0].column_count()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test.pkl'
            processor.save(str(filepath), verbose=False)

            loaded = CorticalTextProcessor.load(str(filepath), verbose=False)

            # Verify state preserved
            self.assertEqual(len(loaded.documents), docs_before)
            self.assertEqual(loaded.layers[0].column_count(), layer0_count)

    def test_save_json_load_json_roundtrip(self):
        """
        Verify JSON save/load roundtrip preserves state.
        """
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural networks")
        processor.process_document("doc2", "machine learning")
        processor.compute_all(verbose=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            processor.save_json(tmpdir, verbose=False)
            loaded = CorticalTextProcessor.load_json(tmpdir, verbose=False)

            self.assertEqual(len(loaded.documents), 2)
            self.assertIn("doc1", loaded.documents)


class TestRAGPassageEdgeCases(unittest.TestCase):
    """Test RAG passage retrieval edge cases."""

    def test_find_passages_empty_corpus(self):
        """
        Verify RAG returns empty list on empty corpus.
        """
        processor = CorticalTextProcessor()

        passages = processor.find_passages_for_query("neural")

        self.assertEqual(passages, [])
        self.assertIsInstance(passages, list)

    def test_find_passages_no_matches(self):
        """
        Verify RAG returns empty list when no matches found.
        """
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "machine learning algorithms")
        processor.compute_all(verbose=False)

        # Query for something not in corpus
        passages = processor.find_passages_for_query("xyznonexistent")

        self.assertEqual(passages, [])

    def test_find_passages_boundary_positions(self):
        """
        Verify passage boundaries are valid positions in document.
        """
        processor = CorticalTextProcessor()
        doc_text = "The quick brown fox jumps over the lazy dog. Neural networks process data efficiently."
        processor.process_document("doc1", doc_text)
        processor.compute_all(verbose=False)

        passages = processor.find_passages_for_query("neural", top_n=1)

        if passages:
            text, doc_id, start, end, score = passages[0]

            # Boundaries must be valid
            self.assertGreaterEqual(start, 0)
            self.assertLessEqual(end, len(doc_text))
            self.assertLess(start, end)

            # Extracted text should match document slice
            self.assertEqual(text, doc_text[start:end])

    def test_find_passages_with_small_chunk_size(self):
        """
        Verify small chunk sizes work correctly.
        """
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Neural networks are powerful. They process data.")
        processor.compute_all(verbose=False)

        passages = processor.find_passages_for_query(
            "neural",
            top_n=3,
            chunk_size=20,
            overlap=5
        )

        # Should return results
        self.assertIsInstance(passages, list)

        # Each passage should respect chunk size approximately
        for text, doc_id, start, end, score in passages:
            self.assertLessEqual(len(text), 50)  # Allow some flexibility

    def test_find_passages_overlap_less_than_chunk(self):
        """
        Verify overlap must be less than chunk_size.
        """
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Neural networks")
        processor.compute_all(verbose=False)

        with self.assertRaises(ValueError):
            processor.find_passages_for_query(
                "neural",
                chunk_size=10,
                overlap=15  # Invalid: overlap >= chunk_size
            )

    def test_rag_retrieve_returns_structured_data(self):
        """
        Verify rag_retrieve returns properly structured dicts.
        """
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Neural networks process information efficiently.")
        processor.compute_all(verbose=False)

        results = processor.rag_retrieve("neural", top_n=1)

        self.assertIsInstance(results, list)
        if results:
            passage = results[0]
            self.assertIn('text', passage)
            self.assertIn('doc_id', passage)
            self.assertIn('start', passage)
            self.assertIn('end', passage)
            self.assertIn('score', passage)


class TestIntrospectionMethods(unittest.TestCase):
    """Test introspection and comparison methods."""

    def test_compare_documents_same_content(self):
        """
        Verify comparing documents with same content shows high similarity.
        """
        processor = CorticalTextProcessor()
        content = "Neural networks are a type of machine learning model"
        processor.process_document("doc1", content)
        processor.process_document("doc2", content)
        processor.compute_all(verbose=False)

        comparison = processor.compare_documents("doc1", "doc2")

        # Should show high similarity
        self.assertIn('jaccard_similarity', comparison)
        self.assertGreater(comparison['jaccard_similarity'], 0.9)

    def test_compare_documents_different_content(self):
        """
        Verify comparing very different documents shows low similarity.
        """
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Neural networks process data")
        processor.process_document("doc2", "Cooking recipes for dinner")
        processor.compute_all(verbose=False)

        comparison = processor.compare_documents("doc1", "doc2")

        # Should show low similarity
        self.assertIn('jaccard_similarity', comparison)
        self.assertLess(comparison['jaccard_similarity'], 0.3)

    def test_what_changed_additions(self):
        """
        Verify what_changed detects added content.
        """
        processor = CorticalTextProcessor()
        processor.process_document("doc", "baseline content")
        processor.compute_all(verbose=False)

        old_content = "neural networks"
        new_content = "neural networks and deep learning"

        diff = processor.what_changed(old_content, new_content)

        self.assertIsInstance(diff, dict)
        # Should have some indication of changes
        self.assertGreater(len(diff), 0)

    def test_fingerprint_consistency(self):
        """
        Verify fingerprint is consistent for same text.
        """
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural networks")
        processor.compute_all(verbose=False)

        text = "Neural networks process data efficiently"
        fp1 = processor.get_fingerprint(text)
        fp2 = processor.get_fingerprint(text)

        # Should be identical
        self.assertEqual(fp1['term_count'], fp2['term_count'])
        self.assertEqual(set(fp1['terms'].keys()), set(fp2['terms'].keys()))

    def test_compare_fingerprints_identical(self):
        """
        Verify comparing identical fingerprints returns similarity ~1.0.
        """
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural networks machine learning")
        processor.compute_all(verbose=False)

        text = "Neural networks and machine learning"
        fp = processor.get_fingerprint(text)

        comparison = processor.compare_fingerprints(fp, fp)

        self.assertIn('overall_similarity', comparison)
        self.assertAlmostEqual(comparison['overall_similarity'], 1.0, places=2)

    def test_summarize_document_short(self):
        """
        Verify summarize returns full text for short documents.
        """
        processor = CorticalTextProcessor()
        short_text = "This is short."
        processor.process_document("doc1", short_text)
        processor.compute_all(verbose=False)

        summary = processor.summarize_document("doc1", num_sentences=3)

        self.assertEqual(summary, short_text)

    def test_summarize_document_not_found(self):
        """
        Verify summarize returns empty string for missing document.
        """
        processor = CorticalTextProcessor()

        summary = processor.summarize_document("nonexistent")

        self.assertEqual(summary, "")

    def test_get_corpus_summary(self):
        """
        Verify corpus summary contains expected fields.
        """
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural networks")
        processor.process_document("doc2", "machine learning")
        processor.compute_all(verbose=False)

        summary = processor.get_corpus_summary()

        self.assertIn('documents', summary)
        self.assertEqual(summary['documents'], 2)
        self.assertIn('total_columns', summary)


class TestQueryCacheManagement(unittest.TestCase):
    """Test query cache management."""

    def test_clear_query_cache(self):
        """
        Verify clear_query_cache removes all entries.
        """
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural networks")
        processor.compute_all(verbose=False)

        # Prime cache
        _ = processor.expand_query_cached("neural")
        _ = processor.expand_query_cached("networks")
        self.assertEqual(len(processor._query_expansion_cache), 2)

        cleared = processor.clear_query_cache()

        self.assertEqual(cleared, 2)
        self.assertEqual(len(processor._query_expansion_cache), 0)

    def test_set_query_cache_size(self):
        """
        Verify set_query_cache_size enforces limit.
        """
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural networks machine learning")
        processor.compute_all(verbose=False)

        # Set small cache size
        processor.set_query_cache_size(2)

        # Add more than cache size
        _ = processor.expand_query_cached("neural")
        _ = processor.expand_query_cached("networks")
        _ = processor.expand_query_cached("machine")

        # Should only keep 2 entries
        self.assertLessEqual(len(processor._query_expansion_cache), 2)

    def test_set_query_cache_size_invalid(self):
        """
        Verify set_query_cache_size rejects invalid sizes.
        """
        processor = CorticalTextProcessor()

        with self.assertRaises(ValueError):
            processor.set_query_cache_size(0)

        with self.assertRaises(ValueError):
            processor.set_query_cache_size(-1)


if __name__ == '__main__':
    unittest.main()
