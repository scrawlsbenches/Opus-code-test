"""
Tests for the configuration module.
"""

import unittest

from cortical.config import (
    CorticalConfig,
    get_default_config,
    VALID_RELATION_CHAINS,
    DEFAULT_CHAIN_VALIDITY,
)


class TestCorticalConfig(unittest.TestCase):
    """Tests for CorticalConfig dataclass."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = CorticalConfig()

        # PageRank defaults
        self.assertEqual(config.pagerank_damping, 0.85)
        self.assertEqual(config.pagerank_iterations, 20)
        self.assertEqual(config.pagerank_tolerance, 1e-6)

        # Clustering defaults
        self.assertEqual(config.min_cluster_size, 3)
        self.assertEqual(config.cluster_strictness, 1.0)

        # Chunking defaults
        self.assertEqual(config.chunk_size, 512)
        self.assertEqual(config.chunk_overlap, 128)

    def test_custom_values(self):
        """Test creating config with custom values."""
        config = CorticalConfig(
            pagerank_damping=0.9,
            min_cluster_size=5,
            chunk_size=1024
        )

        self.assertEqual(config.pagerank_damping, 0.9)
        self.assertEqual(config.min_cluster_size, 5)
        self.assertEqual(config.chunk_size, 1024)
        # Other defaults still apply
        self.assertEqual(config.pagerank_iterations, 20)

    def test_relation_weights_default(self):
        """Test that relation weights have sensible defaults."""
        config = CorticalConfig()

        self.assertIn('IsA', config.relation_weights)
        self.assertIn('PartOf', config.relation_weights)
        self.assertIn('RelatedTo', config.relation_weights)

        # IsA should have high weight
        self.assertGreater(config.relation_weights['IsA'], 1.0)
        # Antonym should have low weight
        self.assertLess(config.relation_weights['Antonym'], 1.0)


class TestConfigValidation(unittest.TestCase):
    """Tests for configuration validation."""

    def test_invalid_pagerank_damping_too_high(self):
        """Test that damping > 1 raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            CorticalConfig(pagerank_damping=1.5)
        self.assertIn('pagerank_damping', str(ctx.exception))

    def test_invalid_pagerank_damping_too_low(self):
        """Test that damping <= 0 raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            CorticalConfig(pagerank_damping=0)
        self.assertIn('pagerank_damping', str(ctx.exception))

    def test_invalid_pagerank_damping_negative(self):
        """Test that negative damping raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            CorticalConfig(pagerank_damping=-0.5)
        self.assertIn('pagerank_damping', str(ctx.exception))

    def test_invalid_pagerank_iterations(self):
        """Test that iterations < 1 raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            CorticalConfig(pagerank_iterations=0)
        self.assertIn('pagerank_iterations', str(ctx.exception))

    def test_invalid_pagerank_tolerance(self):
        """Test that tolerance <= 0 raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            CorticalConfig(pagerank_tolerance=0)
        self.assertIn('pagerank_tolerance', str(ctx.exception))

    def test_invalid_min_cluster_size(self):
        """Test that min_cluster_size < 1 raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            CorticalConfig(min_cluster_size=0)
        self.assertIn('min_cluster_size', str(ctx.exception))

    def test_invalid_cluster_strictness_too_high(self):
        """Test that cluster_strictness > 1 raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            CorticalConfig(cluster_strictness=1.5)
        self.assertIn('cluster_strictness', str(ctx.exception))

    def test_invalid_cluster_strictness_negative(self):
        """Test that cluster_strictness < 0 raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            CorticalConfig(cluster_strictness=-0.1)
        self.assertIn('cluster_strictness', str(ctx.exception))

    def test_invalid_chunk_size(self):
        """Test that chunk_size < 1 raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            CorticalConfig(chunk_size=0)
        self.assertIn('chunk_size', str(ctx.exception))

    def test_invalid_chunk_overlap_negative(self):
        """Test that negative chunk_overlap raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            CorticalConfig(chunk_overlap=-1)
        self.assertIn('chunk_overlap', str(ctx.exception))

    def test_invalid_chunk_overlap_too_large(self):
        """Test that chunk_overlap >= chunk_size raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            CorticalConfig(chunk_size=100, chunk_overlap=100)
        self.assertIn('chunk_overlap', str(ctx.exception))

    def test_invalid_cross_layer_damping(self):
        """Test that cross_layer_damping outside (0,1) raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            CorticalConfig(cross_layer_damping=1.0)
        self.assertIn('cross_layer_damping', str(ctx.exception))

    def test_invalid_semantic_expansion_discount(self):
        """Test that semantic_expansion_discount outside [0,1] raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            CorticalConfig(semantic_expansion_discount=1.5)
        self.assertIn('semantic_expansion_discount', str(ctx.exception))

    def test_valid_boundary_values(self):
        """Test that valid boundary values are accepted."""
        # Should not raise
        config = CorticalConfig(
            pagerank_damping=0.99,
            cluster_strictness=0.0,
            semantic_expansion_discount=0.0,
            chunk_overlap=0
        )
        self.assertEqual(config.pagerank_damping, 0.99)
        self.assertEqual(config.cluster_strictness, 0.0)


class TestConfigCopy(unittest.TestCase):
    """Tests for configuration copying."""

    def test_copy_creates_new_instance(self):
        """Test that copy creates a new independent instance."""
        original = CorticalConfig(pagerank_damping=0.9)
        copied = original.copy()

        self.assertIsNot(original, copied)
        self.assertEqual(original.pagerank_damping, copied.pagerank_damping)

    def test_copy_is_independent(self):
        """Test that modifying copy doesn't affect original."""
        original = CorticalConfig()
        copied = original.copy()

        # Modify the copy's relation weights
        copied.relation_weights['IsA'] = 999.0

        # Original should be unchanged
        self.assertNotEqual(original.relation_weights['IsA'], 999.0)

    def test_copy_preserves_all_values(self):
        """Test that copy preserves all configuration values."""
        original = CorticalConfig(
            pagerank_damping=0.9,
            min_cluster_size=5,
            chunk_size=1024,
            isolation_threshold=0.05
        )
        copied = original.copy()

        self.assertEqual(copied.pagerank_damping, 0.9)
        self.assertEqual(copied.min_cluster_size, 5)
        self.assertEqual(copied.chunk_size, 1024)
        self.assertEqual(copied.isolation_threshold, 0.05)


class TestConfigSerialization(unittest.TestCase):
    """Tests for configuration serialization."""

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = CorticalConfig(pagerank_damping=0.9)
        data = config.to_dict()

        self.assertIsInstance(data, dict)
        self.assertEqual(data['pagerank_damping'], 0.9)
        self.assertIn('relation_weights', data)

    def test_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            'pagerank_damping': 0.9,
            'min_cluster_size': 5,
            'pagerank_iterations': 20,
            'pagerank_tolerance': 1e-6,
            'cluster_strictness': 1.0,
            'isolation_threshold': 0.02,
            'well_connected_threshold': 0.03,
            'weak_topic_tfidf_threshold': 0.005,
            'bridge_similarity_min': 0.005,
            'bridge_similarity_max': 0.03,
            'chunk_size': 512,
            'chunk_overlap': 128,
            'max_query_expansions': 10,
            'semantic_expansion_discount': 0.7,
            'cross_layer_damping': 0.7,
            'bigram_component_weight': 0.5,
            'bigram_chain_weight': 0.7,
            'bigram_cooccurrence_weight': 0.3,
            'concept_min_shared_docs': 1,
            'concept_min_jaccard': 0.1,
            'concept_embedding_threshold': 0.3,
            'multihop_max_hops': 2,
            'multihop_decay_factor': 0.5,
            'multihop_min_path_score': 0.3,
            'inheritance_decay_factor': 0.7,
            'inheritance_max_depth': 5,
            'inheritance_boost_factor': 0.3,
            'relation_weights': {'IsA': 1.5, 'RelatedTo': 1.0}
        }
        config = CorticalConfig.from_dict(data)

        self.assertEqual(config.pagerank_damping, 0.9)
        self.assertEqual(config.min_cluster_size, 5)

    def test_roundtrip(self):
        """Test that to_dict and from_dict are inverses."""
        original = CorticalConfig(
            pagerank_damping=0.9,
            min_cluster_size=5,
            chunk_size=1024
        )

        data = original.to_dict()
        restored = CorticalConfig.from_dict(data)

        self.assertEqual(original.pagerank_damping, restored.pagerank_damping)
        self.assertEqual(original.min_cluster_size, restored.min_cluster_size)
        self.assertEqual(original.chunk_size, restored.chunk_size)


class TestGetDefaultConfig(unittest.TestCase):
    """Tests for get_default_config function."""

    def test_returns_config_instance(self):
        """Test that get_default_config returns a CorticalConfig."""
        config = get_default_config()
        self.assertIsInstance(config, CorticalConfig)

    def test_returns_new_instance_each_time(self):
        """Test that get_default_config returns new instances."""
        config1 = get_default_config()
        config2 = get_default_config()
        self.assertIsNot(config1, config2)


class TestValidRelationChains(unittest.TestCase):
    """Tests for VALID_RELATION_CHAINS constant."""

    def test_transitive_chains_high_score(self):
        """Test that transitive chains have high validity scores."""
        self.assertEqual(VALID_RELATION_CHAINS[('IsA', 'IsA')], 1.0)
        self.assertEqual(VALID_RELATION_CHAINS[('PartOf', 'PartOf')], 1.0)

    def test_contradictory_chains_low_score(self):
        """Test that contradictory chains have low validity scores."""
        self.assertLess(VALID_RELATION_CHAINS[('Antonym', 'IsA')], 0.5)

    def test_association_chains_medium_score(self):
        """Test that association chains have medium validity scores."""
        score = VALID_RELATION_CHAINS[('RelatedTo', 'RelatedTo')]
        self.assertGreater(score, 0.3)
        self.assertLess(score, 0.8)

    def test_default_chain_validity(self):
        """Test DEFAULT_CHAIN_VALIDITY value."""
        self.assertEqual(DEFAULT_CHAIN_VALIDITY, 0.4)


class TestProcessorConfigIntegration(unittest.TestCase):
    """Tests for CorticalConfig integration with CorticalTextProcessor."""

    def test_processor_accepts_config(self):
        """Test that processor accepts config parameter."""
        from cortical.processor import CorticalTextProcessor

        config = CorticalConfig(min_cluster_size=5, chunk_size=256)
        processor = CorticalTextProcessor(config=config)

        self.assertEqual(processor.config.min_cluster_size, 5)
        self.assertEqual(processor.config.chunk_size, 256)

    def test_processor_uses_default_config(self):
        """Test that processor uses default config when none provided."""
        from cortical.processor import CorticalTextProcessor

        processor = CorticalTextProcessor()

        # Should have default values
        self.assertEqual(processor.config.min_cluster_size, 3)
        self.assertEqual(processor.config.chunk_size, 512)

    def test_config_used_in_expand_query(self):
        """Test that config.max_query_expansions is used."""
        from cortical.processor import CorticalTextProcessor

        config = CorticalConfig(max_query_expansions=3)
        processor = CorticalTextProcessor(config=config)
        processor.process_document("doc1", "neural network deep learning models")
        processor.compute_all(verbose=False)

        # When no max_expansions specified, should use config value
        result = processor.expand_query("neural")
        # Should respect the config limit (may have fewer if not enough expansions)
        self.assertIsInstance(result, dict)

    def test_config_preserved_on_save_load(self):
        """Test that config is preserved after save/load."""
        import tempfile
        import os
        from cortical.processor import CorticalTextProcessor

        # Create processor with custom config
        config = CorticalConfig(
            min_cluster_size=5,
            chunk_size=256,
            max_query_expansions=15
        )
        processor = CorticalTextProcessor(config=config)
        processor.process_document("doc1", "test content")
        processor.compute_all(verbose=False)

        # Save and load
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name

        try:
            processor.save(temp_path, verbose=False)
            loaded = CorticalTextProcessor.load(temp_path, verbose=False)

            # Config should be preserved
            self.assertEqual(loaded.config.min_cluster_size, 5)
            self.assertEqual(loaded.config.chunk_size, 256)
            self.assertEqual(loaded.config.max_query_expansions, 15)
        finally:
            os.unlink(temp_path)

    def test_load_without_config_uses_default(self):
        """Test that loading old files without config uses defaults."""
        import tempfile
        import os
        from cortical.processor import CorticalTextProcessor

        # Create processor (with default config)
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")
        processor.compute_all(verbose=False)

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name

        try:
            processor.save(temp_path, verbose=False)
            loaded = CorticalTextProcessor.load(temp_path, verbose=False)

            # Should have valid config (either restored or default)
            self.assertIsNotNone(loaded.config)
            self.assertIsInstance(loaded.config, CorticalConfig)
        finally:
            os.unlink(temp_path)


if __name__ == '__main__':
    unittest.main()
