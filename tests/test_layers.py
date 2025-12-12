"""Tests for Minicolumn, Edge, and Layer classes."""

import unittest
import sys
sys.path.insert(0, '..')

from cortical import Minicolumn, Edge, CorticalLayer, HierarchicalLayer


class TestMinicolumn(unittest.TestCase):
    """Test the Minicolumn class."""
    
    def test_creation(self):
        """Test basic minicolumn creation."""
        col = Minicolumn("L0_test", "test", 0)
        self.assertEqual(col.id, "L0_test")
        self.assertEqual(col.content, "test")
        self.assertEqual(col.layer, 0)
        self.assertEqual(col.activation, 0.0)
    
    def test_lateral_connections(self):
        """Test adding lateral connections."""
        col = Minicolumn("L0_test", "test", 0)
        col.add_lateral_connection("L0_other", 0.5)
        self.assertIn("L0_other", col.lateral_connections)
        self.assertEqual(col.lateral_connections["L0_other"], 0.5)
    
    def test_connection_strengthening(self):
        """Test that repeated connections strengthen."""
        col = Minicolumn("L0_test", "test", 0)
        col.add_lateral_connection("L0_other", 0.5)
        col.add_lateral_connection("L0_other", 0.3)
        self.assertEqual(col.lateral_connections["L0_other"], 0.8)
    
    def test_connection_count(self):
        """Test connection count."""
        col = Minicolumn("L0_test", "test", 0)
        col.add_lateral_connection("L0_a", 1.0)
        col.add_lateral_connection("L0_b", 1.0)
        self.assertEqual(col.connection_count(), 2)
    
    def test_document_ids(self):
        """Test document ID tracking."""
        col = Minicolumn("L0_test", "test", 0)
        col.document_ids.add("doc1")
        col.document_ids.add("doc2")
        self.assertEqual(len(col.document_ids), 2)
    
    def test_serialization(self):
        """Test to_dict and from_dict."""
        col = Minicolumn("L0_test", "test", 0)
        col.activation = 5.0
        col.occurrence_count = 10
        col.document_ids.add("doc1")
        col.add_lateral_connection("L0_other", 2.0)
        
        data = col.to_dict()
        restored = Minicolumn.from_dict(data)
        
        self.assertEqual(restored.id, col.id)
        self.assertEqual(restored.content, col.content)
        self.assertEqual(restored.activation, col.activation)
        self.assertEqual(restored.occurrence_count, col.occurrence_count)


class TestHierarchicalLayer(unittest.TestCase):
    """Test the HierarchicalLayer class."""
    
    def test_creation(self):
        """Test layer creation."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        self.assertEqual(layer.level, CorticalLayer.TOKENS)
        self.assertEqual(len(layer.minicolumns), 0)
    
    def test_get_or_create(self):
        """Test get_or_create_minicolumn."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        col = layer.get_or_create_minicolumn("test")
        self.assertEqual(col.content, "test")
        
        # Should return same column
        col2 = layer.get_or_create_minicolumn("test")
        self.assertIs(col, col2)
    
    def test_get_minicolumn(self):
        """Test get_minicolumn returns None for missing."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        self.assertIsNone(layer.get_minicolumn("missing"))
        
        layer.get_or_create_minicolumn("exists")
        self.assertIsNotNone(layer.get_minicolumn("exists"))
    
    def test_column_count(self):
        """Test column counting."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        layer.get_or_create_minicolumn("a")
        layer.get_or_create_minicolumn("b")
        layer.get_or_create_minicolumn("c")
        self.assertEqual(layer.column_count(), 3)
    
    def test_iteration(self):
        """Test iterating over layer."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        layer.get_or_create_minicolumn("a")
        layer.get_or_create_minicolumn("b")

        contents = [col.content for col in layer]
        self.assertEqual(set(contents), {"a", "b"})

    def test_contains(self):
        """Test __contains__ method."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        layer.get_or_create_minicolumn("exists")

        self.assertTrue("exists" in layer)
        self.assertFalse("missing" in layer)

    def test_remove_minicolumn(self):
        """Test removing a minicolumn."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        layer.get_or_create_minicolumn("test")

        # Verify it exists
        self.assertIn("test", layer)
        self.assertEqual(layer.column_count(), 1)

        # Remove it
        result = layer.remove_minicolumn("test")
        self.assertTrue(result)
        self.assertNotIn("test", layer)
        self.assertEqual(layer.column_count(), 0)

        # Try removing non-existent
        result = layer.remove_minicolumn("missing")
        self.assertFalse(result)

    def test_activation_range_non_empty(self):
        """Test activation_range with non-empty layer."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        col1 = layer.get_or_create_minicolumn("a")
        col2 = layer.get_or_create_minicolumn("b")
        col3 = layer.get_or_create_minicolumn("c")

        col1.activation = 1.0
        col2.activation = 5.0
        col3.activation = 3.0

        min_act, max_act = layer.activation_range()
        self.assertEqual(min_act, 1.0)
        self.assertEqual(max_act, 5.0)

    def test_top_by_pagerank(self):
        """Test top_by_pagerank method."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        col1 = layer.get_or_create_minicolumn("a")
        col2 = layer.get_or_create_minicolumn("b")
        col3 = layer.get_or_create_minicolumn("c")

        col1.pagerank = 0.1
        col2.pagerank = 0.5
        col3.pagerank = 0.3

        top = layer.top_by_pagerank(n=2)
        self.assertEqual(len(top), 2)
        self.assertEqual(top[0][0], "b")  # Highest pagerank
        self.assertEqual(top[0][1], 0.5)
        self.assertEqual(top[1][0], "c")  # Second highest
        self.assertEqual(top[1][1], 0.3)

    def test_top_by_tfidf(self):
        """Test top_by_tfidf method."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        col1 = layer.get_or_create_minicolumn("a")
        col2 = layer.get_or_create_minicolumn("b")
        col3 = layer.get_or_create_minicolumn("c")

        col1.tfidf = 0.2
        col2.tfidf = 0.8
        col3.tfidf = 0.5

        top = layer.top_by_tfidf(n=2)
        self.assertEqual(len(top), 2)
        self.assertEqual(top[0][0], "b")  # Highest tfidf
        self.assertEqual(top[0][1], 0.8)
        self.assertEqual(top[1][0], "c")  # Second highest
        self.assertEqual(top[1][1], 0.5)

    def test_top_by_activation(self):
        """Test top_by_activation method."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        col1 = layer.get_or_create_minicolumn("a")
        col2 = layer.get_or_create_minicolumn("b")
        col3 = layer.get_or_create_minicolumn("c")

        col1.activation = 1.0
        col2.activation = 10.0
        col3.activation = 5.0

        top = layer.top_by_activation(n=2)
        self.assertEqual(len(top), 2)
        self.assertEqual(top[0][0], "b")  # Highest activation
        self.assertEqual(top[0][1], 10.0)
        self.assertEqual(top[1][0], "c")  # Second highest
        self.assertEqual(top[1][1], 5.0)


class TestCorticalLayerEnum(unittest.TestCase):
    """Test the CorticalLayer enum."""

    def test_values(self):
        """Test layer values."""
        self.assertEqual(CorticalLayer.TOKENS.value, 0)
        self.assertEqual(CorticalLayer.BIGRAMS.value, 1)
        self.assertEqual(CorticalLayer.CONCEPTS.value, 2)
        self.assertEqual(CorticalLayer.DOCUMENTS.value, 3)

    def test_description(self):
        """Test layer descriptions."""
        self.assertIn("Token", CorticalLayer.TOKENS.description)
        self.assertIn("Document", CorticalLayer.DOCUMENTS.description)

    def test_analogy_property(self):
        """Test layer analogy property for all layers."""
        # Test TOKENS
        self.assertIn("V1", CorticalLayer.TOKENS.analogy)
        self.assertIn("token", CorticalLayer.TOKENS.analogy.lower())

        # Test BIGRAMS
        self.assertIn("V2", CorticalLayer.BIGRAMS.analogy)
        self.assertIn("pattern", CorticalLayer.BIGRAMS.analogy.lower())

        # Test CONCEPTS
        self.assertIn("V4", CorticalLayer.CONCEPTS.analogy)
        self.assertIn("concept", CorticalLayer.CONCEPTS.analogy.lower())

        # Test DOCUMENTS
        self.assertIn("IT", CorticalLayer.DOCUMENTS.analogy)
        self.assertIn("document", CorticalLayer.DOCUMENTS.analogy.lower())


class TestEdge(unittest.TestCase):
    """Test the Edge dataclass."""

    def test_edge_creation(self):
        """Test basic Edge creation."""
        edge = Edge("L0_target", 0.5)
        self.assertEqual(edge.target_id, "L0_target")
        self.assertEqual(edge.weight, 0.5)
        self.assertEqual(edge.relation_type, 'co_occurrence')
        self.assertEqual(edge.confidence, 1.0)
        self.assertEqual(edge.source, 'corpus')

    def test_edge_with_metadata(self):
        """Test Edge creation with full metadata."""
        edge = Edge(
            target_id="L0_target",
            weight=0.8,
            relation_type='IsA',
            confidence=0.9,
            source='semantic'
        )
        self.assertEqual(edge.relation_type, 'IsA')
        self.assertEqual(edge.confidence, 0.9)
        self.assertEqual(edge.source, 'semantic')

    def test_edge_serialization(self):
        """Test Edge to_dict and from_dict."""
        edge = Edge("L0_target", 0.8, 'RelatedTo', 0.9, 'semantic')
        data = edge.to_dict()

        restored = Edge.from_dict(data)
        self.assertEqual(restored.target_id, edge.target_id)
        self.assertEqual(restored.weight, edge.weight)
        self.assertEqual(restored.relation_type, edge.relation_type)
        self.assertEqual(restored.confidence, edge.confidence)
        self.assertEqual(restored.source, edge.source)

    def test_edge_from_dict_defaults(self):
        """Test Edge.from_dict with minimal data."""
        data = {'target_id': 'L0_test'}
        edge = Edge.from_dict(data)
        self.assertEqual(edge.target_id, 'L0_test')
        self.assertEqual(edge.weight, 1.0)
        self.assertEqual(edge.relation_type, 'co_occurrence')


class TestTypedConnections(unittest.TestCase):
    """Test typed connection functionality on Minicolumn."""

    def test_add_typed_connection(self):
        """Test adding a typed connection."""
        col = Minicolumn("L0_test", "test", 0)
        col.add_typed_connection("L0_other", 0.5, relation_type='RelatedTo')

        self.assertIn("L0_other", col.typed_connections)
        edge = col.typed_connections["L0_other"]
        self.assertEqual(edge.weight, 0.5)
        self.assertEqual(edge.relation_type, 'RelatedTo')

    def test_typed_connection_also_updates_lateral(self):
        """Test that typed connections also update lateral_connections."""
        col = Minicolumn("L0_test", "test", 0)
        col.add_typed_connection("L0_other", 0.5, relation_type='RelatedTo')

        # Should also be in lateral_connections
        self.assertIn("L0_other", col.lateral_connections)
        self.assertEqual(col.lateral_connections["L0_other"], 0.5)

    def test_typed_connection_weight_accumulation(self):
        """Test that typed connection weights accumulate."""
        col = Minicolumn("L0_test", "test", 0)
        col.add_typed_connection("L0_other", 0.5, relation_type='RelatedTo')
        col.add_typed_connection("L0_other", 0.3, relation_type='RelatedTo')

        edge = col.typed_connections["L0_other"]
        self.assertEqual(edge.weight, 0.8)

    def test_typed_connection_relation_type_priority(self):
        """Test that specific relation types take priority over co_occurrence."""
        col = Minicolumn("L0_test", "test", 0)
        col.add_typed_connection("L0_other", 0.5, relation_type='co_occurrence')
        col.add_typed_connection("L0_other", 0.3, relation_type='IsA')

        edge = col.typed_connections["L0_other"]
        self.assertEqual(edge.relation_type, 'IsA')

    def test_typed_connection_source_priority(self):
        """Test that semantic/inferred sources take priority over corpus."""
        col = Minicolumn("L0_test", "test", 0)
        col.add_typed_connection("L0_other", 0.5, source='corpus')
        col.add_typed_connection("L0_other", 0.3, source='semantic')

        edge = col.typed_connections["L0_other"]
        self.assertEqual(edge.source, 'semantic')

    def test_typed_connection_confidence_weighted_average(self):
        """Test that confidence uses weighted average (can increase or decrease)."""
        col = Minicolumn("L0_test", "test", 0)
        col.add_typed_connection("L0_other", 0.5, confidence=0.7)
        col.add_typed_connection("L0_other", 0.3, confidence=0.9)

        edge = col.typed_connections["L0_other"]
        # Weighted average: (0.7 * 0.5 + 0.9 * 0.3) / 0.8 = 0.775
        self.assertAlmostEqual(edge.confidence, 0.775, places=5)

    def test_typed_connection_confidence_can_decrease(self):
        """Test that confidence can decrease with lower-confidence evidence."""
        col = Minicolumn("L0_test", "test", 0)
        col.add_typed_connection("L0_other", 1.0, confidence=0.9)  # High confidence
        col.add_typed_connection("L0_other", 1.0, confidence=0.3)  # Low confidence evidence

        edge = col.typed_connections["L0_other"]
        # Weighted average: (0.9 * 1.0 + 0.3 * 1.0) / 2.0 = 0.6
        self.assertAlmostEqual(edge.confidence, 0.6, places=5)

    def test_get_typed_connection(self):
        """Test retrieving a typed connection."""
        col = Minicolumn("L0_test", "test", 0)
        col.add_typed_connection("L0_other", 0.5, relation_type='IsA')

        edge = col.get_typed_connection("L0_other")
        self.assertIsNotNone(edge)
        self.assertEqual(edge.relation_type, 'IsA')

        # Non-existent connection
        self.assertIsNone(col.get_typed_connection("L0_missing"))

    def test_get_connections_by_type(self):
        """Test filtering connections by relation type."""
        col = Minicolumn("L0_test", "test", 0)
        col.add_typed_connection("L0_a", 0.5, relation_type='IsA')
        col.add_typed_connection("L0_b", 0.3, relation_type='IsA')
        col.add_typed_connection("L0_c", 0.4, relation_type='PartOf')

        is_a_edges = col.get_connections_by_type('IsA')
        self.assertEqual(len(is_a_edges), 2)

        part_of_edges = col.get_connections_by_type('PartOf')
        self.assertEqual(len(part_of_edges), 1)

    def test_get_connections_by_source(self):
        """Test filtering connections by source."""
        col = Minicolumn("L0_test", "test", 0)
        col.add_typed_connection("L0_a", 0.5, source='corpus')
        col.add_typed_connection("L0_b", 0.3, source='semantic')
        col.add_typed_connection("L0_c", 0.4, source='semantic')

        corpus_edges = col.get_connections_by_source('corpus')
        self.assertEqual(len(corpus_edges), 1)

        semantic_edges = col.get_connections_by_source('semantic')
        self.assertEqual(len(semantic_edges), 2)

    def test_typed_connections_serialization(self):
        """Test that typed connections survive serialization."""
        col = Minicolumn("L0_test", "test", 0)
        col.add_typed_connection("L0_other", 0.8, relation_type='IsA', confidence=0.9)

        data = col.to_dict()
        restored = Minicolumn.from_dict(data)

        self.assertIn("L0_other", restored.typed_connections)
        edge = restored.typed_connections["L0_other"]
        self.assertEqual(edge.weight, 0.8)
        self.assertEqual(edge.relation_type, 'IsA')
        self.assertEqual(edge.confidence, 0.9)

    def test_empty_typed_connections_serialization(self):
        """Test serialization with no typed connections."""
        col = Minicolumn("L0_test", "test", 0)

        data = col.to_dict()
        self.assertEqual(data['typed_connections'], {})

        restored = Minicolumn.from_dict(data)
        self.assertEqual(restored.typed_connections, {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
