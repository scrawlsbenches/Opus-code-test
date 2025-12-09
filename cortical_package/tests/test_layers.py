"""Tests for Minicolumn and Layer classes."""

import unittest
import sys
sys.path.insert(0, '..')

from cortical import Minicolumn, CorticalLayer, HierarchicalLayer


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
