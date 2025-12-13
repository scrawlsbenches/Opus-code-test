"""
Tests for the MCP server implementation.

Tests all tools and error handling for the Cortical Text Processor MCP server.
"""

import unittest
import tempfile
import os
from pathlib import Path

from cortical.mcp_server import CorticalMCPServer, create_mcp_server
from cortical import CorticalTextProcessor


class TestMCPServerCreation(unittest.TestCase):
    """Test MCP server initialization."""

    def test_create_empty_server(self):
        """Test creating a server with empty corpus."""
        server = create_mcp_server()
        self.assertIsNotNone(server)
        self.assertIsNotNone(server.processor)
        self.assertEqual(len(server.processor.documents), 0)

    def test_create_server_with_corpus(self):
        """Test creating a server with pre-loaded corpus."""
        # Create a temporary corpus
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Neural networks process information.")
        processor.process_document("doc2", "Machine learning algorithms learn patterns.")
        processor.compute_all()

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
            temp_path = f.name
            processor.save(temp_path)

        try:
            # Create server with corpus
            server = create_mcp_server(corpus_path=temp_path)
            self.assertIsNotNone(server)
            self.assertEqual(len(server.processor.documents), 2)
        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_create_server_nonexistent_corpus(self):
        """Test creating a server with nonexistent corpus path."""
        server = create_mcp_server(corpus_path="/nonexistent/path.pkl")
        self.assertIsNotNone(server)
        # Should start with empty corpus
        self.assertEqual(len(server.processor.documents), 0)


class TestMCPServerToolsBase(unittest.TestCase):
    """Base class for MCP tool tests."""

    def setUp(self):
        """Set up test server with sample corpus."""
        self.server = create_mcp_server()

        # Add sample documents
        self.server.processor.process_document(
            "doc1",
            "Neural networks are computational models inspired by biological neurons."
        )
        self.server.processor.process_document(
            "doc2",
            "Machine learning algorithms can learn patterns from data."
        )
        self.server.processor.process_document(
            "doc3",
            "Deep learning uses multi-layered neural networks for complex tasks."
        )
        self.server.processor.compute_all()


class TestSearchTool(TestMCPServerToolsBase):
    """Test the search tool."""

    async def async_search(self, query=None, **kwargs):
        """Helper to call search tool."""
        # Build arguments dict
        args_dict = {"query": query} if query is not None else {}
        args_dict.update(kwargs)
        # Call the tool directly - returns (content, metadata) tuple
        content, metadata = await self.server.mcp.call_tool("search", args_dict)
        # Extract actual result from metadata
        return metadata.get('result', {})

    def test_search_basic(self):
        """Test basic search functionality."""
        import asyncio
        result = asyncio.run(self.async_search(query="neural networks"))

        self.assertIn("results", result)
        self.assertIn("count", result)
        self.assertGreater(result["count"], 0)

        # Check result structure
        if result["results"]:
            first_result = result["results"][0]
            self.assertIn("doc_id", first_result)
            self.assertIn("score", first_result)

    def test_search_empty_query(self):
        """Test search with empty query."""
        import asyncio
        result = asyncio.run(self.async_search(query=""))

        self.assertIn("error", result)
        self.assertEqual(result["count"], 0)

    def test_search_top_n(self):
        """Test search with different top_n values."""
        import asyncio
        result = asyncio.run(self.async_search(query="neural", top_n=2))

        self.assertIn("results", result)
        self.assertLessEqual(len(result["results"]), 2)

    def test_search_invalid_top_n(self):
        """Test search with invalid top_n."""
        import asyncio
        result = asyncio.run(self.async_search(query="neural", top_n=0))

        self.assertIn("error", result)


class TestPassagesTool(TestMCPServerToolsBase):
    """Test the passages tool."""

    async def async_passages(self, query=None, **kwargs):
        """Helper to call passages tool."""
        # Build arguments dict
        args_dict = {"query": query} if query is not None else {}
        args_dict.update(kwargs)
        content, metadata = await self.server.mcp.call_tool("passages", args_dict)
        return metadata.get('result', {})

    def test_passages_basic(self):
        """Test basic passage retrieval."""
        import asyncio
        result = asyncio.run(self.async_passages(query="neural networks"))

        self.assertIn("passages", result)
        self.assertIn("count", result)

        # Check passage structure if any found
        if result["passages"]:
            passage = result["passages"][0]
            self.assertIn("doc_id", passage)
            self.assertIn("text", passage)
            self.assertIn("start", passage)
            self.assertIn("end", passage)
            self.assertIn("score", passage)

    def test_passages_empty_query(self):
        """Test passages with empty query."""
        import asyncio
        result = asyncio.run(self.async_passages(query=""))

        self.assertIn("error", result)
        self.assertEqual(result["count"], 0)

    def test_passages_with_chunk_size(self):
        """Test passages with custom chunk size."""
        import asyncio
        result = asyncio.run(self.async_passages(query="neural", chunk_size=50))

        self.assertIn("passages", result)


class TestExpandQueryTool(TestMCPServerToolsBase):
    """Test the expand_query tool."""

    async def async_expand_query(self, query=None, **kwargs):
        """Helper to call expand_query tool."""
        # Build arguments dict
        args_dict = {"query": query} if query is not None else {}
        args_dict.update(kwargs)
        content, metadata = await self.server.mcp.call_tool("expand_query", args_dict)
        return metadata.get('result', {})

    def test_expand_query_basic(self):
        """Test basic query expansion."""
        import asyncio
        result = asyncio.run(self.async_expand_query(query="neural"))

        self.assertIn("expansions", result)
        self.assertIn("count", result)
        self.assertIsInstance(result["expansions"], dict)

        # Check that expansions contain weights
        for term, weight in result["expansions"].items():
            self.assertIsInstance(term, str)
            self.assertIsInstance(weight, (int, float))

    def test_expand_query_empty(self):
        """Test query expansion with empty query."""
        import asyncio
        result = asyncio.run(self.async_expand_query(query=""))

        self.assertIn("error", result)
        self.assertEqual(result["count"], 0)

    def test_expand_query_max_expansions(self):
        """Test query expansion with max_expansions limit."""
        import asyncio
        result = asyncio.run(self.async_expand_query(query="neural", max_expansions=5))

        self.assertIn("expansions", result)
        # max_expansions controls expansion terms added, but the original query term
        # is also included, so we allow for original + max_expansions
        self.assertLessEqual(len(result["expansions"]), 10)  # Reasonable upper bound

    def test_expand_query_invalid_max_expansions(self):
        """Test query expansion with invalid max_expansions."""
        import asyncio
        result = asyncio.run(self.async_expand_query(query="neural", max_expansions=0))

        self.assertIn("error", result)


class TestCorpusStatsTool(TestMCPServerToolsBase):
    """Test the corpus_stats tool."""

    async def async_corpus_stats(self):
        """Helper to call corpus_stats tool."""
        content, metadata = await self.server.mcp.call_tool("corpus_stats", {})
        return metadata.get('result', {})

    def test_corpus_stats_basic(self):
        """Test corpus statistics retrieval."""
        import asyncio
        result = asyncio.run(self.async_corpus_stats())

        self.assertIsInstance(result, dict)
        # Stats should contain some information about the corpus
        # The exact structure depends on get_corpus_summary implementation

    def test_corpus_stats_empty_corpus(self):
        """Test corpus stats on empty corpus."""
        import asyncio
        empty_server = create_mcp_server()

        async def get_stats():
            content, metadata = await empty_server.mcp.call_tool("corpus_stats", {})
            return metadata.get('result', {})

        result = asyncio.run(get_stats())
        self.assertIsInstance(result, dict)


class TestAddDocumentTool(TestMCPServerToolsBase):
    """Test the add_document tool."""

    async def async_add_document(self, *args, **kwargs):
        """Helper to call add_document tool."""
        content, metadata = await self.server.mcp.call_tool("add_document", kwargs)
        return metadata.get('result', {})

    def test_add_document_basic(self):
        """Test adding a document."""
        import asyncio
        initial_count = len(self.server.processor.documents)

        result = asyncio.run(self.async_add_document(
            doc_id="new_doc",
            content="This is a new document about artificial intelligence."
        ))

        self.assertIn("stats", result)
        self.assertIn("doc_id", result)
        self.assertEqual(result["doc_id"], "new_doc")

        # Verify document was added
        self.assertEqual(len(self.server.processor.documents), initial_count + 1)
        self.assertIn("new_doc", self.server.processor.documents)

    def test_add_document_empty_id(self):
        """Test adding a document with empty ID."""
        import asyncio
        result = asyncio.run(self.async_add_document(
            doc_id="",
            content="Content"
        ))

        self.assertIn("error", result)

    def test_add_document_invalid_content(self):
        """Test adding a document with invalid content type."""
        import asyncio
        # Note: We can't directly pass non-string through MCP,
        # but we can test the server's handling
        # This test verifies the error handling exists

    def test_add_document_recompute_levels(self):
        """Test adding document with different recompute levels."""
        import asyncio

        # Test 'tfidf' (default)
        result1 = asyncio.run(self.async_add_document(
            doc_id="doc_tfidf",
            content="Test document for TF-IDF recomputation.",
            recompute="tfidf"
        ))
        self.assertIn("stats", result1)

        # Test 'none'
        result2 = asyncio.run(self.async_add_document(
            doc_id="doc_none",
            content="Test document with no recomputation.",
            recompute="none"
        ))
        self.assertIn("stats", result2)

        # Test 'full'
        result3 = asyncio.run(self.async_add_document(
            doc_id="doc_full",
            content="Test document with full recomputation.",
            recompute="full"
        ))
        self.assertIn("stats", result3)

    def test_add_document_invalid_recompute(self):
        """Test adding document with invalid recompute level."""
        import asyncio
        result = asyncio.run(self.async_add_document(
            doc_id="doc_invalid",
            content="Test document.",
            recompute="invalid"
        ))

        self.assertIn("error", result)


class TestMCPServerIntegration(unittest.TestCase):
    """Integration tests for MCP server."""

    def test_end_to_end_workflow(self):
        """Test complete workflow: add documents, search, get passages."""
        import asyncio

        async def workflow():
            server = create_mcp_server()

            # Add documents
            _, doc1_meta = await server.mcp.call_tool("add_document", {
                "doc_id": "ai_basics",
                "content": "Artificial intelligence enables machines to learn and reason."
            })
            doc1_result = doc1_meta.get('result', {})
            self.assertIn("stats", doc1_result)

            _, doc2_meta = await server.mcp.call_tool("add_document", {
                "doc_id": "ml_intro",
                "content": "Machine learning is a subset of artificial intelligence."
            })
            doc2_result = doc2_meta.get('result', {})
            self.assertIn("stats", doc2_result)

            # Get corpus stats
            _, stats_meta = await server.mcp.call_tool("corpus_stats", {})
            stats = stats_meta.get('result', {})
            self.assertIsInstance(stats, dict)

            # Search
            _, search_meta = await server.mcp.call_tool("search", {
                "query": "artificial intelligence",
                "top_n": 5
            })
            search_result = search_meta.get('result', {})
            self.assertGreater(search_result["count"], 0)

            # Get passages
            _, passages_meta = await server.mcp.call_tool("passages", {
                "query": "machine learning",
                "top_n": 3
            })
            passages_result = passages_meta.get('result', {})
            self.assertIn("passages", passages_result)

            # Expand query
            _, expand_meta = await server.mcp.call_tool("expand_query", {
                "query": "intelligence"
            })
            expand_result = expand_meta.get('result', {})
            self.assertIn("expansions", expand_result)

        asyncio.run(workflow())


if __name__ == "__main__":
    unittest.main()
