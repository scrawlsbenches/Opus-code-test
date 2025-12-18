"""
MCP (Model Context Protocol) Server for Cortical Text Processor.

Provides an MCP server interface for AI agents to integrate with the
Cortical Text Processor, enabling semantic search, query expansion,
passage retrieval, and document indexing capabilities.

Example:
    python -m cortical.projects.mcp.server

    Or programmatically:
    from cortical.projects.mcp.server import create_mcp_server
    server = create_mcp_server()
    server.run()
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any

from mcp.server import FastMCP

from cortical.processor import CorticalTextProcessor
from cortical.config import CorticalConfig

logger = logging.getLogger(__name__)


class CorticalMCPServer:
    """
    MCP Server wrapper for CorticalTextProcessor.

    Provides tools for:
    - search: Find relevant documents for a query
    - passages: Retrieve RAG-ready text passages
    - expand_query: Get query expansion terms
    - corpus_stats: Get corpus statistics
    - add_document: Index a new document
    """

    def __init__(
        self,
        corpus_path: Optional[str] = None,
        config: Optional[CorticalConfig] = None,
        name: str = "cortical-text-processor",
        version: str = "2.0.0"
    ):
        """
        Initialize the MCP server with a Cortical Text Processor.

        Args:
            corpus_path: Path to saved corpus pickle file. If None, starts with empty corpus.
            config: Optional CorticalConfig. If None, uses default config.
            name: Server name for MCP protocol
            version: Server version
        """
        self.name = name
        self.version = version

        # Initialize processor
        if corpus_path and os.path.exists(corpus_path):
            logger.info(f"Loading corpus from {corpus_path}")
            self.processor = CorticalTextProcessor.load(corpus_path)
        else:
            logger.info("Starting with empty corpus")
            self.processor = CorticalTextProcessor(config=config)

        # Create FastMCP server
        self.mcp = FastMCP(
            name=self.name,
            instructions=(
                "Cortical Text Processor - A neocortex-inspired semantic search "
                "and text analysis system. Use this server to search documents, "
                "retrieve relevant passages, expand queries, and manage a text corpus."
            )
        )

        # Register tools
        self._register_tools()

    def _register_tools(self):
        """Register all MCP tools."""

        @self.mcp.tool()
        async def search(query: str, top_n: int = 5) -> Dict[str, Any]:
            """
            Search for documents relevant to a query.

            Args:
                query: Search query string
                top_n: Number of top results to return (default: 5)

            Returns:
                Dict with 'results' list of (doc_id, score) tuples and 'count'
            """
            try:
                if not query or not query.strip():
                    return {
                        "error": "Query must be a non-empty string",
                        "results": [],
                        "count": 0
                    }

                if top_n < 1:
                    return {
                        "error": "top_n must be at least 1",
                        "results": [],
                        "count": 0
                    }

                results = self.processor.find_documents_for_query(
                    query_text=query,
                    top_n=top_n
                )

                return {
                    "results": [
                        {"doc_id": doc_id, "score": float(score)}
                        for doc_id, score in results
                    ],
                    "count": len(results)
                }
            except Exception as e:
                # Catch all exceptions to return graceful error responses.
                # Processor can raise various exception types depending on internal state.
                logger.error(f"Error in search: {e}", exc_info=True)
                return {
                    "error": str(e),
                    "results": [],
                    "count": 0
                }

        @self.mcp.tool()
        async def passages(
            query: str,
            top_n: int = 5,
            chunk_size: Optional[int] = None
        ) -> Dict[str, Any]:
            """
            Find relevant text passages for a query (RAG-ready).

            Args:
                query: Search query string
                top_n: Number of top passages to return (default: 5)
                chunk_size: Size of text chunks in characters (default: from config)

            Returns:
                Dict with 'passages' list containing doc_id, text, start, end, score
            """
            try:
                if not query or not query.strip():
                    return {
                        "error": "Query must be a non-empty string",
                        "passages": [],
                        "count": 0
                    }

                if top_n < 1:
                    return {
                        "error": "top_n must be at least 1",
                        "passages": [],
                        "count": 0
                    }

                results = self.processor.find_passages_for_query(
                    query_text=query,
                    top_n=top_n,
                    chunk_size=chunk_size
                )

                return {
                    "passages": [
                        {
                            "doc_id": doc_id,
                            "text": text,
                            "start": start,
                            "end": end,
                            "score": float(score)
                        }
                        for doc_id, text, start, end, score in results
                    ],
                    "count": len(results)
                }
            except Exception as e:
                # Catch all exceptions to return graceful error responses.
                # Processor can raise various exception types depending on internal state.
                logger.error(f"Error in passages: {e}", exc_info=True)
                return {
                    "error": str(e),
                    "passages": [],
                    "count": 0
                }

        @self.mcp.tool()
        async def expand_query(query: str, max_expansions: int = 10) -> Dict[str, Any]:
            """
            Expand a query with related terms using semantic connections.

            Args:
                query: Query string to expand
                max_expansions: Maximum number of expansion terms (default: 10)

            Returns:
                Dict with 'expansions' dict mapping terms to weights and 'count'
            """
            try:
                if not query or not query.strip():
                    return {
                        "error": "Query must be a non-empty string",
                        "expansions": {},
                        "count": 0
                    }

                if max_expansions < 1:
                    return {
                        "error": "max_expansions must be at least 1",
                        "expansions": {},
                        "count": 0
                    }

                expansions = self.processor.expand_query(
                    query_text=query,
                    max_expansions=max_expansions
                )

                # Convert to serializable format
                result = {term: float(weight) for term, weight in expansions.items()}

                return {
                    "expansions": result,
                    "count": len(result)
                }
            except Exception as e:
                # Catch all exceptions to return graceful error responses.
                # Processor can raise various exception types depending on internal state.
                logger.error(f"Error in expand_query: {e}", exc_info=True)
                return {
                    "error": str(e),
                    "expansions": {},
                    "count": 0
                }

        @self.mcp.tool()
        async def corpus_stats() -> Dict[str, Any]:
            """
            Get statistics about the current corpus.

            Returns:
                Dict with corpus statistics including document count, layer stats, etc.
            """
            try:
                stats = self.processor.get_corpus_summary()

                # Ensure all values are JSON-serializable
                def make_serializable(obj):
                    if isinstance(obj, dict):
                        return {k: make_serializable(v) for k, v in obj.items()}
                    elif isinstance(obj, (list, tuple)):
                        return [make_serializable(item) for item in obj]
                    elif isinstance(obj, (int, float, str, bool, type(None))):
                        return obj
                    else:
                        return str(obj)

                return make_serializable(stats)
            except Exception as e:
                # Catch all exceptions to return graceful error responses.
                # Processor can raise various exception types depending on internal state.
                logger.error(f"Error in corpus_stats: {e}", exc_info=True)
                return {
                    "error": str(e)
                }

        @self.mcp.tool()
        async def add_document(
            doc_id: str,
            content: str,
            recompute: str = "tfidf"
        ) -> Dict[str, Any]:
            """
            Add a document to the corpus with incremental updates.

            Args:
                doc_id: Unique identifier for the document
                content: Document text content
                recompute: Recomputation level - 'none', 'tfidf', or 'full' (default: 'tfidf')

            Returns:
                Dict with processing statistics (tokens, bigrams, unique_tokens)
            """
            try:
                if not doc_id or not doc_id.strip():
                    return {
                        "error": "doc_id must be a non-empty string",
                        "stats": {}
                    }

                if not isinstance(content, str):
                    return {
                        "error": "content must be a string",
                        "stats": {}
                    }

                valid_recompute = {'none', 'tfidf', 'full'}
                if recompute not in valid_recompute:
                    return {
                        "error": f"recompute must be one of {valid_recompute}",
                        "stats": {}
                    }

                stats = self.processor.add_document_incremental(
                    doc_id=doc_id,
                    content=content,
                    recompute=recompute
                )

                return {
                    "stats": stats,
                    "doc_id": doc_id
                }
            except Exception as e:
                # Catch all exceptions to return graceful error responses.
                # Processor can raise various exception types depending on internal state.
                logger.error(f"Error in add_document: {e}", exc_info=True)
                return {
                    "error": str(e),
                    "stats": {}
                }

    def run(self, transport: str = "stdio"):
        """
        Run the MCP server.

        Args:
            transport: Transport protocol - 'stdio', 'sse', or 'streamable-http' (default: 'stdio')
        """
        logger.info(f"Starting Cortical MCP Server with {transport} transport")
        self.mcp.run(transport=transport)


def create_mcp_server(
    corpus_path: Optional[str] = None,
    config: Optional[CorticalConfig] = None
) -> CorticalMCPServer:
    """
    Create a Cortical MCP Server instance.

    Args:
        corpus_path: Path to saved corpus file
        config: Optional CorticalConfig

    Returns:
        CorticalMCPServer instance
    """
    return CorticalMCPServer(corpus_path=corpus_path, config=config)


def main():
    """
    Main entry point for running the MCP server from command line.

    Usage:
        python -m cortical.projects.mcp.server

    Environment variables:
        CORTICAL_CORPUS_PATH: Path to corpus file to load
        CORTICAL_LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    # Configure logging
    log_level = os.getenv("CORTICAL_LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Get corpus path from environment
    corpus_path = os.getenv("CORTICAL_CORPUS_PATH")

    # Create and run server
    server = create_mcp_server(corpus_path=corpus_path)
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
