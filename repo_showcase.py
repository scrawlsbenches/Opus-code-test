"""
Repository Showcase - Cortical Text Processor
==============================================

This showcase analyzes the ENTIRE repository using the pre-built
100% corpus index (corpus_dev.pkl), demonstrating the hierarchical
analysis of code, documentation, and tests.

Unlike showcase.py which processes a small samples directory,
this loads the full repository index for comprehensive analysis.

Usage:
    python repo_showcase.py                    # Use existing corpus_dev.pkl
    python repo_showcase.py --rebuild          # Rebuild index first
    python repo_showcase.py --corpus FILE      # Use custom corpus file
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from cortical import CorticalTextProcessor, CorticalLayer


class Timer:
    """Simple timer for measuring operation durations."""

    def __init__(self):
        self.times: Dict[str, float] = {}
        self._start: float = 0
        self._current: str = ""

    def start(self, name: str):
        """Start timing an operation."""
        self._start = time.perf_counter()
        self._current = name

    def stop(self) -> float:
        """Stop timing and record the duration."""
        elapsed = time.perf_counter() - self._start
        self.times[self._current] = elapsed
        return elapsed

    def get(self, name: str) -> float:
        """Get recorded time for an operation."""
        return self.times.get(name, 0)


def print_header(title: str, char: str = "="):
    """Print a formatted section header."""
    width = 70
    print(f"\n{char * width}")
    print(f"{title:^{width}}")
    print(f"{char * width}\n")


def print_subheader(title: str):
    """Print a formatted subsection header."""
    print(f"\n{title}")
    print("-" * len(title))


def render_bar(value: float, max_value: float, width: int = 30) -> str:
    """Render a text-based progress bar."""
    if max_value == 0:
        return " " * width
    filled = int((value / max_value) * width)
    return "█" * filled + "░" * (width - filled)


class RepoShowcase:
    """Showcases the cortical text processor on the entire repository."""

    def __init__(self, corpus_path: str = "corpus_dev.pkl"):
        self.corpus_path = corpus_path
        self.processor: Optional[CorticalTextProcessor] = None
        self.timer = Timer()
        self.doc_count = 0

    def run(self):
        """Run the complete repository analysis."""
        self.print_intro()

        if not self.load_corpus():
            print("Failed to load corpus!")
            return

        self.analyze_hierarchy()
        self.discover_key_concepts()
        self.analyze_tfidf()
        self.find_code_associations()
        self.analyze_document_relationships()
        self.demonstrate_code_queries()
        self.demonstrate_passage_search()
        self.demonstrate_definition_search()
        self.demonstrate_intent_detection()
        self.demonstrate_gap_analysis()
        self.demonstrate_embeddings()
        self.demonstrate_concept_clusters()
        self.print_insights()

    def print_intro(self):
        """Print introduction."""
        print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                                                                      ║
    ║         🧠  REPOSITORY SHOWCASE - CORTICAL TEXT PROCESSOR  🧠        ║
    ║                                                                      ║
    ║        Full codebase analysis with 100% semantic index               ║
    ║                                                                      ║
    ╚══════════════════════════════════════════════════════════════════════╝
        """)

    def load_corpus(self) -> bool:
        """Load the pre-built corpus."""
        print_header("LOADING REPOSITORY INDEX", "═")

        if not os.path.exists(self.corpus_path):
            print(f"  ❌ Corpus not found: {self.corpus_path}")
            print("\n  Build it with:")
            print("    python scripts/index_codebase.py --full-analysis --batch")
            print("    (run repeatedly until complete)")
            return False

        print(f"Loading pre-built corpus from: {self.corpus_path}")
        print("(This contains the full repository with 100% semantic analysis)\n")

        self.timer.start('load_corpus')
        try:
            self.processor = CorticalTextProcessor.load(self.corpus_path)
        except Exception as e:
            print(f"  ❌ Error loading corpus: {e}")
            return False
        load_time = self.timer.stop()

        # Get statistics
        layer0 = self.processor.get_layer(CorticalLayer.TOKENS)
        layer1 = self.processor.get_layer(CorticalLayer.BIGRAMS)
        layer2 = self.processor.get_layer(CorticalLayer.CONCEPTS)
        layer3 = self.processor.get_layer(CorticalLayer.DOCUMENTS)

        self.doc_count = layer3.column_count()

        # Calculate total connections
        total_conns = sum(
            layer.total_connections()
            for layer in self.processor.layers.values()
        )

        # Count semantic relations
        semantic_count = len(getattr(self.processor, 'semantic_relations', []))

        print(f"✓ Loaded {self.doc_count} documents")
        print(f"✓ {layer0.column_count():,} token minicolumns")
        print(f"✓ {layer1.column_count():,} bigram minicolumns")
        print(f"✓ {layer2.column_count()} concept clusters")
        print(f"✓ {total_conns:,} total connections")
        print(f"✓ {semantic_count:,} semantic relations")
        print(f"\n⏱  Corpus load time: {load_time:.2f}s")

        # Verify 100% index
        if layer2.column_count() == 0:
            print("\n⚠️  Warning: Concept clusters not built (partial index)")
            print("   Run: python scripts/index_codebase.py --full-analysis --batch")

        return True

    def analyze_hierarchy(self):
        """Show the hierarchical structure."""
        print_header("HIERARCHICAL STRUCTURE", "═")

        print("Repository organized in 4 cortical layers:\n")

        layers = [
            (CorticalLayer.TOKENS, "Token Layer (V1)", "Individual terms from code/docs"),
            (CorticalLayer.BIGRAMS, "Bigram Layer (V2)", "Term pairs (e.g., 'page rank')"),
            (CorticalLayer.CONCEPTS, "Concept Layer (V4)", "Semantic clusters"),
            (CorticalLayer.DOCUMENTS, "Document Layer (IT)", "Files in repository"),
        ]

        for layer_enum, name, desc in layers:
            layer = self.processor.get_layer(layer_enum)
            count = layer.column_count()
            conns = layer.total_connections()
            print(f"  Layer {layer_enum.value}: {name}")
            print(f"         {count:,} minicolumns, {conns:,} connections")
            print(f"         Purpose: {desc}")

            # Show clustering quality for concept layer
            if layer_enum == CorticalLayer.CONCEPTS and count > 0:
                quality = self.processor.compute_clustering_quality()
                print(f"         Quality: modularity={quality['modularity']:.2f}, "
                      f"silhouette={quality['silhouette']:.2f}, "
                      f"balance={quality['balance']:.2f}")

            print()

    def discover_key_concepts(self):
        """Show most important concepts via PageRank."""
        print_header("KEY CONCEPTS (PageRank)", "═")

        print("PageRank identifies central concepts in the codebase:")
        print("(Higher rank = more connected, more important)\n")

        layer0 = self.processor.get_layer(CorticalLayer.TOKENS)

        # Get top tokens by pagerank
        top_tokens = sorted(layer0.minicolumns.values(),
                           key=lambda c: c.pagerank, reverse=True)[:20]

        if top_tokens:
            max_pr = top_tokens[0].pagerank
            print("  Rank  Concept            PageRank    Docs")
            print("  " + "─" * 55)

            for i, col in enumerate(top_tokens, 1):
                bar = render_bar(col.pagerank, max_pr, 15)
                doc_count = len(col.document_ids)
                print(f"  {i:>3}.  {col.content:<18} {bar} {col.pagerank:.4f}  ({doc_count})")

    def analyze_tfidf(self):
        """Show TF-IDF analysis."""
        print_header("TF-IDF ANALYSIS", "═")

        print("TF-IDF identifies distinctive terms - rare but meaningful:")
        print("(High TF-IDF = important in few files, unique concepts)\n")

        layer0 = self.processor.get_layer(CorticalLayer.TOKENS)

        # Filter out very common terms and get top by TF-IDF
        top_tfidf = sorted(
            [col for col in layer0.minicolumns.values() if len(col.document_ids) < self.doc_count // 2],
            key=lambda c: c.tfidf, reverse=True
        )[:20]

        if top_tfidf:
            max_tfidf = top_tfidf[0].tfidf
            print("  Rank  Term               TF-IDF   Documents")
            print("  " + "─" * 50)

            for i, col in enumerate(top_tfidf, 1):
                bar = render_bar(col.tfidf, max_tfidf, 12)
                doc_count = len(col.document_ids)
                print(f"  {i:>3}.  {col.content:<18} {bar} {col.tfidf:.4f}  ({doc_count} docs)")

    def find_code_associations(self):
        """Show lateral connections between code concepts."""
        print_header("CODE CONCEPT ASSOCIATIONS", "═")

        print("Lateral connections from co-occurrence in code:\n")

        layer0 = self.processor.get_layer(CorticalLayer.TOKENS)

        # Code-relevant concepts to explore
        test_concepts = ["pagerank", "query", "document", "layer", "test", "config"]

        for concept in test_concepts:
            col = layer0.get_minicolumn(concept)
            if col and col.lateral_connections:
                print_subheader(f"🔗 '{concept}' connects to:")

                # Get top connections
                sorted_conns = sorted(col.lateral_connections.items(),
                                     key=lambda x: x[1], reverse=True)[:8]

                for neighbor_id, weight in sorted_conns:
                    neighbor = layer0.get_by_id(neighbor_id)
                    if neighbor:
                        bar_len = int(min(weight, 10) * 2)
                        bar = "─" * bar_len + ">"
                        print(f"    {bar} {neighbor.content} (weight: {weight:.2f})")
                print()

    def analyze_document_relationships(self):
        """Show document-level relationships."""
        print_header("FILE RELATIONSHIPS", "═")

        print("Files connect based on shared concepts and imports:\n")

        layer3 = self.processor.get_layer(CorticalLayer.DOCUMENTS)

        # Find most connected documents
        sorted_docs = sorted(layer3.minicolumns.values(),
                            key=lambda c: c.connection_count(), reverse=True)[:10]

        print("  Most connected files (bridge multiple concepts):")
        print("  " + "─" * 60)

        for col in sorted_docs:
            conns = col.connection_count()
            # Shorten path for display
            name = col.content
            if len(name) > 45:
                name = "..." + name[-42:]
            print(f"  📄 {name:<45} ({conns} connections)")

        # Show relationships for top file
        if sorted_docs:
            doc = sorted_docs[0]
            print(f"\n  '{doc.content}' relates to:")

            related = self.processor.find_related_documents(doc.content)[:5]
            for related_doc, weight in related:
                short_name = related_doc if len(related_doc) < 40 else "..." + related_doc[-37:]
                print(f"    → {short_name} (similarity: {weight:.3f})")

    def demonstrate_code_queries(self):
        """Demonstrate code-focused queries."""
        print_header("CODE SEARCH DEMONSTRATION", "═")

        print("Query expansion with code-aware synonyms:\n")

        # Code-relevant queries
        test_queries = [
            "compute pagerank importance",
            "expand query terms",
            "save load persistence",
            "tokenize text documents",
            "test coverage gaps",
        ]

        total_query_time = 0

        for query in test_queries:
            print_subheader(f"🔍 Query: '{query}'")

            start = time.perf_counter()

            # Show code-aware expansion
            expanded = self.processor.expand_query_for_code(query, max_expansions=8)
            original = set(self.processor.tokenizer.tokenize(query))
            new_terms = [t for t in expanded.keys() if t not in original]

            if new_terms:
                print(f"    Expanded with: {', '.join(new_terms[:6])}")

            # Find documents
            results = self.processor.find_documents_for_query(query, top_n=5)
            elapsed = time.perf_counter() - start
            total_query_time += elapsed

            print(f"\n    Top files:")
            for doc_id, score in results:
                # Categorize file type
                if 'test' in doc_id.lower():
                    marker = "🧪"
                elif doc_id.endswith('.md'):
                    marker = "📖"
                else:
                    marker = "📄"
                short_id = doc_id if len(doc_id) < 45 else "..." + doc_id[-42:]
                print(f"      {marker} {short_id} (score: {score:.3f})")
            print(f"    ⏱  {elapsed*1000:.1f}ms")
            print()

        self.timer.times['queries'] = total_query_time
        print(f"Average query time: {total_query_time/len(test_queries)*1000:.1f}ms")

    def demonstrate_passage_search(self):
        """Demonstrate passage-level retrieval."""
        print_header("PASSAGE RETRIEVAL (RAG)", "═")

        print("Passage search for feeding context to LLMs:\n")

        query = "how does pagerank compute importance scores"
        print_subheader(f"🔍 Query: '{query}'")

        self.timer.start('passage_search')
        passages = self.processor.find_passages_for_query(
            query,
            top_n=3,
            chunk_size=400,
            overlap=50
        )
        passage_time = self.timer.stop()

        if passages:
            print(f"\n    Found {len(passages)} relevant passages:\n")

            for i, (passage_text, doc_id, start, end, score) in enumerate(passages, 1):
                doc_content = self.processor.documents.get(doc_id, "")
                line_num = doc_content[:start].count('\n') + 1

                short_id = doc_id if len(doc_id) < 40 else "..." + doc_id[-37:]
                print(f"    [{i}] {short_id}:{line_num} (score: {score:.3f})")
                print("    " + "─" * 55)

                lines = passage_text.strip().split('\n')[:5]
                for line in lines:
                    if len(line) > 65:
                        line = line[:62] + "..."
                    print(f"      {line}")
                if len(passage_text.strip().split('\n')) > 5:
                    print(f"      ...")
                print()

        print(f"    ⏱  Passage retrieval: {passage_time*1000:.1f}ms")

    def demonstrate_definition_search(self):
        """Demonstrate definition-finding capability."""
        print_header("DEFINITION SEARCH", "═")

        print("Find class and function definitions in the codebase:\n")

        definition_queries = [
            "class CorticalTextProcessor",
            "def compute_pagerank",
            "class Minicolumn",
            "def expand_query",
            "class HierarchicalLayer",
        ]

        for query in definition_queries:
            is_def, def_type, identifier = self.processor.is_definition_query(query)
            print(f"  Query: \"{query}\"")

            if is_def:
                passages = self.processor.find_definition_passages(query)
                if passages:
                    text, doc_id, start, end, score = passages[0]
                    first_line = text.strip().split('\n')[0][:55]
                    short_id = doc_id if len(doc_id) < 35 else "..." + doc_id[-32:]
                    print(f"    Found: {short_id}")
                    print(f"    Match: {first_line}...")
                else:
                    print(f"    (No definition found)")
            print()

    def demonstrate_intent_detection(self):
        """Show query intent detection for code search."""
        print_header("QUERY INTENT DETECTION", "═")

        print("Detecting query type for smarter search:\n")

        test_queries = [
            ("what is pagerank", True),
            ("compute pagerank damping factor", False),
            ("how does query expansion work", True),
            ("find documents for query", False),
            ("explain the layer hierarchy", True),
            ("test coverage report", False),
        ]

        for query, expected_conceptual in test_queries:
            is_conceptual = self.processor.is_conceptual_query(query)
            intent = "conceptual (📖 docs)" if is_conceptual else "implementation (💻 code)"
            marker = "📖" if is_conceptual else "💻"
            print(f"  {marker} \"{query}\"")
            print(f"      → {intent}")
            print()

    def demonstrate_gap_analysis(self):
        """Show knowledge gap detection in the codebase."""
        print_header("CODEBASE GAP ANALYSIS", "═")

        print("Detecting gaps and anomalies in repository coverage:\n")

        gaps = self.processor.analyze_knowledge_gaps()

        print(f"  Coverage Score: {gaps['coverage_score']:.1%}")
        print(f"  Connectivity Score: {gaps['connectivity_score']:.4f}")

        summary = gaps['summary']
        print(f"\n  Total files: {summary['total_documents']}")
        print(f"  Isolated files: {summary['isolated_count']}")
        print(f"  Well-connected: {summary['well_connected_count']}")
        print(f"  Weak topics: {summary['weak_topic_count']}")

        if gaps['isolated_documents']:
            print("\n  📍 Isolated files (might need better docs/tests):")
            for doc in gaps['isolated_documents'][:5]:
                short_id = doc['doc_id']
                if len(short_id) > 45:
                    short_id = "..." + short_id[-42:]
                print(f"    • {short_id} (avg sim: {doc['avg_similarity']:.3f})")

        if gaps['weak_topics']:
            print("\n  📍 Weak topics (thin coverage):")
            for topic in gaps['weak_topics'][:8]:
                print(f"    • '{topic['term']}' - only {topic['doc_count']} file(s)")

    def demonstrate_embeddings(self):
        """Show embedding-based similarity."""
        print_header("GRAPH EMBEDDINGS", "═")

        print("Finding similar terms via graph structure:\n")

        # Check if embeddings exist
        if not hasattr(self.processor, 'embeddings') or not self.processor.embeddings:
            print("  Computing embeddings...")
            stats = self.processor.compute_graph_embeddings(
                dimensions=32, method='random_walk', verbose=False
            )
            print(f"  Created {stats['terms_embedded']} term embeddings\n")

        # Find similar terms for code concepts
        test_terms = ["pagerank", "query", "document", "layer", "test"]

        for term in test_terms:
            similar = self.processor.find_similar_by_embedding(term, top_n=6)
            if similar:
                print(f"  Terms similar to '{term}':")
                for other, sim in similar:
                    print(f"    • {other} (similarity: {sim:.3f})")
                print()

    def demonstrate_concept_clusters(self):
        """Show the concept clusters (Layer 2)."""
        print_header("CONCEPT CLUSTERS", "═")

        layer2 = self.processor.get_layer(CorticalLayer.CONCEPTS)

        if layer2.column_count() == 0:
            print("  No concept clusters built (run full analysis)")
            return

        print(f"Found {layer2.column_count()} semantic clusters:\n")

        # Get clusters sorted by size
        clusters = sorted(
            layer2.minicolumns.values(),
            key=lambda c: len(c.feedforward_connections),
            reverse=True
        )[:10]

        for i, cluster in enumerate(clusters, 1):
            member_count = len(cluster.feedforward_connections)
            print(f"  Cluster {i}: {member_count} terms")

            # Get some member terms
            layer0 = self.processor.get_layer(CorticalLayer.TOKENS)
            members = []
            for term_id in list(cluster.feedforward_connections.keys())[:8]:
                term_col = layer0.get_by_id(term_id)
                if term_col:
                    members.append(term_col.content)

            if members:
                print(f"    Members: {', '.join(members)}")
            print()

    def print_insights(self):
        """Print final insights and summary."""
        print_header("REPOSITORY ANALYSIS SUMMARY", "═")

        layer0 = self.processor.get_layer(CorticalLayer.TOKENS)
        layer1 = self.processor.get_layer(CorticalLayer.BIGRAMS)
        layer2 = self.processor.get_layer(CorticalLayer.CONCEPTS)
        layer3 = self.processor.get_layer(CorticalLayer.DOCUMENTS)

        total_conns = sum(
            layer.total_connections()
            for layer in self.processor.layers.values()
        )

        semantic_count = len(getattr(self.processor, 'semantic_relations', []))

        print("📊 REPOSITORY STATISTICS\n")
        print(f"  Files indexed:           {layer3.column_count()}")
        print(f"  Unique tokens:           {layer0.column_count():,}")
        print(f"  Unique bigrams:          {layer1.column_count():,}")
        print(f"  Concept clusters:        {layer2.column_count()}")
        print(f"  Total connections:       {total_conns:,}")
        print(f"  Semantic relations:      {semantic_count:,}")

        # Find most central token
        top_token = max(layer0.minicolumns.values(), key=lambda c: c.pagerank)
        print(f"\n  Most central concept: '{top_token.content}'")

        # Find most connected document
        if layer3.column_count() > 0:
            top_doc = max(layer3.minicolumns.values(), key=lambda c: c.connection_count())
            print(f"  Most connected file: '{top_doc.content}'")

        # Performance summary
        print("\n⏱  PERFORMANCE\n")
        if 'load_corpus' in self.timer.times:
            print(f"  Corpus load:         {self.timer.get('load_corpus'):.2f}s")
        if 'queries' in self.timer.times:
            avg_query = self.timer.get('queries') / 5 * 1000  # 5 queries
            print(f"  Avg query time:      {avg_query:.1f}ms")
        if 'passage_search' in self.timer.times:
            print(f"  Passage retrieval:   {self.timer.get('passage_search')*1000:.1f}ms")

        # Index completeness
        print("\n📈 INDEX COMPLETENESS\n")
        completeness = []
        completeness.append("✓ Tokens (Layer 0)" if layer0.column_count() > 0 else "✗ Tokens")
        completeness.append("✓ Bigrams (Layer 1)" if layer1.column_count() > 0 else "✗ Bigrams")
        completeness.append("✓ Concepts (Layer 2)" if layer2.column_count() > 0 else "✗ Concepts (run --full-analysis)")
        completeness.append("✓ Documents (Layer 3)" if layer3.column_count() > 0 else "✗ Documents")
        completeness.append("✓ Semantic relations" if semantic_count > 0 else "✗ Semantic relations")

        for item in completeness:
            print(f"  {item}")

        is_100_pct = layer2.column_count() > 0 and semantic_count > 0
        pct = "100%" if is_100_pct else "~80% (fast mode)"
        print(f"\n  Overall: {pct} indexed")

        print("\n" + "═" * 70)
        print("Repository analysis complete!")
        print("  ✓ Loaded full repository index")
        print("  ✓ Analyzed hierarchical structure (4 layers)")
        print("  ✓ Identified key concepts via PageRank")
        print("  ✓ Found distinctive terms via TF-IDF")
        print("  ✓ Mapped code associations")
        print("  ✓ Discovered file relationships")
        print("  ✓ Demonstrated code-aware search")
        print("  ✓ Retrieved passages for RAG")
        print("  ✓ Detected definitions")
        print("  ✓ Analyzed query intent")
        print("  ✓ Found knowledge gaps")
        print("  ✓ Computed graph embeddings")
        if layer2.column_count() > 0:
            print("  ✓ Explored concept clusters")
        print("═" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Repository Showcase - Analyze entire codebase with Cortical Text Processor"
    )
    parser.add_argument(
        '--corpus', '-c',
        default='corpus_dev.pkl',
        help='Path to corpus file (default: corpus_dev.pkl)'
    )
    parser.add_argument(
        '--rebuild', '-r',
        action='store_true',
        help='Rebuild the corpus index before running'
    )

    args = parser.parse_args()

    if args.rebuild:
        print("Rebuilding corpus index...")
        import subprocess
        result = subprocess.run(
            [sys.executable, 'scripts/index_codebase.py', '--full-analysis', '--foreground'],
            cwd=Path(__file__).parent
        )
        if result.returncode != 0:
            print("Index rebuild failed!")
            return 1

    showcase = RepoShowcase(corpus_path=args.corpus)
    showcase.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
