"""
Cortical Text Processor Showcase
================================

This showcase processes a corpus of documents, demonstrating the
hierarchical analysis of relationships between concepts, documents,
and ideas across diverse topics.
"""

import os
import sys
import time
from typing import Dict, List, Tuple

from cortical import CorticalTextProcessor, CorticalLayer


class Timer:
    """Simple timer for measuring operation durations."""

    def __init__(self):
        self.times: Dict[str, float] = {}
        self._start: float = 0

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


class CorticalShowcase:
    """Showcases the cortical text processor with interesting analysis."""

    def __init__(self, samples_dir: str = "samples"):
        self.samples_dir = samples_dir
        self.processor = CorticalTextProcessor()
        self.loaded_files = []
        self.timer = Timer()

    def run(self):
        """Run the complete demo."""
        self.print_intro()

        if not self.ingest_corpus():
            print("No documents found!")
            return

        self.analyze_hierarchy()
        self.discover_key_concepts()
        self.analyze_tfidf()
        self.find_concept_associations()
        self.analyze_document_relationships()
        self.demonstrate_queries()
        self.demonstrate_passage_search()
        self.demonstrate_polysemy()
        self.demonstrate_code_features()
        self.demonstrate_gap_analysis()
        self.demonstrate_embeddings()
        self.print_insights()
    
    def print_intro(self):
        """Print introduction."""
        print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                                                                      ║
    ║            🧠  CORTICAL TEXT PROCESSOR SHOWCASE  🧠                  ║
    ║                                                                      ║
    ║     Mimicking how the neocortex processes and understands text       ║
    ║                                                                      ║
    ╚══════════════════════════════════════════════════════════════════════╝
        """)
    
    def ingest_corpus(self) -> bool:
        """Ingest the document corpus from disk."""
        print_header("DOCUMENT INGESTION", "═")

        print(f"Loading documents from: {self.samples_dir}")
        print("Processing through cortical hierarchy...")
        print("(Like visual information flowing V1 → V2 → V4 → IT)\n")

        if not os.path.exists(self.samples_dir):
            print(f"  ❌ Directory not found: {self.samples_dir}")
            return False

        # Load both .txt and .py files
        txt_files = sorted([f for f in os.listdir(self.samples_dir) if f.endswith('.txt')])
        py_files = sorted([f for f in os.listdir(self.samples_dir) if f.endswith('.py')])
        all_files = txt_files + py_files

        if not all_files:
            return False

        # Time document loading
        self.timer.start('document_loading')
        for filename in all_files:
            filepath = os.path.join(self.samples_dir, filename)
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Handle both .txt and .py extensions
            doc_id = filename.replace('.txt', '').replace('.py', '')
            self.processor.process_document(doc_id, content)
            word_count = len(content.split())
            self.loaded_files.append((doc_id, word_count))
            print(f"  📄 {doc_id:30} ({word_count:3} words)")
        load_time = self.timer.stop()

        # Run all computations with hybrid strategy for better Layer 2 connectivity
        print("\nComputing cortical representations...")
        self.timer.start('compute_all')
        self.processor.compute_all(
            verbose=False,
            connection_strategy='hybrid',
            cluster_strictness=0.5,
            bridge_weight=0.3
        )
        compute_time = self.timer.stop()

        layer0 = self.processor.get_layer(CorticalLayer.TOKENS)
        layer1 = self.processor.get_layer(CorticalLayer.BIGRAMS)

        print(f"\n✓ Processed {len(self.loaded_files)} documents")
        print(f"✓ Created {layer0.column_count()} token minicolumns")
        print(f"✓ Created {layer1.column_count()} bigram minicolumns")
        print(f"✓ Formed {layer0.total_connections()} lateral connections")
        print(f"\n⏱  Document loading: {load_time:.2f}s")
        print(f"⏱  Compute all:      {compute_time:.2f}s")

        return True
    
    def analyze_hierarchy(self):
        """Show the hierarchical structure."""
        print_header("HIERARCHICAL STRUCTURE", "═")
        
        print("The cortical model has 4 layers (like visual cortex V1→IT):\n")
        
        layers = [
            (CorticalLayer.TOKENS, "Token Layer (V1)", "Individual words"),
            (CorticalLayer.BIGRAMS, "Bigram Layer (V2)", "Word pairs"),
            (CorticalLayer.CONCEPTS, "Concept Layer (V4)", "Semantic clusters"),
            (CorticalLayer.DOCUMENTS, "Document Layer (IT)", "Full documents"),
        ]
        
        for layer_enum, name, desc in layers:
            layer = self.processor.get_layer(layer_enum)
            count = layer.column_count()
            conns = layer.total_connections()
            print(f"  Layer {layer_enum.value}: {name}")
            print(f"         {count:,} minicolumns, {conns:,} connections")
            print(f"         Purpose: {desc}\n")
    
    def discover_key_concepts(self):
        """Show most important concepts via PageRank."""
        print_header("KEY CONCEPTS (PageRank)", "═")
        
        print("PageRank identifies central concepts - highly connected 'hub' words:")
        print("(Like identifying influential neurons in a network)\n")
        
        layer0 = self.processor.get_layer(CorticalLayer.TOKENS)
        
        # Get top tokens by pagerank
        top_tokens = sorted(layer0.minicolumns.values(), 
                           key=lambda c: c.pagerank, reverse=True)[:15]
        
        if top_tokens:
            max_pr = top_tokens[0].pagerank
            print("  Rank  Concept            PageRank")
            print("  " + "─" * 45)
            
            for i, col in enumerate(top_tokens, 1):
                bar = render_bar(col.pagerank, max_pr, 20)
                print(f"  {i:>3}.  {col.content:<18} {bar} {col.pagerank:.4f}")
    
    def analyze_tfidf(self):
        """Show TF-IDF analysis."""
        print_header("TF-IDF ANALYSIS", "═")
        
        print("TF-IDF identifies distinctive terms - rare but meaningful:")
        print("(High TF-IDF = important in some docs but rare across corpus)\n")
        
        layer0 = self.processor.get_layer(CorticalLayer.TOKENS)
        
        top_tfidf = sorted(layer0.minicolumns.values(),
                          key=lambda c: c.tfidf, reverse=True)[:15]
        
        if top_tfidf:
            max_tfidf = top_tfidf[0].tfidf
            print("  Rank  Term               TF-IDF   Documents")
            print("  " + "─" * 50)
            
            for i, col in enumerate(top_tfidf, 1):
                bar = render_bar(col.tfidf, max_tfidf, 15)
                doc_count = len(col.document_ids)
                print(f"  {i:>3}.  {col.content:<18} {bar} {col.tfidf:.4f}  ({doc_count} docs)")
    
    def find_concept_associations(self):
        """Show lateral connections between concepts."""
        print_header("CONCEPT ASSOCIATIONS", "═")
        
        print("Lateral connections form from co-occurrence (like Hebbian learning):")
        print("'Neurons that fire together, wire together'\n")
        
        layer0 = self.processor.get_layer(CorticalLayer.TOKENS)
        
        # Find interesting concepts and their connections
        test_concepts = ["neural", "learning", "bread", "systems"]
        
        for concept in test_concepts:
            col = layer0.get_minicolumn(concept)
            if col and col.lateral_connections:
                print_subheader(f"🔗 '{concept}' connects to:")
                
                # Get top connections
                sorted_conns = sorted(col.lateral_connections.items(), 
                                     key=lambda x: x[1], reverse=True)[:6]
                
                for neighbor_id, weight in sorted_conns:
                    # O(1) lookup using _id_index
                    neighbor = layer0.get_by_id(neighbor_id)
                    if neighbor:
                        bar_len = int(min(weight, 10) * 3)
                        bar = "─" * bar_len + ">"
                        print(f"    {bar} {neighbor.content} (weight: {weight:.2f})")
                print()
    
    def analyze_document_relationships(self):
        """Show document-level relationships."""
        print_header("DOCUMENT RELATIONSHIPS", "═")
        
        print("Documents connect based on shared concepts and term overlap:\n")
        
        layer3 = self.processor.get_layer(CorticalLayer.DOCUMENTS)
        
        # Find most connected documents
        sorted_docs = sorted(layer3.minicolumns.values(),
                            key=lambda c: c.connection_count(), reverse=True)[:5]
        
        print("  Most connected documents (bridge topics):")
        print("  " + "─" * 50)
        
        for col in sorted_docs:
            conns = col.connection_count()
            print(f"  📄 {col.content:<30} ({conns} connections)")
        
        # Show a document's relationships
        if sorted_docs:
            doc = sorted_docs[0]
            print(f"\n  '{doc.content}' connects to:")
            
            related = self.processor.find_related_documents(doc.content)[:5]
            for related_doc, weight in related:
                print(f"    → {related_doc} (similarity: {weight:.3f})")
    
    def demonstrate_queries(self):
        """Demonstrate query capability with expansion."""
        print_header("QUERY DEMONSTRATION", "═")

        print("Query expansion adds semantically related terms for better recall:\n")

        test_queries = ["neural networks", "fermentation", "distributed systems"]
        total_query_time = 0

        for query in test_queries:
            print_subheader(f"🔍 Query: '{query}'")

            # Time expansion + search
            start = time.perf_counter()

            # Show expansion
            expanded = self.processor.expand_query(query, max_expansions=6)
            original = set(self.processor.tokenizer.tokenize(query))
            new_terms = [t for t in expanded.keys() if t not in original]

            if new_terms:
                print(f"    Expanded with: {', '.join(new_terms[:6])}")

            # Find documents
            results = self.processor.find_documents_for_query(query, top_n=3)
            elapsed = time.perf_counter() - start
            total_query_time += elapsed

            print(f"\n    Top documents:")
            for doc_id, score in results:
                print(f"      • {doc_id} (score: {score:.3f})")
            print(f"    ⏱  {elapsed*1000:.1f}ms")
            print()

        self.timer.times['queries'] = total_query_time
        print(f"Average query time: {total_query_time/len(test_queries)*1000:.1f}ms")

    def demonstrate_passage_search(self):
        """Demonstrate passage-level retrieval for RAG applications."""
        print_header("PASSAGE RETRIEVAL (RAG)", "═")

        print("Passage search retrieves specific text chunks, ideal for RAG:")
        print("(Retrieval-Augmented Generation - feeding context to LLMs)\n")

        # Demonstrate with a specific query
        query = "PageRank algorithm convergence"
        print_subheader(f"🔍 Query: '{query}'")

        # Time passage retrieval
        self.timer.start('passage_search')

        # Get passages
        passages = self.processor.find_passages_for_query(
            query,
            top_n=3,
            chunk_size=300,
            overlap=50
        )
        passage_time = self.timer.stop()

        if passages:
            print(f"\n    Found {len(passages)} relevant passages:\n")

            for i, (passage_text, doc_id, start, end, score) in enumerate(passages, 1):
                # Calculate line number from character position
                doc_content = self.processor.documents.get(doc_id, "")
                line_num = doc_content[:start].count('\n') + 1

                print(f"    [{i}] {doc_id}:{line_num} (score: {score:.3f})")
                print("    " + "─" * 50)

                # Show truncated passage
                lines = passage_text.strip().split('\n')[:4]
                for line in lines:
                    if len(line) > 60:
                        line = line[:57] + "..."
                    print(f"      {line}")
                if len(passage_text.strip().split('\n')) > 4:
                    print(f"      ...")
                print()

        print(f"    ⏱  Passage retrieval: {passage_time*1000:.1f}ms")
        print("\n    💡 Use case: Feed these passages to an LLM as context")
        print("                 for answering questions about your corpus.")
        print()

    def demonstrate_polysemy(self):
        """Demonstrate polysemy - same word, different meanings."""
        print_header("POLYSEMY DEMONSTRATION", "═")

        print("Polysemy occurs when the same word has multiple meanings.")
        print("This affects retrieval when query terms are ambiguous.\n")

        # Query for "candle sticks"
        query = "candle sticks"
        print_subheader(f"🔍 Query: '{query}'")

        results = self.processor.find_documents_for_query(query, top_n=6)
        print("\n    Results:")
        for doc_id, score in results:
            print(f"      • {doc_id} (score: {score:.3f})")

        # Explain the polysemy
        print("\n    📝 Analysis:")
        print("    The query tokenizes to: ['candle', 'sticks']")
        print()
        print("    'sticks' appears in multiple contexts:")
        print("      • candlestick_patterns - trading chart patterns")
        print("      • letterpress_printing - 'composing sticks' (typesetting tools)")
        print()
        print("    This is a classic word sense disambiguation challenge.")
        print("    The system correctly finds both but cannot distinguish intent.")

        # Show the actual text snippets
        layer0 = self.processor.get_layer(CorticalLayer.TOKENS)
        if 'sticks' in layer0.minicolumns:
            col = layer0.minicolumns['sticks']
            print(f"\n    'sticks' appears in {len(col.document_ids)} documents:")
            for doc_id in col.document_ids:
                print(f"      • {doc_id}")

        print("\n    💡 Potential improvements:")
        print("      • Weight adjacent term matches higher (bigram boost)")
        print("      • Use document context for disambiguation")
        print("      • Implement word sense disambiguation")
        print()

    def demonstrate_code_features(self):
        """Demonstrate code-specific search capabilities."""
        print_header("CODE SEARCH FEATURES", "═")

        print("Features optimized for searching code and technical content:\n")

        # 1. Query intent detection
        print_subheader("🎯 Query Intent Detection")
        print("    The system detects whether queries are conceptual or implementation-focused:\n")

        test_queries = [
            ("what is PageRank", True),
            ("compute pagerank damping", False),
            ("how does TF-IDF work", True),
            ("find documents tfidf", False),
        ]

        for query, expected_conceptual in test_queries:
            is_conceptual = self.processor.is_conceptual_query(query)
            intent = "conceptual" if is_conceptual else "implementation"
            marker = "📖" if is_conceptual else "💻"
            print(f"    {marker} \"{query}\" → {intent}")

        print("\n    💡 Use case: Boost documentation for conceptual queries,")
        print("                 boost code files for implementation queries.")

        # 2. Definition search (NEW - Task #84)
        print_subheader("\n🔎 Definition Search")
        print("    Find class/function definitions directly in code:\n")

        definition_queries = ["class DataProcessor", "def calculate_statistics", "class SearchIndex"]

        for query in definition_queries:
            is_def, def_type, identifier = self.processor.is_definition_query(query)
            print(f"    Query: \"{query}\"")
            print(f"      Is definition query: {is_def} ({def_type} '{identifier}')" if is_def else f"      Is definition query: {is_def}")

            if is_def:
                passages = self.processor.find_definition_passages(query)
                if passages:
                    text, doc_id, start, end, score = passages[0]
                    # Show first line of the definition
                    first_line = text.strip().split('\n')[0][:60]
                    print(f"      Found in: {doc_id}")
                    print(f"      Match: {first_line}...")
                else:
                    print(f"      (No definition found in corpus)")
            print()

        # 3. Doc-type boosting (NEW - Task #66)
        print_subheader("📊 Doc-Type Boosting")
        print("    Apply different weights to docs, code, and test files:\n")

        query = "filter data records"
        print(f"    Query: \"{query}\"\n")

        # Without boosting
        results_normal = self.processor.find_passages_for_query(
            query, top_n=3, apply_doc_boost=False
        )

        # With boosting (prefer docs over tests)
        results_boosted = self.processor.find_passages_for_query(
            query, top_n=3, apply_doc_boost=True, prefer_docs=True
        )

        print("    Without doc-type boost:")
        for text, doc_id, start, end, score in results_normal[:3]:
            is_test = 'test' in doc_id.lower()
            marker = "🧪" if is_test else "📄"
            print(f"      {marker} {doc_id}: {score:.3f}")

        print("\n    With doc-type boost (prefer docs, penalize tests):")
        for text, doc_id, start, end, score in results_boosted[:3]:
            is_test = 'test' in doc_id.lower()
            marker = "🧪" if is_test else "📄"
            print(f"      {marker} {doc_id}: {score:.3f}")

        print("\n    💡 Test files receive a 0.5x penalty to surface source files first.")

        # 4. Code-aware chunking (NEW - Task #86)
        print_subheader("\n✂️  Code-Aware Chunking")
        print("    Split code at semantic boundaries (class/function defs):\n")

        # Find a Python file
        code_doc_id = None
        for doc_id, _ in self.loaded_files:
            if doc_id in ['data_processor', 'search_engine']:
                code_doc_id = doc_id
                break

        if code_doc_id:
            content = self.processor.documents.get(code_doc_id, "")

            # Regular chunking
            from cortical.query import create_chunks, create_code_aware_chunks
            regular_chunks = create_chunks(content, chunk_size=300, overlap=50)

            # Code-aware chunking
            code_chunks = create_code_aware_chunks(content, max_size=300)

            print(f"    File: {code_doc_id}")
            print(f"    Regular chunks: {len(regular_chunks)} (fixed 300-char boundaries)")
            print(f"    Code-aware chunks: {len(code_chunks)} (semantic boundaries)\n")

            print("    Code-aware chunk boundaries:")
            for i, (chunk_text, start, end) in enumerate(code_chunks[:4]):
                first_line = chunk_text.strip().split('\n')[0][:50]
                print(f"      [{i+1}] {first_line}...")
        else:
            print("    (No Python files in corpus)")

        # 5. Code-aware query expansion
        print_subheader("\n🔧 Code-Aware Query Expansion")
        print("    Programming synonyms expand queries for better code search:\n")

        code_queries = ["fetch data", "get results", "process input"]

        for query in code_queries:
            # Regular expansion
            regular = self.processor.expand_query(query, max_expansions=5)
            # Code-aware expansion
            code_exp = self.processor.expand_query_for_code(query, max_expansions=8)

            # Find terms only in code expansion
            regular_terms = set(regular.keys())
            code_terms = set(code_exp.keys())
            new_terms = code_terms - regular_terms

            print(f"    Query: \"{query}\"")
            if new_terms:
                new_list = sorted(new_terms, key=lambda t: -code_exp.get(t, 0))[:4]
                print(f"      + Code terms: {', '.join(new_list)}")
            else:
                print(f"      (corpus lacks programming synonyms for this query)")
            print()

        # 6. Semantic fingerprinting
        print_subheader("🔍 Semantic Fingerprinting")
        print("    Compare text similarity using semantic fingerprints:\n")

        # Get two related documents
        if len(self.loaded_files) >= 2:
            doc1_id = "neural_pagerank" if "neural_pagerank" in self.processor.documents else self.loaded_files[0][0]
            doc2_id = "pagerank_fundamentals" if "pagerank_fundamentals" in self.processor.documents else self.loaded_files[1][0]

            doc1_content = self.processor.documents.get(doc1_id, "")[:500]
            doc2_content = self.processor.documents.get(doc2_id, "")[:500]

            if doc1_content and doc2_content:
                fp1 = self.processor.get_fingerprint(doc1_content, top_n=10)
                fp2 = self.processor.get_fingerprint(doc2_content, top_n=10)

                comparison = self.processor.compare_fingerprints(fp1, fp2)

                print(f"    Comparing: '{doc1_id}' vs '{doc2_id}'")
                print(f"      Similarity: {comparison['overall_similarity']:.1%}")
                print(f"      Shared concepts: {len(comparison.get('shared_concepts', []))}")

                if comparison['shared_terms']:
                    shared = list(comparison['shared_terms'])[:5]
                    print(f"      Common terms: {', '.join(shared)}")

        print("\n    💡 Use case: Find similar code, detect duplicates,")
        print("                 suggest related files when editing.")
        print()

    def demonstrate_gap_analysis(self):
        """Show knowledge gap detection."""
        print_header("KNOWLEDGE GAP ANALYSIS", "═")
        
        print("Detecting gaps and anomalies in the corpus:\n")
        
        gaps = self.processor.analyze_knowledge_gaps()
        
        print(f"  Coverage Score: {gaps['coverage_score']:.1%}")
        print(f"  Connectivity Score: {gaps['connectivity_score']:.4f}")
        
        summary = gaps['summary']
        print(f"\n  Total documents: {summary['total_documents']}")
        print(f"  Isolated documents: {summary['isolated_count']}")
        print(f"  Well-connected: {summary['well_connected_count']}")
        print(f"  Weak topics found: {summary['weak_topic_count']}")
        
        if gaps['isolated_documents']:
            print("\n  📍 Isolated documents (don't fit well):")
            for doc in gaps['isolated_documents'][:3]:
                print(f"    • {doc['doc_id']} (avg sim: {doc['avg_similarity']:.3f})")
        
        if gaps['weak_topics']:
            print("\n  📍 Weak topics (thin coverage):")
            for topic in gaps['weak_topics'][:5]:
                print(f"    • '{topic['term']}' - only {topic['doc_count']} doc(s)")
    
    def demonstrate_embeddings(self):
        """Show embedding-based similarity."""
        print_header("GRAPH EMBEDDINGS", "═")
        
        print("Computing embeddings from graph structure...\n")
        
        stats = self.processor.compute_graph_embeddings(
            dimensions=32, method='adjacency', verbose=False
        )
        print(f"  Created {stats['terms_embedded']} term embeddings")
        
        # Find similar terms
        test_terms = ["neural", "learning", "data"]
        
        for term in test_terms:
            similar = self.processor.find_similar_by_embedding(term, top_n=5)
            if similar:
                print(f"\n  Terms similar to '{term}':")
                for other, sim in similar:
                    print(f"    • {other} (similarity: {sim:.3f})")
    
    def print_insights(self):
        """Print final insights and summary."""
        print_header("INSIGHTS & SUMMARY", "═")
        
        layer0 = self.processor.get_layer(CorticalLayer.TOKENS)
        layer1 = self.processor.get_layer(CorticalLayer.BIGRAMS)
        layer3 = self.processor.get_layer(CorticalLayer.DOCUMENTS)
        
        print("📊 CORPUS ANALYSIS SUMMARY\n")
        
        print(f"  Documents processed:     {len(self.loaded_files)}")
        print(f"  Unique tokens:           {layer0.column_count()}")
        print(f"  Unique bigrams:          {layer1.column_count()}")
        print(f"  Total connections:       {layer0.total_connections():,}")
        
        # Find most central token
        top_token = max(layer0.minicolumns.values(), key=lambda c: c.pagerank)
        print(f"\n  Most central concept: '{top_token.content}'")
        
        # Find most connected document
        if layer3.column_count() > 0:
            top_doc = max(layer3.minicolumns.values(), key=lambda c: c.connection_count())
            print(f"  Most connected document: '{top_doc.content}'")

        # Performance summary
        print("\n⏱  PERFORMANCE SUMMARY\n")
        if 'document_loading' in self.timer.times:
            print(f"  Document loading:    {self.timer.get('document_loading'):.2f}s")
        if 'compute_all' in self.timer.times:
            print(f"  Compute all:         {self.timer.get('compute_all'):.2f}s")
        if 'queries' in self.timer.times:
            avg_query = self.timer.get('queries') / 3 * 1000  # 3 queries
            print(f"  Avg query time:      {avg_query:.1f}ms")
        if 'passage_search' in self.timer.times:
            print(f"  Passage retrieval:   {self.timer.get('passage_search')*1000:.1f}ms")

        print("\n" + "═" * 70)
        print("Demo complete! The cortical text processor successfully:")
        print("  ✓ Built hierarchical representations (Layers 0-3)")
        print("  ✓ Discovered key concepts via PageRank")
        print("  ✓ Computed TF-IDF for discriminative analysis")
        print("  ✓ Found associations through lateral connections")
        print("  ✓ Identified document relationships")
        print("  ✓ Retrieved passages for RAG applications")
        print("  ✓ Demonstrated code search features")
        print("  ✓ Detected knowledge gaps and anomalies")
        print("  ✓ Computed graph embeddings")
        print("  ✓ Enabled semantic queries with expansion")
        print("═" * 70 + "\n")


if __name__ == "__main__":
    showcase = CorticalShowcase(samples_dir="samples")
    showcase.run()
