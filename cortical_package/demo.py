"""
Cortical Text Processor Demo
============================

This demo processes a corpus of documents, analyzing relationships
between concepts, documents, and ideas across diverse topics.
"""

import os
import sys
from typing import Dict, List, Tuple

from cortical import CorticalTextProcessor, CorticalLayer


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


class CorticalDemo:
    """Demonstrates the cortical text processor with interesting analysis."""
    
    def __init__(self, samples_dir: str = "samples"):
        self.samples_dir = samples_dir
        self.processor = CorticalTextProcessor()
        self.loaded_files = []
    
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
        self.demonstrate_gap_analysis()
        self.demonstrate_embeddings()
        self.print_insights()
    
    def print_intro(self):
        """Print introduction."""
        print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                                                                      ║
    ║              🧠  CORTICAL TEXT PROCESSOR DEMO  🧠                    ║
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
        
        txt_files = sorted([f for f in os.listdir(self.samples_dir) if f.endswith('.txt')])
        
        if not txt_files:
            return False
        
        for filename in txt_files:
            filepath = os.path.join(self.samples_dir, filename)
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            doc_id = filename.replace('.txt', '')
            self.processor.process_document(doc_id, content)
            word_count = len(content.split())
            self.loaded_files.append((doc_id, word_count))
            print(f"  📄 {doc_id:30} ({word_count:3} words)")
        
        # Run all computations
        print("\nComputing cortical representations...")
        self.processor.compute_all(verbose=False)
        
        layer0 = self.processor.get_layer(CorticalLayer.TOKENS)
        layer1 = self.processor.get_layer(CorticalLayer.BIGRAMS)
        
        print(f"\n✓ Processed {len(self.loaded_files)} documents")
        print(f"✓ Created {layer0.column_count()} token minicolumns")
        print(f"✓ Created {layer1.column_count()} bigram minicolumns")
        print(f"✓ Formed {layer0.total_connections()} lateral connections")
        
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
                    # Find neighbor content
                    for c in layer0.minicolumns.values():
                        if c.id == neighbor_id:
                            bar_len = int(min(weight, 10) * 3)
                            bar = "─" * bar_len + ">"
                            print(f"    {bar} {c.content} (weight: {weight:.2f})")
                            break
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
        
        for query in test_queries:
            print_subheader(f"🔍 Query: '{query}'")
            
            # Show expansion
            expanded = self.processor.expand_query(query, max_expansions=6)
            original = set(self.processor.tokenizer.tokenize(query))
            new_terms = [t for t in expanded.keys() if t not in original]
            
            if new_terms:
                print(f"    Expanded with: {', '.join(new_terms[:6])}")
            
            # Find documents
            results = self.processor.find_documents_for_query(query, top_n=3)
            print(f"\n    Top documents:")
            for doc_id, score in results:
                print(f"      • {doc_id} (score: {score:.3f})")
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
        
        print("\n" + "═" * 70)
        print("Demo complete! The cortical text processor successfully:")
        print("  ✓ Built hierarchical representations (Layers 0-3)")
        print("  ✓ Discovered key concepts via PageRank")
        print("  ✓ Computed TF-IDF for discriminative analysis")
        print("  ✓ Found associations through lateral connections")
        print("  ✓ Identified document relationships")
        print("  ✓ Detected knowledge gaps and anomalies")
        print("  ✓ Computed graph embeddings")
        print("  ✓ Enabled semantic queries with expansion")
        print("═" * 70 + "\n")


if __name__ == "__main__":
    demo = CorticalDemo(samples_dir="samples")
    demo.run()
