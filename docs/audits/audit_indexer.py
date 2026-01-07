"""
Audit Framework Indexer
=======================

Indexes the audit framework documents for semantic search,
enabling discovery of patterns, learnings, and related experiments.

Usage:
    python audit_indexer.py                      # Compute fresh
    python audit_indexer.py --build-index        # Build and save index only
    python audit_indexer.py --use-index          # Load from pre-built index
    python audit_indexer.py --query "guardrails" # Search the index
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cortical import CorticalTextProcessor, CorticalLayer
from cortical.tokenizer import Tokenizer

# Default locations
DEFAULT_INDEX_PATH = "docs/audits/audit_index.pkl"
AUDIT_DIR = "docs/audits"


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


class AuditIndexer:
    """Indexes audit framework documents for semantic search."""

    def __init__(
        self,
        audit_dir: str = AUDIT_DIR,
        use_index: bool = False,
        index_path: Optional[str] = None
    ):
        self.audit_dir = audit_dir
        self.use_index = use_index
        self.index_path = index_path or DEFAULT_INDEX_PATH
        self.processor = None
        self.loaded_files = []

    def run(self):
        """Run the indexer with demo output."""
        self.print_intro()

        if not self.ingest_documents():
            print("No documents found!")
            return

        self.show_structure()
        self.discover_key_concepts()
        self.find_associations()
        self.show_document_relationships()
        self.print_summary()

    def print_intro(self):
        """Print introduction."""
        print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                                                                      ║
    ║              🔬  AUDIT FRAMEWORK INDEXER  🔬                         ║
    ║                                                                      ║
    ║        Semantic search for experiments, learnings, and patterns      ║
    ║                                                                      ║
    ╚══════════════════════════════════════════════════════════════════════╝
        """)

    def ingest_documents(self) -> bool:
        """Ingest audit documents from disk or load from index."""
        print_header("DOCUMENT INGESTION", "═")

        # Try to load from index if requested
        if self.use_index and os.path.exists(self.index_path):
            return self._load_from_index()

        # Otherwise, compute fresh
        return self._compute_fresh()

    def _load_from_index(self) -> bool:
        """Load processor state from pre-built index."""
        print(f"Loading pre-built index from: {self.index_path}")

        try:
            self.processor = CorticalTextProcessor.load(self.index_path, verbose=False)
        except Exception as e:
            print(f"  Failed to load index: {e}")
            print("  Falling back to fresh computation...\n")
            return self._compute_fresh()

        # Populate loaded_files from processor state
        layer3 = self.processor.get_layer(CorticalLayer.DOCUMENTS)
        for col in layer3.minicolumns.values():
            doc_content = self.processor.documents.get(col.content, "")
            word_count = len(doc_content.split())
            self.loaded_files.append((col.content, word_count))

        layer0 = self.processor.get_layer(CorticalLayer.TOKENS)
        layer1 = self.processor.get_layer(CorticalLayer.BIGRAMS)

        total_conns = sum(
            layer.total_connections()
            for layer in self.processor.layers.values()
        )

        print(f"✓ Loaded {len(self.loaded_files)} documents from index")
        print(f"✓ {layer0.column_count()} token minicolumns ready")
        print(f"✓ {layer1.column_count()} bigram minicolumns ready")
        print(f"✓ {total_conns:,} connections pre-computed")

        return True

    def _compute_fresh(self) -> bool:
        """Compute processor state from scratch."""
        print(f"Loading documents from: {self.audit_dir}")

        if not os.path.exists(self.audit_dir):
            print(f"  Directory not found: {self.audit_dir}")
            return False

        # Find all markdown files recursively
        md_files = list(Path(self.audit_dir).rglob("*.md"))

        if not md_files:
            return False

        # Initialize processor
        tokenizer = Tokenizer(filter_code_noise=False)
        self.processor = CorticalTextProcessor(tokenizer=tokenizer)

        # Process each file
        for filepath in sorted(md_files):
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                # Create doc_id from relative path
                rel_path = filepath.relative_to(self.audit_dir)
                doc_id = str(rel_path).replace('/', ':').replace('.md', '')

                self.processor.process_document(doc_id, content)
                word_count = len(content.split())
                self.loaded_files.append((doc_id, word_count))

                # Categorize by type
                if 'exp-' in doc_id:
                    icon = "🧪"
                elif 'result-' in doc_id:
                    icon = "📊"
                elif 'task-' in doc_id:
                    icon = "📋"
                elif 'learnings' in doc_id:
                    icon = "💡"
                else:
                    icon = "📄"

                print(f"  {icon} {doc_id[:40]:<40} ({word_count:4} words)")

            except Exception as e:
                print(f"  ⚠ Failed to load {filepath}: {e}")

        # Run computations
        print("\nComputing semantic representations...")
        self.processor.compute_all(
            verbose=False,
            connection_strategy='hybrid',
            cluster_strictness=0.5,
            bridge_weight=0.3
        )

        layer0 = self.processor.get_layer(CorticalLayer.TOKENS)
        layer1 = self.processor.get_layer(CorticalLayer.BIGRAMS)

        total_conns = sum(
            layer.total_connections()
            for layer in self.processor.layers.values()
        )

        print(f"\n✓ Processed {len(self.loaded_files)} documents")
        print(f"✓ Created {layer0.column_count()} token minicolumns")
        print(f"✓ Created {layer1.column_count()} bigram minicolumns")
        print(f"✓ Formed {total_conns:,} total connections")

        return True

    def build_index(self) -> bool:
        """Build and save index without running demos."""
        print_header("BUILDING INDEX", "═")
        print(f"Target: {self.index_path}\n")

        if not self._compute_fresh():
            return False

        # Ensure directory exists
        Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving index to {self.index_path}...")
        self.processor.save(self.index_path, verbose=False)

        size_kb = os.path.getsize(self.index_path) / 1024
        print(f"✓ Index saved ({size_kb:.1f} KB)")
        print(f"\n💡 Run with --use-index to skip computation")

        return True

    def show_structure(self):
        """Show document structure by category."""
        print_header("DOCUMENT STRUCTURE", "═")

        categories = {
            'experiments': [],
            'results': [],
            'tasks': [],
            'learnings': [],
            'other': []
        }

        for doc_id, word_count in self.loaded_files:
            if 'exp-' in doc_id:
                categories['experiments'].append((doc_id, word_count))
            elif 'result-' in doc_id:
                categories['results'].append((doc_id, word_count))
            elif 'task-' in doc_id:
                categories['tasks'].append((doc_id, word_count))
            elif 'learnings' in doc_id:
                categories['learnings'].append((doc_id, word_count))
            else:
                categories['other'].append((doc_id, word_count))

        icons = {
            'experiments': '🧪',
            'results': '📊',
            'tasks': '📋',
            'learnings': '💡',
            'other': '📄'
        }

        for category, docs in categories.items():
            if docs:
                total_words = sum(wc for _, wc in docs)
                print(f"  {icons[category]} {category.upper()}: {len(docs)} files ({total_words:,} words)")

    def discover_key_concepts(self):
        """Show most important concepts via PageRank."""
        print_header("KEY CONCEPTS (PageRank)", "═")

        print("Central concepts in the audit framework:\n")

        layer0 = self.processor.get_layer(CorticalLayer.TOKENS)

        top_tokens = sorted(
            layer0.minicolumns.values(),
            key=lambda c: c.pagerank,
            reverse=True
        )[:15]

        if top_tokens:
            max_pr = top_tokens[0].pagerank
            print("  Rank  Concept            PageRank")
            print("  " + "─" * 45)

            for i, col in enumerate(top_tokens, 1):
                bar = render_bar(col.pagerank, max_pr, 20)
                print(f"  {i:>3}.  {col.content:<18} {bar} {col.pagerank:.4f}")

    def find_associations(self):
        """Show lateral connections between concepts."""
        print_header("CONCEPT ASSOCIATIONS", "═")

        print("Key concepts and their connections:\n")

        layer0 = self.processor.get_layer(CorticalLayer.TOKENS)

        # Test relevant concepts
        test_concepts = ["guardrail", "agent", "template", "experiment", "misleading"]

        for concept in test_concepts:
            col = layer0.get_minicolumn(concept)
            if col and col.lateral_connections:
                print_subheader(f"🔗 '{concept}' connects to:")

                sorted_conns = sorted(
                    col.lateral_connections.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]

                for neighbor_id, weight in sorted_conns:
                    neighbor = layer0.get_by_id(neighbor_id)
                    if neighbor:
                        bar_len = int(min(weight, 10) * 3)
                        bar = "─" * bar_len + ">"
                        print(f"    {bar} {neighbor.content} (weight: {weight:.2f})")
                print()

    def show_document_relationships(self):
        """Show document-level relationships."""
        print_header("DOCUMENT RELATIONSHIPS", "═")

        print("Documents connected by shared concepts:\n")

        layer3 = self.processor.get_layer(CorticalLayer.DOCUMENTS)

        sorted_docs = sorted(
            layer3.minicolumns.values(),
            key=lambda c: c.connection_count(),
            reverse=True
        )[:5]

        print("  Most connected documents:")
        print("  " + "─" * 50)

        for col in sorted_docs:
            conns = col.connection_count()
            print(f"  📄 {col.content:<40} ({conns} connections)")

        # Show relationships for top document
        if sorted_docs:
            doc = sorted_docs[0]
            print(f"\n  '{doc.content}' relates to:")

            related = self.processor.find_related_documents(doc.content)[:5]
            for related_doc, weight in related:
                print(f"    → {related_doc} (similarity: {weight:.3f})")

    def query(self, query_str: str, top_n: int = 5):
        """Search the index for matching documents."""
        print_header(f"SEARCH: '{query_str}'", "═")

        # Expand query
        expanded = self.processor.expand_query(query_str, max_expansions=5)
        original = set(self.processor.tokenizer.tokenize(query_str))
        new_terms = [t for t in expanded.keys() if t not in original]

        if new_terms:
            print(f"  Expanded with: {', '.join(new_terms[:5])}\n")

        # Find documents
        results = self.processor.find_documents_for_query(query_str, top_n=top_n)

        print("  Matching documents:")
        print("  " + "─" * 50)

        for doc_id, score in results:
            # Categorize
            if 'exp-' in doc_id:
                icon = "🧪"
            elif 'result-' in doc_id:
                icon = "📊"
            elif 'task-' in doc_id:
                icon = "📋"
            elif 'learnings' in doc_id:
                icon = "💡"
            else:
                icon = "📄"

            print(f"  {icon} {doc_id:<45} (score: {score:.3f})")

        # Show passages
        print("\n  Relevant passages:")
        print("  " + "─" * 50)

        passages = self.processor.find_passages_for_query(
            query_str,
            top_n=3,
            chunk_size=200,
            overlap=30
        )

        for i, (passage_text, doc_id, start, end, score) in enumerate(passages, 1):
            print(f"\n  [{i}] {doc_id} (score: {score:.3f})")

            # Show truncated passage
            lines = passage_text.strip().split('\n')[:3]
            for line in lines:
                if len(line) > 65:
                    line = line[:62] + "..."
                print(f"      {line}")

    def print_summary(self):
        """Print summary."""
        print_header("SUMMARY", "═")

        layer0 = self.processor.get_layer(CorticalLayer.TOKENS)
        layer1 = self.processor.get_layer(CorticalLayer.BIGRAMS)

        total_conns = sum(
            layer.total_connections()
            for layer in self.processor.layers.values()
        )

        print(f"  Documents indexed:     {len(self.loaded_files)}")
        print(f"  Unique tokens:         {layer0.column_count()}")
        print(f"  Unique bigrams:        {layer1.column_count()}")
        print(f"  Total connections:     {total_conns:,}")

        # Find most central token
        top_token = max(layer0.minicolumns.values(), key=lambda c: c.pagerank)
        print(f"\n  Most central concept: '{top_token.content}'")

        print("\n" + "═" * 70)
        print("Index complete! Use --query to search:")
        print("  python audit_indexer.py --use-index --query 'guardrail patterns'")
        print("═" * 70 + "\n")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Audit Framework Indexer - semantic search for experiments and learnings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python audit_indexer.py                          # Index and show demo
  python audit_indexer.py --build-index            # Build index only
  python audit_indexer.py --use-index              # Load from index
  python audit_indexer.py --query "binary questions"  # Search
        """
    )
    parser.add_argument(
        "--audit-dir",
        default=AUDIT_DIR,
        help=f"Directory containing audit documents (default: {AUDIT_DIR})"
    )
    parser.add_argument(
        "--use-index",
        action="store_true",
        help="Load from pre-built index"
    )
    parser.add_argument(
        "--build-index",
        action="store_true",
        help="Build and save index without running demos"
    )
    parser.add_argument(
        "--index-path",
        default=DEFAULT_INDEX_PATH,
        help=f"Path to index file (default: {DEFAULT_INDEX_PATH})"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Search query to run against the index"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    indexer = AuditIndexer(
        audit_dir=args.audit_dir,
        use_index=args.use_index,
        index_path=args.index_path
    )

    if args.build_index:
        success = indexer.build_index()
        sys.exit(0 if success else 1)
    elif args.query:
        # Query mode
        if not indexer.ingest_documents():
            print("Failed to load index!")
            sys.exit(1)
        indexer.query(args.query)
    else:
        # Full demo
        indexer.run()
