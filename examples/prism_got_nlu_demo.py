#!/usr/bin/env python3
"""
PRISM-GoT Natural Language Understanding Demo

A focused demo showing how PRISM-GoT answers natural language questions
by learning from corpus content. No simulation - real query understanding.

Run with: python examples/prism_got_nlu_demo.py
"""

import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from cortical.processor import CorticalTextProcessor
from cortical.layers import CorticalLayer
from cortical.reasoning import (
    NodeType,
    EdgeType,
    SynapticMemoryGraph,
    IncrementalReasoner,
)


class KnowledgeBase:
    """Simple knowledge base built from corpus with NLU capabilities."""

    def __init__(self, samples_dir: Path):
        self.processor = CorticalTextProcessor()
        self.graph = SynapticMemoryGraph()
        self.reasoner = IncrementalReasoner(self.graph)
        self.docs = {}
        self.doc_summaries = {}

        self._load_corpus(samples_dir)
        self._build_index()

    def _load_corpus(self, samples_dir: Path):
        """Load all documents."""
        for pattern in ["**/*.txt", "**/*.md"]:
            for f in samples_dir.glob(pattern):
                try:
                    content = f.read_text(encoding="utf-8")
                    doc_id = str(f.relative_to(samples_dir))
                    self.docs[doc_id] = content

                    # Extract first meaningful line as summary
                    lines = [l.strip() for l in content.split('\n') if l.strip()]
                    summary = lines[0].strip('# ').strip()[:100] if lines else doc_id
                    self.doc_summaries[doc_id] = summary
                except:
                    pass

    def _build_index(self):
        """Build cortical index and knowledge graph."""
        # Index with cortical processor
        for doc_id, content in self.docs.items():
            self.processor.process_document(doc_id, content)
        self.processor.compute_all()

        # Add documents to knowledge graph
        for doc_id, summary in self.doc_summaries.items():
            self.graph.add_node(f"DOC:{doc_id}", NodeType.ARTIFACT, summary)

    def _tokenize_query(self, query: str) -> list:
        """Simple query tokenization."""
        # Remove common question words and punctuation
        stop_words = {
            'what', 'how', 'why', 'when', 'where', 'who', 'which',
            'is', 'are', 'do', 'does', 'can', 'could', 'would', 'should',
            'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'about', 'explain', 'tell', 'me', 'show', 'find', 'give',
            'i', 'need', 'want', 'looking', 'search', 'help'
        }
        words = query.lower().replace('?', '').replace('.', '').split()
        return [w for w in words if w not in stop_words and len(w) > 2]

    def _score_document(self, doc_id: str, query_terms: list) -> float:
        """Score a document's relevance to query terms."""
        content = self.docs.get(doc_id, "").lower()
        score = 0.0

        layer0 = self.processor.get_layer(CorticalLayer.TOKENS)

        for term in query_terms:
            # Direct term match
            if term in content:
                col = layer0.get_minicolumn(term)
                if col and doc_id in col.document_ids:
                    # Use TF-IDF weight
                    tfidf = col.tfidf_per_doc.get(doc_id, col.tfidf)
                    score += tfidf * 2

            # Check for related terms via lateral connections
            col = layer0.get_minicolumn(term)
            if col:
                for related_id, weight in list(col.lateral_connections.items())[:5]:
                    related_term = related_id.replace("L0_", "")
                    if related_term in content:
                        score += weight * 0.5

        return score

    def ask(self, question: str, top_k: int = 5) -> list:
        """
        Answer a natural language question by finding relevant documents.

        Returns list of (doc_id, summary, score, snippet) tuples.
        """
        query_terms = self._tokenize_query(question)

        if not query_terms:
            return []

        # Score all documents
        scores = {}
        for doc_id in self.docs:
            score = self._score_document(doc_id, query_terms)
            if score > 0:
                scores[doc_id] = score

        # Sort by score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Build results with snippets
        results = []
        for doc_id, score in ranked:
            content = self.docs[doc_id]
            summary = self.doc_summaries[doc_id]

            # Find best snippet containing query terms
            snippet = self._extract_snippet(content, query_terms)

            results.append((doc_id, summary, score, snippet))

            # Record this in the knowledge graph for learning
            q_node_id = f"Q:{hash(question) % 10000}"
            if q_node_id not in self.graph.nodes:
                self.graph.add_node(q_node_id, NodeType.QUESTION, question[:50])

            doc_node_id = f"DOC:{doc_id}"
            edge_key = (q_node_id, doc_node_id, EdgeType.ANSWERS)
            if edge_key not in self.graph.synaptic_edges:
                self.graph.add_synaptic_edge(
                    q_node_id, doc_node_id, EdgeType.ANSWERS,
                    weight=min(score / 10, 1.0)
                )

        return results

    def _extract_snippet(self, content: str, query_terms: list, max_len: int = 150) -> str:
        """Extract a relevant snippet from content."""
        content_lower = content.lower()
        sentences = content.replace('\n', ' ').split('. ')

        # Score sentences by term presence
        best_sentence = ""
        best_score = 0

        for sentence in sentences:
            sentence_lower = sentence.lower()
            score = sum(1 for t in query_terms if t in sentence_lower)
            if score > best_score:
                best_score = score
                best_sentence = sentence

        if best_sentence:
            snippet = best_sentence.strip()
            if len(snippet) > max_len:
                snippet = snippet[:max_len] + "..."
            return snippet

        # Fallback to first sentence
        return sentences[0][:max_len] + "..." if sentences else ""

    def learn(self, question: str, helpful_doc: str):
        """Mark a document as helpful for a question (reinforcement)."""
        q_node_id = f"Q:{hash(question) % 10000}"
        doc_node_id = f"DOC:{helpful_doc}"

        # Strengthen the connection
        for (src, tgt, etype), edge in self.graph.synaptic_edges.items():
            if src == q_node_id and tgt == doc_node_id:
                edge.strengthen(0.2)
                edge.record_prediction_outcome(correct=True)
                return True
        return False

    def get_stats(self) -> dict:
        """Get knowledge base statistics."""
        return {
            "documents": len(self.docs),
            "tokens": self.processor.get_layer(CorticalLayer.TOKENS).column_count(),
            "graph_nodes": self.graph.node_count(),
            "graph_edges": len(self.graph.synaptic_edges),
        }


def main():
    print("\n" + "="*60)
    print("  PRISM-GoT Natural Language Understanding")
    print("="*60)

    samples_dir = Path(__file__).parent.parent / "samples"

    print("\nBuilding knowledge base from corpus...")
    kb = KnowledgeBase(samples_dir)

    stats = kb.get_stats()
    print(f"\nKnowledge base ready:")
    print(f"  Documents: {stats['documents']}")
    print(f"  Tokens:    {stats['tokens']:,}")

    # Example questions to demonstrate NLU
    questions = [
        "How does memory consolidation work during sleep?",
        "What is PageRank and how does it rank importance?",
        "How do agents coordinate in parallel systems?",
        "What are the principles of semantic search?",
        "How does Hebbian learning strengthen connections?",
        "What is the role of attention in neural networks?",
        "How do you handle knowledge gaps?",
        "What patterns help with code refactoring?",
    ]

    print("\n" + "-"*60)
    print("  ANSWERING NATURAL LANGUAGE QUESTIONS")
    print("-"*60)

    for question in questions:
        print(f"\n Q: {question}")

        results = kb.ask(question, top_k=3)

        if results:
            print(" A: Found relevant knowledge:")
            for i, (doc_id, summary, score, snippet) in enumerate(results, 1):
                print(f"    {i}. [{score:.1f}] {doc_id}")
                print(f"       \"{snippet}\"")
        else:
            print(" A: No relevant documents found.")

    print("\n" + "-"*60)
    print("  INTERACTIVE MODE")
    print("-"*60)
    print("\nAsk questions about the corpus (or 'quit' to exit):\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not question:
            continue
        if question.lower() in ('quit', 'exit', 'q'):
            break

        results = kb.ask(question, top_k=3)

        if results:
            print("\nKB: Here's what I found:\n")
            for i, (doc_id, summary, score, snippet) in enumerate(results, 1):
                print(f"  {i}. {doc_id}")
                print(f"     \"{snippet}\"\n")

            # Ask for feedback
            try:
                feedback = input("Was result #1 helpful? (y/n): ").strip().lower()
                if feedback == 'y' and results:
                    kb.learn(question, results[0][0])
                    print("Thanks! I'll remember that.\n")
            except:
                pass
        else:
            print("\nKB: I couldn't find relevant information for that question.\n")

    print("\nFinal knowledge graph stats:")
    stats = kb.get_stats()
    print(f"  Graph nodes: {stats['graph_nodes']}")
    print(f"  Graph edges: {stats['graph_edges']}")
    print("\nGoodbye!")


if __name__ == "__main__":
    main()
