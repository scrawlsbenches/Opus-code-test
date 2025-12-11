#!/usr/bin/env python3
"""
Ask the Codebase - Interactive Q&A using RAG.

This script provides a conversational interface for asking questions
about the codebase, using the Cortical Text Processor's passage retrieval
to find relevant context.

Usage:
    python scripts/ask_codebase.py                    # Interactive mode
    python scripts/ask_codebase.py "How does X work?" # Single question
    python scripts/ask_codebase.py --sources          # Show source references
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cortical.processor import CorticalTextProcessor


def find_line_number(doc_content: str, char_position: int) -> int:
    """Find the line number for a character position."""
    return doc_content[:char_position].count('\n') + 1


def format_reference(doc_id: str, line_num: int) -> str:
    """Format a file:line reference."""
    return f"{doc_id}:{line_num}"


def get_doc_type_emoji(doc_id: str) -> str:
    """Get emoji indicator for document type."""
    if doc_id.endswith('.md'):
        return "📖"
    elif doc_id.startswith('tests/'):
        return "🧪"
    else:
        return "💻"


class CodebaseQA:
    """Interactive Q&A system for the codebase."""

    def __init__(self, processor: CorticalTextProcessor):
        self.processor = processor

    def find_relevant_passages(
        self,
        question: str,
        top_n: int = 5,
        chunk_size: int = 400
    ) -> List[Tuple[str, str, int, float]]:
        """
        Find passages relevant to the question.

        Returns:
            List of (passage_text, reference, line_num, score) tuples
        """
        # Detect if this is a conceptual or implementation question
        is_conceptual = self.processor.is_conceptual_query(question)

        # Get relevant documents with boosting
        doc_results = self.processor.find_documents_with_boost(
            question,
            top_n=top_n * 2,
            auto_detect_intent=True
        )

        # Get passages from those documents
        doc_ids = [doc_id for doc_id, _ in doc_results]
        passages = self.processor.find_passages_for_query(
            question,
            top_n=top_n,
            chunk_size=chunk_size,
            overlap=100,
            doc_filter=doc_ids if doc_ids else None
        )

        results = []
        for passage_text, doc_id, start, end, score in passages:
            doc_content = self.processor.documents.get(doc_id, "")
            line_num = find_line_number(doc_content, start)
            reference = format_reference(doc_id, line_num)
            results.append((passage_text, reference, line_num, score))

        return results

    def format_answer(
        self,
        question: str,
        passages: List[Tuple[str, str, int, float]],
        show_sources: bool = True,
        verbose: bool = False
    ) -> str:
        """
        Format the answer with context from passages.

        This creates a structured response showing:
        1. Question type (conceptual vs implementation)
        2. Relevant passages with references
        3. Source list for verification
        """
        lines = []

        # Detect question type
        is_conceptual = self.processor.is_conceptual_query(question)
        intent = "conceptual" if is_conceptual else "implementation"

        lines.append(f"\n{'─' * 60}")
        lines.append(f"Question: {question}")
        lines.append(f"Type: {intent}")
        lines.append(f"{'─' * 60}\n")

        if not passages:
            lines.append("No relevant passages found for this question.")
            lines.append("Try rephrasing or using different keywords.")
            return '\n'.join(lines)

        lines.append("📚 Relevant Context:\n")

        for i, (passage_text, reference, line_num, score) in enumerate(passages, 1):
            doc_id = reference.split(':')[0]
            emoji = get_doc_type_emoji(doc_id)

            lines.append(f"[{i}] {emoji} {reference} (relevance: {score:.2f})")
            lines.append("─" * 50)

            # Show passage content
            if verbose:
                # Full passage
                for line in passage_text.strip().split('\n'):
                    lines.append(f"  {line}")
            else:
                # Truncated passage (first 5 lines)
                passage_lines = passage_text.strip().split('\n')[:5]
                for line in passage_lines:
                    if len(line) > 70:
                        line = line[:67] + "..."
                    lines.append(f"  {line}")
                if len(passage_text.strip().split('\n')) > 5:
                    lines.append(f"  ... (more content in source)")

            lines.append("")

        if show_sources:
            lines.append("\n📌 Sources:")
            seen_docs = set()
            for passage_text, reference, line_num, score in passages:
                doc_id = reference.split(':')[0]
                if doc_id not in seen_docs:
                    emoji = get_doc_type_emoji(doc_id)
                    lines.append(f"  {emoji} {doc_id}")
                    seen_docs.add(doc_id)

        lines.append("")
        return '\n'.join(lines)

    def answer(
        self,
        question: str,
        top_n: int = 3,
        show_sources: bool = True,
        verbose: bool = False
    ) -> str:
        """Answer a question about the codebase."""
        passages = self.find_relevant_passages(question, top_n=top_n)
        return self.format_answer(question, passages, show_sources, verbose)


def interactive_mode(qa: CodebaseQA, verbose: bool = False):
    """Run interactive Q&A mode."""
    print("\n" + "=" * 60)
    print("       🧠 ASK THE CODEBASE - Interactive Q&A")
    print("=" * 60)
    print("\nAsk questions about the codebase in natural language.")
    print("The system will find relevant passages to help answer.\n")
    print("Commands:")
    print("  /verbose    - Toggle verbose mode (show full passages)")
    print("  /top N      - Set number of results (default: 3)")
    print("  /help       - Show this help")
    print("  /quit       - Exit")
    print()

    top_n = 3
    show_verbose = verbose

    while True:
        try:
            question = input("Ask> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue

        if question.startswith('/'):
            cmd_parts = question.split(maxsplit=1)
            cmd = cmd_parts[0].lower()

            if cmd in ('/quit', '/exit', '/q'):
                print("Goodbye!")
                break
            elif cmd == '/help':
                print("\nCommands: /verbose, /top N, /quit")
                print("Or just type your question!\n")
            elif cmd == '/verbose':
                show_verbose = not show_verbose
                status = "ON" if show_verbose else "OFF"
                print(f"Verbose mode: {status}")
            elif cmd == '/top' and len(cmd_parts) > 1:
                try:
                    top_n = int(cmd_parts[1])
                    print(f"Now showing top {top_n} results")
                except ValueError:
                    print("Usage: /top N (where N is a number)")
            else:
                print(f"Unknown command: {cmd}")
        else:
            answer = qa.answer(question, top_n=top_n, verbose=show_verbose)
            print(answer)


def main():
    parser = argparse.ArgumentParser(
        description='Ask questions about the codebase',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Interactive mode
  %(prog)s "How does PageRank work?"    # Single question
  %(prog)s "where is TF-IDF computed" --verbose
  %(prog)s "authentication" --top 5     # More results
        """
    )
    parser.add_argument('question', nargs='?', help='Question to ask')
    parser.add_argument('--corpus', '-c', default='corpus_dev.pkl',
                        help='Corpus file path (default: corpus_dev.pkl)')
    parser.add_argument('--top', '-n', type=int, default=3,
                        help='Number of passages to retrieve (default: 3)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show full passage content')
    parser.add_argument('--no-sources', action='store_true',
                        help='Hide source list')
    args = parser.parse_args()

    base_path = Path(__file__).parent.parent
    corpus_path = base_path / args.corpus

    # Check if corpus exists
    if not corpus_path.exists():
        print(f"Error: Corpus file not found: {corpus_path}")
        print("Run 'python scripts/index_codebase.py' first to create it.")
        sys.exit(1)

    # Load corpus
    print(f"Loading corpus from {corpus_path}...")
    processor = CorticalTextProcessor.load(str(corpus_path))
    print(f"Loaded {len(processor.documents)} documents")

    qa = CodebaseQA(processor)

    if args.question:
        # Single question mode
        answer = qa.answer(
            args.question,
            top_n=args.top,
            show_sources=not args.no_sources,
            verbose=args.verbose
        )
        print(answer)
    else:
        # Interactive mode
        interactive_mode(qa, verbose=args.verbose)


if __name__ == '__main__':
    main()
