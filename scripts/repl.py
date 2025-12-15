#!/usr/bin/env python3
"""
Cortical Text Processor REPL
=============================

Interactive Read-Eval-Print Loop for the Cortical Text Processor.

Usage:
    python scripts/repl.py [corpus_file]
    python scripts/repl.py corpus_dev.pkl

Features:
    - Tab completion for commands
    - Command history via readline
    - Built-in help system
    - All processor operations accessible

Commands:
    load <file>              Load a corpus file
    search <query>           Search documents
    expand <term>            Show query expansion
    stats                    Show corpus statistics
    concepts [n]             List top n concept clusters (default: 10)
    fingerprint <text>       Get semantic fingerprint
    patterns <doc_id>        Show code patterns in document
    metrics                  Show observability metrics
    similar <file:line>      Find similar code to reference
    docs <query>             Search with documentation boost
    code <query>             Search with code-aware expansion
    intent <query>           Parse and search by intent
    passages <query>         Find relevant passages (RAG)
    relations [n]            Show semantic relations (default: 10)
    stale                    Show stale computations
    compute [type]           Compute all or specific type (tfidf, pagerank, etc.)
    save <file>              Save corpus to file
    export <file> [format]   Export corpus (json, csv, txt)
    clear                    Clear metrics
    reset                    Reset metrics collection
    help [command]           Show help for command(s)
    quit                     Exit REPL

Example session:
    >>> load corpus_dev.pkl
    Loaded 125 documents
    >>> search "pagerank algorithm"
    [1] cortical/analysis.py:45 (score: 0.850)
    >>> expand "neural"
    neural: 1.000, network: 0.650, neuron: 0.450...
    >>> metrics
    Operation          Count    Avg (ms)    Min (ms)    Max (ms)
    >>> quit
"""

import cmd
import sys
import os
import shlex
from pathlib import Path
from typing import Optional, Dict, Any, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cortical.processor import CorticalTextProcessor
from cortical.layers import CorticalLayer

# Try to import readline for history/completion
try:
    import readline
    READLINE_AVAILABLE = True
except ImportError:
    READLINE_AVAILABLE = False


class CorticalREPL(cmd.Cmd):
    """Interactive REPL for Cortical Text Processor."""

    intro = """
╔════════════════════════════════════════════════════════════════╗
║         Cortical Text Processor REPL v1.0                      ║
║         Type 'help' for commands, 'quit' to exit               ║
╚════════════════════════════════════════════════════════════════╝
"""
    prompt = '>>> '

    def __init__(self, corpus_file: Optional[str] = None):
        """
        Initialize REPL.

        Args:
            corpus_file: Optional corpus file to load on startup
        """
        super().__init__()
        self.processor: Optional[CorticalTextProcessor] = None
        self.corpus_file: Optional[str] = None

        # Enable metrics by default for observability
        if corpus_file:
            try:
                self.do_load(corpus_file)
            except Exception as e:
                print(f"Warning: Could not load {corpus_file}: {e}")
                print("Use 'load <file>' to load a corpus.\n")

    # =========================================================================
    # CORPUS MANAGEMENT COMMANDS
    # =========================================================================

    def do_load(self, arg: str) -> None:
        """
        Load a corpus file.

        Usage: load <file>

        Example:
            >>> load corpus_dev.pkl
            Loaded 125 documents
        """
        if not arg.strip():
            print("Error: Please specify a file to load")
            print("Usage: load <file>")
            return

        file_path = arg.strip()
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            return

        try:
            print(f"Loading corpus from {file_path}...")
            self.processor = CorticalTextProcessor.load(file_path)
            # Enable metrics after loading
            self.processor.enable_metrics()
            self.corpus_file = file_path
            print(f"✓ Loaded {len(self.processor.documents)} documents")

            # Show quick stats
            layer0 = self.processor.layers[CorticalLayer.TOKENS]
            layer1 = self.processor.layers[CorticalLayer.BIGRAMS]
            layer2 = self.processor.layers[CorticalLayer.CONCEPTS]
            print(f"  Tokens: {layer0.column_count()}, "
                  f"Bigrams: {layer1.column_count()}, "
                  f"Concepts: {layer2.column_count()}")
        except Exception as e:
            print(f"Error loading corpus: {e}")

    def do_save(self, arg: str) -> None:
        """
        Save corpus to file.

        Usage: save <file>

        Example:
            >>> save my_corpus.pkl
            Saved to my_corpus.pkl
        """
        if not self._require_processor():
            return

        if not arg.strip():
            print("Error: Please specify a file to save to")
            print("Usage: save <file>")
            return

        file_path = arg.strip()
        try:
            self.processor.save(file_path)
            print(f"✓ Saved to {file_path}")
        except Exception as e:
            print(f"Error saving corpus: {e}")

    def do_export(self, arg: str) -> None:
        """
        Export corpus to various formats.

        Usage: export <dir> [type]
        Types: json (default), graph, embeddings, relations

        Example:
            >>> export corpus_state        # Export full state to JSON
            >>> export graph.json graph    # Export graph only
        """
        if not self._require_processor():
            return

        parts = arg.strip().split()
        if not parts:
            print("Error: Please specify a path to export to")
            print("Usage: export <dir> [type]")
            return

        path = parts[0]
        export_type = parts[1] if len(parts) > 1 else 'json'

        if export_type not in ['json', 'graph', 'embeddings', 'relations']:
            print(f"Error: Unknown export type '{export_type}'")
            print("Use: json, graph, embeddings, relations")
            return

        try:
            if export_type == 'json':
                # Export full state to JSON directory
                self.processor.save_json(path, verbose=True)
                print(f"✓ Exported full state to {path}/")
            elif export_type == 'graph':
                # Export graph structure
                self.processor.export_graph(path)
                print(f"✓ Exported graph to {path}")
            else:
                print(f"Error: Export type '{export_type}' not yet implemented")
                print("Currently supported: json, graph")
        except Exception as e:
            print(f"Error exporting corpus: {e}")

    # =========================================================================
    # SEARCH COMMANDS
    # =========================================================================

    def do_search(self, arg: str) -> None:
        """
        Search documents for a query.

        Usage: search <query>

        Example:
            >>> search "pagerank algorithm"
            [1] cortical/analysis.py (score: 0.850)
            [2] docs/architecture.md (score: 0.720)
        """
        if not self._require_processor():
            return

        query = arg.strip()
        if not query:
            print("Error: Please provide a search query")
            return

        try:
            results = self.processor.find_documents_for_query(query, top_n=10)
            if not results:
                print("No results found.")
                return

            print(f"\n{'='*70}")
            print(f"Results for: {query}")
            print(f"{'='*70}\n")

            for i, (doc_id, score) in enumerate(results, 1):
                print(f"[{i}] {doc_id}")
                print(f"    Score: {score:.3f}\n")
        except Exception as e:
            print(f"Error searching: {e}")

    def do_docs(self, arg: str) -> None:
        """
        Search with documentation boost.

        Usage: docs <query>

        Example:
            >>> docs "what is pagerank"
            [1] docs/architecture.md (score: 1.200)
        """
        if not self._require_processor():
            return

        query = arg.strip()
        if not query:
            print("Error: Please provide a search query")
            return

        try:
            results = self.processor.find_documents_with_boost(
                query, top_n=10, prefer_docs=True
            )
            if not results:
                print("No results found.")
                return

            print(f"\n{'='*70}")
            print(f"Results (docs boosted): {query}")
            print(f"{'='*70}\n")

            for i, (doc_id, score) in enumerate(results, 1):
                print(f"[{i}] {doc_id}")
                print(f"    Score: {score:.3f}\n")
        except Exception as e:
            print(f"Error searching: {e}")

    def do_code(self, arg: str) -> None:
        """
        Search with code-aware expansion.

        Usage: code <query>

        Example:
            >>> code "fetch data"
            Expands to: fetch, get, load, retrieve, read...
        """
        if not self._require_processor():
            return

        query = arg.strip()
        if not query:
            print("Error: Please provide a search query")
            return

        try:
            # Show expansion first
            expanded = self.processor.expand_query_for_code(query)
            print("\nCode-aware expansion:")
            for term, weight in sorted(expanded.items(), key=lambda x: -x[1])[:8]:
                print(f"  {term}: {weight:.3f}")

            # Then search
            results = self.processor.find_documents_for_query(query, top_n=10)
            if not results:
                print("\nNo results found.")
                return

            print(f"\n{'='*70}")
            print(f"Results: {query}")
            print(f"{'='*70}\n")

            for i, (doc_id, score) in enumerate(results, 1):
                print(f"[{i}] {doc_id}")
                print(f"    Score: {score:.3f}\n")
        except Exception as e:
            print(f"Error searching: {e}")

    def do_passages(self, arg: str) -> None:
        """
        Find relevant passages for RAG systems.

        Usage: passages <query>

        Example:
            >>> passages "how does pagerank work"
            [1] cortical/analysis.py:45-65
                PageRank implementation...
        """
        if not self._require_processor():
            return

        query = arg.strip()
        if not query:
            print("Error: Please provide a search query")
            return

        try:
            results = self.processor.find_passages_for_query(query, top_n=5)
            if not results:
                print("No passages found.")
                return

            print(f"\n{'='*70}")
            print(f"Passages for: {query}")
            print(f"{'='*70}\n")

            for i, (passage, doc_id, start, end, score) in enumerate(results, 1):
                # Find approximate line numbers
                doc_content = self.processor.documents.get(doc_id, '')
                line_start = doc_content[:start].count('\n') + 1
                line_end = doc_content[:end].count('\n') + 1

                print(f"[{i}] {doc_id}:{line_start}-{line_end}")
                print(f"    Score: {score:.3f}")
                print(f"    {'-'*60}")
                # Show first 5 lines
                lines = passage.split('\n')[:5]
                for line in lines:
                    if len(line) > 70:
                        line = line[:67] + '...'
                    print(f"    {line}")
                if len(passage.split('\n')) > 5:
                    print(f"    ... ({len(passage.split(chr(10))) - 5} more lines)")
                print()
        except Exception as e:
            print(f"Error finding passages: {e}")

    def do_intent(self, arg: str) -> None:
        """
        Parse query intent and search.

        Usage: intent <query>

        Example:
            >>> intent "where do we handle authentication"
            Intent: location, Action: handle, Subject: authentication
        """
        if not self._require_processor():
            return

        query = arg.strip()
        if not query:
            print("Error: Please provide a query")
            return

        try:
            # Parse intent
            parsed = self.processor.parse_intent_query(query)
            print(f"\nParsed intent:")
            print(f"  Intent: {parsed.get('intent', 'unknown')}")
            print(f"  Action: {parsed.get('action', 'N/A')}")
            print(f"  Subject: {parsed.get('subject', 'N/A')}")
            if parsed.get('modifiers'):
                print(f"  Modifiers: {', '.join(parsed['modifiers'])}")

            # Search by intent
            results = self.processor.search_by_intent(query, top_n=5)
            if not results:
                print("\nNo results found.")
                return

            print(f"\n{'='*70}")
            print(f"Results:")
            print(f"{'='*70}\n")

            for i, (doc_id, score) in enumerate(results, 1):
                print(f"[{i}] {doc_id}")
                print(f"    Score: {score:.3f}\n")
        except Exception as e:
            print(f"Error with intent search: {e}")

    def do_similar(self, arg: str) -> None:
        """
        Find code similar to a file:line reference.

        Usage: similar <file:line>

        Example:
            >>> similar cortical/processor.py:100
            [1] cortical/analysis.py:250 (85% similar)
        """
        if not self._require_processor():
            return

        target = arg.strip()
        if not target:
            print("Error: Please provide file:line reference")
            print("Usage: similar <file:line>")
            return

        try:
            # Parse file:line
            if ':' not in target:
                print("Error: Use format file:line (e.g., processor.py:100)")
                return

            parts = target.split(':')
            file_path = parts[0]
            try:
                line_num = int(parts[1]) if len(parts) > 1 else 1
            except ValueError:
                line_num = 1

            # Get content from document
            doc_content = self.processor.documents.get(file_path, '')
            if not doc_content:
                # Try to find matching document
                for doc_id in self.processor.documents:
                    if doc_id.endswith(file_path) or file_path in doc_id:
                        doc_content = self.processor.documents[doc_id]
                        file_path = doc_id
                        break

            if not doc_content:
                print(f"Error: Document not found: {file_path}")
                return

            # Extract passage around line
            lines = doc_content.split('\n')
            start_line = max(0, line_num - 1)
            end_line = min(len(lines), start_line + 20)
            target_text = '\n'.join(lines[start_line:end_line])

            if not target_text.strip():
                print("Error: No content at specified line")
                return

            # Get fingerprint
            target_fp = self.processor.get_fingerprint(target_text, top_n=20)

            # Compare against all documents
            results = []
            for doc_id, content in self.processor.documents.items():
                if doc_id == file_path:
                    continue  # Skip source document

                # Chunk and compare
                chunk_size = 400
                for start in range(0, len(content), chunk_size // 2):
                    end = min(start + chunk_size, len(content))
                    chunk = content[start:end]

                    if len(chunk.strip()) < 50:
                        continue

                    chunk_fp = self.processor.get_fingerprint(chunk, top_n=20)
                    comparison = self.processor.compare_fingerprints(target_fp, chunk_fp)

                    similarity = comparison.get('overall_similarity', 0)
                    if similarity > 0.1:
                        chunk_line = content[:start].count('\n') + 1
                        results.append({
                            'doc_id': doc_id,
                            'line': chunk_line,
                            'similarity': similarity,
                            'shared': list(comparison.get('shared_terms', []))[:5]
                        })

            # Sort and display
            results.sort(key=lambda x: x['similarity'], reverse=True)
            results = results[:10]

            if not results:
                print("No similar code found.")
                return

            print(f"\n{'='*70}")
            print(f"Code similar to: {target}")
            print(f"{'='*70}\n")

            for i, result in enumerate(results, 1):
                print(f"[{i}] {result['doc_id']}:{result['line']}")
                print(f"    Similarity: {result['similarity']:.1%}")
                if result.get('shared'):
                    print(f"    Shared: {', '.join(result['shared'])}")
                print()

        except Exception as e:
            print(f"Error finding similar code: {e}")

    # =========================================================================
    # QUERY EXPANSION & ANALYSIS
    # =========================================================================

    def do_expand(self, arg: str) -> None:
        """
        Show query expansion for a term.

        Usage: expand <term>

        Example:
            >>> expand "neural"
            neural: 1.000, network: 0.650, neuron: 0.450...
        """
        if not self._require_processor():
            return

        term = arg.strip()
        if not term:
            print("Error: Please provide a term to expand")
            return

        try:
            expanded = self.processor.expand_query(term, max_expansions=15)
            if not expanded:
                print(f"No expansions found for: {term}")
                return

            print(f"\nQuery expansion for '{term}':")
            print(f"{'─'*50}")
            for t, weight in sorted(expanded.items(), key=lambda x: -x[1])[:15]:
                bar = '█' * int(weight * 20)
                print(f"  {t:.<30} {weight:.3f} {bar}")
        except Exception as e:
            print(f"Error expanding query: {e}")

    def do_fingerprint(self, arg: str) -> None:
        """
        Get semantic fingerprint of text.

        Usage: fingerprint <text>

        Example:
            >>> fingerprint "neural networks process data"
            Top terms: neural, network, process, data...
        """
        if not self._require_processor():
            return

        text = arg.strip()
        if not text:
            print("Error: Please provide text to fingerprint")
            return

        try:
            fp = self.processor.get_fingerprint(text, top_n=20)
            explanation = self.processor.explain_fingerprint(fp, top_n=10)

            print(f"\nFingerprint:")
            print(f"{'─'*50}")
            print(f"Term count: {fp.get('term_count', 0)}")

            if explanation.get('top_terms'):
                print(f"\nTop terms:")
                for term, weight in explanation['top_terms']:
                    print(f"  {term}: {weight:.3f}")

            if explanation.get('concepts'):
                print(f"\nConcepts: {', '.join(explanation['concepts'][:5])}")

            if explanation.get('bigrams'):
                print(f"\nBigrams: {', '.join(explanation['bigrams'][:5])}")

        except Exception as e:
            print(f"Error computing fingerprint: {e}")

    # =========================================================================
    # INTROSPECTION COMMANDS
    # =========================================================================

    def do_stats(self, arg: str) -> None:
        """
        Show corpus statistics.

        Usage: stats

        Example:
            >>> stats
            Documents: 125
            Tokens: 5420
            Bigrams: 8930
            Concepts: 34
        """
        if not self._require_processor():
            return

        try:
            summary = self.processor.get_corpus_summary()
            print(f"\nCorpus Statistics:")
            print(f"{'═'*50}")
            print(f"  Documents:     {summary.get('document_count', 0):,}")
            print(f"  Tokens:        {summary.get('token_count', 0):,}")
            print(f"  Bigrams:       {summary.get('bigram_count', 0):,}")
            print(f"  Concepts:      {summary.get('concept_count', 0):,}")
            print(f"  Relations:     {len(self.processor.semantic_relations):,}")

            # Stale computations
            stale = self.processor.get_stale_computations()
            if stale:
                print(f"\n  Stale:         {', '.join(sorted(stale))}")
            else:
                print(f"\n  All computations up-to-date")

        except Exception as e:
            print(f"Error getting stats: {e}")

    def do_concepts(self, arg: str) -> None:
        """
        List concept clusters.

        Usage: concepts [n]
        Default: n=10

        Example:
            >>> concepts 5
            [1] neural network deep learning...
            [2] algorithm computation analysis...
        """
        if not self._require_processor():
            return

        try:
            n = int(arg.strip()) if arg.strip() else 10
        except ValueError:
            print("Error: Please provide a valid number")
            return

        try:
            layer2 = self.processor.layers[CorticalLayer.CONCEPTS]
            concepts = sorted(
                layer2.minicolumns.values(),
                key=lambda c: c.pagerank if hasattr(c, 'pagerank') else 0,
                reverse=True
            )[:n]

            print(f"\nTop {n} Concept Clusters:")
            print(f"{'═'*70}")

            for i, concept in enumerate(concepts, 1):
                content = concept.content[:60]
                if len(concept.content) > 60:
                    content += '...'
                pr = getattr(concept, 'pagerank', 0)
                print(f"[{i}] {content}")
                print(f"    PageRank: {pr:.4f}, Docs: {len(concept.document_ids)}\n")

        except Exception as e:
            print(f"Error listing concepts: {e}")

    def do_relations(self, arg: str) -> None:
        """
        Show semantic relations.

        Usage: relations [n]
        Default: n=10

        Example:
            >>> relations 5
            network --is_a--> system (0.85)
            algorithm --uses--> data (0.72)
        """
        if not self._require_processor():
            return

        try:
            n = int(arg.strip()) if arg.strip() else 10
        except ValueError:
            print("Error: Please provide a valid number")
            return

        try:
            relations = self.processor.semantic_relations[:n]
            if not relations:
                print("No semantic relations found.")
                print("Run 'compute semantics' first.")
                return

            print(f"\nTop {n} Semantic Relations:")
            print(f"{'═'*70}")

            for i, (term1, rel, term2, weight) in enumerate(relations, 1):
                print(f"[{i}] {term1} --{rel}--> {term2}")
                print(f"    Weight: {weight:.3f}\n")

        except Exception as e:
            print(f"Error showing relations: {e}")

    def do_patterns(self, arg: str) -> None:
        """
        Show code patterns in a document.

        Usage: patterns <doc_id>

        Example:
            >>> patterns cortical/processor.py
            Decorator: 15 occurrences
            Context Manager: 8 occurrences
        """
        if not self._require_processor():
            return

        doc_id = arg.strip()
        if not doc_id:
            print("Error: Please provide a document ID")
            return

        try:
            patterns = self.processor.detect_patterns(doc_id)
            if not patterns:
                print(f"No patterns found in: {doc_id}")
                return

            print(f"\nCode Patterns in {doc_id}:")
            print(f"{'═'*70}")

            # Group by category
            from collections import defaultdict
            by_category = defaultdict(list)

            # Get pattern definitions for categories
            from cortical.patterns import PATTERN_DEFINITIONS

            for pattern_name, occurrences in patterns.items():
                if not occurrences:
                    continue
                category = PATTERN_DEFINITIONS.get(pattern_name, ('', '', 'other'))[2]
                by_category[category].append((pattern_name, len(occurrences)))

            # Display by category
            for category in sorted(by_category.keys()):
                print(f"\n{category.upper()}:")
                for pattern_name, count in sorted(by_category[category], key=lambda x: -x[1]):
                    print(f"  {pattern_name:.<40} {count:>3} occurrences")

        except Exception as e:
            print(f"Error detecting patterns: {e}")

    def do_metrics(self, arg: str) -> None:
        """
        Show observability metrics.

        Usage: metrics

        Example:
            >>> metrics
            Operation          Count    Avg (ms)    Min (ms)    Max (ms)
            compute_all            1      125.30       125.30      125.30
        """
        if not self._require_processor():
            return

        try:
            summary = self.processor.get_metrics_summary()
            if not summary or summary.strip() == "No metrics collected yet.":
                print("\nNo metrics collected yet.")
                print("Metrics are collected as you use the processor.")
                return

            print(f"\n{summary}")
        except Exception as e:
            print(f"Error getting metrics: {e}")

    def do_stale(self, arg: str) -> None:
        """
        Show stale computations.

        Usage: stale

        Example:
            >>> stale
            Stale: pagerank, concepts
        """
        if not self._require_processor():
            return

        try:
            stale = self.processor.get_stale_computations()
            if not stale:
                print("All computations are up-to-date.")
            else:
                print(f"\nStale computations: {', '.join(sorted(stale))}")
                print("Use 'compute' to update them.")
        except Exception as e:
            print(f"Error checking staleness: {e}")

    # =========================================================================
    # COMPUTATION COMMANDS
    # =========================================================================

    def do_compute(self, arg: str) -> None:
        """
        Run computations.

        Usage: compute [type]
        Types: tfidf, pagerank, concepts, semantics, all (default)

        Example:
            >>> compute
            Computing all...
            >>> compute pagerank
            Computing PageRank...
        """
        if not self._require_processor():
            return

        comp_type = arg.strip().lower() or 'all'

        try:
            if comp_type == 'all':
                print("Computing all...")
                self.processor.compute_all()
                print("✓ Done")
            elif comp_type == 'tfidf':
                print("Computing TF-IDF...")
                self.processor.compute_tfidf()
                print("✓ Done")
            elif comp_type == 'pagerank':
                print("Computing PageRank...")
                self.processor.compute_importance()
                print("✓ Done")
            elif comp_type == 'concepts':
                print("Computing concepts...")
                self.processor.build_concept_clusters()
                print("✓ Done")
            elif comp_type == 'semantics':
                print("Computing semantic relations...")
                self.processor.extract_corpus_semantics()
                print("✓ Done")
            elif comp_type == 'embeddings':
                print("Computing embeddings...")
                self.processor.compute_graph_embeddings()
                print("✓ Done")
            else:
                print(f"Error: Unknown computation type: {comp_type}")
                print("Use: tfidf, pagerank, concepts, semantics, embeddings, all")

        except Exception as e:
            print(f"Error computing: {e}")

    # =========================================================================
    # UTILITY COMMANDS
    # =========================================================================

    def do_clear(self, arg: str) -> None:
        """
        Clear metrics.

        Usage: clear

        Example:
            >>> clear
            Metrics cleared.
        """
        if not self._require_processor():
            return

        try:
            self.processor.reset_metrics()
            print("✓ Metrics cleared")
        except Exception as e:
            print(f"Error clearing metrics: {e}")

    def do_reset(self, arg: str) -> None:
        """
        Reset metrics collection.

        Usage: reset

        Example:
            >>> reset
            Metrics reset.
        """
        if not self._require_processor():
            return

        try:
            self.processor.reset_metrics()
            self.processor.enable_metrics()
            print("✓ Metrics reset and re-enabled")
        except Exception as e:
            print(f"Error resetting metrics: {e}")

    def do_quit(self, arg: str) -> bool:
        """
        Exit the REPL.

        Usage: quit

        Example:
            >>> quit
            Goodbye!
        """
        print("\nGoodbye!")
        return True

    def do_exit(self, arg: str) -> bool:
        """Alias for quit."""
        return self.do_quit(arg)

    def do_EOF(self, arg: str) -> bool:
        """Handle Ctrl+D."""
        print()  # New line after ^D
        return self.do_quit(arg)

    # =========================================================================
    # COMPLETION SUPPORT
    # =========================================================================

    def completedefault(self, text: str, line: str, begidx: int, endidx: int) -> List[str]:
        """
        Provide completion for document IDs in commands that accept them.
        """
        if not self.processor:
            return []

        # Commands that take doc_id as argument
        doc_commands = ['patterns', 'similar']

        # Get the command
        parts = line.split()
        if not parts:
            return []

        cmd = parts[0]

        # If command takes doc_id, complete with document IDs
        if cmd in doc_commands and len(parts) >= 1:
            docs = list(self.processor.documents.keys())
            return [d for d in docs if d.startswith(text)]

        return []

    def complete_load(self, text: str, line: str, begidx: int, endidx: int) -> List[str]:
        """Complete file paths for load command."""
        # Simple file completion - just list .pkl files in current dir
        import glob
        matches = glob.glob(text + '*.pkl')
        return matches

    def complete_save(self, text: str, line: str, begidx: int, endidx: int) -> List[str]:
        """Complete file paths for save command."""
        import glob
        matches = glob.glob(text + '*.pkl')
        return matches

    def complete_export(self, text: str, line: str, begidx: int, endidx: int) -> List[str]:
        """Complete types for export command."""
        parts = line.split()
        if len(parts) == 3 or (len(parts) == 2 and not line.endswith(' ')):
            # Completing export type
            types = ['json', 'graph', 'embeddings', 'relations']
            return [t for t in types if t.startswith(text)]
        return []

    def complete_compute(self, text: str, line: str, begidx: int, endidx: int) -> List[str]:
        """Complete computation types."""
        types = ['all', 'tfidf', 'pagerank', 'concepts', 'semantics', 'embeddings']
        return [t for t in types if t.startswith(text)]

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _require_processor(self) -> bool:
        """Check if processor is loaded, print error if not."""
        if not self.processor:
            print("Error: No corpus loaded. Use 'load <file>' first.")
            return False
        return True

    def emptyline(self) -> bool:
        """Do nothing on empty line (override default repeat behavior)."""
        return False

    def default(self, line: str) -> None:
        """Handle unknown commands."""
        print(f"Unknown command: {line.split()[0] if line else ''}")
        print("Type 'help' for available commands.")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Cortical Text Processor REPL',
        epilog="""
Examples:
  %(prog)s                       # Start REPL without corpus
  %(prog)s corpus_dev.pkl        # Start with corpus loaded
  %(prog)s my_corpus.pkl         # Start with custom corpus
        """
    )
    parser.add_argument('corpus', nargs='?', help='Corpus file to load (optional)')
    args = parser.parse_args()

    # Create and run REPL
    repl = CorticalREPL(corpus_file=args.corpus)
    try:
        repl.cmdloop()
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
        sys.exit(0)


if __name__ == '__main__':
    main()
