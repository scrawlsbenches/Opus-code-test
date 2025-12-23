#!/usr/bin/env python3
"""
SparkSLM Code Assistant - Intelligent code analysis and suggestions.

A practical tool that uses SparkSLM to:
1. Suggest code completions based on learned patterns
2. Find similar code patterns across the codebase
3. Detect unusual/anomalous code that might need review
4. Suggest related files when working on a task
5. Generate smart search queries

Usage:
    python scripts/spark_code_assistant.py train              # Train on codebase
    python scripts/spark_code_assistant.py complete "def test"  # Suggest completions
    python scripts/spark_code_assistant.py similar "pagerank"   # Find similar patterns
    python scripts/spark_code_assistant.py anomaly FILE         # Check file for anomalies
    python scripts/spark_code_assistant.py related FILE         # Find related files
    python scripts/spark_code_assistant.py search "query"       # Smart search with expansion
    python scripts/spark_code_assistant.py interactive          # Interactive mode
"""

import argparse
import json
import os
import pickle
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cortical.spark import NGramModel, SparkPredictor, AnomalyDetector


class CodePattern:
    """Represents a learned code pattern."""

    def __init__(self, pattern: str, frequency: int, files: Set[str]):
        self.pattern = pattern
        self.frequency = frequency
        self.files = files

    def __repr__(self):
        return f"CodePattern('{self.pattern}', freq={self.frequency}, files={len(self.files)})"


class FileRelationship:
    """Tracks relationships between files based on shared patterns."""

    def __init__(self):
        self.file_patterns: Dict[str, Set[str]] = defaultdict(set)
        self.pattern_files: Dict[str, Set[str]] = defaultdict(set)
        self.file_imports: Dict[str, Set[str]] = defaultdict(set)

    def add_pattern(self, file_path: str, pattern: str):
        self.file_patterns[file_path].add(pattern)
        self.pattern_files[pattern].add(file_path)

    def add_import(self, file_path: str, imported: str):
        self.file_imports[file_path].add(imported)

    def get_related_files(self, file_path: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """Find files related to the given file based on shared patterns."""
        if file_path not in self.file_patterns:
            return []

        my_patterns = self.file_patterns[file_path]
        scores = Counter()

        for pattern in my_patterns:
            for other_file in self.pattern_files[pattern]:
                if other_file != file_path:
                    scores[other_file] += 1

        # Normalize by total patterns
        results = []
        for other_file, shared in scores.most_common(top_n):
            total = len(my_patterns | self.file_patterns[other_file])
            similarity = shared / total if total > 0 else 0
            results.append((other_file, similarity))

        return results


class SparkCodeAssistant:
    """Intelligent code assistant powered by SparkSLM."""

    MODEL_FILE = ".spark_assistant_model.pkl"

    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root).resolve()
        self.model_path = self.repo_root / self.MODEL_FILE

        # Core models
        self.ngram: Optional[NGramModel] = None
        self.predictor: Optional[SparkPredictor] = None
        self.anomaly_detector: Optional[AnomalyDetector] = None

        # Metadata
        self.file_contents: Dict[str, str] = {}
        self.file_relationships = FileRelationship()
        self.code_patterns: Dict[str, CodePattern] = {}
        self.function_index: Dict[str, List[str]] = defaultdict(list)  # func_name -> [file_paths]
        self.class_index: Dict[str, List[str]] = defaultdict(list)

        # Stats
        self.total_files = 0
        self.total_tokens = 0
        self.trained = False

    def train(self, extensions: List[str] = None, verbose: bool = True):
        """Train the assistant on the codebase."""
        if extensions is None:
            extensions = ['.py', '.md', '.txt', '.rst']

        if verbose:
            print("=" * 60)
            print("SparkSLM Code Assistant - Training")
            print("=" * 60)
            print(f"\nScanning {self.repo_root} for: {', '.join(extensions)}")

        start_time = time.perf_counter()

        # Collect files
        documents = []
        for ext in extensions:
            for file_path in self.repo_root.rglob(f"*{ext}"):
                # Skip hidden directories and common exclusions
                if any(part.startswith('.') for part in file_path.parts):
                    continue
                if any(skip in str(file_path) for skip in ['__pycache__', 'node_modules', '.git', 'venv', 'env']):
                    continue

                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    rel_path = str(file_path.relative_to(self.repo_root))

                    self.file_contents[rel_path] = content
                    documents.append(content)
                    self.total_files += 1

                    # Extract patterns for Python files
                    if ext == '.py':
                        self._extract_python_patterns(rel_path, content)

                except Exception as e:
                    if verbose:
                        print(f"  Warning: Could not read {file_path}: {e}")

        if not documents:
            print("No documents found!")
            return

        if verbose:
            print(f"\nProcessing {self.total_files} files...")

        # Train n-gram model
        self.ngram = NGramModel(n=3)
        self.ngram.train(documents)
        self.total_tokens = self.ngram.total_tokens

        # Create predictor
        self.predictor = SparkPredictor(ngram_order=3)
        self.predictor.ngram = self.ngram
        self.predictor._trained = True

        # Train anomaly detector
        self.anomaly_detector = AnomalyDetector(ngram_model=self.ngram)
        # Calibrate with sample code snippets
        samples = [doc[:500] for doc in documents[:50] if len(doc) > 100]
        if samples:
            self.anomaly_detector.calibrate(samples)

        # Build code patterns
        self._build_code_patterns()

        train_time = time.perf_counter() - start_time
        self.trained = True

        if verbose:
            print(f"\n{'='*60}")
            print("Training Complete")
            print(f"{'='*60}")
            print(f"  Files:       {self.total_files:,}")
            print(f"  Tokens:      {self.total_tokens:,}")
            print(f"  Vocabulary:  {len(self.ngram.vocab):,}")
            print(f"  Contexts:    {len(self.ngram.counts):,}")
            print(f"  Functions:   {sum(len(v) for v in self.function_index.values()):,}")
            print(f"  Classes:     {sum(len(v) for v in self.class_index.values()):,}")
            print(f"  Time:        {train_time:.2f}s")
            print(f"  Throughput:  {self.total_tokens/train_time:,.0f} tokens/sec")

        # Save model
        self.save()
        if verbose:
            print(f"\nModel saved to {self.model_path}")

    def _extract_python_patterns(self, file_path: str, content: str):
        """Extract patterns from Python code."""
        lines = content.split('\n')

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Extract function definitions
            if stripped.startswith('def '):
                func_name = stripped[4:].split('(')[0].strip()
                self.function_index[func_name].append(file_path)
                self.file_relationships.add_pattern(file_path, f"def:{func_name}")

            # Extract class definitions
            elif stripped.startswith('class '):
                class_name = stripped[6:].split('(')[0].split(':')[0].strip()
                self.class_index[class_name].append(file_path)
                self.file_relationships.add_pattern(file_path, f"class:{class_name}")

            # Extract imports
            elif stripped.startswith('import ') or stripped.startswith('from '):
                parts = stripped.replace('import ', '').replace('from ', '').split()
                if parts:
                    module = parts[0].split('.')[0]
                    self.file_relationships.add_import(file_path, module)
                    self.file_relationships.add_pattern(file_path, f"import:{module}")

            # Extract common patterns (2-3 word sequences)
            words = stripped.split()[:5]
            if len(words) >= 2:
                pattern = ' '.join(words[:2])
                self.file_relationships.add_pattern(file_path, pattern)

    def _build_code_patterns(self):
        """Build index of common code patterns."""
        pattern_counts = Counter()
        pattern_files = defaultdict(set)

        for file_path, patterns in self.file_relationships.file_patterns.items():
            for pattern in patterns:
                pattern_counts[pattern] += 1
                pattern_files[pattern].add(file_path)

        # Keep patterns that appear in multiple files
        for pattern, count in pattern_counts.most_common(1000):
            if count >= 2:
                self.code_patterns[pattern] = CodePattern(
                    pattern, count, pattern_files[pattern]
                )

    def save(self):
        """Save the trained model."""
        data = {
            'ngram': self.ngram,
            'file_contents': self.file_contents,
            'file_relationships': self.file_relationships,
            'code_patterns': self.code_patterns,
            'function_index': dict(self.function_index),
            'class_index': dict(self.class_index),
            'total_files': self.total_files,
            'total_tokens': self.total_tokens,
        }
        with open(self.model_path, 'wb') as f:
            pickle.dump(data, f)

    def load(self) -> bool:
        """Load a previously trained model."""
        if not self.model_path.exists():
            return False

        try:
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)

            self.ngram = data['ngram']
            self.file_contents = data['file_contents']
            self.file_relationships = data['file_relationships']
            self.code_patterns = data['code_patterns']
            self.function_index = defaultdict(list, data['function_index'])
            self.class_index = defaultdict(list, data['class_index'])
            self.total_files = data['total_files']
            self.total_tokens = data['total_tokens']

            # Rebuild predictor and anomaly detector
            self.predictor = SparkPredictor(ngram_order=3)
            self.predictor.ngram = self.ngram
            self.predictor._trained = True

            self.anomaly_detector = AnomalyDetector(ngram_model=self.ngram)
            samples = [c[:500] for c in list(self.file_contents.values())[:50] if len(c) > 100]
            if samples:
                self.anomaly_detector.calibrate(samples)

            self.trained = True
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def complete(self, prefix: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """Suggest completions for the given code prefix."""
        if not self.trained:
            return []

        words = prefix.lower().split()
        if len(words) < 2:
            words = ['<s>'] + words

        context = tuple(words[-2:])
        predictions = self.ngram.predict(list(context), top_k=top_n)

        return predictions

    def complete_sequence(self, prefix: str, length: int = 5) -> str:
        """Generate a sequence completion."""
        if not self.trained:
            return prefix

        words = prefix.lower().split()
        sequence = self.ngram.predict_sequence(words, length=length)
        return prefix + ' ' + ' '.join(sequence)

    def find_similar_patterns(self, query: str, top_n: int = 10) -> List[Tuple[str, int, Set[str]]]:
        """Find code patterns similar to the query."""
        if not self.trained:
            return []

        query_lower = query.lower()
        results = []

        for pattern, cp in self.code_patterns.items():
            if query_lower in pattern.lower():
                results.append((cp.pattern, cp.frequency, cp.files))

        # Sort by frequency
        results.sort(key=lambda x: -x[1])
        return results[:top_n]

    def check_anomaly(self, file_path: str) -> Dict:
        """Check a file for anomalous code patterns."""
        if not self.trained:
            return {'error': 'Model not trained'}

        path = Path(file_path)
        if not path.exists():
            # Try relative to repo root
            path = self.repo_root / file_path

        if not path.exists():
            return {'error': f'File not found: {file_path}'}

        content = path.read_text(encoding='utf-8', errors='ignore')

        # Split into chunks and analyze each
        lines = content.split('\n')
        anomalies = []

        chunk_size = 10
        for i in range(0, len(lines), chunk_size):
            chunk = '\n'.join(lines[i:i+chunk_size])
            if len(chunk.strip()) < 20:
                continue

            result = self.anomaly_detector.check(chunk)
            if result.is_anomalous:
                anomalies.append({
                    'lines': f"{i+1}-{min(i+chunk_size, len(lines))}",
                    'score': result.confidence,
                    'reasons': result.reasons,
                    'preview': chunk[:100] + '...' if len(chunk) > 100 else chunk
                })

        # Calculate overall perplexity
        perplexity = self.ngram.perplexity(content[:5000])  # First 5000 chars

        return {
            'file': str(path),
            'lines': len(lines),
            'perplexity': perplexity,
            'anomaly_count': len(anomalies),
            'anomalies': anomalies[:10],  # Top 10
        }

    def find_related_files(self, file_path: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """Find files related to the given file."""
        if not self.trained:
            return []

        # Normalize path
        try:
            rel_path = str(Path(file_path).relative_to(self.repo_root))
        except ValueError:
            rel_path = file_path

        return self.file_relationships.get_related_files(rel_path, top_n)

    def smart_search(self, query: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """Search with query expansion using learned patterns."""
        if not self.trained:
            return []

        # Get completions to expand query
        words = query.lower().split()
        expansions = set(words)

        # Add predicted continuations
        if len(words) >= 1:
            preds = self.complete(' '.join(words[-2:]) if len(words) >= 2 else words[-1], top_n=5)
            for word, _ in preds:
                expansions.add(word)

        # Search files
        scores = Counter()
        query_terms = expansions

        for file_path, content in self.file_contents.items():
            content_lower = content.lower()
            score = 0
            for term in query_terms:
                count = content_lower.count(term)
                if count > 0:
                    # TF-like scoring
                    score += count * (2 if term in words else 1)

            if score > 0:
                scores[file_path] = score

        return scores.most_common(top_n)

    def find_function(self, func_name: str) -> List[str]:
        """Find files containing a function definition."""
        if not self.trained:
            return []
        return self.function_index.get(func_name, [])

    def find_class(self, class_name: str) -> List[str]:
        """Find files containing a class definition."""
        if not self.trained:
            return []
        return self.class_index.get(class_name, [])

    def get_stats(self) -> Dict:
        """Get assistant statistics."""
        return {
            'trained': self.trained,
            'files': self.total_files,
            'tokens': self.total_tokens,
            'vocabulary': len(self.ngram.vocab) if self.ngram else 0,
            'contexts': len(self.ngram.counts) if self.ngram else 0,
            'functions': sum(len(v) for v in self.function_index.values()),
            'classes': sum(len(v) for v in self.class_index.values()),
            'patterns': len(self.code_patterns),
        }

    def interactive(self):
        """Run interactive mode."""
        print("=" * 60)
        print("SparkSLM Code Assistant - Interactive Mode")
        print("=" * 60)
        print("\nCommands:")
        print("  complete <prefix>     - Suggest completions")
        print("  sequence <prefix>     - Generate sequence")
        print("  similar <query>       - Find similar patterns")
        print("  search <query>        - Smart search")
        print("  related <file>        - Find related files")
        print("  anomaly <file>        - Check for anomalies")
        print("  func <name>           - Find function")
        print("  class <name>          - Find class")
        print("  stats                 - Show statistics")
        print("  quit                  - Exit")
        print()

        while True:
            try:
                line = input("assistant> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not line:
                continue

            parts = line.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd in ('quit', 'exit', 'q'):
                print("Goodbye!")
                break

            elif cmd == 'complete':
                if not arg:
                    print("Usage: complete <prefix>")
                    continue
                results = self.complete(arg)
                print(f"\nCompletions for '{arg}':")
                for word, prob in results:
                    print(f"  {word} ({prob:.3f})")

            elif cmd == 'sequence':
                if not arg:
                    print("Usage: sequence <prefix>")
                    continue
                result = self.complete_sequence(arg, length=8)
                print(f"\n  {result}")

            elif cmd == 'similar':
                if not arg:
                    print("Usage: similar <query>")
                    continue
                results = self.find_similar_patterns(arg)
                print(f"\nPatterns matching '{arg}':")
                for pattern, freq, files in results[:10]:
                    print(f"  {pattern} (freq={freq}, files={len(files)})")

            elif cmd == 'search':
                if not arg:
                    print("Usage: search <query>")
                    continue
                results = self.smart_search(arg)
                print(f"\nSearch results for '{arg}':")
                for file_path, score in results:
                    print(f"  {file_path} (score={score})")

            elif cmd == 'related':
                if not arg:
                    print("Usage: related <file>")
                    continue
                results = self.find_related_files(arg)
                print(f"\nFiles related to '{arg}':")
                for file_path, sim in results:
                    print(f"  {file_path} ({sim:.2%})")

            elif cmd == 'anomaly':
                if not arg:
                    print("Usage: anomaly <file>")
                    continue
                result = self.check_anomaly(arg)
                if 'error' in result:
                    print(f"Error: {result['error']}")
                else:
                    print(f"\nAnomaly report for '{result['file']}':")
                    print(f"  Lines: {result['lines']}")
                    print(f"  Perplexity: {result['perplexity']:.2f}")
                    print(f"  Anomalies found: {result['anomaly_count']}")
                    for a in result['anomalies'][:5]:
                        print(f"\n  Lines {a['lines']} (score={a['score']:.2f}):")
                        print(f"    {a['preview'][:60]}...")

            elif cmd == 'func':
                if not arg:
                    print("Usage: func <name>")
                    continue
                results = self.find_function(arg)
                print(f"\nFunction '{arg}' found in:")
                for f in results:
                    print(f"  {f}")

            elif cmd == 'class':
                if not arg:
                    print("Usage: class <name>")
                    continue
                results = self.find_class(arg)
                print(f"\nClass '{arg}' found in:")
                for f in results:
                    print(f"  {f}")

            elif cmd == 'stats':
                stats = self.get_stats()
                print("\nAssistant Statistics:")
                for k, v in stats.items():
                    if isinstance(v, int):
                        print(f"  {k}: {v:,}")
                    else:
                        print(f"  {k}: {v}")

            else:
                print(f"Unknown command: {cmd}")

            print()


def main():
    parser = argparse.ArgumentParser(
        description="SparkSLM Code Assistant - Intelligent code analysis"
    )
    parser.add_argument(
        'command',
        choices=['train', 'complete', 'sequence', 'similar', 'search',
                 'related', 'anomaly', 'func', 'class', 'stats', 'interactive'],
        help="Command to run"
    )
    parser.add_argument(
        'argument',
        nargs='?',
        help="Command argument (query, file path, etc.)"
    )
    parser.add_argument(
        '--top', '-n',
        type=int,
        default=10,
        help="Number of results to show"
    )
    parser.add_argument(
        '--repo',
        default='.',
        help="Repository root directory"
    )

    args = parser.parse_args()

    assistant = SparkCodeAssistant(args.repo)

    if args.command == 'train':
        assistant.train(verbose=True)
        return

    # Load model for other commands
    if not assistant.load():
        print("No trained model found. Run 'train' first:")
        print("  python scripts/spark_code_assistant.py train")
        return

    if args.command == 'interactive':
        assistant.interactive()

    elif args.command == 'complete':
        if not args.argument:
            print("Usage: complete <prefix>")
            return
        results = assistant.complete(args.argument, top_n=args.top)
        print(f"Completions for '{args.argument}':")
        for word, prob in results:
            print(f"  {word} ({prob:.3f})")

    elif args.command == 'sequence':
        if not args.argument:
            print("Usage: sequence <prefix>")
            return
        result = assistant.complete_sequence(args.argument, length=8)
        print(f"Sequence: {result}")

    elif args.command == 'similar':
        if not args.argument:
            print("Usage: similar <query>")
            return
        results = assistant.find_similar_patterns(args.argument, top_n=args.top)
        print(f"Patterns matching '{args.argument}':")
        for pattern, freq, files in results:
            print(f"  {pattern}")
            print(f"    frequency: {freq}, files: {len(files)}")
            for f in list(files)[:3]:
                print(f"      - {f}")

    elif args.command == 'search':
        if not args.argument:
            print("Usage: search <query>")
            return
        results = assistant.smart_search(args.argument, top_n=args.top)
        print(f"Search results for '{args.argument}':")
        for file_path, score in results:
            print(f"  {file_path} (score={score})")

    elif args.command == 'related':
        if not args.argument:
            print("Usage: related <file>")
            return
        results = assistant.find_related_files(args.argument, top_n=args.top)
        print(f"Files related to '{args.argument}':")
        for file_path, sim in results:
            print(f"  {file_path} ({sim:.1%} similarity)")

    elif args.command == 'anomaly':
        if not args.argument:
            print("Usage: anomaly <file>")
            return
        result = assistant.check_anomaly(args.argument)
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Anomaly Report: {result['file']}")
            print(f"  Lines: {result['lines']}")
            print(f"  Perplexity: {result['perplexity']:.2f}")
            print(f"  Anomalies: {result['anomaly_count']}")
            if result['anomalies']:
                print("\n  Suspicious sections:")
                for a in result['anomalies']:
                    print(f"\n  Lines {a['lines']} (score={a['score']:.2f})")
                    for reason in a['reasons']:
                        print(f"    - {reason}")

    elif args.command == 'func':
        if not args.argument:
            print("Usage: func <name>")
            return
        results = assistant.find_function(args.argument)
        if results:
            print(f"Function '{args.argument}' defined in:")
            for f in results:
                print(f"  {f}")
        else:
            print(f"Function '{args.argument}' not found")

    elif args.command == 'class':
        if not args.argument:
            print("Usage: class <name>")
            return
        results = assistant.find_class(args.argument)
        if results:
            print(f"Class '{args.argument}' defined in:")
            for f in results:
                print(f"  {f}")
        else:
            print(f"Class '{args.argument}' not found")

    elif args.command == 'stats':
        stats = assistant.get_stats()
        print("SparkSLM Code Assistant Statistics")
        print("=" * 40)
        for k, v in stats.items():
            if isinstance(v, int):
                print(f"  {k}: {v:,}")
            else:
                print(f"  {k}: {v}")


if __name__ == '__main__':
    main()
