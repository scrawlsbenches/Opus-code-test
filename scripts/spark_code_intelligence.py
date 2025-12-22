#!/usr/bin/env python3
"""
SparkCodeIntelligence CLI - Hybrid AST + N-gram Code Intelligence Engine
=========================================================================

A powerful code intelligence system combining:
1. AST-based structural analysis (classes, functions, calls, imports)
2. Code-aware tokenization (preserves punctuation and operators)
3. N-gram language model for pattern prediction
4. Semantic queries (find callers, inheritance, related code)

This is SparkSLM v2 - designed specifically for code understanding.

Usage:
    python scripts/spark_code_intelligence.py train [--verbose]
    python scripts/spark_code_intelligence.py complete "self."
    python scripts/spark_code_intelligence.py find-callers "method_name"
    python scripts/spark_code_intelligence.py find-class "ClassName"
    python scripts/spark_code_intelligence.py inheritance "ClassName"
    python scripts/spark_code_intelligence.py imports "module_name"
    python scripts/spark_code_intelligence.py related "file.py"
    python scripts/spark_code_intelligence.py query "natural language query"
    python scripts/spark_code_intelligence.py interactive
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add project root to path
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cortical.spark import SparkCodeIntelligence


# =============================================================================
# COVERAGE ESTIMATOR - Fast coverage prediction
# =============================================================================

class CoverageEstimator:
    """
    Fast coverage estimator based on test file relationships.

    Instead of running the full test suite (which can take many minutes),
    this estimates coverage by analyzing:
    1. Test file <-> source file naming conventions
    2. Import relationships between tests and sources
    3. Historical patterns (if available)
    """

    def __init__(self, engine: SparkCodeIntelligence):
        """
        Initialize coverage estimator.

        Args:
            engine: Trained SparkCodeIntelligence instance
        """
        self.engine = engine
        self.test_source_map: Dict[str, List[str]] = {}  # test -> sources it tests
        self.source_test_map: Dict[str, List[str]] = {}  # source -> tests that cover it
        self.coverage_history: Dict[str, float] = {}     # file -> last known coverage

    def analyze(self, verbose: bool = True) -> 'CoverageEstimator':
        """
        Analyze test-source relationships.

        Returns:
            self for method chaining
        """
        if verbose:
            print("Analyzing test-source relationships...")

        # Find all test files and source files
        test_files = set()
        source_files = set()

        for file_path in self.engine.ast_index.functions_by_file.keys():
            filename = Path(file_path).name

            # Test file detection (broad)
            if '/tests/' in file_path or filename.startswith('test_'):
                test_files.add(file_path)
            elif file_path.endswith('.py'):
                # Source file if it's in cortical/ but not a test
                if '/cortical/' in file_path:
                    if not filename.startswith('test_'):
                        source_files.add(file_path)

        test_files = list(test_files)
        source_files = list(source_files)

        if verbose:
            print(f"  Found {len(test_files)} test files")
            print(f"  Found {len(source_files)} source files")

        # Build test-source relationships
        for test_file in test_files:
            sources = self._find_tested_sources(test_file, source_files)
            self.test_source_map[test_file] = sources
            for src in sources:
                if src not in self.source_test_map:
                    self.source_test_map[src] = []
                self.source_test_map[src].append(test_file)

        # Calculate coverage based on test existence
        covered = len([s for s in source_files if s in self.source_test_map])
        if verbose:
            print(f"  Source files with tests: {covered}/{len(source_files)}")
            if source_files:
                print(f"  Estimated file coverage: {100*covered/len(source_files):.1f}%")
            else:
                print(f"  No source files found in cortical/")

        return self

    def _find_tested_sources(self, test_file: str, source_files: List[str]) -> List[str]:
        """Find source files that a test file likely tests."""
        tested = []

        # Extract test file name pattern
        test_name = Path(test_file).stem  # e.g., "test_processor"

        # Pattern 1: test_X.py tests X.py
        if test_name.startswith('test_'):
            target = test_name[5:]  # Remove "test_" prefix
            for src in source_files:
                src_stem = Path(src).stem
                if src_stem == target or src_stem.endswith(f'/{target}'):
                    tested.append(src)

        # Pattern 2: Check imports in the test file
        for imp in self.engine.ast_index.imports:
            if imp.file_path == test_file:
                # Look for source files matching the import
                for src in source_files:
                    src_module = Path(src).stem
                    if src_module in imp.module or imp.module.endswith(src_module):
                        if src not in tested:
                            tested.append(src)

        return tested

    def estimate_coverage(self, changed_files: List[str] = None) -> Dict[str, Any]:
        """
        Estimate current coverage without running tests.

        Args:
            changed_files: Optional list of recently changed files

        Returns:
            Dict with coverage estimates
        """
        # Count files with test coverage
        total_source = len([f for f in self.engine.ast_index.functions_by_file
                          if '/cortical/' in f and not Path(f).name.startswith('test_')])

        covered_source = len(self.source_test_map)

        if total_source == 0:
            return {
                'estimated_line_coverage': 0,
                'file_coverage_rate': 0,
                'source_files': 0,
                'covered_files': 0,
                'uncovered_files': 0,
                'change_impact': None,
                'confidence': 'low',
            }

        # Estimate line coverage based on function coverage
        # Heuristic: files with tests typically have 70-90% line coverage
        # Files without tests have 0%
        avg_coverage_with_tests = 0.80  # Assume 80% for files with tests
        file_coverage_rate = covered_source / max(total_source, 1)

        estimated_line_coverage = file_coverage_rate * avg_coverage_with_tests * 100

        # Adjust for changed files
        if changed_files:
            changed_coverage = []
            for cf in changed_files:
                if cf in self.source_test_map:
                    changed_coverage.append(avg_coverage_with_tests)
                else:
                    changed_coverage.append(0.0)
            if changed_coverage:
                change_impact = sum(changed_coverage) / len(changed_coverage)
            else:
                change_impact = 0.5
        else:
            change_impact = None

        return {
            'estimated_line_coverage': round(estimated_line_coverage, 1),
            'file_coverage_rate': round(file_coverage_rate * 100, 1),
            'source_files': total_source,
            'covered_files': covered_source,
            'uncovered_files': total_source - covered_source,
            'change_impact': round(change_impact * 100, 1) if change_impact else None,
            'confidence': 'medium',  # Can improve with historical data
        }

    def find_uncovered(self, top_n: int = 20) -> List[Tuple[str, int]]:
        """
        Find source files without test coverage.

        Returns:
            List of (file_path, function_count) for uncovered files
        """
        uncovered = []

        for file_path, funcs in self.engine.ast_index.functions_by_file.items():
            # Only check cortical/ source files
            if '/cortical/' not in file_path:
                continue
            if Path(file_path).name.startswith('test_'):
                continue
            if not file_path.endswith('.py'):
                continue
            if file_path not in self.source_test_map:
                uncovered.append((file_path, len(funcs)))

        # Sort by function count (more functions = higher priority)
        uncovered.sort(key=lambda x: -x[1])
        return uncovered[:top_n]

    def suggest_tests(self, file_path: str) -> List[str]:
        """
        Suggest what tests should cover a file.

        Args:
            file_path: Source file path

        Returns:
            List of suggested test patterns
        """
        suggestions = []

        # Get functions in this file
        funcs = self.engine.ast_index.functions_by_file.get(file_path, [])

        file_stem = Path(file_path).stem

        # Suggest test file name
        suggestions.append(f"tests/test_{file_stem}.py")
        suggestions.append(f"tests/unit/test_{file_stem}.py")

        # Suggest test methods for key functions
        for func_name in funcs[:5]:
            if func_name.startswith('_'):
                continue
            name_parts = func_name.split('.')
            method_name = name_parts[-1]
            suggestions.append(f"  def test_{method_name}(self):")

        return suggestions


# =============================================================================
# BENCHMARKS
# =============================================================================

def run_benchmarks(engine: SparkCodeIntelligence) -> Dict[str, Any]:
    """
    Run comprehensive benchmarks on SparkCodeIntelligence.

    Returns:
        Dict with benchmark results
    """
    results = {}

    # Benchmark 1: Completion latency
    completions = ['self.', 'def ', 'import ', 'from cortical', 'NGramModel.']
    completion_times = []

    for prefix in completions:
        start = time.perf_counter()
        engine.complete(prefix, top_n=10)
        elapsed = (time.perf_counter() - start) * 1000
        completion_times.append(elapsed)

    results['completion'] = {
        'avg_ms': round(sum(completion_times) / len(completion_times), 2),
        'min_ms': round(min(completion_times), 2),
        'max_ms': round(max(completion_times), 2),
    }

    # Benchmark 2: Caller search
    test_funcs = ['compute_pagerank', 'process_document', 'expand_query', 'find_callers']
    caller_times = []

    for func in test_funcs:
        start = time.perf_counter()
        engine.find_callers(func)
        elapsed = (time.perf_counter() - start) * 1000
        caller_times.append(elapsed)

    results['find_callers'] = {
        'avg_ms': round(sum(caller_times) / len(caller_times), 2),
        'min_ms': round(min(caller_times), 2),
        'max_ms': round(max(caller_times), 2),
    }

    # Benchmark 3: Class lookup
    test_classes = ['NGramModel', 'CorticalTextProcessor', 'HierarchicalLayer', 'ASTIndex']
    class_times = []

    for cls in test_classes:
        start = time.perf_counter()
        engine.find_class(cls)
        elapsed = (time.perf_counter() - start) * 1000
        class_times.append(elapsed)

    results['find_class'] = {
        'avg_ms': round(sum(class_times) / len(class_times), 2),
        'min_ms': round(min(class_times), 2),
        'max_ms': round(max(class_times), 2),
    }

    # Benchmark 4: Natural language query
    test_queries = [
        'what calls compute_tfidf',
        'where is PageRank',
        'class that inherits Layer',
    ]
    query_times = []

    for q in test_queries:
        start = time.perf_counter()
        engine.query(q)
        elapsed = (time.perf_counter() - start) * 1000
        query_times.append(elapsed)

    results['query'] = {
        'avg_ms': round(sum(query_times) / len(query_times), 2),
        'min_ms': round(min(query_times), 2),
        'max_ms': round(max(query_times), 2),
    }

    # Benchmark 5: Related files
    test_files = list(engine._file_tokens.keys())[:3]
    related_times = []

    for f in test_files:
        start = time.perf_counter()
        engine.find_related_files(f, top_n=10)
        elapsed = (time.perf_counter() - start) * 1000
        related_times.append(elapsed)

    if related_times:
        results['find_related'] = {
            'avg_ms': round(sum(related_times) / len(related_times), 2),
            'min_ms': round(min(related_times), 2),
            'max_ms': round(max(related_times), 2),
        }
    else:
        results['find_related'] = {'avg_ms': 0, 'min_ms': 0, 'max_ms': 0}

    # Memory estimate
    model_size = sys.getsizeof(engine.ngram_model.counts) / (1024 * 1024)
    results['memory'] = {
        'ngram_counts_mb': round(model_size, 1),
        'contexts': len(engine.ngram_model.counts),
        'vocab': len(engine.ngram_model.vocab),
    }

    return results


def print_benchmarks(results: Dict[str, Any]) -> None:
    """Print benchmark results in a nice format."""
    print("\n" + "=" * 60)
    print("SparkCodeIntelligence Benchmarks")
    print("=" * 60)

    print("\n📊 Latency (milliseconds):")
    print(f"  {'Operation':<20} {'Avg':>10} {'Min':>10} {'Max':>10}")
    print("  " + "-" * 50)

    for op in ['completion', 'find_callers', 'find_class', 'query', 'find_related']:
        if op in results:
            r = results[op]
            print(f"  {op:<20} {r['avg_ms']:>10.2f} {r['min_ms']:>10.2f} {r['max_ms']:>10.2f}")

    print("\n💾 Memory:")
    if 'memory' in results:
        m = results['memory']
        print(f"  N-gram counts: {m['ngram_counts_mb']:.1f} MB")
        print(f"  Contexts: {m['contexts']:,}")
        print(f"  Vocabulary: {m['vocab']:,}")

    print()


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Main CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="SparkCodeIntelligence - Hybrid AST + N-gram Code Intelligence"
    )
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train on codebase')
    train_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    train_parser.add_argument('--extensions', '-e', nargs='+', default=['.py'], help='File extensions')

    # Complete command
    complete_parser = subparsers.add_parser('complete', help='Code completion')
    complete_parser.add_argument('prefix', help='Code prefix to complete')
    complete_parser.add_argument('--top', '-n', type=int, default=10, help='Number of suggestions')

    # Find callers command
    callers_parser = subparsers.add_parser('find-callers', help='Find function callers')
    callers_parser.add_argument('function', help='Function name')

    # Find class command
    class_parser = subparsers.add_parser('find-class', help='Find class info')
    class_parser.add_argument('name', help='Class name')

    # Inheritance command
    inherit_parser = subparsers.add_parser('inheritance', help='Show inheritance tree')
    inherit_parser.add_argument('class_name', help='Class name')

    # Imports command
    imports_parser = subparsers.add_parser('imports', help='Find imports of module')
    imports_parser.add_argument('module', help='Module name')

    # Related files command
    related_parser = subparsers.add_parser('related', help='Find related files')
    related_parser.add_argument('file', help='File path')
    related_parser.add_argument('--top', '-n', type=int, default=10, help='Number of results')

    # Query command
    query_parser = subparsers.add_parser('query', help='Natural language query')
    query_parser.add_argument('question', help='Query string')

    # Interactive command
    subparsers.add_parser('interactive', help='Interactive mode')

    # Stats command
    subparsers.add_parser('stats', help='Show statistics')

    # Benchmark command
    subparsers.add_parser('benchmark', help='Run performance benchmarks')

    # Coverage commands
    coverage_parser = subparsers.add_parser('coverage', help='Estimate code coverage')
    coverage_parser.add_argument('--uncovered', '-u', action='store_true', help='Show uncovered files')
    coverage_parser.add_argument('--suggest', '-s', type=str, help='Suggest tests for a file')
    coverage_parser.add_argument('--changed', '-c', nargs='+', help='Changed files to analyze')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Initialize engine
    engine = SparkCodeIntelligence()

    # Load or train
    if args.command == 'train':
        engine.train(extensions=args.extensions, verbose=args.verbose)
        engine.save()
        print(f"\n💾 Model saved to {SparkCodeIntelligence.MODEL_FILE}")
    else:
        # Try to load existing model
        try:
            engine.load()
        except FileNotFoundError:
            print("❌ No trained model found. Run 'train' first.")
            print(f"   python {__file__} train")
            return

        if args.command == 'complete':
            results = engine.complete(args.prefix, top_n=args.top)
            print(f"\nCompletions for '{args.prefix}':")
            for suggestion, conf, source in results:
                print(f"  {suggestion:<30} ({conf:.2f}) [{source}]")

        elif args.command == 'find-callers':
            results = engine.find_callers(args.function)
            print(f"\nCallers of '{args.function}':")
            if results:
                for r in results:
                    print(f"  {r['caller']:<40} {r['file']}:{r['line']}")
            else:
                print("  (none found)")

        elif args.command == 'find-class':
            result = engine.find_class(args.name)
            if result:
                print(f"\nClass: {result['name']}")
                print(f"  File: {result['file']}:{result['line']}")
                print(f"  Bases: {', '.join(result['bases']) or 'object'}")
                print(f"  Methods ({len(result['methods'])}):")
                for m in result['methods'][:15]:
                    print(f"    - {m}()")
                if len(result['methods']) > 15:
                    print(f"    ... and {len(result['methods']) - 15} more")
                print(f"  Attributes ({len(result['attributes'])}):")
                for a in sorted(result['attributes'])[:15]:
                    print(f"    - {a}")
            else:
                print(f"Class '{args.name}' not found")

        elif args.command == 'inheritance':
            result = engine.get_inheritance(args.class_name)
            print(f"\nInheritance for '{args.class_name}':")
            print(f"  Parents: {', '.join(result['parents']) or 'object'}")
            print(f"  Children:")
            if result['children']:
                def print_tree(node, indent=4):
                    print(' ' * indent + f"- {node['name']}")
                    for child in node.get('children', []):
                        print_tree(child, indent + 2)
                for child in result['children']:
                    print_tree(child)
            else:
                print("    (none)")

        elif args.command == 'imports':
            results = engine.find_imports(args.module)
            print(f"\nFiles importing '{args.module}':")
            if results:
                for r in results:
                    print(f"  {r['file']}:{r['line']}")
            else:
                print("  (none found)")

        elif args.command == 'related':
            results = engine.find_related_files(args.file, top_n=args.top)
            print(f"\nFiles related to '{args.file}':")
            if results:
                for path, score in results:
                    print(f"  {path:<50} (score: {score:.2f})")
            else:
                print("  (none found)")

        elif args.command == 'query':
            results = engine.query(args.question)
            print(f"\nResults for: '{args.question}'")
            if results:
                for r in results:
                    if r.get('type') == 'callers':
                        print(f"  Callers of {r['function']}:")
                        for caller in r['results'][:10]:
                            print(f"    - {caller['caller']} ({caller['file']}:{caller['line']})")
                    elif r.get('type') == 'inheritance':
                        print(f"  Classes inheriting from {r['parent']}:")
                        for child in r['children']:
                            print(f"    - {child}")
                    else:
                        print(f"  {r}")
            else:
                print("  (no results)")

        elif args.command == 'interactive':
            engine.interactive()

        elif args.command == 'stats':
            stats = engine.get_stats()
            print("\nSparkCodeIntelligence Statistics:")
            print("=" * 40)
            for key, value in stats.items():
                print(f"  {key}: {value:,}")

        elif args.command == 'benchmark':
            print("Running benchmarks...")
            results = run_benchmarks(engine)
            print_benchmarks(results)

        elif args.command == 'coverage':
            estimator = CoverageEstimator(engine)
            estimator.analyze(verbose=True)

            if args.suggest:
                print(f"\nSuggested tests for '{args.suggest}':")
                suggestions = estimator.suggest_tests(args.suggest)
                for s in suggestions:
                    print(f"  {s}")

            elif args.uncovered:
                print("\nUncovered source files (by function count):")
                uncovered = estimator.find_uncovered(top_n=20)
                for path, func_count in uncovered:
                    print(f"  {path:<60} ({func_count} functions)")

            else:
                print("\nCoverage Estimate:")
                estimate = estimator.estimate_coverage(changed_files=args.changed)
                print(f"  Estimated line coverage: {estimate['estimated_line_coverage']}%")
                print(f"  File coverage rate: {estimate['file_coverage_rate']}%")
                print(f"  Source files: {estimate['source_files']}")
                print(f"  Files with tests: {estimate['covered_files']}")
                print(f"  Files without tests: {estimate['uncovered_files']}")
                if estimate['change_impact'] is not None:
                    print(f"  Changed files coverage: {estimate['change_impact']}%")
                print(f"  Confidence: {estimate['confidence']}")


if __name__ == '__main__':
    main()
