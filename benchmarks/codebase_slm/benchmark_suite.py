#!/usr/bin/env python3
"""
Benchmark Suite for Repository-Native SLM.

Tracks model performance over time across multiple dimensions:
- File location accuracy
- Concept understanding
- Code completion quality
- Command recall
- Perplexity on codebase text

Usage:
    # Run all benchmarks
    python -m benchmarks.codebase_slm.benchmark_suite --full

    # Quick benchmark
    python -m benchmarks.codebase_slm.benchmark_suite --quick

    # Compare with baseline
    python -m benchmarks.codebase_slm.benchmark_suite --compare baseline.json

    # Save results
    python -m benchmarks.codebase_slm.benchmark_suite --output results/run_001.json
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cortical.reasoning.prism_slm import PRISMLanguageModel


@dataclass
class BenchmarkResult:
    """Result of a single benchmark query."""
    category: str
    query: str
    expected: str
    generated: str
    term_matches: int
    total_terms: int
    match_percentage: float
    passed: bool
    latency_ms: float


@dataclass
class BenchmarkSummary:
    """Summary of benchmark run."""
    timestamp: str
    model_vocab_size: int
    model_transitions: int
    corpus_size: int
    total_queries: int
    passed_queries: int
    pass_rate: float
    avg_latency_ms: float
    avg_perplexity: float
    category_scores: Dict[str, float]
    results: List[Dict[str, Any]]


# Benchmark queries organized by category
BENCHMARK_QUERIES = {
    'file_location': [
        {'query': 'Q: Where is PageRank implemented? A:', 'expected': 'cortical analysis pagerank'},
        {'query': 'Q: Where is TF-IDF implemented? A:', 'expected': 'cortical analysis tfidf'},
        {'query': 'Q: Where is GoTManager defined? A:', 'expected': 'cortical got api'},
        {'query': 'Q: Where is the tokenizer? A:', 'expected': 'cortical tokenizer'},
        {'query': 'Q: Where is clustering implemented? A:', 'expected': 'cortical analysis clustering'},
        {'query': 'Q: Where is query expansion? A:', 'expected': 'cortical query expansion'},
        {'query': 'Q: Where is the Minicolumn class? A:', 'expected': 'cortical minicolumn'},
        {'query': 'Q: Where is persistence implemented? A:', 'expected': 'cortical persistence'},
    ],
    'concept': [
        {'query': 'Q: What is Hebbian learning? A:', 'expected': 'neurons fire together wire together'},
        {'query': 'Q: What is PRISM? A:', 'expected': 'statistical language model'},
        {'query': 'Q: What is Woven Mind? A:', 'expected': 'dual process cognitive'},
        {'query': 'Q: What is GoT? A:', 'expected': 'graph thought task tracking'},
        {'query': 'Q: What is BM25? A:', 'expected': 'scoring algorithm search'},
        {'query': 'Q: What are the 4 layers? A:', 'expected': 'tokens bigrams concepts documents'},
    ],
    'how_to': [
        {'query': 'Q: How to create a task? A:', 'expected': 'python scripts got_utils task create'},
        {'query': 'Q: How to run tests? A:', 'expected': 'pytest make test'},
        {'query': 'Q: How to search the codebase? A:', 'expected': 'python scripts search_codebase'},
        {'query': 'Q: How to index the codebase? A:', 'expected': 'python scripts index_codebase'},
    ],
    'completion': [
        {'query': 'from cortical.got import', 'expected': 'gotmanager'},
        {'query': 'from cortical.processor import', 'expected': 'corticaltextprocessor'},
        {'query': 'from cortical.analysis import', 'expected': 'pagerank tfidf clustering'},
        {'query': 'CorticalTextProcessor.', 'expected': 'process_document compute_all'},
    ],
    'process': [
        {'query': 'Q: What is the work priority order? A:', 'expected': 'security bugs features documentation'},
        {'query': 'Q: What is TDD? A:', 'expected': 'test driven development red green refactor'},
        {'query': 'Q: What should I do before writing code? A:', 'expected': 'write tests first'},
    ],
}

# Perplexity test texts (actual codebase content)
PERPLEXITY_TESTS = [
    "The Cortical Text Processor uses PageRank to compute term importance.",
    "Hebbian learning strengthens connections between co-occurring terms.",
    "Use python scripts got_utils.py task create to create a new task.",
    "The processor has 4 layers: tokens, bigrams, concepts, and documents.",
    "BM25 is the default scoring algorithm, optimized for code search.",
]


def load_model_and_corpus(corpus_path: Path, limit: int = None) -> Tuple[PRISMLanguageModel, int]:
    """Load training corpus and train model."""
    patterns = []

    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found at {corpus_path}")

    with open(corpus_path, 'r') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            try:
                data = json.loads(line.strip())
                pattern_type = data.get('pattern_type', 'unknown')

                if pattern_type == 'qa':
                    text = f"Q: {data['input_text']} A: {data['target_text']}"
                elif pattern_type == 'completion':
                    text = f"{data['input_text']} → {data['target_text']}"
                elif pattern_type == 'association':
                    text = f"{data['input_text']} relates to {data['target_text']}"
                elif pattern_type == 'explanation':
                    text = f"{data['input_text']}: {data['target_text']}"
                else:
                    text = f"{data['input_text']} {data['target_text']}"

                patterns.append(text)
            except json.JSONDecodeError:
                continue

    # Train model
    model = PRISMLanguageModel(context_size=3)
    for text in patterns:
        model.train(text)

    return model, len(patterns)


def evaluate_query(model: PRISMLanguageModel, query: str, expected: str) -> BenchmarkResult:
    """Evaluate a single query."""
    start = time.time()
    generated = model.generate(query, max_tokens=20, temperature=0.3)
    latency_ms = (time.time() - start) * 1000

    # Count term matches
    expected_terms = expected.lower().split()
    generated_lower = generated.lower()
    matches = sum(1 for term in expected_terms if term in generated_lower)
    match_pct = matches / len(expected_terms) * 100 if expected_terms else 0

    return BenchmarkResult(
        category='',  # Set by caller
        query=query,
        expected=expected,
        generated=generated,
        term_matches=matches,
        total_terms=len(expected_terms),
        match_percentage=match_pct,
        passed=match_pct >= 50,
        latency_ms=latency_ms,
    )


def run_benchmarks(
    model: PRISMLanguageModel,
    corpus_size: int,
    categories: Optional[List[str]] = None,
) -> BenchmarkSummary:
    """Run full benchmark suite."""
    results = []
    category_scores = {}
    total_latency = 0.0

    for category, queries in BENCHMARK_QUERIES.items():
        if categories and category not in categories:
            continue

        category_passed = 0
        for q in queries:
            result = evaluate_query(model, q['query'], q['expected'])
            result.category = category
            results.append(result)
            total_latency += result.latency_ms

            if result.passed:
                category_passed += 1

        category_scores[category] = category_passed / len(queries) * 100 if queries else 0

    # Calculate perplexity
    total_ppl = 0.0
    for text in PERPLEXITY_TESTS:
        try:
            ppl = model.perplexity(text)
            total_ppl += ppl
        except:
            total_ppl += 100.0  # Default high perplexity on error

    avg_ppl = total_ppl / len(PERPLEXITY_TESTS) if PERPLEXITY_TESTS else 0

    # Summary
    passed = sum(1 for r in results if r.passed)

    return BenchmarkSummary(
        timestamp=datetime.utcnow().isoformat(),
        model_vocab_size=model.vocab_size,
        model_transitions=sum(len(t) for t in model.graph._transitions.values()),
        corpus_size=corpus_size,
        total_queries=len(results),
        passed_queries=passed,
        pass_rate=passed / len(results) * 100 if results else 0,
        avg_latency_ms=total_latency / len(results) if results else 0,
        avg_perplexity=avg_ppl,
        category_scores=category_scores,
        results=[asdict(r) for r in results],
    )


def print_summary(summary: BenchmarkSummary, verbose: bool = False):
    """Print benchmark summary."""
    print("\n" + "=" * 70)
    print("REPOSITORY-NATIVE SLM BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Timestamp: {summary.timestamp}")
    print(f"Corpus Size: {summary.corpus_size:,} patterns")
    print(f"Model: {summary.model_vocab_size:,} vocab, {summary.model_transitions:,} transitions")
    print()

    print("OVERALL METRICS:")
    print(f"  Pass Rate: {summary.pass_rate:.1f}% ({summary.passed_queries}/{summary.total_queries})")
    print(f"  Avg Latency: {summary.avg_latency_ms:.2f} ms")
    print(f"  Avg Perplexity: {summary.avg_perplexity:.2f}")
    print()

    print("CATEGORY SCORES:")
    for category, score in summary.category_scores.items():
        bar = "█" * int(score / 10) + "░" * (10 - int(score / 10))
        print(f"  {category:15} [{bar}] {score:.1f}%")

    if verbose:
        print()
        print("DETAILED RESULTS:")
        for r in summary.results:
            status = "✓" if r['passed'] else "✗"
            print(f"\n  [{r['category']}] {status}")
            print(f"    Query: {r['query'][:60]}...")
            print(f"    Generated: {r['generated'][:60]}...")
            print(f"    Match: {r['term_matches']}/{r['total_terms']} ({r['match_percentage']:.0f}%)")

    print()
    print("=" * 70)


def compare_results(current: BenchmarkSummary, baseline: BenchmarkSummary):
    """Compare current results with baseline."""
    print("\n" + "=" * 70)
    print("COMPARISON WITH BASELINE")
    print("=" * 70)
    print(f"Baseline: {baseline.timestamp}")
    print(f"Current:  {current.timestamp}")
    print()

    # Overall comparison
    pass_diff = current.pass_rate - baseline.pass_rate
    latency_diff = current.avg_latency_ms - baseline.avg_latency_ms
    ppl_diff = current.avg_perplexity - baseline.avg_perplexity

    print("CHANGES:")
    print(f"  Pass Rate:   {baseline.pass_rate:.1f}% → {current.pass_rate:.1f}% ({pass_diff:+.1f}%)")
    print(f"  Latency:     {baseline.avg_latency_ms:.2f}ms → {current.avg_latency_ms:.2f}ms ({latency_diff:+.2f}ms)")
    print(f"  Perplexity:  {baseline.avg_perplexity:.2f} → {current.avg_perplexity:.2f} ({ppl_diff:+.2f})")
    print()

    print("CATEGORY CHANGES:")
    for category in current.category_scores:
        if category in baseline.category_scores:
            curr = current.category_scores[category]
            base = baseline.category_scores[category]
            diff = curr - base
            arrow = "↑" if diff > 0 else "↓" if diff < 0 else "→"
            print(f"  {category:15} {base:.1f}% → {curr:.1f}% ({arrow} {abs(diff):.1f}%)")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Repository-Native SLM Benchmarks')
    parser.add_argument('--quick', action='store_true', help='Quick benchmark (1000 patterns)')
    parser.add_argument('--full', action='store_true', help='Full benchmark (all patterns)')
    parser.add_argument('--categories', type=str, nargs='+',
                        choices=list(BENCHMARK_QUERIES.keys()),
                        help='Categories to benchmark')
    parser.add_argument('--corpus', type=str,
                        default='benchmarks/codebase_slm/corpus/training_patterns.jsonl',
                        help='Path to training corpus')
    parser.add_argument('--output', type=str, help='Save results to JSON file')
    parser.add_argument('--compare', type=str, help='Compare with baseline JSON file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    limit = 1000 if args.quick else None

    print("Loading and training model...")
    start = time.time()
    try:
        model, corpus_size = load_model_and_corpus(corpus_path, limit)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run: python -m benchmarks.codebase_slm.generate_corpus --full")
        return 1

    print(f"Trained on {corpus_size:,} patterns in {time.time() - start:.1f}s")

    # Run benchmarks
    print("Running benchmarks...")
    summary = run_benchmarks(model, corpus_size, args.categories)

    # Print results
    print_summary(summary, args.verbose)

    # Compare with baseline if provided
    if args.compare:
        baseline_path = Path(args.compare)
        if baseline_path.exists():
            with open(baseline_path, 'r') as f:
                baseline_data = json.load(f)
                baseline = BenchmarkSummary(
                    timestamp=baseline_data['timestamp'],
                    model_vocab_size=baseline_data['model_vocab_size'],
                    model_transitions=baseline_data['model_transitions'],
                    corpus_size=baseline_data['corpus_size'],
                    total_queries=baseline_data['total_queries'],
                    passed_queries=baseline_data['passed_queries'],
                    pass_rate=baseline_data['pass_rate'],
                    avg_latency_ms=baseline_data['avg_latency_ms'],
                    avg_perplexity=baseline_data['avg_perplexity'],
                    category_scores=baseline_data['category_scores'],
                    results=baseline_data['results'],
                )
                compare_results(summary, baseline)
        else:
            print(f"Warning: Baseline file not found: {baseline_path}")

    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(asdict(summary), f, indent=2)
        print(f"\nResults saved to: {output_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
