#!/usr/bin/env python3
"""
Generate training corpus for Repository-Native SLM.

================================================================================
STEP 1 OF TWO-STEP TRAINING PIPELINE
================================================================================

This script is STEP 1. After running this, run STEP 2 to train:

    python -m benchmarks.codebase_slm.generate_corpus --full    # STEP 1 (this)
    python -m benchmarks.codebase_slm.train_augmented --dry-run # STEP 2 (train)

This script orchestrates the extraction and pattern generation pipeline
with batching, progress reporting, and timeout protection.

What it extracts:
  - Python files in cortical/    → functions, classes, imports, docstrings
  - Markdown files in docs/      → sections, Q&A pairs, definitions
  - Markdown files in samples/   → knowledge-base entries, memories
  - GoT entities in .got/        → tasks, decisions, metadata

Output:
  - corpus/training_patterns.jsonl  (~35,000 patterns)
  - This file is gitignored - regenerate after codebase changes!

Usage:
    # Quick test (5 files each)
    python -m benchmarks.codebase_slm.generate_corpus --quick

    # Full generation
    python -m benchmarks.codebase_slm.generate_corpus --full

    # Resume from cache
    python -m benchmarks.codebase_slm.generate_corpus --resume

See train_augmented.py docstring for the full pipeline documentation.
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from benchmarks.codebase_slm.generators.code_extractor import CodeExtractor
from benchmarks.codebase_slm.generators.doc_extractor import DocExtractor
from benchmarks.codebase_slm.generators.meta_extractor import MetaExtractor
from benchmarks.codebase_slm.generators.pattern_generator import PatternGenerator


def progress_callback(done: int, total: int, current: str) -> None:
    """Print progress."""
    bar_len = 30
    filled = int(bar_len * done / total) if total > 0 else 0
    bar = '█' * filled + '░' * (bar_len - filled)
    short_path = current.split('/')[-1][:30] if current else ''
    print(f"\r  [{bar}] {done}/{total} {short_path:30}", end='', flush=True)


def main():
    parser = argparse.ArgumentParser(description='Generate training corpus')
    parser.add_argument('--quick', action='store_true', help='Quick test mode (limited files)')
    parser.add_argument('--full', action='store_true', help='Full generation')
    parser.add_argument('--resume', action='store_true', help='Resume from cache')
    parser.add_argument('--output', type=str, default='benchmarks/codebase_slm/corpus',
                        help='Output directory')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size for extraction')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Repository-Native SLM Corpus Generation")
    print("=" * 60)
    print(f"Output: {output_dir}")
    print(f"Mode: {'quick' if args.quick else 'full'}")
    print(f"Batch size: {args.batch_size}")
    print()

    start_time = time.time()

    # Initialize extractors with caching
    code_extractor = CodeExtractor(cache_dir=output_dir)
    doc_extractor = DocExtractor(cache_dir=output_dir)
    meta_extractor = MetaExtractor(cache_dir=output_dir)

    # Determine source directories
    project_root = Path(__file__).parent.parent.parent

    # Step 1: Extract code patterns
    print("Step 1: Extracting code patterns...")
    code_start = time.time()

    if args.quick:
        # Quick mode: just cortical/ core files
        code_patterns = []
        for batch in code_extractor.extract_batched(
            project_root / 'cortical',
            batch_size=10,
            progress_callback=progress_callback,
            force_refresh=not args.resume
        ):
            code_patterns.extend(batch)
            if len(code_patterns) >= 50:  # Limit for quick mode
                break
    else:
        code_patterns = code_extractor.extract_all(
            project_root / 'cortical',
            progress_callback=progress_callback,
            force_refresh=not args.resume
        )

    print(f"\n  ✓ {len(code_patterns)} files in {time.time() - code_start:.1f}s")
    print(f"  Stats: {code_extractor.get_statistics()}")

    # Step 2: Extract documentation patterns
    print("\nStep 2: Extracting documentation patterns...")
    doc_start = time.time()

    doc_sources = [project_root / 'docs', project_root / 'samples']
    doc_patterns = []

    for source in doc_sources:
        if source.exists():
            for batch in doc_extractor.extract_batched(
                source,
                batch_size=args.batch_size,
                progress_callback=progress_callback,
                force_refresh=not args.resume
            ):
                doc_patterns.extend(batch)
                if args.quick and len(doc_patterns) >= 30:
                    break

    print(f"\n  ✓ {len(doc_patterns)} files in {time.time() - doc_start:.1f}s")
    print(f"  Stats: {doc_extractor.get_statistics()}")

    # Step 3: Extract metadata patterns
    print("\nStep 3: Extracting metadata patterns...")
    meta_start = time.time()

    commit_limit = 50 if args.quick else 500
    meta_patterns = meta_extractor.extract_all(commit_limit=commit_limit)

    print(f"  ✓ Extracted in {time.time() - meta_start:.1f}s")
    print(f"  Stats: {meta_extractor.get_statistics()}")

    # Step 4: Generate training patterns
    print("\nStep 4: Generating training patterns...")
    gen_start = time.time()

    generator = PatternGenerator()
    training_patterns = generator.generate_all(
        code_patterns=code_patterns,
        doc_patterns=doc_patterns,
        meta_patterns=meta_patterns,
    )

    print(f"  ✓ Generated in {time.time() - gen_start:.1f}s")
    print(f"  Stats: {generator.get_statistics()}")

    # Step 5: Save corpus
    print("\nStep 5: Saving corpus...")

    # Save as JSONL (structured)
    jsonl_path = output_dir / 'training_patterns.jsonl'
    generator.save_corpus(jsonl_path, format='jsonl')
    print(f"  ✓ Saved to {jsonl_path}")

    # Save as text (for direct training)
    text_path = output_dir / 'training_corpus.txt'
    generator.save_corpus(text_path, format='text')
    print(f"  ✓ Saved to {text_path}")

    # Summary
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("CORPUS GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total time: {total_time:.1f}s")
    print(f"Patterns generated: {len(training_patterns)}")
    print(f"\nBreakdown:")
    for ptype, count in generator.get_statistics().items():
        if ptype != 'total':
            print(f"  {ptype}: {count}")

    # Show samples
    print("\n" + "=" * 60)
    print("SAMPLE PATTERNS")
    print("=" * 60)
    for p in generator.sample_patterns(5):
        print(f"\n[{p.pattern_type}]")
        print(f"  Input:  {p.input_text[:60]}...")
        print(f"  Target: {p.target_text[:60]}...")

    return 0


if __name__ == '__main__':
    sys.exit(main())
