#!/usr/bin/env python3
"""
Migrate ML data from JSON files to CALI (Content-Addressable Log with Index) storage.

This script migrates:
- .git-ml/tracked/commits.jsonl -> CALI store
- .git-ml/chats/**/*.json -> CALI store
- .git-ml/sessions/*.json -> CALI store
- .git-ml/actions/**/*.json -> CALI store

Usage:
    python scripts/migrate_ml_to_cali.py                    # Migrate all
    python scripts/migrate_ml_to_cali.py --dry-run          # Show what would migrate
    python scripts/migrate_ml_to_cali.py --type commits     # Migrate only commits
    python scripts/migrate_ml_to_cali.py --benchmark        # Run performance comparison

Benefits of CALI:
    - O(1) existence checks (vs O(n) JSONL scan)
    - O(1) lookups by ID (vs O(n) file search)
    - O(log n) range queries (vs O(n) full scan)
    - Automatic deduplication (content-addressed)
    - 10-100x faster for common operations
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cortical.ml_storage import (
    MLStore,
    migrate_from_jsonl,
    migrate_from_json_dir
)


def get_git_ml_dir() -> Path:
    """Get .git-ml directory."""
    return Path(__file__).parent.parent / '.git-ml'


def migrate_commits(store: MLStore, dry_run: bool = False) -> dict:
    """Migrate commits from JSONL."""
    git_ml = get_git_ml_dir()
    commits_file = git_ml / 'tracked' / 'commits.jsonl'

    if not commits_file.exists():
        return {'status': 'not_found', 'path': str(commits_file)}

    if dry_run:
        count = sum(1 for line in open(commits_file) if line.strip())
        return {'status': 'dry_run', 'would_migrate': count}

    return migrate_from_jsonl(
        commits_file, store, 'commit',
        id_field='hash', timestamp_field='timestamp'
    )


def migrate_chats(store: MLStore, dry_run: bool = False) -> dict:
    """Migrate chats from JSON files."""
    git_ml = get_git_ml_dir()
    chats_dir = git_ml / 'chats'

    if not chats_dir.exists():
        return {'status': 'not_found', 'path': str(chats_dir)}

    if dry_run:
        count = sum(1 for _ in chats_dir.rglob('*.json'))
        return {'status': 'dry_run', 'would_migrate': count}

    return migrate_from_json_dir(
        chats_dir, store, 'chat',
        id_field='id', timestamp_field='timestamp'
    )


def migrate_sessions(store: MLStore, dry_run: bool = False) -> dict:
    """Migrate sessions from JSONL."""
    git_ml = get_git_ml_dir()
    sessions_file = git_ml / 'tracked' / 'sessions.jsonl'

    if not sessions_file.exists():
        return {'status': 'not_found', 'path': str(sessions_file)}

    if dry_run:
        count = sum(1 for line in open(sessions_file) if line.strip())
        return {'status': 'dry_run', 'would_migrate': count}

    return migrate_from_jsonl(
        sessions_file, store, 'session',
        id_field='session_id', timestamp_field='started_at'
    )


def migrate_actions(store: MLStore, dry_run: bool = False) -> dict:
    """Migrate actions from JSON files."""
    git_ml = get_git_ml_dir()
    actions_dir = git_ml / 'actions'

    if not actions_dir.exists():
        return {'status': 'not_found', 'path': str(actions_dir)}

    if dry_run:
        count = sum(1 for _ in actions_dir.rglob('*.json'))
        return {'status': 'dry_run', 'would_migrate': count}

    return migrate_from_json_dir(
        actions_dir, store, 'action',
        id_field='id', timestamp_field='timestamp'
    )


def run_benchmark(store: MLStore) -> dict:
    """
    Benchmark CALI vs JSON file operations.

    Tests:
    1. Existence check (bloom filter vs file read)
    2. Get by ID (index lookup vs directory scan)
    3. Sequential iteration (log read vs glob + parse)
    """
    results = {}
    n_tests = 100

    git_ml = get_git_ml_dir()
    commits_file = git_ml / 'tracked' / 'commits.jsonl'

    if not commits_file.exists():
        return {'error': 'No commits.jsonl found for benchmark'}

    # Load some commit hashes for testing
    test_hashes = []
    with open(commits_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= n_tests:
                break
            try:
                data = json.loads(line)
                test_hashes.append(data.get('hash', f'test_{i}'))
            except:
                test_hashes.append(f'test_{i}')

    if len(test_hashes) < 10:
        return {'error': f'Not enough test data ({len(test_hashes)} commits)'}

    # Benchmark 1: Existence check
    print(f"\nBenchmarking existence check ({n_tests} checks)...")

    # JSON method: read file and search
    start = time.perf_counter()
    for test_hash in test_hashes:
        with open(commits_file, 'r') as f:
            found = any(test_hash in line for line in f)
    json_exists_time = time.perf_counter() - start

    # CALI method: bloom filter
    start = time.perf_counter()
    for test_hash in test_hashes:
        found = store.exists('commit', test_hash)
    cali_exists_time = time.perf_counter() - start

    results['existence_check'] = {
        'json_ms': json_exists_time * 1000,
        'cali_ms': cali_exists_time * 1000,
        'speedup': f"{json_exists_time / cali_exists_time:.1f}x" if cali_exists_time > 0 else "N/A"
    }

    # Benchmark 2: Get by ID
    print(f"Benchmarking get by ID ({min(10, len(test_hashes))} lookups)...")
    test_subset = test_hashes[:10]

    # JSON method: read file and find
    start = time.perf_counter()
    for test_hash in test_subset:
        with open(commits_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data.get('hash') == test_hash:
                        break
                except:
                    pass
    json_get_time = time.perf_counter() - start

    # CALI method: index lookup
    start = time.perf_counter()
    for test_hash in test_subset:
        record = store.get('commit', test_hash)
    cali_get_time = time.perf_counter() - start

    results['get_by_id'] = {
        'json_ms': json_get_time * 1000,
        'cali_ms': cali_get_time * 1000,
        'speedup': f"{json_get_time / cali_get_time:.1f}x" if cali_get_time > 0 else "N/A"
    }

    # Benchmark 3: Sequential iteration
    print("Benchmarking sequential iteration...")

    # JSON method: read and parse all
    start = time.perf_counter()
    json_records = []
    with open(commits_file, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    json_records.append(json.loads(line))
                except:
                    pass
    json_iter_time = time.perf_counter() - start

    # CALI method: iterate log
    start = time.perf_counter()
    cali_records = list(store.iterate('commit'))
    cali_iter_time = time.perf_counter() - start

    results['iteration'] = {
        'json_count': len(json_records),
        'cali_count': len(cali_records),
        'json_ms': json_iter_time * 1000,
        'cali_ms': cali_iter_time * 1000,
        'speedup': f"{json_iter_time / cali_iter_time:.1f}x" if cali_iter_time > 0 else "N/A"
    }

    return results


def print_results(results: dict, title: str):
    """Pretty print migration/benchmark results."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)

    for key, value in results.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")


def main():
    parser = argparse.ArgumentParser(
        description='Migrate ML data to CALI storage',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Show what would be migrated without actually migrating'
    )
    parser.add_argument(
        '--type', choices=['commits', 'chats', 'sessions', 'actions', 'all'],
        default='all', help='Type of data to migrate'
    )
    parser.add_argument(
        '--benchmark', action='store_true',
        help='Run performance benchmark after migration'
    )
    parser.add_argument(
        '--store-path', default='.git-ml/cali',
        help='Path for CALI store (default: .git-ml/cali)'
    )
    parser.add_argument(
        '--compact', action='store_true',
        help='Compact store after migration'
    )
    args = parser.parse_args()

    print("CALI Migration Utility")
    print("=" * 60)

    # Create store
    store_path = Path(args.store_path)
    if args.dry_run:
        print(f"[DRY RUN] Would create store at: {store_path.absolute()}")
    else:
        print(f"Creating/opening CALI store at: {store_path.absolute()}")

    store = MLStore(store_path)

    # Migration functions
    migrations = {
        'commits': migrate_commits,
        'chats': migrate_chats,
        'sessions': migrate_sessions,
        'actions': migrate_actions,
    }

    # Run migrations
    all_results = {}

    if args.type == 'all':
        for name, migrate_fn in migrations.items():
            print(f"\nMigrating {name}...")
            result = migrate_fn(store, dry_run=args.dry_run)
            all_results[name] = result
            print(f"  Result: {result}")
    else:
        print(f"\nMigrating {args.type}...")
        result = migrations[args.type](store, dry_run=args.dry_run)
        all_results[args.type] = result
        print(f"  Result: {result}")

    # Print summary
    print_results(all_results, "Migration Results")

    # Print store stats
    if not args.dry_run:
        print_results(store.stats(), "CALI Store Statistics")

    # Compact if requested
    if args.compact and not args.dry_run:
        print("\nCompacting store...")
        compact_stats = store.compact()
        print_results(compact_stats, "Compaction Results")

    # Run benchmark if requested
    if args.benchmark and not args.dry_run:
        benchmark_results = run_benchmark(store)
        print_results(benchmark_results, "Performance Benchmark")

    # Close store
    store.close()

    print("\n" + "=" * 60)
    print("Migration complete!")
    if not args.dry_run:
        print(f"Store location: {store_path.absolute()}")
        print("\nTo use the new store in code:")
        print("  from cortical.ml_storage import MLStore")
        print(f"  store = MLStore('{store_path}')")
        print("  record = store.get('commit', 'abc123')")


if __name__ == '__main__':
    main()
