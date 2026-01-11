#!/usr/bin/env python3
"""
Incremental batch training script for CognitiveAgent.

This script provides a safe, repeatable way to train the cognitive agent
in controlled batches. Use this when you want to:
  - Train a specific number of documents at a time
  - Monitor progress and resource usage
  - Resume training across sessions

Usage:
    # Train next 5 documents (safe test batch)
    python scripts/train_batch.py --batch-size 5

    # Train next 25 documents with checkpoints every 10
    python scripts/train_batch.py --batch-size 25 --checkpoint 10

    # Show status without training
    python scripts/train_batch.py --status

    # Dry run - show what would be trained
    python scripts/train_batch.py --batch-size 10 --dry-run
"""

import argparse
import time
from pathlib import Path

import sys
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cortical.cognitive.training import IncrementalTrainer
from cortical.cognitive.graph import CognitiveAgent
from cortical.common.filesystem import RealFileSystem


def load_trainer(model_dir: Path) -> IncrementalTrainer:
    """Load the existing trainer with all state."""
    filesystem = RealFileSystem(base_dir=Path.cwd())
    agent = CognitiveAgent(filesystem=filesystem)
    return IncrementalTrainer(agent, model_dir, filesystem)


def show_status(trainer: IncrementalTrainer):
    """Display current training status."""
    status = trainer.status()

    print("=" * 60)
    print("TRAINING STATUS")
    print("=" * 60)
    print(f"Documents trained: {status['total_documents_trained']}")
    print(f"Vocabulary size:   {status['vocabulary_size']}")
    print(f"Last training:     {status.get('last_training', 'Never')}")

    # Get untrained count
    all_files = list(trainer.scan_directory('samples/', '*.txt', recursive=True))
    untrained = trainer.manifest.get_untrained(all_files)

    print(f"Documents remaining: {len(untrained)}")
    print(f"Total on disk:     {len(all_files)}")
    print("=" * 60)


def train_batch(
    trainer: IncrementalTrainer,
    batch_size: int,
    checkpoint_interval: int,
    dry_run: bool = False,
):
    """Train a batch of documents."""
    # Scan for untrained documents
    all_files = list(trainer.scan_directory('samples/', '*.txt', recursive=True))
    untrained = trainer.manifest.get_untrained(all_files)

    if not untrained:
        print("All documents are already trained. Nothing to do.")
        return

    # Select batch
    batch = untrained[:batch_size]
    paths = [path for path, _ in batch]

    print("=" * 60)
    print(f"BATCH TRAINING: {len(batch)} documents")
    print("=" * 60)
    print(f"Checkpoint interval: {checkpoint_interval}")
    print(f"Remaining after this batch: {len(untrained) - len(batch)}")
    print()
    print("Documents to train:")
    for i, path in enumerate(paths, 1):
        print(f"  {i}. {path}")
    print()

    if dry_run:
        print("[DRY RUN] No changes made.")
        return

    # Build full paths
    base_dir = Path("samples/")
    full_paths = [base_dir / path for path in paths]

    # Train with timing
    print("Training...")
    start = time.perf_counter()

    stats = trainer.train_files(
        file_paths=full_paths,
        base_dir=base_dir,
        show_progress=True,
    )

    elapsed = time.perf_counter() - start

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Documents trained: {stats.new_documents + stats.modified_documents}")
    print(f"Atoms created:     {stats.atoms_created}")
    print(f"Links created:     {stats.links_created}")
    print(f"Vocabulary size:   {stats.vocabulary_size}")
    print(f"Time elapsed:      {elapsed:.2f}s")
    if stats.new_documents > 0:
        print(f"Time per document: {elapsed / stats.new_documents:.2f}s")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Incremental batch training for CognitiveAgent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--batch-size", "-n",
        type=int,
        default=5,
        help="Number of documents to train (default: 5)",
    )
    parser.add_argument(
        "--checkpoint", "-c",
        type=int,
        default=10,
        help="Checkpoint interval (default: 10)",
    )
    parser.add_argument(
        "--status", "-s",
        action="store_true",
        help="Show status and exit",
    )
    parser.add_argument(
        "--dry-run", "-d",
        action="store_true",
        help="Show what would be trained without training",
    )
    parser.add_argument(
        "--model-dir", "-m",
        type=Path,
        default=Path("models/cognitive_agent"),
        help="Model directory (default: models/cognitive_agent)",
    )

    args = parser.parse_args()

    # Load trainer
    print(f"Loading model from {args.model_dir}...")
    load_start = time.perf_counter()
    trainer = load_trainer(args.model_dir)
    load_time = time.perf_counter() - load_start
    print(f"Model loaded in {load_time:.2f}s")
    print()

    if args.status:
        show_status(trainer)
    else:
        train_batch(
            trainer,
            batch_size=args.batch_size,
            checkpoint_interval=args.checkpoint,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
