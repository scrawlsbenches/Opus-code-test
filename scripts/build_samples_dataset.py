#!/usr/bin/env python3
"""
Build Training Dataset from Samples Directory

Converts text files in the samples/ directory into a preprocessed training
dataset for language model training.

Features:
- Recursively finds all text files (.txt, .md, .py)
- Optionally chunks long documents for better training
- Weights documents by category/type
- Creates train/val/test splits
- Exports as JSONL format

Usage:
    # Build dataset with defaults (document-level)
    python scripts/build_samples_dataset.py

    # Chunk long documents (better for training)
    python scripts/build_samples_dataset.py --chunk-size 2048

    # Include code files with higher weight
    python scripts/build_samples_dataset.py --weight-code 1.5

    # Custom output
    python scripts/build_samples_dataset.py --output datasets/samples.jsonl
"""

import argparse
import json
import logging
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
SAMPLES_DIR = PROJECT_ROOT / "samples"
DEFAULT_OUTPUT = PROJECT_ROOT / "datasets" / "samples_training_data.jsonl"


@dataclass
class Document:
    """A document from the samples directory."""
    path: str
    content: str
    category: str
    file_type: str
    char_count: int
    word_count: int
    weight: float = 1.0


class SamplesDatasetBuilder:
    """
    Builds training datasets from the samples directory.

    Category Weights (configurable):
        code (.py): 1.5× (structured, high-quality)
        technical: 1.2× (domain knowledge)
        general: 1.0× (baseline)

    Chunking Strategy:
        - Short docs (< chunk_size): Keep as single example
        - Long docs: Split into overlapping chunks
        - Overlap prevents losing context at boundaries
    """

    # File extensions to process
    TEXT_EXTENSIONS = {'.txt', '.md', '.py', '.rst', '.json'}

    # Category weights based on directory name patterns
    CATEGORY_WEIGHTS = {
        'code': 1.5,           # Python files
        'technical': 1.2,      # Technical documentation
        'cognitive': 1.3,      # Cognitive science, AI
        'examples': 1.4,       # Example code, BDD specs
        'default': 1.0,
    }

    def __init__(
        self,
        samples_dir: Path = SAMPLES_DIR,
        chunk_size: int = 0,  # 0 = no chunking
        chunk_overlap: int = 200,
        code_weight: float = 1.5,
        min_doc_length: int = 100,
    ):
        """
        Initialize the dataset builder.

        Args:
            samples_dir: Path to samples directory
            chunk_size: Max characters per chunk (0 = no chunking)
            chunk_overlap: Characters to overlap between chunks
            code_weight: Weight multiplier for code files
            min_doc_length: Minimum characters to include a document
        """
        self.samples_dir = samples_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.code_weight = code_weight
        self.min_doc_length = min_doc_length

    def get_category(self, path: Path) -> str:
        """Extract category from file path."""
        # Get relative path parts
        rel_path = path.relative_to(self.samples_dir)
        parts = rel_path.parts

        if len(parts) > 1:
            # File is in a subdirectory - use subdirectory name
            return parts[0]
        else:
            # Root level file - categorize by content/name
            name = path.stem.lower()
            if any(kw in name for kw in ['code', 'programming', 'algorithm']):
                return 'programming'
            elif any(kw in name for kw in ['api', 'system', 'architecture']):
                return 'technical'
            else:
                return 'general'

    def get_weight(self, doc: Document) -> float:
        """Compute weight for a document."""
        weight = 1.0

        # Code files get higher weight
        if doc.file_type == '.py':
            weight *= self.code_weight

        # Category-based weights
        category_lower = doc.category.lower()
        if 'cognitive' in category_lower or 'ai' in category_lower:
            weight *= self.CATEGORY_WEIGHTS['cognitive']
        elif 'bdd' in category_lower or 'example' in category_lower:
            weight *= self.CATEGORY_WEIGHTS['examples']
        elif 'technical' in category_lower or 'engineering' in category_lower:
            weight *= self.CATEGORY_WEIGHTS['technical']

        return weight

    def load_documents(self) -> List[Document]:
        """Load all documents from the samples directory."""
        documents = []

        if not self.samples_dir.exists():
            logger.error(f"Samples directory not found: {self.samples_dir}")
            return []

        # Find all text files
        for ext in self.TEXT_EXTENSIONS:
            for path in self.samples_dir.rglob(f"*{ext}"):
                try:
                    content = path.read_text(encoding='utf-8', errors='ignore')

                    # Skip empty or very short files
                    if len(content.strip()) < self.min_doc_length:
                        continue

                    doc = Document(
                        path=str(path.relative_to(self.samples_dir)),
                        content=content,
                        category=self.get_category(path),
                        file_type=path.suffix,
                        char_count=len(content),
                        word_count=len(content.split()),
                    )
                    doc.weight = self.get_weight(doc)
                    documents.append(doc)

                except Exception as e:
                    logger.warning(f"Error reading {path}: {e}")

        logger.info(f"Loaded {len(documents)} documents from {self.samples_dir}")
        return documents

    def chunk_document(self, doc: Document) -> List[Dict]:
        """Split a document into chunks if needed."""
        content = doc.content.strip()

        # No chunking requested or document is small enough
        if self.chunk_size == 0 or len(content) <= self.chunk_size:
            return [{
                "text": content,
                "weight": round(doc.weight, 4),
                "metadata": {
                    "source": doc.path,
                    "category": doc.category,
                    "file_type": doc.file_type,
                    "char_count": doc.char_count,
                    "word_count": doc.word_count,
                    "chunk": 0,
                    "total_chunks": 1,
                }
            }]

        # Split into chunks with overlap
        chunks = []
        start = 0
        chunk_idx = 0

        while start < len(content):
            end = start + self.chunk_size

            # Try to break at a sentence or paragraph boundary
            if end < len(content):
                # Look for paragraph break
                para_break = content.rfind('\n\n', start + self.chunk_size // 2, end)
                if para_break > start:
                    end = para_break + 2
                else:
                    # Look for sentence break
                    sent_break = content.rfind('. ', start + self.chunk_size // 2, end)
                    if sent_break > start:
                        end = sent_break + 2

            chunk_text = content[start:end].strip()

            if len(chunk_text) >= self.min_doc_length:
                chunks.append({
                    "text": chunk_text,
                    "weight": round(doc.weight, 4),
                    "metadata": {
                        "source": doc.path,
                        "category": doc.category,
                        "file_type": doc.file_type,
                        "char_count": len(chunk_text),
                        "word_count": len(chunk_text.split()),
                        "chunk": chunk_idx,
                        "total_chunks": -1,  # Will update later
                    }
                })
                chunk_idx += 1

            # Move to next chunk with overlap
            start = end - self.chunk_overlap
            if start >= len(content) - self.min_doc_length:
                break

        # Update total_chunks
        for chunk in chunks:
            chunk["metadata"]["total_chunks"] = len(chunks)

        return chunks

    def build_examples(self, documents: List[Document]) -> List[Dict]:
        """Convert documents to training examples."""
        examples = []

        for doc in documents:
            chunks = self.chunk_document(doc)
            examples.extend(chunks)

        logger.info(f"Created {len(examples)} training examples from {len(documents)} documents")
        return examples

    def compute_stats(self, documents: List[Document], examples: List[Dict]) -> Dict:
        """Compute dataset statistics."""
        if not documents:
            return {}

        # Category distribution
        categories = {}
        for doc in documents:
            categories[doc.category] = categories.get(doc.category, 0) + 1

        # Weight stats
        weights = [doc.weight for doc in documents]

        # Content stats
        total_chars = sum(doc.char_count for doc in documents)
        total_words = sum(doc.word_count for doc in documents)

        return {
            "total_documents": len(documents),
            "total_examples": len(examples),
            "total_characters": total_chars,
            "total_words": total_words,
            "avg_doc_length": total_chars // len(documents),
            "weight_stats": {
                "min": round(min(weights), 4),
                "max": round(max(weights), 4),
                "mean": round(sum(weights) / len(weights), 4),
            },
            "file_types": {
                ext: sum(1 for d in documents if d.file_type == ext)
                for ext in set(d.file_type for d in documents)
            },
            "top_categories": dict(sorted(
                categories.items(), key=lambda x: -x[1]
            )[:10]),
        }

    def split_dataset(
        self,
        examples: List[Dict],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split examples into train/val/test sets."""
        random.seed(seed)
        shuffled = examples.copy()
        random.shuffle(shuffled)

        n = len(shuffled)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]

    def export(
        self,
        examples: List[Dict],
        output_path: Path,
        split: bool = True
    ) -> Dict[str, int]:
        """Export examples to JSONL files."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if split:
            train, val, test = self.split_dataset(examples)
            counts = {"train": len(train), "val": len(val), "test": len(test)}

            for name, data in [("train", train), ("val", val), ("test", test)]:
                split_path = output_path.parent / f"{output_path.stem}_{name}.jsonl"
                with open(split_path, 'w') as f:
                    for ex in data:
                        f.write(json.dumps(ex) + '\n')
                logger.info(f"Wrote {len(data)} examples to {split_path}")
        else:
            with open(output_path, 'w') as f:
                for ex in examples:
                    f.write(json.dumps(ex) + '\n')
            counts = {"total": len(examples)}
            logger.info(f"Wrote {len(examples)} examples to {output_path}")

        return counts


def main():
    parser = argparse.ArgumentParser(
        description="Build training dataset from samples directory"
    )
    parser.add_argument(
        "--samples-dir", type=str, default=str(SAMPLES_DIR),
        help=f"Path to samples directory (default: {SAMPLES_DIR})"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=0,
        help="Max characters per chunk (0 = no chunking, recommended: 2048)"
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=200,
        help="Characters to overlap between chunks (default: 200)"
    )
    parser.add_argument(
        "--weight-code", type=float, default=1.5,
        help="Weight multiplier for code files (default: 1.5)"
    )
    parser.add_argument(
        "--min-length", type=int, default=100,
        help="Minimum document length in characters (default: 100)"
    )
    parser.add_argument(
        "--output", type=str, default=str(DEFAULT_OUTPUT),
        help=f"Output path (default: {DEFAULT_OUTPUT})"
    )
    parser.add_argument(
        "--no-split", action="store_true",
        help="Don't split into train/val/test"
    )
    parser.add_argument(
        "--stats-only", action="store_true",
        help="Only show statistics, don't export"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print("=" * 60)
    print("Building Training Dataset from Samples Directory")
    print("=" * 60)

    # Initialize builder
    builder = SamplesDatasetBuilder(
        samples_dir=Path(args.samples_dir),
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        code_weight=args.weight_code,
        min_doc_length=args.min_length,
    )

    # Load documents
    documents = builder.load_documents()
    if not documents:
        print("No documents found!")
        return 1

    # Build examples
    print(f"\nProcessing {len(documents)} documents...")
    if args.chunk_size > 0:
        print(f"  Chunking: {args.chunk_size} chars with {args.chunk_overlap} overlap")
    examples = builder.build_examples(documents)

    # Compute and show stats
    stats = builder.compute_stats(documents, examples)
    print(f"\n📊 Dataset Statistics:")
    print(f"   Documents: {stats['total_documents']}")
    print(f"   Examples: {stats['total_examples']}")
    print(f"   Total characters: {stats['total_characters']:,}")
    print(f"   Total words: {stats['total_words']:,}")
    print(f"   Avg doc length: {stats['avg_doc_length']:,} chars")
    print(f"\n   Weight range: {stats['weight_stats']['min']} - {stats['weight_stats']['max']}")
    print(f"   Mean weight: {stats['weight_stats']['mean']}")
    print(f"\n   File types:")
    for ftype, count in stats['file_types'].items():
        print(f"     {ftype}: {count}")
    print(f"\n   Top categories:")
    for cat, count in list(stats['top_categories'].items())[:5]:
        print(f"     {cat}: {count}")

    if args.stats_only:
        return 0

    # Export
    output_path = Path(args.output)
    counts = builder.export(examples, output_path, split=not args.no_split)

    print(f"\n✅ Dataset created!")
    for name, count in counts.items():
        print(f"   {name}: {count} examples")

    return 0


if __name__ == "__main__":
    exit(main())
