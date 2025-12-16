#!/usr/bin/env python3
"""
Generate the Cortical Chronicles - a self-documenting living book.

This script orchestrates the generation of book chapters from various sources:
- Algorithm documentation from docs/VISION.md
- Module documentation from .ai_meta files
- Commit narratives from git history
- Concept clusters from Louvain output

Usage:
    python scripts/generate_book.py              # Generate full book
    python scripts/generate_book.py --chapter foundations
    python scripts/generate_book.py --dry-run   # Show what would be generated
    python scripts/generate_book.py --verbose   # Detailed output
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod

BOOK_DIR = Path(__file__).parent.parent / "book"

class ChapterGenerator(ABC):
    """Base class for chapter generators."""

    def __init__(self, book_dir: Path = BOOK_DIR):
        self.book_dir = book_dir
        self.generated_files: List[Path] = []

    @property
    @abstractmethod
    def name(self) -> str:
        """Generator name for logging."""
        pass

    @property
    @abstractmethod
    def output_dir(self) -> str:
        """Subdirectory in book/ for output."""
        pass

    @abstractmethod
    def generate(self, dry_run: bool = False, verbose: bool = False) -> Dict[str, Any]:
        """
        Generate chapter content.

        Returns dict with:
            - files: List of generated file paths
            - stats: Generation statistics
            - errors: Any errors encountered
        """
        pass

    def write_chapter(self, filename: str, content: str, dry_run: bool = False) -> Optional[Path]:
        """Write a chapter file with standard handling."""
        output_path = self.book_dir / self.output_dir / filename
        if dry_run:
            print(f"  Would write: {output_path}")
            return None
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content)
        self.generated_files.append(output_path)
        return output_path

    def generate_frontmatter(self, title: str, tags: List[str], source_files: List[str]) -> str:
        """Generate YAML frontmatter for a chapter."""
        return f"""---
title: "{title}"
generated: "{datetime.utcnow().isoformat()}Z"
generator: "{self.name}"
source_files:
{chr(10).join(f'  - "{f}"' for f in source_files)}
tags:
{chr(10).join(f'  - {t}' for t in tags)}
---

"""


class BookBuilder:
    """Orchestrates book generation from multiple generators."""

    def __init__(self, book_dir: Path = BOOK_DIR, verbose: bool = False):
        self.book_dir = book_dir
        self.verbose = verbose
        self.generators: Dict[str, ChapterGenerator] = {}

    def register_generator(self, generator: ChapterGenerator) -> None:
        """Register a chapter generator."""
        self.generators[generator.name] = generator
        if self.verbose:
            print(f"Registered generator: {generator.name}")

    def generate_all(self, dry_run: bool = False) -> Dict[str, Any]:
        """Generate all chapters from all registered generators."""
        results = {
            "generated_at": datetime.utcnow().isoformat(),
            "dry_run": dry_run,
            "chapters": {},
            "total_files": 0,
            "errors": []
        }

        for name, generator in self.generators.items():
            print(f"\n{'[DRY RUN] ' if dry_run else ''}Generating: {name}")
            try:
                chapter_result = generator.generate(dry_run=dry_run, verbose=self.verbose)
                results["chapters"][name] = chapter_result
                results["total_files"] += len(chapter_result.get("files", []))
            except Exception as e:
                error_msg = f"Generator '{name}' failed: {e}"
                print(f"  ERROR: {error_msg}")
                results["errors"].append(error_msg)

        return results

    def generate_chapter(self, chapter_name: str, dry_run: bool = False) -> Dict[str, Any]:
        """Generate a specific chapter."""
        if chapter_name not in self.generators:
            available = ", ".join(self.generators.keys())
            raise ValueError(f"Unknown chapter: {chapter_name}. Available: {available}")

        generator = self.generators[chapter_name]
        print(f"\n{'[DRY RUN] ' if dry_run else ''}Generating: {chapter_name}")
        return generator.generate(dry_run=dry_run, verbose=self.verbose)


# Placeholder generators (to be implemented in Wave 2)
class PlaceholderGenerator(ChapterGenerator):
    """Placeholder generator for testing the framework."""

    def __init__(self, gen_name: str, output: str, book_dir: Path = BOOK_DIR):
        super().__init__(book_dir)
        self._name = gen_name
        self._output_dir = output

    @property
    def name(self) -> str:
        return self._name

    @property
    def output_dir(self) -> str:
        return self._output_dir

    def generate(self, dry_run: bool = False, verbose: bool = False) -> Dict[str, Any]:
        content = self.generate_frontmatter(
            title=f"{self._name.title()} Chapter",
            tags=[self._name, "placeholder"],
            source_files=["(to be implemented)"]
        )
        content += f"# {self._name.title()}\n\n"
        content += "*This chapter will be auto-generated in a future update.*\n"

        self.write_chapter("index.md", content, dry_run=dry_run)

        return {
            "files": [str(f) for f in self.generated_files],
            "stats": {"placeholder": True},
            "errors": []
        }


def main():
    parser = argparse.ArgumentParser(
        description="Generate the Cortical Chronicles - a self-documenting living book",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s                    Generate full book
    %(prog)s --chapter foundations  Generate only foundations chapter
    %(prog)s --dry-run          Show what would be generated
    %(prog)s --verbose          Detailed progress output
    %(prog)s --list             List available generators
        """
    )
    parser.add_argument("--chapter", "-c", help="Generate specific chapter only")
    parser.add_argument("--dry-run", "-n", action="store_true", help="Show what would be generated without writing")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--list", "-l", action="store_true", help="List available generators")
    parser.add_argument("--output", "-o", type=Path, default=BOOK_DIR, help="Output directory (default: book/)")

    args = parser.parse_args()

    # Initialize builder
    builder = BookBuilder(book_dir=args.output, verbose=args.verbose)

    # Register placeholder generators (will be replaced with real ones)
    builder.register_generator(PlaceholderGenerator("foundations", "01-foundations"))
    builder.register_generator(PlaceholderGenerator("architecture", "02-architecture"))
    builder.register_generator(PlaceholderGenerator("decisions", "03-decisions"))
    builder.register_generator(PlaceholderGenerator("evolution", "04-evolution"))
    builder.register_generator(PlaceholderGenerator("future", "05-future"))

    # List mode
    if args.list:
        print("Available generators:")
        for name in builder.generators:
            print(f"  - {name}")
        return

    # Generate
    if args.chapter:
        results = builder.generate_chapter(args.chapter, dry_run=args.dry_run)
    else:
        results = builder.generate_all(dry_run=args.dry_run)

    # Summary
    print(f"\n{'=' * 50}")
    if args.dry_run:
        print("DRY RUN COMPLETE")
    else:
        print("GENERATION COMPLETE")
    print(f"Total files: {results.get('total_files', len(results.get('files', [])))}")
    if results.get("errors"):
        print(f"Errors: {len(results['errors'])}")
        for err in results["errors"]:
            print(f"  - {err}")


if __name__ == "__main__":
    main()
