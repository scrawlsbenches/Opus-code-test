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
import re
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from abc import ABC, abstractmethod
from collections import defaultdict

BOOK_DIR = Path(__file__).parent.parent / "book"
DOCS_DIR = Path(__file__).parent.parent / "docs"
CORTICAL_DIR = Path(__file__).parent.parent / "cortical"

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

    def generate_chapter(self, chapter_name: str, dry_run: bool = False, force: bool = False) -> Dict[str, Any]:
        """Generate a specific chapter."""
        if chapter_name not in self.generators:
            available = ", ".join(self.generators.keys())
            raise ValueError(f"Unknown chapter: {chapter_name}. Available: {available}")

        generator = self.generators[chapter_name]
        print(f"\n{'[DRY RUN] ' if dry_run else ''}Generating: {chapter_name}")
        # Pass force if the generator supports it (e.g., MarkdownBookGenerator)
        try:
            return generator.generate(dry_run=dry_run, verbose=self.verbose, force=force)
        except TypeError:
            # Generator doesn't support force parameter
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


class AlgorithmChapterGenerator(ChapterGenerator):
    """Generate algorithm chapters from docs/VISION.md."""

    @property
    def name(self) -> str:
        return "foundations"

    @property
    def output_dir(self) -> str:
        return "01-foundations"

    def generate(self, dry_run: bool = False, verbose: bool = False) -> Dict[str, Any]:
        """Generate algorithm chapters from VISION.md."""
        errors = []
        stats = {
            "algorithms_found": 0,
            "chapters_written": 0
        }

        # Read VISION.md
        vision_path = DOCS_DIR / "VISION.md"
        if not vision_path.exists():
            errors.append(f"Source file not found: {vision_path}")
            return {"files": [], "stats": stats, "errors": errors}

        vision_content = vision_path.read_text()

        # Extract algorithm sections
        algorithms = self._extract_algorithms(vision_content)
        stats["algorithms_found"] = len(algorithms)

        if verbose:
            print(f"  Found {len(algorithms)} algorithms in VISION.md")

        # Generate a chapter for each algorithm
        for algo_name, algo_content in algorithms:
            filename = self._generate_filename(algo_name)
            chapter_content = self._generate_chapter(algo_name, algo_content)

            if verbose:
                print(f"  Generating: {filename}")

            self.write_chapter(filename, chapter_content, dry_run=dry_run)
            stats["chapters_written"] += 1

        return {
            "files": [str(f) for f in self.generated_files],
            "stats": stats,
            "errors": errors
        }

    def _extract_algorithms(self, content: str) -> List[Tuple[str, str]]:
        """
        Extract algorithm sections from VISION.md.

        Returns list of (algorithm_name, content) tuples.
        """
        algorithms = []

        # Find the "Deep Algorithm Analysis" section
        deep_analysis_match = re.search(
            r'## Deep Algorithm Analysis\n(.*?)(?=\n## |\Z)',
            content,
            re.DOTALL
        )

        if not deep_analysis_match:
            return algorithms

        analysis_section = deep_analysis_match.group(1)

        # Extract individual algorithm sections (### Algorithm N: Title)
        algo_pattern = r'### (Algorithm \d+:? [^\n]+)\n(.*?)(?=\n### Algorithm |\n### Algorithm Synergy|\Z)'
        matches = re.finditer(algo_pattern, analysis_section, re.DOTALL)

        for match in matches:
            algo_title = match.group(1).strip()
            algo_content = match.group(2).strip()

            # Clean up the title (remove "Algorithm N:" prefix)
            clean_title = re.sub(r'^Algorithm \d+:?\s*', '', algo_title)

            algorithms.append((clean_title, algo_content))

        return algorithms

    def _generate_filename(self, algo_name: str) -> str:
        """Generate filename from algorithm name."""
        # Map algorithm names to file slugs
        name_map = {
            "PageRank — Importance Discovery": "alg-pagerank.md",
            "PageRank": "alg-pagerank.md",
            "BM25/TF-IDF — Distinctiveness Scoring": "alg-bm25.md",
            "BM25/TF-IDF": "alg-bm25.md",
            "Louvain Community Detection — Concept Discovery": "alg-louvain.md",
            "Louvain Community Detection": "alg-louvain.md",
            "Query Expansion — Semantic Bridging": "alg-query-expansion.md",
            "Query Expansion": "alg-query-expansion.md",
            "Graph-Boosted Search (GB-BM25) — Hybrid Ranking": "alg-graph-boosted-search.md",
            "Graph-Boosted Search": "alg-graph-boosted-search.md",
            "Semantic Relation Extraction — Knowledge Graph Construction": "alg-semantic-extraction.md",
            "Semantic Relation Extraction": "alg-semantic-extraction.md",
        }

        return name_map.get(algo_name, self._slugify(algo_name) + ".md")

    def _slugify(self, text: str) -> str:
        """Convert text to filename-safe slug."""
        # Remove em dashes and special chars
        text = re.sub(r'[—–−]', '-', text)
        text = re.sub(r'[^\w\s-]', '', text.lower())
        text = re.sub(r'[-\s]+', '-', text)
        return f"alg-{text.strip('-')}"

    def _generate_chapter(self, algo_name: str, algo_content: str) -> str:
        """Generate a complete chapter markdown file."""
        # Extract implementation path if present
        impl_match = re.search(r'\*\*Implementation:\*\*\s*`([^`]+)`', algo_content)
        source_files = ["docs/VISION.md"]
        if impl_match:
            source_files.append(impl_match.group(1))

        # Generate frontmatter
        tags = ["algorithms", "foundations", "ir-theory"]
        frontmatter = self.generate_frontmatter(
            title=algo_name,
            tags=tags,
            source_files=source_files
        )

        # Clean up algorithm content (remove trailing separator if present)
        algo_content = algo_content.strip()
        if algo_content.endswith('---'):
            algo_content = algo_content[:-3].strip()

        # Build chapter content
        chapter = frontmatter
        chapter += f"# {algo_name}\n\n"
        chapter += algo_content + "\n\n"

        # Add footer
        chapter += "---\n\n"
        chapter += "*This chapter is part of [The Cortical Chronicles](../README.md), "
        chapter += "a self-documenting book generated by the Cortical Text Processor.*\n"

        return chapter


class ModuleDocGenerator(ChapterGenerator):
    """Generate architecture documentation from .ai_meta files."""

    @property
    def name(self) -> str:
        return "architecture"

    @property
    def output_dir(self) -> str:
        return "02-architecture"

    def generate(self, dry_run: bool = False, verbose: bool = False) -> Dict[str, Any]:
        """Generate module documentation chapters."""
        errors = []
        stats = {
            "metadata_files_found": 0,
            "modules_documented": 0,
            "chapters_written": 0
        }

        # Find all .ai_meta files
        metadata_files = list(CORTICAL_DIR.glob("**/*.ai_meta"))
        stats["metadata_files_found"] = len(metadata_files)

        if verbose:
            print(f"  Found {len(metadata_files)} .ai_meta files")

        if not metadata_files:
            errors.append(f"No .ai_meta files found in {CORTICAL_DIR}")
            return {"files": [], "stats": stats, "errors": errors}

        # Parse all metadata
        modules = []
        for meta_file in metadata_files:
            try:
                metadata = self._parse_metadata(meta_file)
                if metadata:
                    modules.append(metadata)
                    stats["modules_documented"] += 1
            except Exception as e:
                errors.append(f"Failed to parse {meta_file.name}: {e}")

        if verbose:
            print(f"  Successfully parsed {len(modules)} modules")

        # Group modules by category
        grouped = self._group_modules(modules)

        # Generate chapter for each group
        for group_name, group_modules in grouped.items():
            filename = f"mod-{group_name}.md"
            content = self._generate_group_chapter(group_name, group_modules)

            if verbose:
                print(f"  Generating: {filename}")

            self.write_chapter(filename, content, dry_run=dry_run)
            stats["chapters_written"] += 1

        # Generate index with module overview
        index_content = self._generate_index(grouped, modules)
        self.write_chapter("index.md", index_content, dry_run=dry_run)
        stats["chapters_written"] += 1

        return {
            "files": [str(f) for f in self.generated_files],
            "stats": stats,
            "errors": errors
        }

    def _parse_metadata(self, meta_file: Path) -> Optional[Dict[str, Any]]:
        """Parse a .ai_meta YAML file."""
        try:
            content = meta_file.read_text()
            # Parse YAML (skip the comment header)
            lines = content.split('\n')
            yaml_lines = [line for line in lines if not line.startswith('#')]
            yaml_content = '\n'.join(yaml_lines)
            metadata = yaml.safe_load(yaml_content)

            if not metadata:
                return None

            # Add the filename for reference
            metadata['meta_file'] = meta_file.name
            return metadata

        except Exception as e:
            print(f"  Warning: Failed to parse {meta_file.name}: {e}")
            return None

    def _group_modules(self, modules: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group modules by functional category."""
        groups = defaultdict(list)

        for module in modules:
            filename = module.get('filename', '')
            filepath = module.get('file', '')

            # Categorize by file path first (for package modules)
            if '/processor/' in filepath or '/processor\\' in filepath:
                groups['processor'].append(module)
            elif '/analysis/' in filepath or '/analysis\\' in filepath:
                groups['analysis'].append(module)
            elif '/query/' in filepath or '/query\\' in filepath:
                groups['query'].append(module)
            # Then by filename patterns (for top-level modules)
            elif filename in ['persistence.py', 'chunk_index.py', 'state_storage.py']:
                groups['persistence'].append(module)
            elif filename in ['minicolumn.py', 'layers.py', 'types.py']:
                groups['data-structures'].append(module)
            elif filename in ['tokenizer.py', 'semantics.py', 'embeddings.py']:
                groups['nlp'].append(module)
            elif filename in ['config.py', 'validation.py', 'constants.py']:
                groups['configuration'].append(module)
            elif filename in ['observability.py', 'progress.py', 'results.py']:
                groups['observability'].append(module)
            else:
                groups['utilities'].append(module)

        return dict(groups)

    def _generate_group_chapter(self, group_name: str, modules: List[Dict[str, Any]]) -> str:
        """Generate a chapter for a group of modules."""
        # Group title mapping
        title_map = {
            'processor': 'Core Processor',
            'analysis': 'Graph Algorithms',
            'query': 'Search & Retrieval',
            'persistence': 'Persistence Layer',
            'data-structures': 'Data Structures',
            'nlp': 'NLP Components',
            'configuration': 'Configuration',
            'observability': 'Observability',
            'utilities': 'Utilities'
        }

        title = title_map.get(group_name, group_name.replace('-', ' ').title())

        # Collect source files
        source_files = [m.get('file', '') for m in modules if m.get('file')]

        # Generate frontmatter
        frontmatter = self.generate_frontmatter(
            title=title,
            tags=['architecture', 'modules', group_name],
            source_files=source_files
        )

        # Build chapter
        chapter = frontmatter
        chapter += f"# {title}\n\n"

        # Add group overview
        chapter += self._generate_group_overview(group_name, modules)
        chapter += "\n\n"

        # Document each module
        for module in sorted(modules, key=lambda m: m.get('filename', '')):
            chapter += self._generate_module_section(module)
            chapter += "\n\n"

        # Add footer
        chapter += "---\n\n"
        chapter += "*This chapter is part of [The Cortical Chronicles](../README.md), "
        chapter += "a self-documenting book generated by the Cortical Text Processor.*\n"

        return chapter

    def _generate_group_overview(self, group_name: str, modules: List[Dict[str, Any]]) -> str:
        """Generate an overview section for a module group."""
        overview_map = {
            'processor': 'The core processor orchestrates all text processing operations.',
            'analysis': 'Graph algorithms for computing importance, relevance, and clusters.',
            'query': 'Search and retrieval components for finding relevant documents and passages.',
            'persistence': 'Save and load functionality for maintaining processor state.',
            'data-structures': 'Fundamental data structures used throughout the system.',
            'nlp': 'Natural language processing components for tokenization and semantics.',
            'configuration': 'Configuration management and validation.',
            'observability': 'Metrics collection and progress tracking.',
            'utilities': 'Utility modules supporting various features.'
        }

        overview = overview_map.get(group_name, 'Supporting modules for the processor.')

        section = f"{overview}\n\n"
        section += "## Modules\n\n"

        for module in sorted(modules, key=lambda m: m.get('filename', '')):
            filename = module.get('filename', 'unknown')
            doc = module.get('module_doc', '').split('\n')[0].strip()
            section += f"- **{filename}**: {doc}\n"

        return section

    def _generate_module_section(self, module: Dict[str, Any]) -> str:
        """Generate documentation section for a single module."""
        filename = module.get('filename', 'unknown')
        module_doc = module.get('module_doc', 'No documentation available.')

        section = f"## {filename}\n\n"

        # Module docstring
        section += f"{module_doc}\n\n"

        # Classes
        classes = module.get('classes', {})
        if classes:
            section += "### Classes\n\n"
            for class_name, class_info in classes.items():
                doc = class_info.get('doc', 'No documentation.')
                section += f"#### {class_name}\n\n"
                section += f"{doc}\n\n"

                # Methods
                methods = class_info.get('methods', [])
                if methods:
                    section += "**Methods:**\n\n"
                    for method in methods:
                        section += f"- `{method}`\n"
                    section += "\n"

        # Functions
        functions = module.get('functions', {})
        if functions:
            section += "### Functions\n\n"
            # Filter out private functions
            public_funcs = {name: info for name, info in functions.items()
                          if not info.get('private', False)}

            for func_name, func_info in public_funcs.items():
                signature = func_info.get('signature', '()')
                doc = func_info.get('doc', 'No documentation.')
                section += f"#### {func_name}\n\n"
                section += f"```python\n{func_name}{signature}\n```\n\n"
                section += f"{doc}\n\n"

        # Imports
        imports = module.get('imports', {})
        if imports:
            section += "### Dependencies\n\n"

            stdlib = imports.get('stdlib', [])
            if stdlib:
                section += "**Standard Library:**\n\n"
                for imp in stdlib[:5]:  # Limit to first 5
                    section += f"- `{imp}`\n"
                if len(stdlib) > 5:
                    section += f"- ... and {len(stdlib) - 5} more\n"
                section += "\n"

            local = imports.get('local', [])
            if local:
                section += "**Local Imports:**\n\n"
                for imp in local:
                    section += f"- `{imp}`\n"
                section += "\n"

        return section

    def _generate_index(self, grouped: Dict[str, List[Dict]], all_modules: List[Dict]) -> str:
        """Generate an index page with module overview."""
        frontmatter = self.generate_frontmatter(
            title="Architecture Overview",
            tags=['architecture', 'index', 'modules'],
            source_files=[m.get('file', '') for m in all_modules if m.get('file')]
        )

        content = frontmatter
        content += "# Architecture Overview\n\n"
        content += "This section documents the architecture of the Cortical Text Processor "
        content += "through automatically extracted module metadata.\n\n"

        # Module statistics
        content += "## Statistics\n\n"
        content += f"- **Total Modules**: {len(all_modules)}\n"
        content += f"- **Module Groups**: {len(grouped)}\n"

        total_classes = sum(len(m.get('classes', {})) for m in all_modules)
        total_functions = sum(len(m.get('functions', {})) for m in all_modules)
        content += f"- **Classes**: {total_classes}\n"
        content += f"- **Functions**: {total_functions}\n\n"

        # Module groups
        content += "## Module Groups\n\n"
        for group_name, modules in sorted(grouped.items()):
            content += f"### [{group_name.replace('-', ' ').title()}](mod-{group_name}.md)\n\n"
            content += f"{len(modules)} modules:\n\n"
            for module in sorted(modules, key=lambda m: m.get('filename', '')):
                filename = module.get('filename', 'unknown')
                content += f"- `{filename}`\n"
            content += "\n"

        # Footer
        content += "---\n\n"
        content += "*This chapter is part of [The Cortical Chronicles](../README.md), "
        content += "a self-documenting book generated by the Cortical Text Processor.*\n"

        return content


class SearchIndexGenerator(ChapterGenerator):
    """Generate search index from all book chapters."""

    @property
    def name(self) -> str:
        return "search"

    @property
    def output_dir(self) -> str:
        return ""  # Output to book/ root

    def generate(self, dry_run: bool = False, verbose: bool = False) -> Dict[str, Any]:
        """Generate search index and full-text search data."""
        errors = []
        stats = {
            "chapters_indexed": 0,
            "sections_found": 0,
            "total_keywords": 0
        }

        if verbose:
            print("  Scanning book chapters...")

        # Find all .md files except README and TEMPLATE
        chapters = []
        for md_file in sorted(self.book_dir.glob("**/*.md")):
            # Skip root-level meta files
            if md_file.name in ["README.md", "TEMPLATE.md"]:
                continue
            # Skip files in book root
            if md_file.parent == self.book_dir:
                continue

            try:
                chapter_data = self._parse_chapter(md_file)
                if chapter_data:
                    chapters.append(chapter_data)
                    stats["chapters_indexed"] += 1
            except Exception as e:
                errors.append(f"Failed to parse {md_file.name}: {e}")

        if verbose:
            print(f"  Parsed {len(chapters)} chapters")

        # Group by section
        sections = self._group_by_section(chapters)
        stats["sections_found"] = len(sections)

        # Generate index.json
        index_data = {
            "generated": datetime.utcnow().isoformat() + "Z",
            "version": "1.0.0",
            "chapters": chapters,
            "sections": sections,
            "stats": {
                "total_chapters": len(chapters),
                "total_sections": len(sections)
            }
        }

        index_path = self.book_dir / "index.json"
        if not dry_run:
            index_path.write_text(json.dumps(index_data, indent=2))
            self.generated_files.append(index_path)
        else:
            print(f"  Would write: {index_path}")

        # Generate search.json
        search_data = {
            "generated": datetime.utcnow().isoformat() + "Z",
            "documents": [
                {
                    "id": ch["path"].replace(".md", ""),
                    "title": ch["title"],
                    "content": ch.get("full_content", ""),
                    "keywords": ch.get("keywords", [])
                }
                for ch in chapters
            ]
        }

        search_path = self.book_dir / "search.json"
        if not dry_run:
            search_path.write_text(json.dumps(search_data, indent=2))
            self.generated_files.append(search_path)
        else:
            print(f"  Would write: {search_path}")

        stats["total_keywords"] = sum(len(ch.get("keywords", [])) for ch in chapters)

        return {
            "files": [str(f) for f in self.generated_files],
            "stats": stats,
            "errors": errors
        }

    def _parse_chapter(self, md_file: Path) -> Optional[Dict[str, Any]]:
        """Parse a chapter markdown file and extract metadata."""
        content = md_file.read_text()

        # Extract YAML frontmatter
        frontmatter = self._extract_frontmatter(content)

        # Get relative path from book dir
        rel_path = md_file.relative_to(self.book_dir)

        # Extract section from path (e.g., "01-foundations" -> "foundations")
        section = rel_path.parts[0] if len(rel_path.parts) > 1 else "root"
        section_clean = section.split("-", 1)[-1] if "-" in section else section

        # Remove frontmatter from content
        content_without_fm = self._remove_frontmatter(content)

        # Extract excerpt (first 200 chars of actual content, skipping title)
        excerpt = self._extract_excerpt(content_without_fm)

        # Extract keywords
        keywords = self._extract_keywords(content_without_fm)

        return {
            "path": str(rel_path.as_posix()),
            "title": frontmatter.get("title", md_file.stem),
            "section": section_clean,
            "tags": frontmatter.get("tags", []),
            "source_files": frontmatter.get("source_files", []),
            "excerpt": excerpt,
            "keywords": keywords,
            "full_content": content_without_fm  # For search.json
        }

    def _extract_frontmatter(self, content: str) -> Dict[str, Any]:
        """Extract YAML frontmatter from markdown content."""
        # Match YAML frontmatter between --- delimiters
        match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
        if not match:
            return {}

        yaml_content = match.group(1)

        # Try yaml.safe_load first
        try:
            import yaml
            return yaml.safe_load(yaml_content) or {}
        except (ImportError, Exception):
            pass

        # Fallback: regex parsing
        frontmatter = {}
        try:
            # Parse simple key: value pairs
            for line in yaml_content.split('\n'):
                if ':' in line and not line.startswith(' '):
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip().strip('"')
                    frontmatter[key] = value
                elif line.startswith('  - '):
                    # List item
                    value = line.strip()[2:].strip('"')
                    # Append to last key if it exists
                    if frontmatter:
                        last_key = list(frontmatter.keys())[-1]
                        if not isinstance(frontmatter[last_key], list):
                            frontmatter[last_key] = [frontmatter[last_key]]
                        frontmatter[last_key].append(value)
        except Exception:
            pass

        return frontmatter

    def _remove_frontmatter(self, content: str) -> str:
        """Remove YAML frontmatter from content."""
        match = re.match(r'^---\s*\n.*?\n---\s*\n', content, re.DOTALL)
        if match:
            return content[match.end():]
        return content

    def _extract_excerpt(self, content: str, max_length: int = 200) -> str:
        """Extract excerpt from content (skip title, get first paragraph)."""
        lines = content.split('\n')

        # Skip title (first # line) and empty lines
        text_lines = []
        skip_title = True
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if skip_title and line.startswith('#'):
                skip_title = False
                continue
            if line.startswith('---'):
                break
            # Skip markdown formatting
            if not line.startswith('#') and not line.startswith('*') and not line.startswith('-'):
                text_lines.append(line)

        # Join and truncate
        text = ' '.join(text_lines)
        if len(text) > max_length:
            text = text[:max_length].rsplit(' ', 1)[0] + '...'

        return text

    def _extract_keywords(self, content: str, top_n: int = 10) -> List[str]:
        """Extract keywords from content using word frequency."""
        # Common stopwords
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'should', 'could', 'may', 'might', 'can', 'this', 'that',
            'these', 'those', 'it', 'its', 'they', 'them', 'their', 'we', 'our',
            'you', 'your', 'he', 'him', 'his', 'she', 'her', 'not', 'no', 'yes',
            'if', 'then', 'so', 'when', 'where', 'what', 'why', 'how', 'which',
            'who', 'whom', 'than', 'more', 'most', 'all', 'some', 'any', 'each',
            'every', 'both', 'few', 'many', 'much', 'such', 'only', 'own', 'same',
            'other', 'another', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'up', 'down', 'out', 'over', 'under',
            'again', 'further', 'once', 'here', 'there', 'also', 'just', 'now'
        }

        # Extract words (including from code blocks)
        words = re.findall(r'\b[a-z_][a-z0-9_]{2,}\b', content.lower())

        # Count frequencies
        word_freq = {}
        for word in words:
            if word not in stopwords and len(word) > 2:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Get top N
        sorted_words = sorted(word_freq.items(), key=lambda x: -x[1])
        keywords = [word for word, freq in sorted_words[:top_n]]

        return keywords

    def _group_by_section(self, chapters: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Group chapters by section and generate section metadata."""
        sections = {}
        section_descriptions = {
            "preface": "Introduction to the book and how it works",
            "foundations": "Core algorithms and IR theory",
            "architecture": "Module documentation and system design",
            "decisions": "Architectural decision records",
            "evolution": "Project history and development narrative",
            "future": "Future plans and roadmap"
        }

        for chapter in chapters:
            section = chapter["section"]
            if section not in sections:
                sections[section] = {
                    "count": 0,
                    "description": section_descriptions.get(section, "")
                }
            sections[section]["count"] += 1

        return sections


class MarkdownBookGenerator(ChapterGenerator):
    """Generate a single consolidated markdown document from all chapters."""

    def __init__(self, book_dir: Path = BOOK_DIR, include_timestamp: bool = False):
        super().__init__(book_dir)
        self.include_timestamp = include_timestamp
        self._hash_file = book_dir / ".book_source_hash"

    @property
    def name(self) -> str:
        return "markdown"

    @property
    def output_dir(self) -> str:
        return ""  # Output to book/ root

    def _compute_source_hash(self) -> str:
        """Compute hash of all source chapter files for change detection."""
        import hashlib
        hasher = hashlib.sha256()

        # Hash all chapter files in sorted order for determinism
        for section_dir in sorted(self.book_dir.iterdir()):
            if not section_dir.is_dir():
                continue
            if section_dir.name.startswith('.') or section_dir.name in ('assets', 'docs'):
                continue

            for md_file in sorted(section_dir.glob("*.md")):
                if md_file.name in ("README.md", "TEMPLATE.md"):
                    continue
                try:
                    hasher.update(str(md_file).encode())
                    hasher.update(md_file.read_bytes())
                except Exception:
                    pass

        return hasher.hexdigest()

    def _read_cached_hash(self) -> Optional[str]:
        """Read the cached source hash from last generation."""
        try:
            if self._hash_file.exists():
                return self._hash_file.read_text().strip()
        except Exception:
            pass
        return None

    def _write_cached_hash(self, hash_value: str) -> None:
        """Write source hash for future change detection."""
        try:
            self._hash_file.write_text(hash_value)
        except Exception:
            pass

    def generate(self, dry_run: bool = False, verbose: bool = False, force: bool = False) -> Dict[str, Any]:
        """Generate a single consolidated markdown file from all chapters.

        Args:
            dry_run: If True, show what would be generated without writing
            verbose: If True, show detailed progress
            force: If True, regenerate even if sources haven't changed
        """
        errors = []
        stats = {
            "chapters_included": 0,
            "sections_found": 0,
            "total_lines": 0,
            "skipped": False,
            "unchanged": False
        }

        output_path = self.book_dir / "BOOK.md"

        # Check if sources changed since last generation
        source_hash = self._compute_source_hash()
        cached_hash = self._read_cached_hash()

        if not force and source_hash == cached_hash and output_path.exists():
            if verbose:
                print("  Source files unchanged, skipping generation")
            stats["skipped"] = True
            return {
                "files": [],
                "stats": stats,
                "errors": errors
            }

        if verbose:
            print("  Scanning book chapters for consolidation...")

        # Find all .md files organized by section
        sections = self._collect_sections()
        stats["sections_found"] = len(sections)

        if verbose:
            print(f"  Found {len(sections)} sections")

        # Build the consolidated markdown
        content = self._generate_header()
        content += self._generate_table_of_contents(sections)

        # Add each section and its chapters
        for section_name, section_info in sections.items():
            if verbose:
                print(f"  Processing section: {section_name}")

            content += self._generate_section(section_name, section_info)
            stats["chapters_included"] += len(section_info["chapters"])

        # Add footer
        content += self._generate_footer()

        stats["total_lines"] = len(content.split('\n'))

        # Compare with existing file to avoid unnecessary writes
        if output_path.exists():
            existing_content = output_path.read_text()
            if existing_content == content:
                if verbose:
                    print("  Content unchanged, skipping write")
                stats["unchanged"] = True
                # Update hash cache even if content unchanged
                if not dry_run:
                    self._write_cached_hash(source_hash)
                return {
                    "files": [],
                    "stats": stats,
                    "errors": errors
                }

        # Write the consolidated file
        if not dry_run:
            output_path.write_text(content)
            self._write_cached_hash(source_hash)
            self.generated_files.append(output_path)
            if verbose:
                print(f"  Written: {output_path}")
        else:
            print(f"  Would write: {output_path}")

        return {
            "files": [str(f) for f in self.generated_files],
            "stats": stats,
            "errors": errors
        }

    def _collect_sections(self) -> Dict[str, Dict[str, Any]]:
        """Collect all chapters organized by section."""
        sections = {}

        # Section order and display names
        section_config = {
            "00-preface": {"order": 0, "title": "Preface"},
            "01-foundations": {"order": 1, "title": "Foundations: Core Algorithms"},
            "02-architecture": {"order": 2, "title": "Architecture: System Design"},
            "03-decisions": {"order": 3, "title": "Decisions: ADRs"},
            "04-evolution": {"order": 4, "title": "Evolution: Project History"},
            "05-future": {"order": 5, "title": "Future: Roadmap"},
        }

        # Find all section directories
        for section_dir in sorted(self.book_dir.iterdir()):
            if not section_dir.is_dir():
                continue
            if section_dir.name.startswith('.') or section_dir.name == 'assets' or section_dir.name == 'docs':
                continue

            section_name = section_dir.name
            config = section_config.get(section_name, {"order": 99, "title": section_name.replace('-', ' ').title()})

            chapters = []
            for md_file in sorted(section_dir.glob("*.md")):
                if md_file.name in ["README.md", "TEMPLATE.md"]:
                    continue

                chapter_data = self._parse_chapter_file(md_file)
                if chapter_data:
                    chapters.append(chapter_data)

            if chapters:
                sections[section_name] = {
                    "order": config["order"],
                    "title": config["title"],
                    "chapters": chapters
                }

        # Sort by order
        return dict(sorted(sections.items(), key=lambda x: x[1]["order"]))

    def _parse_chapter_file(self, md_file: Path) -> Optional[Dict[str, Any]]:
        """Parse a chapter markdown file."""
        try:
            content = md_file.read_text()

            # Extract title from frontmatter or first heading
            title = md_file.stem.replace('-', ' ').title()

            # Check for YAML frontmatter
            if content.startswith('---'):
                match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
                if match:
                    try:
                        frontmatter = yaml.safe_load(match.group(1))
                        if frontmatter and 'title' in frontmatter:
                            title = frontmatter['title']
                        # Remove frontmatter from content
                        content = content[match.end():]
                    except Exception:
                        pass

            # Also try to get title from first # heading if not in frontmatter
            first_heading = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
            if first_heading and title == md_file.stem.replace('-', ' ').title():
                title = first_heading.group(1).strip()

            return {
                "file": md_file,
                "filename": md_file.name,
                "title": title,
                "content": content.strip()
            }
        except Exception as e:
            print(f"  Warning: Failed to parse {md_file.name}: {e}")
            return None

    def _generate_header(self) -> str:
        """Generate the book header."""
        header = """# The Cortical Chronicles

*A Self-Documenting Living Book*

"""
        if self.include_timestamp:
            header += f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n"

        header += """---

This document is automatically generated from the Cortical Text Processor codebase.
It consolidates all book chapters into a single markdown file for offline reading,
PDF generation, or direct viewing on GitHub.

---

"""
        return header

    def _generate_table_of_contents(self, sections: Dict[str, Dict[str, Any]]) -> str:
        """Generate a table of contents."""
        toc = "## Table of Contents\n\n"

        for section_name, section_info in sections.items():
            # Section heading
            section_anchor = self._make_anchor(section_info["title"])
            toc += f"### [{section_info['title']}](#{section_anchor})\n\n"

            # Chapter links
            for chapter in section_info["chapters"]:
                chapter_anchor = self._make_anchor(chapter["title"])
                toc += f"- [{chapter['title']}](#{chapter_anchor})\n"

            toc += "\n"

        toc += "---\n\n"
        return toc

    def _make_anchor(self, text: str) -> str:
        """Convert text to a markdown anchor."""
        # Lowercase, replace spaces with hyphens, remove special chars
        anchor = text.lower()
        anchor = re.sub(r'[^\w\s-]', '', anchor)
        anchor = re.sub(r'[-\s]+', '-', anchor)
        return anchor.strip('-')

    def _generate_section(self, section_name: str, section_info: Dict[str, Any]) -> str:
        """Generate content for a section."""
        content = f"# {section_info['title']}\n\n"

        for chapter in section_info["chapters"]:
            content += self._format_chapter(chapter)

        return content

    def _format_chapter(self, chapter: Dict[str, Any]) -> str:
        """Format a single chapter for inclusion."""
        content = chapter["content"]

        # Ensure chapter starts with a level-2 heading (##)
        # If it starts with # (level 1), convert to ##
        lines = content.split('\n')
        if lines and lines[0].startswith('# ') and not lines[0].startswith('## '):
            lines[0] = '#' + lines[0]  # # -> ##
            content = '\n'.join(lines)

        # Add separator after chapter
        content += "\n\n---\n\n"

        return content

    def _generate_footer(self) -> str:
        """Generate the book footer."""
        footer = """---

## About This Book

**The Cortical Chronicles** is a self-documenting book generated by the Cortical Text Processor.
It documents its own architecture, algorithms, and evolution through automated extraction
of code metadata, git history, and architectural decision records.

### How to Regenerate

```bash
# Generate individual chapters
python scripts/generate_book.py

# Generate consolidated markdown
python scripts/generate_book.py --markdown

# Force regeneration (ignore cache)
python scripts/generate_book.py --markdown --force
```

### Source Code

The source code and generation scripts are available at the project repository.
"""
        if self.include_timestamp:
            footer += f"\n---\n\n*Generated on {datetime.utcnow().strftime('%Y-%m-%d at %H:%M UTC')}*\n"

        return footer


class DecisionStoryGenerator(ChapterGenerator):
    """Generate enriched decision stories from ADRs with conversation context."""

    def __init__(self, book_dir: Path = BOOK_DIR, repo_root: Optional[Path] = None):
        super().__init__(book_dir)
        self.repo_root = repo_root or Path(__file__).parent.parent
        self.decisions_dir = self.repo_root / "samples" / "decisions"
        self.sessions_file = self.repo_root / ".git-ml" / "tracked" / "sessions.jsonl"
        self.commits_file = self.repo_root / ".git-ml" / "tracked" / "commits.jsonl"

    @property
    def name(self) -> str:
        return "decisions"

    @property
    def output_dir(self) -> str:
        return "03-decisions"

    def generate(self, dry_run: bool = False, verbose: bool = False) -> Dict[str, Any]:
        """Generate enriched decision stories from ADRs with conversation context."""
        errors = []
        stats = {
            "adrs_found": 0,
            "adrs_with_context": 0,
            "chapters_written": 0
        }

        if verbose:
            print("  Loading ADRs from samples/decisions/...")

        # Load ADR files
        adrs = self._load_adrs()
        stats["adrs_found"] = len(adrs)

        if not adrs:
            errors.append(f"No ADR files found in {self.decisions_dir}")
            # Generate empty index
            index_content = self._generate_index([])
            self.write_chapter("index.md", index_content, dry_run)
            return {
                "files": [str(f) for f in self.generated_files],
                "stats": stats,
                "errors": errors
            }

        if verbose:
            print(f"  Found {len(adrs)} ADRs")

        # Load session/chat data
        sessions = self._load_sessions()

        if verbose:
            print(f"  Loaded {len(sessions)} sessions with chat data")

        # Generate decision story for each ADR
        for adr in adrs:
            if verbose:
                print(f"  Generating story for: {adr['title']}")

            # Find related chats
            related_chats = self._find_related_chats(adr, sessions)
            if related_chats:
                stats["adrs_with_context"] += 1

            # Find related commits
            related_commits = self._find_related_commits(adr)

            # Generate story
            story_content = self._generate_story(adr, related_chats, related_commits)
            filename = f"decision-{adr['slug']}.md"

            self.write_chapter(filename, story_content, dry_run)
            stats["chapters_written"] += 1

        # Generate index
        index_content = self._generate_index(adrs)
        self.write_chapter("index.md", index_content, dry_run)
        stats["chapters_written"] += 1

        return {
            "files": [str(f) for f in self.generated_files],
            "stats": stats,
            "errors": errors
        }

    def _load_adrs(self) -> List[Dict]:
        """Load ADR files from samples/decisions/"""
        adrs = []

        if not self.decisions_dir.exists():
            return adrs

        for adr_file in sorted(self.decisions_dir.glob("adr-*.md")):
            try:
                parsed = self._parse_adr(adr_file)
                if parsed:
                    adrs.append(parsed)
            except Exception as e:
                print(f"  Warning: Failed to parse {adr_file.name}: {e}")

        return adrs

    def _parse_adr(self, filepath: Path) -> Optional[Dict]:
        """Parse ADR file into structured data"""
        try:
            content = filepath.read_text()
            lines = content.split('\n')

            # Extract title (first line, strip # and whitespace)
            title = lines[0].strip('# ').strip() if lines else filepath.stem

            # Parse metadata (bold markdown after title, before first ---)
            metadata = {}
            section_start = 0

            for i, line in enumerate(lines[1:], 1):
                if line.strip() == '---':
                    section_start = i + 1
                    break
                # Parse **Key:** Value format
                if line.startswith('**') and ':**' in line:
                    # Extract key and value
                    match = re.match(r'\*\*(.+?):\*\*\s*(.+)', line)
                    if match:
                        key = match.group(1).strip()
                        value = match.group(2).strip()
                        metadata[key] = value

            # Extract body sections (after the metadata ---)
            body = '\n'.join(lines[section_start:]) if section_start else content
            sections = self._extract_sections(body)

            # Create slug from filename
            slug = filepath.stem  # adr-xxx

            return {
                'file': filepath,
                'filename': filepath.name,
                'slug': slug,
                'title': title,
                'status': metadata.get('Status', 'Unknown'),
                'date': metadata.get('Date', 'Unknown'),
                'tags': metadata.get('Tags', ''),
                'sections': sections,
                'keywords': self._extract_keywords_from_adr(title, sections)
            }

        except Exception as e:
            print(f"  Error parsing {filepath.name}: {e}")
            return None

    def _extract_sections(self, content: str) -> Dict[str, str]:
        """Extract sections from ADR content"""
        sections = {}
        current_section = None
        current_content = []

        for line in content.split('\n'):
            # Check for section headers (## Section Name)
            if line.startswith('## '):
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = line[3:].strip()
                current_content = []
            elif current_section:
                current_content.append(line)

        # Save last section
        if current_section:
            sections[current_section] = '\n'.join(current_content).strip()

        return sections

    def _extract_keywords_from_adr(self, title: str, sections: Dict[str, str]) -> List[str]:
        """Extract keywords from ADR for chat matching"""
        keywords = set()

        # Extract from title
        title_words = re.findall(r'\b[a-z]{3,}\b', title.lower())
        keywords.update(title_words)

        # Extract from context/problem statement
        context = sections.get('Context and Problem Statement', sections.get('Context', ''))
        context_words = re.findall(r'\b[a-z]{4,}\b', context.lower())
        keywords.update(context_words[:10])  # Top 10 from context

        # Remove common stopwords
        stopwords = {'this', 'that', 'with', 'from', 'have', 'were', 'been', 'were', 'their'}
        keywords = {w for w in keywords if w not in stopwords}

        return list(keywords)

    def _load_sessions(self) -> List[Dict]:
        """Load chat sessions from .git-ml/tracked/sessions.jsonl"""
        sessions = []

        if not self.sessions_file.exists():
            return sessions

        try:
            with open(self.sessions_file, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            session_data = json.loads(line)
                            sessions.append(session_data)
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            print(f"  Warning: Failed to load sessions: {e}")

        return sessions

    def _find_related_chats(self, adr: Dict, sessions: List[Dict]) -> List[Dict]:
        """Search sessions for discussions about this ADR topic"""
        related = []
        keywords = adr['keywords']

        for session in sessions:
            queries = session.get('queries', [])
            for query in queries:
                query_lower = query.lower()
                # Check if any keyword appears in the query
                matches = [kw for kw in keywords if kw in query_lower]
                if len(matches) >= 2:  # At least 2 keyword matches
                    related.append({
                        'session_id': session.get('session_id', 'unknown'),
                        'timestamp': session.get('timestamp', ''),
                        'query': query,
                        'matched_keywords': matches,
                        'tools_used': session.get('tools_used', {}),
                        'files_read': session.get('files_read', [])
                    })

        return related

    def _find_related_commits(self, adr: Dict) -> List[Dict]:
        """Find commits related to this ADR"""
        related = []

        if not self.commits_file.exists():
            return related

        # Extract ADR number from slug (adr-xxx -> xxx)
        adr_num = adr['slug'].split('-', 1)[-1] if '-' in adr['slug'] else ''

        try:
            with open(self.commits_file, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        commit_data = json.loads(line)
                        message = commit_data.get('message', '').lower()

                        # Check for ADR reference in commit message
                        if adr_num and (f'adr-{adr_num}' in message or f'adr {adr_num}' in message):
                            related.append(commit_data)
                        # Also check for keyword matches
                        elif any(kw in message for kw in adr['keywords'][:3]):  # Top 3 keywords
                            related.append(commit_data)

                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass

        return related[:10]  # Limit to 10 most recent

    def _generate_story(self, adr: Dict, chats: List[Dict], commits: List[Dict]) -> str:
        """Generate enriched decision story"""
        sections = adr['sections']

        # Generate frontmatter
        frontmatter = self.generate_frontmatter(
            title=adr['title'],
            tags=['decisions', 'adr', adr['slug']],
            source_files=[str(adr['file'].relative_to(self.repo_root))]
        )

        story = frontmatter
        story += f"# {adr['title']}\n\n"

        # Metadata
        story += f"**Status:** {adr['status']}  \n"
        story += f"**Date:** {adr['date']}  \n"
        if adr['tags']:
            story += f"**Tags:** {adr['tags']}  \n"
        story += "\n---\n\n"

        # The Question
        story += "## The Question\n\n"
        context = sections.get('Context and Problem Statement', sections.get('Context', '*No context documented.*'))
        story += context + "\n\n"

        # The Debate (if we have chat context)
        if chats:
            story += "## The Conversation\n\n"
            story += f"*This decision emerged from {len(chats)} recorded discussion(s).*\n\n"

            for i, chat in enumerate(chats[:3], 1):  # Show top 3 most relevant
                story += f"### Discussion {i}\n\n"
                story += f"**When:** {chat['timestamp'][:10]}  \n"
                story += f"**Matched Keywords:** {', '.join(chat['matched_keywords'])}  \n"
                story += f"\n**Query:**\n\n> {chat['query'][:500]}{'...' if len(chat['query']) > 500 else ''}\n\n"

                if chat['files_read']:
                    story += f"**Files Explored:** {', '.join(f'`{f}`' for f in chat['files_read'][:5])}\n\n"

        # Options Considered
        options_section = sections.get('Considered Options', '')
        if options_section:
            story += "## Options Considered\n\n"
            story += options_section + "\n\n"

        # The Decision
        story += "## The Decision\n\n"
        decision = sections.get('Decision Outcome', sections.get('Decision', '*No decision documented.*'))
        story += decision + "\n\n"

        # Implementation Details
        impl_section = sections.get('Implementation', '')
        if impl_section:
            story += "## Implementation\n\n"
            story += impl_section + "\n\n"

        # Consequences
        consequences = sections.get('Consequences', '')
        if consequences:
            story += "## Consequences\n\n"
            story += consequences + "\n\n"

        # In Hindsight (related commits)
        if commits:
            story += "## In Hindsight\n\n"
            story += f"*This decision has been referenced in {len(commits)} subsequent commit(s).*\n\n"

            for commit in commits[:5]:  # Show top 5
                msg = commit.get('message', '')
                hash_short = commit.get('hash', '')[:7]
                timestamp = commit.get('timestamp', '')[:10]
                story += f"- **{timestamp}** (`{hash_short}`): {msg}\n"

            story += "\n"

        # Footer
        story += "---\n\n"
        story += f"*This decision story was enriched with conversation context from {len(chats)} chat session(s). "
        story += f"Source: [{adr['filename']}](../../samples/decisions/{adr['filename']})*\n"

        return story

    def _generate_index(self, adrs: List[Dict]) -> str:
        """Generate index page for decisions section"""
        frontmatter = self.generate_frontmatter(
            title="Architectural Decision Records",
            tags=['decisions', 'adr', 'index'],
            source_files=['samples/decisions/']
        )

        content = frontmatter
        content += "# Architectural Decision Records\n\n"
        content += "*Enriched with conversation context and implementation history.*\n\n"
        content += "---\n\n"

        if not adrs:
            content += "*No ADRs found. Decisions will appear here as they are documented.*\n\n"
        else:
            # Statistics
            content += "## Overview\n\n"
            content += f"**Total Decisions:** {len(adrs)}  \n"

            # Count by status
            by_status = {}
            for adr in adrs:
                status = adr['status']
                by_status[status] = by_status.get(status, 0) + 1

            for status, count in sorted(by_status.items()):
                content += f"**{status}:** {count}  \n"

            content += "\n"

            # List all decisions
            content += "## Decision Catalog\n\n"

            for adr in sorted(adrs, key=lambda a: a['date'], reverse=True):
                content += f"### [{adr['title']}](decision-{adr['slug']}.md)\n\n"
                content += f"**Status:** {adr['status']} | **Date:** {adr['date']}\n\n"

                # Show excerpt from context
                context = adr['sections'].get('Context and Problem Statement',
                                            adr['sections'].get('Context', ''))
                if context:
                    excerpt = context.split('\n')[0][:200]
                    content += f"{excerpt}...\n\n"

        # Footer
        content += "---\n\n"
        content += "*This chapter is part of [The Cortical Chronicles](../README.md), "
        content += "a self-documenting book generated by the Cortical Text Processor.*\n"

        return content


class CaseStudyGenerator(ChapterGenerator):
    """Generate case study chapters from ML session data."""

    # Narrative templates for storytelling
    CASE_STUDY_TEMPLATES = {
        'opening': [
            "It started with {problem}.",
            "The first sign of trouble was {symptom}.",
            "Nobody expected {situation} to cause {consequence}.",
            "Everything seemed fine until {trigger}.",
        ],
        'investigation': [
            "The obvious suspect was {suspect}. But assumptions are dangerous.",
            "We started with the usual tools: {tools}.",
            "The investigation began by examining {approach}.",
            "First step: {action}.",
        ],
        'breakthrough': [
            "Then we saw it. {discovery}.",
            "The profiler revealed something unexpected: {finding}.",
            "The breakthrough came when {moment}.",
            "Everything clicked: {realization}.",
        ],
        'lesson': [
            "The lesson? {principle}.",
            "This taught us: {wisdom}.",
            "Key takeaway: {insight}.",
            "What we learned: {moral}.",
        ]
    }

    def __init__(self, book_dir: Path = BOOK_DIR, repo_root: Optional[Path] = None):
        super().__init__(book_dir)
        self.repo_root = repo_root or Path(__file__).parent.parent
        self.sessions_dir = self.repo_root / ".git-ml" / "sessions"
        self.chats_dir = self.repo_root / ".git-ml" / "chats"
        self.sessions_file = self.repo_root / ".git-ml" / "tracked" / "sessions.jsonl"
        self.commits_file = self.repo_root / ".git-ml" / "tracked" / "commits.jsonl"

    @property
    def name(self) -> str:
        return "case-studies"

    @property
    def output_dir(self) -> str:
        return "05-case-studies"

    def generate(self, dry_run: bool = False, verbose: bool = False) -> Dict[str, Any]:
        """Generate case study chapters from ML session data."""
        errors = []
        stats = {
            "sessions_found": 0,
            "narrative_worthy": 0,
            "chapters_written": 0
        }

        if verbose:
            print("  Loading ML sessions from .git-ml/...")

        # Load session data
        sessions = self._load_sessions()
        stats["sessions_found"] = len(sessions)

        if not sessions:
            errors.append("No session data found in .git-ml/")
            # Generate empty index
            index_content = self._generate_index([])
            self.write_chapter("index.md", index_content, dry_run)
            return {
                "files": [str(f) for f in self.generated_files],
                "stats": stats,
                "errors": errors
            }

        if verbose:
            print(f"  Found {len(sessions)} sessions")

        # Filter for narrative-worthy sessions
        worthy_sessions = []
        for session in sessions:
            if self._is_narrative_worthy(session):
                worthy_sessions.append(session)

        stats["narrative_worthy"] = len(worthy_sessions)

        if verbose:
            print(f"  {len(worthy_sessions)} sessions are narrative-worthy")

        # Sort by exchange count and select top 10
        worthy_sessions.sort(key=lambda s: s.get('exchange_count', 0), reverse=True)
        top_sessions = worthy_sessions[:10]

        # Load commits for context
        commits = self._load_commits()

        # Generate case studies
        for session in top_sessions:
            if verbose:
                print(f"  Generating case study for session {session.get('session_id', 'unknown')}")

            try:
                case_study = self._generate_case_study(session, commits)
                filename = f"case-{session.get('session_id', 'unknown')[:8]}.md"
                self.write_chapter(filename, case_study, dry_run)
                stats["chapters_written"] += 1
            except Exception as e:
                errors.append(f"Failed to generate case study for {session.get('session_id')}: {e}")
                if verbose:
                    print(f"  Warning: {errors[-1]}")

        # Generate index
        index_content = self._generate_index(top_sessions)
        self.write_chapter("index.md", index_content, dry_run)
        stats["chapters_written"] += 1

        return {
            "files": [str(f) for f in self.generated_files],
            "stats": stats,
            "errors": errors
        }

    def _load_sessions(self) -> List[Dict]:
        """Load sessions from both .git-ml/sessions/*.json and tracked/sessions.jsonl"""
        sessions = []

        # Load from JSONL (aggregated data)
        if self.sessions_file.exists():
            try:
                with open(self.sessions_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            try:
                                session_data = json.loads(line)
                                sessions.append(session_data)
                            except json.JSONDecodeError:
                                continue
            except Exception as e:
                print(f"  Warning: Failed to load sessions.jsonl: {e}")

        # Also load individual session files for richer data
        if self.sessions_dir.exists():
            for session_file in self.sessions_dir.glob("*.json"):
                try:
                    with open(session_file, 'r') as f:
                        session_data = json.load(f)
                        # Merge with existing session if found
                        existing = next((s for s in sessions if s.get('session_id') == session_data.get('id')), None)
                        if existing:
                            # Enrich with detailed data
                            existing['detailed'] = session_data
                        else:
                            # Add as new session
                            sessions.append({
                                'session_id': session_data.get('id'),
                                'detailed': session_data
                            })
                except Exception:
                    continue

        return sessions

    def _load_commits(self) -> Dict[str, Dict]:
        """Load commits indexed by hash"""
        commits = {}

        if not self.commits_file.exists():
            return commits

        try:
            with open(self.commits_file, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        commit_data = json.loads(line)
                        commit_hash = commit_data.get('hash', commit_data.get('commit_hash'))
                        if commit_hash:
                            commits[commit_hash] = commit_data
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"  Warning: Failed to load commits: {e}")

        return commits

    def _is_narrative_worthy(self, session: Dict) -> bool:
        """Determine if a session makes a good case study story."""
        # Minimum exchange count (substantial investigation)
        exchange_count = session.get('exchange_count', session.get('exchanges', 0))
        if exchange_count < 5:
            return False

        # Check for multiple tools used (shows exploration)
        tools_used = session.get('tools_used', {})
        if isinstance(tools_used, dict):
            num_tools = len(tools_used)
        else:
            num_tools = len(tools_used) if tools_used else 0

        if num_tools < 2:
            return False

        # Check for commits (resulted in actual work)
        commits_made = session.get('commits_made', [])
        files_modified = session.get('files_modified', [])

        if not commits_made and not files_modified:
            # No concrete outcome
            return False

        # Check for problem-solving queries
        queries = session.get('queries', [])
        if not queries:
            return False

        first_query = queries[0].lower()
        problem_indicators = ['fix', 'why', 'how do', 'help me', 'error', 'failing', 'issue', 'problem', 'debug']
        has_problem = any(indicator in first_query for indicator in problem_indicators)

        return has_problem

    def _generate_case_study(self, session: Dict, commits: Dict[str, Dict]) -> str:
        """Generate a narrative case study from session data."""
        session_id = session.get('session_id', 'unknown')
        queries = session.get('queries', [])
        tools_used = session.get('tools_used', {})
        files_read = session.get('files_read', [])
        files_modified = session.get('files_modified', [])
        commits_made = session.get('commits_made', [])

        # Extract problem statement
        problem = self._extract_problem_statement(queries)
        title = self._generate_title(problem)

        # Build narrative
        content = f"# Case Study: {title}\n\n"

        # Frontmatter
        content += f"**Session ID:** {session_id[:8]}  \n"
        content += f"**Duration:** {session.get('exchange_count', 0)} exchanges  \n"
        content += f"**Tools:** {', '.join(tools_used.keys()) if isinstance(tools_used, dict) else 'N/A'}  \n"
        content += "\n"

        # The Problem
        content += "## The Problem\n\n"
        content += f"{problem}\n\n"

        if len(queries) > 1:
            content += "The investigation evolved through several questions:\n\n"
            for i, query in enumerate(queries[1:4], 1):  # Show up to 3 follow-up queries
                # Truncate long queries
                query_excerpt = query[:150] + "..." if len(query) > 150 else query
                content += f"{i}. {query_excerpt}\n"
            content += "\n"

        # The Investigation
        content += "## The Investigation\n\n"

        # Tools timeline
        if tools_used:
            content += "### Tools Used\n\n"
            content += "The investigation proceeded with:\n\n"
            for tool, count in (tools_used.items() if isinstance(tools_used, dict) else []):
                content += f"- **{tool}** ({count} times)\n"
            content += "\n"

        # Files explored
        if files_read:
            content += "### Files Explored\n\n"
            # Show up to 10 most relevant files
            for filepath in files_read[:10]:
                content += f"- `{filepath}`\n"
            if len(files_read) > 10:
                content += f"- ...and {len(files_read) - 10} more\n"
            content += "\n"

        # What We Tried
        content += "### What We Tried\n\n"
        content += f"Through {session.get('exchange_count', 0)} exchanges, the session explored "
        content += f"{len(files_read)} files and used {len(tools_used)} different tools. "
        content += "The investigation focused on understanding the problem space before implementing solutions.\n\n"

        # The Solution
        if files_modified or commits_made:
            content += "## The Solution\n\n"

            if files_modified:
                content += "### Files Changed\n\n"
                for filepath in files_modified[:10]:
                    content += f"- `{filepath}`\n"
                content += "\n"

            if commits_made:
                content += "### Commits Made\n\n"
                for commit_ref in commits_made:
                    commit_data = commits.get(commit_ref, {})
                    commit_msg = commit_data.get('message', commit_ref)
                    content += f"- `{commit_ref[:7]}` - {commit_msg}\n"
                content += "\n"

        # The Lesson
        content += "## The Lesson\n\n"
        content += self._extract_lesson(session, problem)
        content += "\n"

        # Try It Yourself
        content += "## Try It Yourself\n\n"
        content += self._generate_exercise(session, problem)
        content += "\n"

        # Footer
        content += "---\n\n"
        content += "*This case study was automatically generated from ML session data. "
        content += "It represents a real problem-solving session during the development of the Cortical Text Processor.*\n"

        return content

    def _extract_problem_statement(self, queries: List[str]) -> str:
        """Extract a clear problem statement from the first query."""
        if not queries:
            return "An investigation into an unknown issue."

        first_query = queries[0]

        # Truncate if too long
        if len(first_query) > 300:
            return first_query[:297] + "..."

        return first_query

    def _generate_title(self, problem: str) -> str:
        """Generate an engaging title from the problem statement."""
        # Extract key words
        problem_lower = problem.lower()

        # Common patterns
        if 'fix' in problem_lower:
            return "Fixing the Unexpected"
        elif 'why' in problem_lower:
            return "Investigating the Mystery"
        elif 'failing' in problem_lower or 'error' in problem_lower:
            return "Debugging the Failure"
        elif 'performance' in problem_lower or 'slow' in problem_lower:
            return "Optimizing Performance"
        elif 'test' in problem_lower:
            return "Test-Driven Investigation"
        elif 'implement' in problem_lower:
            return "Building the Feature"
        else:
            # Extract first meaningful phrase
            words = problem.split()[:6]
            return ' '.join(words) + ("..." if len(problem.split()) > 6 else "")

    def _extract_lesson(self, session: Dict, problem: str) -> str:
        """Extract a generalized lesson from the session."""
        tools_used = session.get('tools_used', {})
        exchange_count = session.get('exchange_count', 0)

        lessons = []

        # Tool diversity lesson
        if len(tools_used) >= 3:
            lessons.append("**Use multiple perspectives.** This investigation benefited from using diverse tools to understand the problem from different angles.")

        # Iteration lesson
        if exchange_count >= 10:
            lessons.append("**Iterate deliberately.** Complex problems require patience and systematic exploration.")

        # Files read ratio
        files_read = session.get('files_read', [])
        files_modified = session.get('files_modified', [])
        if files_read and files_modified:
            ratio = len(files_read) / max(len(files_modified), 1)
            if ratio > 3:
                lessons.append("**Understand before modifying.** This session read significantly more files than it modified, demonstrating the importance of thorough investigation.")

        if not lessons:
            lessons.append("**Systematic investigation pays off.** Breaking down complex problems into smaller steps leads to better solutions.")

        return '\n\n'.join(lessons)

    def _generate_exercise(self, session: Dict, problem: str) -> str:
        """Generate a hands-on exercise based on the case study."""
        tools_used = session.get('tools_used', {})

        exercise = "**Challenge:** Try solving a similar problem in your own codebase:\n\n"
        exercise += f"1. Identify a situation similar to: \"{problem[:100]}...\"\n"
        exercise += f"2. Use the same tools that proved effective: {', '.join(list(tools_used.keys())[:3])}\n"
        exercise += "3. Document your investigation process as you go\n"
        exercise += "4. Compare your approach with this case study\n"

        return exercise

    def _generate_index(self, sessions: List[Dict]) -> str:
        """Generate index page for case studies."""
        content = "# Case Studies\n\n"
        content += "*Real problem-solving sessions from the development of the Cortical Text Processor*\n\n"

        if not sessions:
            content += "No case studies available yet. Case studies are generated from ML session data "
            content += "when sessions demonstrate significant problem-solving narratives.\n\n"
            content += "**What makes a good case study?**\n\n"
            content += "- At least 5 exchanges (substantial investigation)\n"
            content += "- Multiple tools used (shows exploration)\n"
            content += "- Resulted in commits (concrete outcome)\n"
            content += "- Clear problem statement (queries starting with 'fix', 'why', 'how do', etc.)\n"
        else:
            content += "## Featured Case Studies\n\n"
            content += "Each case study tells the story of a real development session, showing:\n\n"
            content += "- **The Problem** - What triggered the investigation\n"
            content += "- **The Investigation** - Tools used and approaches taken\n"
            content += "- **The Solution** - How it was resolved\n"
            content += "- **The Lesson** - What we learned\n"
            content += "- **Try It Yourself** - Hands-on exercises\n\n"

            content += "---\n\n"

            for session in sessions:
                session_id = session.get('session_id', 'unknown')[:8]
                queries = session.get('queries', [])
                problem = self._extract_problem_statement(queries)
                title = self._generate_title(problem)

                exchange_count = session.get('exchange_count', 0)
                tools_used = session.get('tools_used', {})

                content += f"### [Case Study: {title}](case-{session_id}.md)\n\n"
                content += f"**{exchange_count} exchanges** | "
                content += f"**{len(tools_used)} tools**\n\n"

                # Problem excerpt
                problem_excerpt = problem[:200] + "..." if len(problem) > 200 else problem
                content += f"{problem_excerpt}\n\n"

                content += "---\n\n"

        # Footer
        content += "\n"
        content += "*These case studies are automatically generated from ML session data collected during development. "
        content += "They demonstrate real problem-solving workflows and serve as both documentation and learning material.*\n"

        return content


class CommitNarrativeGenerator(ChapterGenerator):
    """Generate evolution narrative from git history."""

    def __init__(self, book_dir: Path = BOOK_DIR, repo_root: Optional[Path] = None):
        super().__init__(book_dir)
        self.repo_root = repo_root or Path(__file__).parent.parent
        self.ml_data_file = self.repo_root / ".git-ml" / "tracked" / "commits.jsonl"
        self.decisions_dir = self.repo_root / "samples" / "decisions"

    @property
    def name(self) -> str:
        return "evolution"

    @property
    def output_dir(self) -> str:
        return "04-evolution"

    def _run_git(self, *args) -> str:
        """Run git command and return output."""
        import subprocess
        result = subprocess.run(
            ["git", *args],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()

    def _load_commits(self, limit: int = 200) -> List[Dict[str, Any]]:
        """Load commits from git history."""
        try:
            log_output = self._run_git(
                "log",
                f"-{limit}",
                "--format=%H|%aI|%s|%an"
            )
        except Exception as e:
            print(f"  Warning: Failed to read git history: {e}")
            return []

        commits = []
        for line in log_output.split('\n'):
            if not line.strip():
                continue
            parts = line.split('|', 3)
            if len(parts) < 4:
                continue
            hash_val, timestamp, message, author = parts
            commits.append({
                'hash': hash_val,
                'short_hash': hash_val[:7],
                'timestamp': timestamp,
                'message': message,
                'author': author,
                'type': self._extract_commit_type(message),
                'date': timestamp[:10]  # YYYY-MM-DD
            })
        return commits

    def _extract_commit_type(self, message: str) -> str:
        """Extract conventional commit type from message."""
        if message.startswith("Merge pull request") or message.startswith("Merge "):
            return "merge"
        for prefix in ["feat:", "fix:", "refactor:", "docs:", "chore:", "test:", "perf:", "security:", "data:"]:
            if message.lower().startswith(prefix):
                return prefix.rstrip(":")
        return "other"

    def _load_ml_commits(self) -> Dict[str, Dict[str, Any]]:
        """Load ML commit data if available."""
        if not self.ml_data_file.exists():
            return {}

        ml_data = {}
        try:
            with open(self.ml_data_file) as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    ml_data[data['hash']] = data
        except Exception as e:
            print(f"  Warning: Failed to load ML data: {e}")
        return ml_data

    def _find_adr_references(self, message: str) -> List[str]:
        """Find ADR references in commit message."""
        # Match patterns like ADR-001, adr-001, ADR 001, etc.
        matches = re.findall(r'(?:adr[- ]?)(\d{3})', message, re.IGNORECASE)
        return [f"adr-{num}" for num in matches]

    def _load_adrs(self) -> Dict[str, Dict[str, Any]]:
        """Load all ADRs from samples/decisions/."""
        adrs = {}
        if not self.decisions_dir.exists():
            return adrs

        for adr_file in self.decisions_dir.glob("adr-*.md"):
            try:
                content = adr_file.read_text()
                # Parse minimal metadata
                title = content.split('\n')[0].strip('# ')
                adrs[adr_file.stem] = {
                    'file': adr_file.name,
                    'title': title,
                    'path': adr_file
                }
            except Exception:
                pass
        return adrs

    def generate(self, dry_run: bool = False, verbose: bool = False) -> Dict[str, Any]:
        """Generate evolution chapters from git history."""
        if verbose:
            print("  Loading git history...")

        commits = self._load_commits()
        ml_data = self._load_ml_commits()
        adrs = self._load_adrs()

        if not commits:
            return {
                "files": [],
                "stats": {"commits": 0, "error": "No git history found"},
                "errors": ["Could not read git history"]
            }

        # Group commits by type
        by_type = {}
        for commit in commits:
            commit_type = commit['type']
            if commit_type not in by_type:
                by_type[commit_type] = []
            by_type[commit_type].append(commit)

        # Generate chapters
        stats = {
            "total_commits": len(commits),
            "by_type": {k: len(v) for k, v in by_type.items()},
            "with_ml_data": len(ml_data),
            "adrs_found": len(adrs)
        }

        if verbose:
            print(f"  Found {len(commits)} commits, {len(adrs)} ADRs")

        # 1. Timeline
        self._generate_timeline(commits, adrs, dry_run, verbose)

        # 2. Features
        self._generate_features(by_type.get('feat', []), ml_data, adrs, dry_run, verbose)

        # 3. Bug fixes
        self._generate_bugfixes(by_type.get('fix', []), ml_data, adrs, dry_run, verbose)

        # 4. Refactorings
        self._generate_refactors(by_type.get('refactor', []), ml_data, adrs, dry_run, verbose)

        return {
            "files": [str(f) for f in self.generated_files],
            "stats": stats,
            "errors": []
        }

    def _generate_timeline(self, commits: List[Dict], adrs: Dict, dry_run: bool, verbose: bool):
        """Generate chronological timeline."""
        if verbose:
            print("  Generating timeline...")

        # Group by month and week
        by_month = defaultdict(lambda: defaultdict(list))
        for commit in commits:
            try:
                dt = datetime.fromisoformat(commit['timestamp'].replace('Z', '+00:00'))
                month_key = dt.strftime("%Y-%m")
                # Calculate week start (Monday)
                week_start = dt - timedelta(days=dt.weekday())
                week_key = week_start.strftime("%b %d")
                by_month[month_key][week_key].append(commit)
            except Exception:
                continue

        content = self.generate_frontmatter(
            title="Project Timeline",
            tags=["timeline", "chronology", "evolution"],
            source_files=["git log"]
        )

        content += "# Project Timeline\n\n"
        content += "*A chronological journey through the Cortical Text Processor's development.*\n\n"
        content += "---\n\n"

        # Sort months descending (most recent first)
        for month_key in sorted(by_month.keys(), reverse=True):
            month_name = datetime.strptime(month_key, "%Y-%m").strftime("%B %Y")
            content += f"## {month_name}\n\n"

            weeks = by_month[month_key]
            for week_key in sorted(weeks.keys(), reverse=True):
                week_commits = weeks[week_key]
                content += f"### Week of {week_key}\n\n"

                # Show up to 10 significant commits per week
                significant = [c for c in week_commits if c['type'] not in ['chore', 'data', 'merge']][:10]

                for commit in significant:
                    date_str = commit['timestamp'][:10]
                    msg = commit['message']
                    # Check for ADR references
                    adr_refs = self._find_adr_references(msg)
                    adr_links = ""
                    if adr_refs:
                        adr_link_list = []
                        for adr_id in adr_refs:
                            if adr_id in adrs:
                                adr_link_list.append(f"[{adr_id.upper()}](../../samples/decisions/{adrs[adr_id]['file']})")
                        if adr_link_list:
                            adr_links = f" (See: {', '.join(adr_link_list)})"

                    content += f"- **{date_str}**: {msg}{adr_links}\n"

                content += "\n"

        self.write_chapter("timeline.md", content, dry_run)

    def _generate_features(self, commits: List[Dict], ml_data: Dict, adrs: Dict, dry_run: bool, verbose: bool):
        """Generate feature additions narrative."""
        if verbose:
            print("  Generating features chapter...")

        content = self.generate_frontmatter(
            title="Feature Evolution",
            tags=["features", "capabilities", "growth"],
            source_files=["git log --grep=feat:"]
        )

        content += "# Feature Evolution\n\n"
        content += "*How the Cortical Text Processor gained its capabilities.*\n\n"
        content += "---\n\n"

        if not commits:
            content += "*No feature commits found in recent history.*\n"
        else:
            content += f"## Overview\n\n"
            content += f"The system has evolved through **{len(commits)} feature additions**. "
            content += "Below is the narrative of how each capability came to be.\n\n"

            # Group by major themes (keywords in commit messages)
            themes = self._group_by_themes(commits)

            for theme, theme_commits in themes.items():
                content += f"## {theme.title()} Capabilities\n\n"

                for commit in theme_commits[:20]:  # Limit per theme
                    msg = commit['message']
                    # Remove "feat: " prefix
                    if msg.lower().startswith("feat:"):
                        msg = msg[5:].strip()

                    content += f"### {msg}\n\n"
                    content += f"**Commit:** `{commit['short_hash']}`  \n"
                    content += f"**Date:** {commit['timestamp'][:10]}  \n"

                    # Add ML data if available
                    if commit['hash'] in ml_data:
                        data = ml_data[commit['hash']]
                        files = data.get('files_changed', [])
                        if files:
                            content += f"**Files Modified:** {len(files)}  \n"

                    # Check for ADR references
                    adr_refs = self._find_adr_references(msg)
                    if adr_refs:
                        for adr_id in adr_refs:
                            if adr_id in adrs:
                                content += f"**Related Decision:** [{adrs[adr_id]['title']}](../../samples/decisions/{adrs[adr_id]['file']})  \n"

                    content += "\n"

        self.write_chapter("features.md", content, dry_run)

    def _generate_bugfixes(self, commits: List[Dict], ml_data: Dict, adrs: Dict, dry_run: bool, verbose: bool):
        """Generate bug fix stories."""
        if verbose:
            print("  Generating bugfixes chapter...")

        content = self.generate_frontmatter(
            title="Bug Fixes and Lessons",
            tags=["bugs", "fixes", "lessons-learned"],
            source_files=["git log --grep=fix:"]
        )

        content += "# Bug Fixes and Lessons\n\n"
        content += "*What broke, how we fixed it, and what we learned.*\n\n"
        content += "---\n\n"

        if not commits:
            content += "*No bug fix commits found in recent history.*\n"
        else:
            content += f"## Overview\n\n"
            content += f"**{len(commits)} bugs** have been identified and resolved. "
            content += "Each fix taught us something about the system.\n\n"

            content += "## Bug Fix History\n\n"

            for commit in commits[:50]:  # Top 50 fixes
                msg = commit['message']
                # Remove "fix: " prefix
                if msg.lower().startswith("fix:"):
                    msg = msg[4:].strip()

                content += f"### {msg}\n\n"
                content += f"**Commit:** `{commit['short_hash']}`  \n"
                content += f"**Date:** {commit['timestamp'][:10]}  \n"

                # Add ML data if available
                if commit['hash'] in ml_data:
                    data = ml_data[commit['hash']]
                    files = data.get('files_changed', [])
                    if files:
                        content += f"**Files Changed:** {', '.join(files[:5])}  \n"
                        if len(files) > 5:
                            content += f"*(and {len(files) - 5} more)*  \n"

                content += "\n"

        self.write_chapter("bugfixes.md", content, dry_run)

    def _generate_refactors(self, commits: List[Dict], ml_data: Dict, adrs: Dict, dry_run: bool, verbose: bool):
        """Generate refactoring decisions narrative."""
        if verbose:
            print("  Generating refactors chapter...")

        content = self.generate_frontmatter(
            title="Refactorings and Architecture Evolution",
            tags=["refactoring", "architecture", "design"],
            source_files=["git log --grep=refactor:"]
        )

        content += "# Refactorings and Architecture Evolution\n\n"
        content += "*How the codebase structure improved over time.*\n\n"
        content += "---\n\n"

        if not commits:
            content += "*No refactoring commits found in recent history.*\n"
        else:
            content += f"## Overview\n\n"
            content += f"The codebase has undergone **{len(commits)} refactorings**. "
            content += "Each improved code quality, maintainability, or performance.\n\n"

            content += "## Refactoring History\n\n"

            for commit in commits[:30]:  # Top 30 refactorings
                msg = commit['message']
                # Remove "refactor: " prefix
                if msg.lower().startswith("refactor:"):
                    msg = msg[9:].strip()

                content += f"### {msg}\n\n"
                content += f"**Commit:** `{commit['short_hash']}`  \n"
                content += f"**Date:** {commit['timestamp'][:10]}  \n"

                # Add ML data if available
                if commit['hash'] in ml_data:
                    data = ml_data[commit['hash']]
                    insertions = data.get('insertions', 0)
                    deletions = data.get('deletions', 0)
                    if insertions or deletions:
                        content += f"**Changes:** +{insertions}/-{deletions} lines  \n"
                    files = data.get('files_changed', [])
                    if files:
                        content += f"**Scope:** {len(files)} files affected  \n"

                # Check for ADR references
                adr_refs = self._find_adr_references(msg)
                if adr_refs:
                    for adr_id in adr_refs:
                        if adr_id in adrs:
                            content += f"**Design Decision:** [{adrs[adr_id]['title']}](../../samples/decisions/{adrs[adr_id]['file']})  \n"

                content += "\n"

        self.write_chapter("refactors.md", content, dry_run)

    def _group_by_themes(self, commits: List[Dict]) -> Dict[str, List[Dict]]:
        """Group commits by keywords/themes."""
        themes = defaultdict(list)
        keywords = {
            'ml': ['ml', 'machine learning', 'model', 'prediction', 'training'],
            'search': ['search', 'query', 'retrieval', 'ranking', 'bm25', 'tfidf'],
            'performance': ['performance', 'optimize', 'speed', 'cache', 'fast'],
            'testing': ['test', 'coverage', 'benchmark', 'validation'],
            'documentation': ['docs', 'documentation', 'guide', 'readme'],
            'api': ['api', 'interface', 'method', 'function'],
            'data': ['data', 'storage', 'persistence', 'save', 'load'],
            'analysis': ['analysis', 'algorithm', 'pagerank', 'clustering', 'graph'],
        }

        for commit in commits:
            msg_lower = commit['message'].lower()
            matched = False
            for theme, words in keywords.items():
                if any(word in msg_lower for word in words):
                    themes[theme].append(commit)
                    matched = True
                    break
            if not matched:
                themes['other'].append(commit)

        return dict(themes)


class LessonExtractor(ChapterGenerator):
    """Generate lesson chapters from bugfixes, refactors, and performance commits."""

    def __init__(self, book_dir: Path = BOOK_DIR, repo_root: Optional[Path] = None):
        super().__init__(book_dir)
        self.repo_root = repo_root or Path(__file__).parent.parent
        self.ml_data_file = self.repo_root / ".git-ml" / "tracked" / "commits.jsonl"

    @property
    def name(self) -> str:
        return "lessons"

    @property
    def output_dir(self) -> str:
        return "06-lessons"

    # Lesson categories with keywords for detection
    LESSON_CATEGORIES = {
        'performance': ['perf:', 'optimize', 'slow', 'timeout', 'O(n', 'fast', 'speed', 'cache'],
        'correctness': ['fix:', 'bug', 'wrong', 'incorrect', 'fail', 'error', 'broken'],
        'architecture': ['refactor:', 'extract', 'split', 'modular', 'reorganize', 'structure'],
        'testing': ['test:', 'coverage', 'assert', 'mock', 'fixture', 'validate']
    }

    def _run_git(self, *args) -> str:
        """Run git command and return output."""
        import subprocess
        result = subprocess.run(
            ["git", *args],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()

    def _load_commits(self, limit: int = 300) -> List[Dict[str, Any]]:
        """Load commits from git history."""
        try:
            log_output = self._run_git(
                "log",
                f"-{limit}",
                "--format=%H|%aI|%s|%an"
            )
        except Exception as e:
            print(f"  Warning: Failed to read git history: {e}")
            return []

        commits = []
        for line in log_output.split('\n'):
            if not line.strip():
                continue
            parts = line.split('|', 3)
            if len(parts) < 4:
                continue
            hash_val, timestamp, message, author = parts
            commits.append({
                'hash': hash_val,
                'short_hash': hash_val[:7],
                'timestamp': timestamp,
                'message': message,
                'author': author,
                'date': timestamp[:10]  # YYYY-MM-DD
            })
        return commits

    def _load_ml_commits(self) -> Dict[str, Dict[str, Any]]:
        """Load ML commit data if available."""
        if not self.ml_data_file.exists():
            return {}

        ml_data = {}
        try:
            with open(self.ml_data_file) as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    ml_data[data['hash']] = data
        except Exception as e:
            print(f"  Warning: Failed to load ML data: {e}")
        return ml_data

    def _categorize_lesson(self, commit: Dict[str, Any]) -> Optional[str]:
        """Categorize a commit into a lesson type."""
        message = commit['message'].lower()

        # Check each category
        for category, keywords in self.LESSON_CATEGORIES.items():
            for keyword in keywords:
                if keyword.lower() in message:
                    return category

        return None

    def _get_files_changed(self, commit_hash: str) -> List[str]:
        """Get list of files changed in a commit."""
        try:
            output = self._run_git("show", "--pretty=format:", "--name-only", commit_hash)
            files = [f.strip() for f in output.split('\n') if f.strip()]
            return files
        except Exception:
            return []

    def _extract_principle(self, commit: Dict[str, Any], category: str) -> str:
        """Extract the general principle from a commit."""
        message = commit['message']

        # Remove conventional commit prefix
        for prefix in ['fix:', 'refactor:', 'perf:', 'test:', 'chore:', 'docs:']:
            if message.lower().startswith(prefix):
                message = message[len(prefix):].strip()
                break

        # Generate principle based on category and message patterns
        # Look for common patterns in commit messages
        message_lower = message.lower()

        if category == 'performance':
            if 'timeout' in message_lower or 'slow' in message_lower:
                return f"Profile before optimizing. The lesson? {message}"
            elif 'cache' in message_lower:
                return f"Cache expensive operations. This taught us: {message}"
            elif 'o(n' in message_lower or 'complexity' in message_lower:
                return f"Watch for O(n²) patterns. The wisdom: {message}"
            else:
                return f"Optimize based on evidence. The lesson? {message}"

        elif category == 'correctness':
            if 'edge case' in message_lower or 'corner case' in message_lower:
                return f"Test edge cases thoroughly. This taught us: {message}"
            elif 'validation' in message_lower or 'check' in message_lower:
                return f"Validate inputs early. The lesson? {message}"
            elif 'null' in message_lower or 'none' in message_lower or 'empty' in message_lower:
                return f"Handle empty/null cases. This taught us: {message}"
            else:
                return f"Verify assumptions with tests. The wisdom: {message}"

        elif category == 'architecture':
            if 'extract' in message_lower or 'split' in message_lower:
                return f"Keep modules focused. The lesson? {message}"
            elif 'coupling' in message_lower or 'dependency' in message_lower:
                return f"Minimize coupling. This taught us: {message}"
            elif 'duplicate' in message_lower or 'dry' in message_lower:
                return f"Don't repeat yourself. The wisdom: {message}"
            else:
                return f"Maintain clear structure. The lesson? {message}"

        elif category == 'testing':
            if 'coverage' in message_lower:
                return f"Measure coverage to find gaps. The lesson? {message}"
            elif 'fixture' in message_lower or 'setup' in message_lower:
                return f"Reuse test fixtures. This taught us: {message}"
            elif 'mock' in message_lower:
                return f"Mock external dependencies. The wisdom: {message}"
            else:
                return f"Test what you build. The lesson? {message}"

        return message

    def generate(self, dry_run: bool = False, verbose: bool = False) -> Dict[str, Any]:
        """Generate lesson chapters from git history."""
        if verbose:
            print("  Loading git history for lessons...")

        commits = self._load_commits()
        ml_data = self._load_ml_commits()

        if not commits:
            return {
                "files": [],
                "stats": {"commits": 0, "error": "No git history found"},
                "errors": ["Could not read git history"]
            }

        # Categorize commits
        lessons_by_category = {
            'performance': [],
            'correctness': [],
            'architecture': [],
            'testing': []
        }

        for commit in commits:
            category = self._categorize_lesson(commit)
            if category:
                lessons_by_category[category].append(commit)

        stats = {
            'total_commits': len(commits),
            'lessons_found': sum(len(v) for v in lessons_by_category.values()),
            'by_category': {k: len(v) for k, v in lessons_by_category.items()}
        }

        if verbose:
            print(f"  Found {stats['lessons_found']} lesson commits")

        # Generate index
        self._generate_index(lessons_by_category, stats, dry_run, verbose)

        # Generate category-specific chapters
        for category, lessons in lessons_by_category.items():
            if lessons:
                self._generate_category_chapter(category, lessons, ml_data, dry_run, verbose)

        return {
            "files": [str(f) for f in self.generated_files],
            "stats": stats,
            "errors": []
        }

    def _generate_index(self, lessons_by_category: Dict, stats: Dict, dry_run: bool, verbose: bool):
        """Generate index with overview and statistics."""
        if verbose:
            print("  Generating lessons index...")

        content = self.generate_frontmatter(
            title="Lessons Learned",
            tags=["lessons", "wisdom", "experience"],
            source_files=["git log"]
        )

        content += "# Lessons Learned\n\n"
        content += "*What the Cortical Text Processor taught us about building IR systems.*\n\n"
        content += "---\n\n"

        content += "## Overview\n\n"
        content += f"Through **{stats['lessons_found']} lessons** extracted from development history, "
        content += "we've learned how to build better search systems. Each bug fixed, each optimization made, "
        content += "and each refactoring completed taught us something valuable.\n\n"

        content += "## Statistics\n\n"
        content += f"- **Total Commits Analyzed**: {stats['total_commits']}\n"
        content += f"- **Lessons Extracted**: {stats['lessons_found']}\n\n"

        content += "### By Category\n\n"
        for category, count in stats['by_category'].items():
            if count > 0:
                content += f"- **{category.title()}**: {count} lessons\n"
        content += "\n"

        content += "## Lesson Categories\n\n"

        categories_desc = {
            'performance': ("Performance Lessons", "How we learned to optimize search and graph algorithms"),
            'correctness': ("Correctness Lessons", "Bugs we fixed and edge cases we discovered"),
            'architecture': ("Architecture Lessons", "How we evolved the codebase structure"),
            'testing': ("Testing Lessons", "What we learned about verifying correctness")
        }

        for category, (title, desc) in categories_desc.items():
            count = stats['by_category'].get(category, 0)
            if count > 0:
                content += f"### [{title}](lessons-{category}.md)\n\n"
                content += f"{desc}\n\n"
                content += f"**{count} lessons** from development history.\n\n"

        content += "---\n\n"
        content += "*This chapter is part of [The Cortical Chronicles](../README.md), "
        content += "a self-documenting book generated by the Cortical Text Processor.*\n"

        self.write_chapter("index.md", content, dry_run)

    def _generate_category_chapter(self, category: str, lessons: List[Dict], ml_data: Dict, dry_run: bool, verbose: bool):
        """Generate a chapter for a specific lesson category."""
        if verbose:
            print(f"  Generating {category} lessons chapter...")

        titles = {
            'performance': 'Performance Lessons',
            'correctness': 'Correctness Lessons',
            'architecture': 'Architecture Lessons',
            'testing': 'Testing Lessons'
        }

        content = self.generate_frontmatter(
            title=titles[category],
            tags=["lessons", category, "experience"],
            source_files=[f"git log --grep={category}"]
        )

        content += f"# {titles[category]}\n\n"

        descriptions = {
            'performance': "What we learned about making the system fast and efficient.",
            'correctness': "Bugs we encountered and how we fixed them.",
            'architecture': "How we evolved the code structure over time.",
            'testing': "Insights from writing and maintaining tests."
        }

        content += f"*{descriptions[category]}*\n\n"
        content += "---\n\n"

        content += f"## Overview\n\n"
        content += f"This chapter captures **{len(lessons)} lessons** from {category} work. "
        content += "Each entry shows the problem, the solution, and the principle we extracted.\n\n"

        # Limit to top 50 lessons per category
        for lesson in lessons[:50]:
            content += self._format_lesson(lesson, category, ml_data)

        content += "---\n\n"
        content += "*This chapter is part of [The Cortical Chronicles](../README.md), "
        content += "a self-documenting book generated by the Cortical Text Processor.*\n"

        self.write_chapter(f"lessons-{category}.md", content, dry_run)

    def _format_lesson(self, commit: Dict[str, Any], category: str, ml_data: Dict) -> str:
        """Format a single lesson entry."""
        message = commit['message']

        # Remove conventional commit prefix
        clean_message = message
        for prefix in ['fix:', 'refactor:', 'perf:', 'test:', 'chore:', 'docs:']:
            if clean_message.lower().startswith(prefix):
                clean_message = clean_message[len(prefix):].strip()
                break

        # Capitalize first letter
        clean_message = clean_message[0].upper() + clean_message[1:] if clean_message else message

        content = f"### {clean_message}\n\n"
        content += f"**Commit:** `{commit['short_hash']}`  \n"
        content += f"**Date:** {commit['date']}  \n"

        # Add files changed
        files = self._get_files_changed(commit['hash'])
        if files:
            content += f"**Files Changed:** {len(files)}  \n"
            # Show first 3 files
            for f in files[:3]:
                content += f"  - `{f}`\n"
            if len(files) > 3:
                content += f"  - *(and {len(files) - 3} more)*\n"

        # Add ML data if available
        if commit['hash'] in ml_data:
            data = ml_data[commit['hash']]
            insertions = data.get('insertions', 0)
            deletions = data.get('deletions', 0)
            if insertions or deletions:
                content += f"**Changes:** +{insertions}/-{deletions} lines  \n"

        # Extract and add the principle
        principle = self._extract_principle(commit, category)
        content += f"\n**The Lesson:** {principle}\n\n"

        return content


class ConceptEvolutionGenerator(ChapterGenerator):
    """Generate concept evolution chapters showing how ideas grew over time."""

    def __init__(self, book_dir: Path = BOOK_DIR, repo_root: Optional[Path] = None):
        super().__init__(book_dir)
        self.repo_root = repo_root or Path(__file__).parent.parent
        self.ml_data_file = self.repo_root / ".git-ml" / "tracked" / "commits.jsonl"

    @property
    def name(self) -> str:
        return "concepts"

    @property
    def output_dir(self) -> str:
        return "07-concepts"

    # Key concepts to track in this domain
    KEY_CONCEPTS = [
        'pagerank', 'tfidf', 'bm25', 'louvain', 'semantic',
        'query expansion', 'fingerprint', 'bigram', 'minicolumn',
        'hebbian', 'lateral connections', 'staleness', 'observability',
        'clustering', 'graph', 'embeddings', 'tokenization', 'persistence',
        'incremental', 'search', 'retrieval', 'passage', 'intent',
        'definition', 'analogy', 'similarity', 'context'
    ]

    def _run_git(self, *args) -> str:
        """Run git command and return output."""
        import subprocess
        result = subprocess.run(
            ["git", *args],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()

    def _load_commits_with_timestamps(self) -> List[Dict[str, Any]]:
        """Load commits from git history with full timestamp data."""
        try:
            log_output = self._run_git(
                "log",
                "--all",
                "--format=%H|%aI|%s|%an"
            )
        except Exception as e:
            print(f"  Warning: Failed to read git history: {e}")
            return []

        commits = []
        for line in log_output.split('\n'):
            if not line.strip():
                continue
            parts = line.split('|', 3)
            if len(parts) < 4:
                continue
            hash_val, timestamp, message, author = parts

            # Parse timestamp
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except Exception:
                continue

            commits.append({
                'hash': hash_val,
                'short_hash': hash_val[:7],
                'timestamp': timestamp,
                'datetime': dt,
                'message': message,
                'author': author,
                'date': timestamp[:10],  # YYYY-MM-DD
                'week': dt.strftime("%Y-W%U"),  # Week number
                'month': dt.strftime("%Y-%m")
            })

        # Sort chronologically (oldest first for evolution tracking)
        commits.sort(key=lambda c: c['datetime'])
        return commits

    def _identify_concepts(self, commits: List[Dict[str, Any]]) -> List[str]:
        """Identify which concepts appear in commit history."""
        concept_mentions = defaultdict(int)

        for commit in commits:
            message_lower = commit['message'].lower()
            for concept in self.KEY_CONCEPTS:
                if concept.lower() in message_lower:
                    concept_mentions[concept] += 1

        # Return concepts with at least 3 mentions, sorted by frequency
        significant_concepts = [
            concept for concept, count in concept_mentions.items()
            if count >= 3
        ]
        significant_concepts.sort(key=lambda c: concept_mentions[c], reverse=True)

        return significant_concepts[:15]  # Top 15 concepts

    def _track_concept_evolution(self, concept: str, commits: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Track how a concept evolved over time."""
        evolution = {
            'concept': concept,
            'first_seen': None,
            'first_commit': None,
            'total_mentions': 0,
            'by_week': defaultdict(list),
            'by_month': defaultdict(list),
            'peak_week': None,
            'peak_count': 0,
            'all_commits': []
        }

        for commit in commits:
            if concept.lower() in commit['message'].lower():
                if evolution['first_seen'] is None:
                    evolution['first_seen'] = commit['date']
                    evolution['first_commit'] = commit

                evolution['total_mentions'] += 1
                evolution['by_week'][commit['week']].append(commit)
                evolution['by_month'][commit['month']].append(commit)
                evolution['all_commits'].append(commit)

        # Find peak activity week
        for week, week_commits in evolution['by_week'].items():
            if len(week_commits) > evolution['peak_count']:
                evolution['peak_count'] = len(week_commits)
                evolution['peak_week'] = week

        return evolution

    def _group_by_time_period(self, commits: List[Dict[str, Any]], period: str = 'month') -> Dict[str, List[Dict[str, Any]]]:
        """Group commits by time period (week or month)."""
        grouped = defaultdict(list)

        for commit in commits:
            if period == 'week':
                key = commit['week']
            elif period == 'month':
                key = commit['month']
            else:
                key = commit['date']
            grouped[key].append(commit)

        return dict(grouped)

    def _find_related_concepts(self, concept: str, commits: List[Dict[str, Any]]) -> List[Tuple[str, int]]:
        """Find concepts that co-occur with the given concept."""
        co_occurrence = defaultdict(int)

        # Find commits that mention this concept
        concept_commits = [c for c in commits if concept.lower() in c['message'].lower()]

        # Check for other concepts in those commits
        for commit in concept_commits:
            message_lower = commit['message'].lower()
            for other_concept in self.KEY_CONCEPTS:
                if other_concept != concept and other_concept.lower() in message_lower:
                    co_occurrence[other_concept] += 1

        # Return sorted by frequency
        related = [(c, count) for c, count in co_occurrence.items() if count > 0]
        related.sort(key=lambda x: x[1], reverse=True)
        return related[:5]  # Top 5 related concepts

    def _generate_concept_chapter(self, concept: str, evolution: Dict[str, Any],
                                  related: List[Tuple[str, int]], dry_run: bool = False) -> Optional[Path]:
        """Generate a chapter for a single concept's evolution."""

        content = self.generate_frontmatter(
            title=f"Concept Evolution: {concept.title()}",
            tags=["concept", "evolution", concept.lower().replace(' ', '-')],
            source_files=["git log"]
        )

        content += f"# Concept Evolution: {concept.title()}\n\n"
        content += f"*Tracking the emergence and growth of '{concept}' through commit history.*\n\n"
        content += "---\n\n"

        # Birth section
        content += "## Birth\n\n"
        if evolution['first_commit']:
            fc = evolution['first_commit']
            content += f"**First Appearance:** {fc['date']}\n\n"
            content += f"The concept of '{concept}' first emerged in commit `{fc['short_hash']}`:\n\n"
            content += f"> {fc['message']}\n\n"
            content += f"*By {fc['author']}*\n\n"
        else:
            content += f"The exact birth of '{concept}' is not captured in recent commit history.\n\n"

        # Growth Timeline
        content += "## Growth Timeline\n\n"

        # Group by month for timeline
        by_month = self._group_by_time_period(evolution['all_commits'], period='month')

        if by_month:
            content += f"The concept has been mentioned in **{evolution['total_mentions']} commits** "
            content += f"across **{len(by_month)} months** of development.\n\n"

            # Show first few months of activity
            sorted_months = sorted(by_month.keys())
            for i, month in enumerate(sorted_months[:5]):  # First 5 months
                month_commits = by_month[month]
                month_name = datetime.strptime(month, "%Y-%m").strftime("%B %Y")

                if i == 0:
                    content += f"### {month_name}: Emergence\n\n"
                elif i == len(sorted_months) - 1 or i == 4:
                    content += f"### {month_name}: Current State\n\n"
                else:
                    content += f"### {month_name}: Expansion\n\n"

                content += f"**{len(month_commits)} commits** mentioning this concept.\n\n"

                # Show key commits (max 3 per month)
                for commit in month_commits[:3]:
                    content += f"- `{commit['short_hash']}`: {commit['message']}\n"

                if len(month_commits) > 3:
                    content += f"- *(and {len(month_commits) - 3} more)*\n"

                content += "\n"

            if len(sorted_months) > 5:
                content += f"*...and {len(sorted_months) - 5} more months of development.*\n\n"

        # Peak Activity
        if evolution['peak_week']:
            content += "## Peak Activity\n\n"
            content += f"The concept saw its most intensive development during week **{evolution['peak_week']}** "
            content += f"with **{evolution['peak_count']} commits**.\n\n"

        # Related Concepts
        if related:
            content += "## Related Concepts\n\n"
            content += f"The '{concept}' concept frequently appears alongside:\n\n"
            for related_concept, count in related:
                content += f"- **{related_concept.title()}** ({count} co-occurrences)\n"
            content += "\n"

        # The Concept Today
        content += "## The Concept Today\n\n"
        if evolution['all_commits']:
            recent = evolution['all_commits'][-1]
            content += f"Most recent mention was on {recent['date']}:\n\n"
            content += f"> {recent['message']}\n\n"
            content += f"This concept has evolved from its initial appearance to become "
            if evolution['total_mentions'] >= 20:
                content += "a **core component** of the system.\n\n"
            elif evolution['total_mentions'] >= 10:
                content += "an **important feature** of the architecture.\n\n"
            else:
                content += "an **emerging aspect** of the design.\n\n"

        # Key Commits
        content += "## Key Commits\n\n"
        content += "Notable commits that shaped this concept:\n\n"

        # Show first, middle, and last commits
        key_commits = []
        if len(evolution['all_commits']) >= 1:
            key_commits.append(('First', evolution['all_commits'][0]))
        if len(evolution['all_commits']) >= 3:
            mid_idx = len(evolution['all_commits']) // 2
            key_commits.append(('Midpoint', evolution['all_commits'][mid_idx]))
        if len(evolution['all_commits']) >= 2:
            key_commits.append(('Latest', evolution['all_commits'][-1]))

        for label, commit in key_commits:
            content += f"### {label}: `{commit['short_hash']}`\n\n"
            content += f"**Date:** {commit['date']}\n\n"
            content += f"**Message:** {commit['message']}\n\n"

        # Write the chapter
        filename = f"{concept.lower().replace(' ', '-')}.md"
        return self.write_chapter(filename, content, dry_run)

    def generate(self, dry_run: bool = False, verbose: bool = False) -> Dict[str, Any]:
        """Generate concept evolution chapters from git history."""
        if verbose:
            print("  Loading git history for concept tracking...")

        commits = self._load_commits_with_timestamps()

        if not commits:
            return {
                "files": [],
                "stats": {"commits": 0, "error": "No git history found"},
                "errors": ["Could not read git history"]
            }

        if verbose:
            print(f"  Analyzing {len(commits)} commits...")

        # Identify concepts
        concepts = self._identify_concepts(commits)

        if verbose:
            print(f"  Found {len(concepts)} significant concepts")

        if not concepts:
            return {
                "files": [],
                "stats": {"commits": len(commits), "concepts": 0},
                "errors": ["No significant concepts found"]
            }

        # Generate chapter for each concept
        concept_evolutions = []
        for concept in concepts:
            if verbose:
                print(f"    Tracking '{concept}'...")

            evolution = self._track_concept_evolution(concept, commits)
            related = self._find_related_concepts(concept, commits)

            concept_evolutions.append({
                'concept': concept,
                'evolution': evolution,
                'related': related
            })

            # Generate chapter
            self._generate_concept_chapter(concept, evolution, related, dry_run)

        # Generate index
        self._generate_index(concept_evolutions, dry_run, verbose)

        stats = {
            "total_commits": len(commits),
            "concepts_tracked": len(concepts),
            "chapters_generated": len(concepts) + 1  # +1 for index
        }

        return {
            "files": [str(f) for f in self.generated_files],
            "stats": stats,
            "errors": []
        }

    def _generate_index(self, concept_evolutions: List[Dict[str, Any]],
                       dry_run: bool = False, verbose: bool = False):
        """Generate index of all concepts."""
        if verbose:
            print("  Generating concept index...")

        content = self.generate_frontmatter(
            title="Concept Evolution Index",
            tags=["concepts", "index", "evolution"],
            source_files=["git log"]
        )

        content += "# Concept Evolution Index\n\n"
        content += "*A guide to how key concepts emerged and grew in the Cortical Text Processor.*\n\n"
        content += "---\n\n"

        content += "## Overview\n\n"
        content += f"This section tracks the evolution of **{len(concept_evolutions)} core concepts** "
        content += "through the project's commit history. Each concept chapter shows:\n\n"
        content += "- When the concept first appeared\n"
        content += "- How it grew over time\n"
        content += "- Related concepts and connections\n"
        content += "- Current state and importance\n\n"

        content += "## Concepts by Importance\n\n"

        # Sort by total mentions
        sorted_concepts = sorted(
            concept_evolutions,
            key=lambda x: x['evolution']['total_mentions'],
            reverse=True
        )

        for item in sorted_concepts:
            concept = item['concept']
            evolution = item['evolution']

            content += f"### [{concept.title()}]({concept.lower().replace(' ', '-')}.md)\n\n"
            content += f"**Mentions:** {evolution['total_mentions']} commits\n\n"

            if evolution['first_seen']:
                content += f"**First seen:** {evolution['first_seen']}\n\n"

            if item['related']:
                related_names = [r[0].title() for r in item['related'][:3]]
                content += f"**Related to:** {', '.join(related_names)}\n\n"

            # Brief description based on frequency
            if evolution['total_mentions'] >= 20:
                content += "*A core component of the system architecture.*\n\n"
            elif evolution['total_mentions'] >= 10:
                content += "*An important feature in the implementation.*\n\n"
            else:
                content += "*An emerging concept in recent development.*\n\n"

        content += "---\n\n"
        content += "*Each concept chapter provides detailed evolution timeline and key commits.*\n"

        self.write_chapter("index.md", content, dry_run)


class ReaderJourneyGenerator(ChapterGenerator):
    """Generate reader journey chapters with progressive learning paths."""

    def __init__(self, book_dir: Path = BOOK_DIR, repo_root: Optional[Path] = None):
        super().__init__(book_dir)
        self.repo_root = repo_root or Path(__file__).parent.parent
        self.ml_data_file = self.repo_root / ".git-ml" / "tracked" / "commits.jsonl"

    @property
    def name(self) -> str:
        return "journey"

    @property
    def output_dir(self) -> str:
        return "09-journey"

    # Concept levels based on complexity and prerequisites
    CONCEPT_LEVELS = {
        'beginner': [
            'tokenization', 'stemming', 'stop words', 'tf-idf',
            'document processing', 'text analysis', 'term frequency'
        ],
        'intermediate': [
            'bigrams', 'pagerank', 'bm25', 'query expansion',
            'lateral connections', 'minicolumn', 'hierarchical layers',
            'persistence', 'incremental indexing'
        ],
        'advanced': [
            'louvain', 'semantic relations', 'graph embeddings', 'fingerprinting',
            'hebbian learning', 'concept clustering', 'retrofitting',
            'observability', 'staleness tracking'
        ]
    }

    # Reading time estimates (in minutes)
    READING_TIMES = {
        'beginner': 15,
        'intermediate': 25,
        'advanced': 35
    }

    def _run_git(self, *args) -> str:
        """Run git command and return output."""
        import subprocess
        result = subprocess.run(
            ["git", *args],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()

    def _load_commits_with_timestamps(self) -> List[Dict[str, Any]]:
        """Load commits from git history with timestamps."""
        try:
            log_output = self._run_git(
                "log",
                "--all",
                "--format=%H|%aI|%s"
            )
        except Exception as e:
            print(f"  Warning: Failed to read git history: {e}")
            return []

        commits = []
        for line in log_output.split('\n'):
            if not line.strip():
                continue
            parts = line.split('|', 2)
            if len(parts) < 3:
                continue
            hash_val, timestamp, message = parts

            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except Exception:
                continue

            commits.append({
                'hash': hash_val,
                'timestamp': timestamp,
                'datetime': dt,
                'message': message,
                'date': timestamp[:10]
            })

        commits.sort(key=lambda c: c['datetime'])
        return commits

    def _classify_concepts(self, commits: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Classify concepts by level based on commit history analysis."""
        concept_data = {
            'beginner': [],
            'intermediate': [],
            'advanced': []
        }

        # Analyze each concept level
        for level, concepts in self.CONCEPT_LEVELS.items():
            for concept in concepts:
                # Find when this concept first appeared
                first_mention = None
                total_mentions = 0
                related_concepts = set()

                for commit in commits:
                    message_lower = commit['message'].lower()
                    if concept.lower() in message_lower:
                        if first_mention is None:
                            first_mention = commit['date']
                        total_mentions += 1

                        # Find co-occurring concepts
                        for other_level, other_concepts in self.CONCEPT_LEVELS.items():
                            for other_concept in other_concepts:
                                if other_concept != concept and other_concept.lower() in message_lower:
                                    related_concepts.add(other_concept)

                if total_mentions > 0:
                    concept_data[level].append({
                        'concept': concept,
                        'first_seen': first_mention,
                        'mentions': total_mentions,
                        'related': sorted(list(related_concepts))[:3],
                        'reading_time': self.READING_TIMES[level]
                    })

        # Sort each level by first appearance (foundational concepts appear first)
        for level in concept_data:
            concept_data[level].sort(key=lambda x: (x['first_seen'] or '9999', -x['mentions']))

        return concept_data

    def _find_prerequisites(self, concept: str, all_concepts: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        """Find prerequisites for a concept based on related concepts in lower levels."""
        prerequisites = []

        # Find the level of this concept
        concept_level = None
        for level, concepts in self.CONCEPT_LEVELS.items():
            if concept in concepts:
                concept_level = level
                break

        if not concept_level:
            return []

        # Look for related concepts in prerequisite levels
        level_order = ['beginner', 'intermediate', 'advanced']
        if concept_level in level_order:
            level_idx = level_order.index(concept_level)
            prereq_levels = level_order[:level_idx]

            for level in prereq_levels:
                for concept_data in all_concepts[level]:
                    if concept in concept_data.get('related', []) or \
                       concept_data['concept'] in self._get_related_terms(concept):
                        prerequisites.append(concept_data['concept'])

        return prerequisites[:3]  # Max 3 prerequisites

    def _get_related_terms(self, concept: str) -> Set[str]:
        """Get terms related to a concept based on domain knowledge."""
        relations = {
            'pagerank': {'lateral connections', 'graph'},
            'bm25': {'tf-idf', 'term frequency'},
            'query expansion': {'pagerank', 'lateral connections'},
            'semantic relations': {'query expansion', 'graph embeddings'},
            'louvain': {'clustering', 'graph'},
            'fingerprinting': {'semantic relations', 'embeddings'},
            'hebbian learning': {'lateral connections', 'co-occurrence'},
            'concept clustering': {'louvain', 'graph'},
            'staleness tracking': {'incremental indexing', 'persistence'}
        }
        return relations.get(concept, set())

    def _estimate_reading_time(self, concepts: List[Dict[str, Any]]) -> int:
        """Estimate total reading time for a list of concepts."""
        return sum(c['reading_time'] for c in concepts)

    def _generate_journey_chapter(self, level: str, concepts: List[Dict[str, Any]],
                                  all_concepts: Dict[str, List[Dict[str, Any]]],
                                  dry_run: bool = False) -> Optional[Path]:
        """Generate a journey chapter for a specific learning level."""
        level_titles = {
            'beginner': 'Start Here (Foundational)',
            'intermediate': 'Going Deeper (Intermediate)',
            'advanced': 'Mastery Path (Advanced)'
        }

        level_descriptions = {
            'beginner': 'Core concepts that unlock everything else. Start here if you\'re new to information retrieval.',
            'intermediate': 'Build on the foundations. These concepts add power and flexibility to your understanding.',
            'advanced': 'Deep dives into sophisticated algorithms. For those ready to master the full system.'
        }

        content = self.generate_frontmatter(
            title=f"Learning Journey: {level.title()}",
            tags=["journey", "learning-path", level],
            source_files=["git log", "CLAUDE.md"]
        )

        content += f"# {level_titles[level]}\n\n"
        content += f"*{level_descriptions[level]}*\n\n"
        content += "---\n\n"

        if not concepts:
            content += f"*No {level} concepts found in commit history.*\n\n"
        else:
            # Summary stats
            total_time = self._estimate_reading_time(concepts)
            content += f"**Concepts:** {len(concepts)}\n\n"
            content += f"**Estimated Time:** ~{total_time} minutes\n\n"
            content += "---\n\n"

            # List each concept with details
            for i, concept_data in enumerate(concepts, 1):
                concept = concept_data['concept']
                content += f"## {i}. {concept.title()}\n\n"

                # Description and context
                content += f"**First Introduced:** {concept_data['first_seen']}\n\n"
                content += f"**Mentions in History:** {concept_data['mentions']} commits\n\n"
                content += f"**Reading Time:** ~{concept_data['reading_time']} min\n\n"

                # Prerequisites
                prerequisites = self._find_prerequisites(concept, all_concepts)
                if prerequisites:
                    content += f"**Prerequisites:**\n"
                    for prereq in prerequisites:
                        content += f"- {prereq.title()}\n"
                    content += "\n"
                else:
                    content += "**Prerequisites:** None (foundational concept)\n\n"

                # Related concepts
                if concept_data['related']:
                    content += f"**Related Concepts:**\n"
                    for related in concept_data['related']:
                        content += f"- {related.title()}\n"
                    content += "\n"

                # Where to find it
                content += "**Where to Learn:**\n"
                content += f"- Foundations: `book/01-foundations/`\n"
                content += f"- Modules: `book/02-modules/`\n"
                content += f"- Evolution: `book/07-concepts/{concept.lower().replace(' ', '-')}.md`\n"
                content += "\n"

                # Key takeaways (generic, could be enhanced with actual content)
                content += "**Key Takeaway:**\n"
                content += self._generate_key_takeaway(concept, level)
                content += "\n"
                content += "---\n\n"

        # Write the chapter
        filename = f"journey-{level}.md"
        return self.write_chapter(filename, content, dry_run)

    def _generate_key_takeaway(self, concept: str, level: str) -> str:
        """Generate a key takeaway for a concept."""
        takeaways = {
            'tokenization': 'Learn how text is broken into processable units - the foundation of all text analysis.',
            'stemming': 'Understand how words are reduced to their roots to improve matching and recall.',
            'stop words': 'See why common words are filtered to focus on meaningful content.',
            'tf-idf': 'Master the classic algorithm for term importance - weighing frequency against distinctiveness.',
            'bigrams': 'Discover how word pairs capture context and improve search relevance.',
            'pagerank': 'Learn how graph algorithms measure importance through connections.',
            'bm25': 'Understand the modern scoring function that improves on TF-IDF with saturation.',
            'query expansion': 'See how searches become smarter by exploring related terms.',
            'lateral connections': 'Grasp the Hebbian-inspired network of term relationships.',
            'louvain': 'Explore community detection for automatic concept clustering.',
            'semantic relations': 'Dive into pattern-based extraction of typed relationships.',
            'graph embeddings': 'Master vector representations derived from graph structure.',
            'fingerprinting': 'Understand semantic signatures for document similarity.',
            'staleness tracking': 'Learn the incremental computation system that keeps data fresh.',
            'observability': 'See how metrics collection enables performance optimization.'
        }
        return takeaways.get(concept, f'Understand the role of {concept} in the information retrieval pipeline.')

    def _generate_study_schedule(self, concept_data: Dict[str, List[Dict[str, Any]]]) -> str:
        """Generate a week-by-week study schedule."""
        content = "# Suggested Study Schedule\n\n"
        content += "*A practical 4-week plan to master the Cortical Text Processor.*\n\n"
        content += "---\n\n"

        # Week 1: Beginner
        beginner_concepts = concept_data['beginner']
        if beginner_concepts:
            content += "## Week 1: Foundations\n\n"
            content += "**Goal:** Understand the core building blocks\n\n"
            content += "**Concepts:**\n"
            for concept in beginner_concepts[:5]:  # First 5
                content += f"- {concept['concept'].title()} (~{concept['reading_time']} min)\n"
            content += "\n"
            total_time = sum(c['reading_time'] for c in beginner_concepts[:5])
            content += f"**Total Time:** ~{total_time} minutes\n\n"
            content += "**Activities:**\n"
            content += "- Read foundation chapters\n"
            content += "- Run `showcase.py` to see concepts in action\n"
            content += "- Experiment with basic tokenization and TF-IDF\n\n"

        # Week 2: Intermediate Part 1
        intermediate_concepts = concept_data['intermediate']
        if intermediate_concepts:
            mid_point = len(intermediate_concepts) // 2
            content += "## Week 2: Building Complexity\n\n"
            content += "**Goal:** Add graph-based features to your mental model\n\n"
            content += "**Concepts:**\n"
            for concept in intermediate_concepts[:mid_point]:
                content += f"- {concept['concept'].title()} (~{concept['reading_time']} min)\n"
            content += "\n"
            total_time = sum(c['reading_time'] for c in intermediate_concepts[:mid_point])
            content += f"**Total Time:** ~{total_time} minutes\n\n"
            content += "**Activities:**\n"
            content += "- Study PageRank and BM25 implementations\n"
            content += "- Index a sample corpus with `scripts/index_codebase.py`\n"
            content += "- Explore query expansion behavior\n\n"

            # Week 3: Intermediate Part 2
            content += "## Week 3: Advanced Structures\n\n"
            content += "**Goal:** Understand hierarchical organization and persistence\n\n"
            content += "**Concepts:**\n"
            for concept in intermediate_concepts[mid_point:]:
                content += f"- {concept['concept'].title()} (~{concept['reading_time']} min)\n"
            content += "\n"
            total_time = sum(c['reading_time'] for c in intermediate_concepts[mid_point:])
            content += f"**Total Time:** ~{total_time} minutes\n\n"
            content += "**Activities:**\n"
            content += "- Review minicolumn and layer architecture\n"
            content += "- Practice save/load operations\n"
            content += "- Test incremental indexing\n\n"

        # Week 4: Advanced
        advanced_concepts = concept_data['advanced']
        if advanced_concepts:
            content += "## Week 4: Mastery\n\n"
            content += "**Goal:** Master sophisticated algorithms and optimization\n\n"
            content += "**Concepts:**\n"
            for concept in advanced_concepts[:6]:  # Top 6 advanced
                content += f"- {concept['concept'].title()} (~{concept['reading_time']} min)\n"
            content += "\n"
            total_time = sum(c['reading_time'] for c in advanced_concepts[:6])
            content += f"**Total Time:** ~{total_time} minutes\n\n"
            content += "**Activities:**\n"
            content += "- Study Louvain clustering implementation\n"
            content += "- Experiment with semantic relations extraction\n"
            content += "- Profile performance with `scripts/profile_full_analysis.py`\n"
            content += "- Implement a custom feature using the library\n\n"

        # Summary
        content += "## Tips for Success\n\n"
        content += "1. **Follow the order** - Prerequisites build on each other\n"
        content += "2. **Code along** - Run examples from `showcase.py` and scripts\n"
        content += "3. **Read tests** - `tests/` directory shows real usage patterns\n"
        content += "4. **Ask questions** - Use the semantic search: `python scripts/search_codebase.py \"your question\"`\n"
        content += "5. **Build something** - Best way to learn is to apply the concepts\n\n"

        return content

    def generate(self, dry_run: bool = False, verbose: bool = False) -> Dict[str, Any]:
        """Generate reader journey chapters with progressive learning paths."""
        if verbose:
            print("  Loading commit history for journey generation...")

        commits = self._load_commits_with_timestamps()

        if not commits:
            # Generate empty journey if no commits
            if verbose:
                print("  Warning: No commit history found, generating minimal journey")

            empty_content = self.generate_frontmatter(
                title="Learning Journey",
                tags=["journey", "learning-path"],
                source_files=["CLAUDE.md"]
            )
            empty_content += "# Your Learning Journey\n\n"
            empty_content += "*Journey chapters will be generated once commit history is available.*\n\n"

            self.write_chapter("index.md", empty_content, dry_run)

            return {
                "files": [str(f) for f in self.generated_files],
                "stats": {"commits": 0, "concepts": 0},
                "errors": ["No commit history available"]
            }

        if verbose:
            print(f"  Analyzing {len(commits)} commits for concept classification...")

        # Classify concepts by level
        concept_data = self._classify_concepts(commits)

        total_concepts = sum(len(concepts) for concepts in concept_data.values())

        if verbose:
            print(f"  Found {total_concepts} concepts across all levels")
            for level, concepts in concept_data.items():
                print(f"    {level}: {len(concepts)} concepts")

        # Generate journey chapters for each level
        for level in ['beginner', 'intermediate', 'advanced']:
            if verbose:
                print(f"  Generating {level} journey chapter...")
            self._generate_journey_chapter(level, concept_data[level], concept_data, dry_run)

        # Generate index with overview and study schedule
        if verbose:
            print("  Generating journey index...")

        index_content = self.generate_frontmatter(
            title="Your Learning Journey",
            tags=["journey", "learning-path", "index"],
            source_files=["git log", "CLAUDE.md"]
        )

        index_content += "# Your Learning Journey\n\n"
        index_content += "*A progressive path through the Cortical Text Processor, designed for learners at all levels.*\n\n"
        index_content += "---\n\n"

        index_content += "## Overview\n\n"
        index_content += "This learning journey is organized into three progressive stages, "
        index_content += "each building on the previous one. The concepts are ordered based on:\n\n"
        index_content += "- **Dependencies:** What you need to know first\n"
        index_content += "- **Complexity:** From simple to sophisticated\n"
        index_content += "- **Historical emergence:** When concepts first appeared in development\n\n"

        # Learning paths summary
        index_content += "## Learning Paths\n\n"

        for level in ['beginner', 'intermediate', 'advanced']:
            concepts = concept_data[level]
            if concepts:
                total_time = self._estimate_reading_time(concepts)
                index_content += f"### [{level.title()} Path](journey-{level}.md)\n\n"
                index_content += f"**Concepts:** {len(concepts)}  \n"
                index_content += f"**Time:** ~{total_time} minutes  \n"

                # Show first 3 concepts
                preview_concepts = [c['concept'].title() for c in concepts[:3]]
                index_content += f"**Preview:** {', '.join(preview_concepts)}"
                if len(concepts) > 3:
                    index_content += f", and {len(concepts) - 3} more"
                index_content += "\n\n"

        # Add study schedule
        index_content += "---\n\n"
        index_content += self._generate_study_schedule(concept_data)

        # Write index
        self.write_chapter("index.md", index_content, dry_run)

        stats = {
            "total_commits": len(commits),
            "total_concepts": total_concepts,
            "beginner": len(concept_data['beginner']),
            "intermediate": len(concept_data['intermediate']),
            "advanced": len(concept_data['advanced']),
            "chapters_generated": 4  # index + 3 levels
        }

        return {
            "files": [str(f) for f in self.generated_files],
            "stats": stats,
            "errors": []
        }


class ExerciseGenerator(ChapterGenerator):
    """Generate exercise chapters from test cases."""

    # Topic mapping for organizing exercises
    EXERCISE_TOPICS = {
        'foundations': {
            'files': ['test_analysis.py', 'test_layers.py', 'test_tokenizer.py', 'test_minicolumn.py'],
            'difficulty': 'Beginner',
            'output': 'ex-foundations.md'
        },
        'search': {
            'files': ['test_query_search.py', 'test_query_expansion.py', 'test_query_ranking.py',
                     'test_query_passages.py', 'test_query_definitions.py'],
            'difficulty': 'Intermediate',
            'output': 'ex-search.md'
        },
        'advanced': {
            'files': ['test_semantics.py', 'test_fingerprint.py', 'test_embeddings.py',
                     'test_gaps.py', 'test_patterns.py'],
            'difficulty': 'Advanced',
            'output': 'ex-advanced.md'
        }
    }

    def __init__(self, book_dir: Path = BOOK_DIR, repo_root: Optional[Path] = None):
        super().__init__(book_dir)
        self.repo_root = repo_root or Path(__file__).parent.parent
        self.tests_dir = self.repo_root / "tests" / "unit"

    @property
    def name(self) -> str:
        return "exercises"

    @property
    def output_dir(self) -> str:
        return "08-exercises"

    def _scan_test_files(self, topic_files: List[str]) -> List[Path]:
        """Scan for test files matching topic."""
        found_files = []
        for filename in topic_files:
            test_file = self.tests_dir / filename
            if test_file.exists():
                found_files.append(test_file)
        return found_files

    def _extract_test_methods(self, test_file: Path) -> List[Dict]:
        """Extract test methods from a test file."""
        import ast
        import inspect

        try:
            source = test_file.read_text()
            tree = ast.parse(source)
        except Exception:
            return []

        tests = []
        current_class = None

        for node in ast.walk(tree):
            # Track current class for context
            if isinstance(node, ast.ClassDef):
                current_class = node.name
                class_doc = ast.get_docstring(node) or ""

            # Find test methods
            if isinstance(node, ast.FunctionDef):
                if node.name.startswith('test_'):
                    docstring = ast.get_docstring(node) or ""

                    # Extract source code for this method
                    try:
                        method_source = ast.get_source_segment(source, node)
                    except:
                        method_source = ""

                    tests.append({
                        'name': node.name,
                        'class': current_class,
                        'docstring': docstring,
                        'source': method_source,
                        'file': test_file.name,
                        'lineno': node.lineno
                    })

        return tests

    def _is_good_exercise(self, test: Dict) -> bool:
        """Determine if a test makes a good exercise."""
        # Must have a clear docstring
        if not test['docstring'] or len(test['docstring']) < 20:
            return False

        # Must have source code
        if not test['source'] or len(test['source']) < 50:
            return False

        # Avoid fixture setup tests
        if 'fixture' in test['name'].lower():
            return False

        # Avoid internal implementation tests
        if test['name'].startswith('test_internal') or test['name'].startswith('test__'):
            return False

        # Prefer tests with clear conceptual focus
        good_keywords = ['basic', 'simple', 'empty', 'single', 'chain', 'cycle',
                        'expansion', 'similarity', 'compare', 'compute']
        name_lower = test['name'].lower()

        return any(keyword in name_lower for keyword in good_keywords)

    def _generate_hints(self, test: Dict) -> List[str]:
        """Generate progressive hints for an exercise."""
        hints = []
        source = test['source']

        # Hint 1: General approach
        if 'empty' in test['name'].lower():
            hints.append("Start by considering the edge case of empty input.")
        elif 'single' in test['name'].lower():
            hints.append("Focus on the simplest case with just one element.")
        elif 'chain' in test['name'].lower() or 'cycle' in test['name'].lower():
            hints.append("Think about how elements connect in sequence.")
        else:
            hints.append("Break down the problem into smaller steps.")

        # Hint 2: API methods (extract from imports/usage)
        if 'compute_fingerprint' in source:
            hints.append("Use the `compute_fingerprint()` function from cortical.fingerprint")
        elif 'expand_query' in source:
            hints.append("The `expand_query()` function can help find related terms")
        elif 'pagerank' in source.lower():
            hints.append("PageRank is computed with `compute_pagerank()` or `compute_importance()`")
        elif 'MockLayers' in source or 'MockMinicolumn' in source:
            hints.append("You may need to create mock layers or minicolumns for testing")

        # Hint 3: Expected behavior (from assertions)
        if 'assert' in source:
            # Extract first assertion as hint about expected behavior
            lines = source.split('\n')
            for line in lines:
                if 'assert' in line and '==' in line:
                    hints.append("Check the expected value and comparison in the assertion")
                    break

        return hints[:3]  # Maximum 3 hints

    def _transform_to_exercise(self, test: Dict, difficulty: str, topic: str) -> str:
        """Transform a test into an exercise format."""
        # Clean up test name for title
        title = test['name'].replace('test_', '').replace('_', ' ').title()

        # Extract concept from docstring (first line)
        concept = test['docstring'].split('\n')[0].strip('. ')

        # Estimate time based on complexity
        time = "10"
        if 'advanced' in topic.lower() or len(test['source']) > 200:
            time = "20"
        elif len(test['source']) < 100:
            time = "5"

        # Generate hints
        hints = self._generate_hints(test)

        # Build exercise content
        content = f"## Exercise: {title}\n\n"
        content += f"**Concept:** {concept}\n\n"
        content += f"**Difficulty:** {difficulty}\n\n"
        content += f"**Time:** ~{time} minutes\n\n"
        content += f"**Source:** `{test['file']}`\n\n"

        content += "### Setup\n\n"
        content += "```python\n"
        content += "from cortical import CorticalTextProcessor\n"

        # Add specific imports based on test content
        if 'Tokenizer' in test['source']:
            content += "from cortical.tokenizer import Tokenizer\n"
        if 'fingerprint' in test['source']:
            content += "from cortical.fingerprint import compute_fingerprint, compare_fingerprints\n"
        if 'query.expansion' in test['source']:
            content += "from cortical.query.expansion import expand_query\n"
        if 'analysis' in test['source']:
            content += "from cortical.analysis import _pagerank_core, _tfidf_core\n"

        content += "```\n\n"

        content += "### Your Task\n\n"
        content += f"{test['docstring']}\n\n"

        # Add hints
        if hints:
            content += "### Hints\n\n"
            for i, hint in enumerate(hints, 1):
                content += f"<details>\n"
                content += f"<summary>Hint {i}</summary>\n\n"
                content += f"{hint}\n\n"
                content += f"</details>\n\n"

        # Add solution
        content += "### Solution\n\n"
        content += "<details>\n"
        content += "<summary>Click to reveal</summary>\n\n"
        content += "```python\n"
        content += test['source']
        content += "\n```\n\n"
        content += "</details>\n\n"

        content += "---\n\n"

        return content

    def _generate_topic_chapter(self, topic: str, config: Dict, tests: List[Dict],
                               dry_run: bool = False, verbose: bool = False) -> Optional[Path]:
        """Generate a chapter for a specific topic."""
        if verbose:
            print(f"    Generating {topic} exercises...")

        difficulty = config['difficulty']

        content = self.generate_frontmatter(
            title=f"Exercises: {topic.title()}",
            tags=["exercises", topic, difficulty.lower()],
            source_files=config['files']
        )

        content += f"# {topic.title()} Exercises\n\n"
        content += f"*Hands-on coding exercises to master {topic} concepts.*\n\n"
        content += f"**Difficulty Level:** {difficulty}\n\n"
        content += "---\n\n"

        # Add introduction
        if topic == 'foundations':
            content += "## Introduction\n\n"
            content += "These exercises cover the fundamental algorithms and data structures "
            content += "of the Cortical Text Processor:\n\n"
            content += "- PageRank for term importance\n"
            content += "- TF-IDF for relevance scoring\n"
            content += "- Graph structures and connections\n"
            content += "- Tokenization and text processing\n\n"
        elif topic == 'search':
            content += "## Introduction\n\n"
            content += "Master the search and retrieval capabilities:\n\n"
            content += "- Query expansion techniques\n"
            content += "- Document ranking algorithms\n"
            content += "- Passage retrieval\n"
            content += "- Definition extraction\n\n"
        elif topic == 'advanced':
            content += "## Introduction\n\n"
            content += "Challenge yourself with advanced features:\n\n"
            content += "- Semantic relation extraction\n"
            content += "- Fingerprint-based similarity\n"
            content += "- Graph embeddings\n"
            content += "- Knowledge gap detection\n\n"

        # Add exercises
        exercise_count = 0
        for test in tests:
            if self._is_good_exercise(test):
                content += self._transform_to_exercise(test, difficulty, topic)
                exercise_count += 1

                # Limit exercises per topic
                if exercise_count >= 10:
                    break

        if exercise_count == 0:
            content += "*No exercises available yet for this topic.*\n\n"

        content += "---\n\n"
        content += f"*Completed {exercise_count} exercises? "
        content += f"Check out the other topics for more challenges!*\n"

        return self.write_chapter(config['output'], content, dry_run)

    def _generate_index(self, stats: Dict, dry_run: bool = False, verbose: bool = False):
        """Generate exercise index."""
        if verbose:
            print("  Generating exercise index...")

        content = self.generate_frontmatter(
            title="Exercise Index",
            tags=["exercises", "index", "learning"],
            source_files=["tests/unit/*.py"]
        )

        content += "# Exercises\n\n"
        content += "*Hands-on coding exercises derived from the test suite.*\n\n"
        content += "---\n\n"

        content += "## Overview\n\n"
        content += "Learn by doing! These exercises are extracted from the Cortical Text Processor's "
        content += "test suite and transformed into learning challenges.\n\n"
        content += "Each exercise includes:\n\n"
        content += "- **Clear task description** - What you need to implement\n"
        content += "- **Progressive hints** - Guidance without spoilers\n"
        content += "- **Complete solution** - Reference implementation from tests\n"
        content += "- **Verification** - How to check your answer\n\n"

        content += "## Exercise Topics\n\n"

        # List topics
        for topic, config in self.EXERCISE_TOPICS.items():
            count = stats.get(topic, {}).get('exercise_count', 0)
            content += f"### [{topic.title()}]({config['output']})\n\n"
            content += f"**Difficulty:** {config['difficulty']}\n\n"
            content += f"**Exercises:** {count}\n\n"

            if topic == 'foundations':
                content += "Core algorithms and data structures. Start here if you're new!\n\n"
            elif topic == 'search':
                content += "Search, ranking, and retrieval techniques. Build on foundations.\n\n"
            elif topic == 'advanced':
                content += "Advanced features and complex algorithms. For experienced users.\n\n"

        content += "## Learning Path\n\n"
        content += "**Recommended progression:**\n\n"
        content += "1. Start with **Foundations** exercises\n"
        content += "2. Move to **Search** once comfortable\n"
        content += "3. Challenge yourself with **Advanced** topics\n\n"

        content += "## Tips for Success\n\n"
        content += "- **Read the test carefully** - The docstring explains what's being tested\n"
        content += "- **Use hints progressively** - Try solving first, then reveal hints as needed\n"
        content += "- **Run the solution** - Verify your understanding by executing the code\n"
        content += "- **Experiment** - Modify parameters and see how behavior changes\n\n"

        content += "---\n\n"
        content += f"*Total exercises: {sum(s.get('exercise_count', 0) for s in stats.values())}*\n"

        self.write_chapter("index.md", content, dry_run)

    def generate(self, dry_run: bool = False, verbose: bool = False) -> Dict[str, Any]:
        """Generate exercise chapters from test cases."""
        errors = []
        stats = {}

        if verbose:
            print(f"  Scanning tests in {self.tests_dir}...")

        # Check if tests directory exists
        if not self.tests_dir.exists():
            errors.append(f"Tests directory not found: {self.tests_dir}")
            return {
                "files": [],
                "stats": {"error": "Tests directory not found"},
                "errors": errors
            }

        # Generate chapter for each topic
        for topic, config in self.EXERCISE_TOPICS.items():
            if verbose:
                print(f"  Processing {topic} topic...")

            # Scan test files
            test_files = self._scan_test_files(config['files'])

            if not test_files and verbose:
                print(f"    Warning: No test files found for {topic}")

            # Extract tests from all files
            all_tests = []
            for test_file in test_files:
                tests = self._extract_test_methods(test_file)
                all_tests.extend(tests)

            # Generate topic chapter
            self._generate_topic_chapter(topic, config, all_tests, dry_run, verbose)

            # Track stats
            good_exercises = [t for t in all_tests if self._is_good_exercise(t)]
            stats[topic] = {
                'files_scanned': len(test_files),
                'tests_found': len(all_tests),
                'exercise_count': min(len(good_exercises), 10)  # Cap at 10
            }

        # Generate index
        self._generate_index(stats, dry_run, verbose)

        # Summary stats
        total_exercises = sum(s.get('exercise_count', 0) for s in stats.values())
        total_tests = sum(s.get('tests_found', 0) for s in stats.values())

        return {
            "files": [str(f) for f in self.generated_files],
            "stats": {
                "topics": len(self.EXERCISE_TOPICS),
                "total_tests_scanned": total_tests,
                "total_exercises": total_exercises,
                "by_topic": stats
            },
            "errors": errors
        }


def main():
    parser = argparse.ArgumentParser(
        description="Generate the Cortical Chronicles - a self-documenting living book",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s                    Generate full book (individual chapters)
    %(prog)s --markdown         Generate consolidated markdown file (BOOK.md)
    %(prog)s --markdown --force Force regeneration (ignore cache)
    %(prog)s --chapter foundations  Generate only foundations chapter
    %(prog)s --dry-run          Show what would be generated
    %(prog)s --verbose          Detailed progress output
    %(prog)s --list             List available generators
        """
    )
    parser.add_argument("--chapter", "-c", help="Generate specific chapter only")
    parser.add_argument("--markdown", "-m", action="store_true", help="Generate consolidated markdown file (BOOK.md)")
    parser.add_argument("--force", "-f", action="store_true", help="Force regeneration even if sources unchanged")
    parser.add_argument("--timestamp", "-t", action="store_true", help="Include generation timestamps in output")
    parser.add_argument("--dry-run", "-n", action="store_true", help="Show what would be generated without writing")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--list", "-l", action="store_true", help="List available generators")
    parser.add_argument("--output", "-o", type=Path, default=BOOK_DIR, help="Output directory (default: book/)")

    args = parser.parse_args()

    # Initialize builder
    builder = BookBuilder(book_dir=args.output, verbose=args.verbose)

    # Register real generators
    builder.register_generator(AlgorithmChapterGenerator(book_dir=args.output))
    builder.register_generator(ModuleDocGenerator(book_dir=args.output))
    builder.register_generator(DecisionStoryGenerator(book_dir=args.output))
    builder.register_generator(CommitNarrativeGenerator(book_dir=args.output))
    builder.register_generator(LessonExtractor(book_dir=args.output))
    builder.register_generator(ConceptEvolutionGenerator(book_dir=args.output))
    builder.register_generator(ExerciseGenerator(book_dir=args.output))
    builder.register_generator(CaseStudyGenerator(book_dir=args.output))
    builder.register_generator(ReaderJourneyGenerator(book_dir=args.output))
    builder.register_generator(SearchIndexGenerator(book_dir=args.output))
    builder.register_generator(MarkdownBookGenerator(
        book_dir=args.output,
        include_timestamp=getattr(args, 'timestamp', False)
    ))

    # Register placeholder generators (will be replaced with real ones)
    builder.register_generator(PlaceholderGenerator("future", "05-future"))

    # List mode
    if args.list:
        print("Available generators:")
        for name in builder.generators:
            print(f"  - {name}")
        return

    # Generate
    if args.markdown:
        # Generate only the consolidated markdown file
        results = builder.generate_chapter("markdown", dry_run=args.dry_run, force=args.force)
    elif args.chapter:
        results = builder.generate_chapter(args.chapter, dry_run=args.dry_run, force=args.force)
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
