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

    # Register real generators
    builder.register_generator(AlgorithmChapterGenerator(book_dir=args.output))
    builder.register_generator(ModuleDocGenerator(book_dir=args.output))
    builder.register_generator(CommitNarrativeGenerator(book_dir=args.output))
    builder.register_generator(SearchIndexGenerator(book_dir=args.output))

    # Register placeholder generators (will be replaced with real ones)
    builder.register_generator(PlaceholderGenerator("decisions", "03-decisions"))
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
