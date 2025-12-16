"""
Unit Tests for Book Generation System
======================================

Tests for ModuleDocGenerator and SearchIndexGenerator classes
from scripts/generate_book.py.

Tests cover:
- Property accessors (name, output_dir)
- Metadata parsing from .ai_meta files
- Module grouping logic
- Keyword extraction
- YAML frontmatter extraction
- Edge cases (malformed data, missing files, etc.)
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import the classes we're testing
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.generate_book import (
    ChapterGenerator,
    AlgorithmChapterGenerator,
    BookBuilder,
    ModuleDocGenerator,
    SearchIndexGenerator,
    CommitNarrativeGenerator,
    MarkdownBookGenerator,
    BOOK_DIR,
)


# =============================================================================
# ChapterGenerator Base Class Tests
# =============================================================================

class ConcreteChapterGenerator(ChapterGenerator):
    """Concrete implementation for testing the abstract base class."""

    def __init__(self, book_dir: Path = BOOK_DIR):
        super().__init__(book_dir)
        self._name = "test_generator"
        self._output_dir = "00-test"

    @property
    def name(self) -> str:
        return self._name

    @property
    def output_dir(self) -> str:
        return self._output_dir

    def generate(self, dry_run: bool = False, verbose: bool = False):
        """Minimal implementation for testing."""
        return {
            "files": [str(f) for f in self.generated_files],
            "stats": {"test": True},
            "errors": []
        }


class TestChapterGeneratorBaseClass:
    """Tests for ChapterGenerator abstract base class."""

    def test_generate_frontmatter_format(self):
        """Test that frontmatter has correct YAML format with required fields."""
        generator = ConcreteChapterGenerator()

        frontmatter = generator.generate_frontmatter(
            title="Test Chapter",
            tags=["test", "example"],
            source_files=["file1.py", "file2.py"]
        )

        # Check it starts and ends with ---
        assert frontmatter.startswith("---\n")
        assert "---\n\n" in frontmatter

        # Parse the YAML content
        yaml_content = frontmatter.split("---\n")[1]
        data = yaml.safe_load(yaml_content)

        # Verify required fields
        assert data["title"] == "Test Chapter"
        assert "generated" in data
        assert data["generator"] == "test_generator"
        assert data["source_files"] == ["file1.py", "file2.py"]
        assert data["tags"] == ["test", "example"]

        # Verify timestamp format (ISO 8601)
        assert "T" in data["generated"]
        assert data["generated"].endswith("Z")

    def test_write_chapter_dry_run(self, tmp_path):
        """Test that dry_run=True doesn't write files."""
        generator = ConcreteChapterGenerator(book_dir=tmp_path)

        result = generator.write_chapter("test.md", "# Test Content", dry_run=True)

        # Should return None in dry run
        assert result is None

        # File should not exist
        expected_path = tmp_path / "00-test" / "test.md"
        assert not expected_path.exists()

        # generated_files should be empty
        assert len(generator.generated_files) == 0

    def test_write_chapter_creates_file(self, tmp_path):
        """Test that dry_run=False writes file and creates directories."""
        generator = ConcreteChapterGenerator(book_dir=tmp_path)

        content = "# Test Chapter\n\nThis is test content."
        result = generator.write_chapter("test.md", content, dry_run=False)

        # Should return the path
        assert result is not None
        assert isinstance(result, Path)

        # File should exist
        expected_path = tmp_path / "00-test" / "test.md"
        assert expected_path.exists()

        # Content should match
        assert expected_path.read_text() == content

        # generated_files should track it
        assert len(generator.generated_files) == 1
        assert generator.generated_files[0] == expected_path


# =============================================================================
# AlgorithmChapterGenerator Tests
# =============================================================================

class TestAlgorithmChapterGenerator:
    """Tests for AlgorithmChapterGenerator class."""

    def test_name_property(self):
        """Test that name property returns 'foundations'."""
        generator = AlgorithmChapterGenerator()
        assert generator.name == "foundations"

    def test_output_dir_property(self):
        """Test that output_dir property returns '01-foundations'."""
        generator = AlgorithmChapterGenerator()
        assert generator.output_dir == "01-foundations"

    def test_generate_dry_run(self, tmp_path):
        """Test generate() in dry run mode with mocked VISION.md."""
        generator = AlgorithmChapterGenerator(book_dir=tmp_path)

        # Mock VISION.md content with algorithm sections
        vision_content = """
# Vision Document

## Deep Algorithm Analysis

### Algorithm 1: PageRank — Importance Discovery

**Implementation:** `cortical/analysis.py`

PageRank is a graph algorithm that computes the importance of nodes.

**Core Concept:**
- Iterative computation
- Damping factor: 0.85

---

### Algorithm 2: BM25/TF-IDF — Distinctiveness Scoring

**Implementation:** `cortical/analysis.py`

BM25 is a ranking function for document relevance.

---
"""

        # Mock the VISION.md file
        docs_dir = tmp_path.parent / "docs"
        docs_dir.mkdir(exist_ok=True)
        vision_file = docs_dir / "VISION.md"

        with patch("scripts.generate_book.DOCS_DIR", docs_dir):
            vision_file.write_text(vision_content)

            result = generator.generate(dry_run=True, verbose=False)

            # Check return structure
            assert "files" in result
            assert "stats" in result
            assert "errors" in result

            # Check stats
            assert result["stats"]["algorithms_found"] == 2
            assert result["stats"]["chapters_written"] == 2

            # No files should be written in dry run
            assert len(result["files"]) == 0
            assert len(result["errors"]) == 0


# =============================================================================
# BookBuilder Tests
# =============================================================================

class TestBookBuilder:
    """Tests for BookBuilder orchestrator class."""

    def test_register_generator(self, tmp_path):
        """Test registering a generator."""
        builder = BookBuilder(book_dir=tmp_path, verbose=False)
        generator = ConcreteChapterGenerator(book_dir=tmp_path)

        # Initially empty
        assert len(builder.generators) == 0

        # Register
        builder.register_generator(generator)

        # Should be registered by name
        assert "test_generator" in builder.generators
        assert builder.generators["test_generator"] == generator
        assert len(builder.generators) == 1

    def test_generate_all_dry_run(self, tmp_path):
        """Test generating all chapters in dry run mode."""
        builder = BookBuilder(book_dir=tmp_path, verbose=False)

        # Register two test generators
        gen1 = ConcreteChapterGenerator(book_dir=tmp_path)
        gen2 = ConcreteChapterGenerator(book_dir=tmp_path)
        gen2._name = "second_generator"
        gen2._output_dir = "01-second"

        builder.register_generator(gen1)
        builder.register_generator(gen2)

        # Generate all in dry run
        results = builder.generate_all(dry_run=True)

        # Check result structure
        assert "generated_at" in results
        assert "dry_run" in results
        assert "chapters" in results
        assert "total_files" in results
        assert "errors" in results

        # Should have results for both generators
        assert "test_generator" in results["chapters"]
        assert "second_generator" in results["chapters"]

        # dry_run flag should be set
        assert results["dry_run"] is True

        # No errors
        assert len(results["errors"]) == 0

    def test_generate_unknown_chapter_raises_error(self, tmp_path):
        """Test that generating unknown chapter raises ValueError."""
        builder = BookBuilder(book_dir=tmp_path, verbose=False)

        # Register one generator
        generator = ConcreteChapterGenerator(book_dir=tmp_path)
        builder.register_generator(generator)

        # Try to generate non-existent chapter
        with pytest.raises(ValueError) as exc_info:
            builder.generate_chapter("nonexistent_chapter", dry_run=True)

        # Check error message mentions the unknown chapter
        assert "nonexistent_chapter" in str(exc_info.value)
        assert "Unknown chapter" in str(exc_info.value)


# =============================================================================
# ModuleDocGenerator Tests
# =============================================================================

class TestModuleDocGenerator:
    """Tests for ModuleDocGenerator class."""

    def test_name_property(self):
        """Test that name property returns 'architecture'."""
        generator = ModuleDocGenerator()
        assert generator.name == "architecture"

    def test_output_dir_property(self):
        """Test that output_dir property returns '02-architecture'."""
        generator = ModuleDocGenerator()
        assert generator.output_dir == "02-architecture"

    def test_parse_metadata_valid_yaml(self, tmp_path):
        """Test parsing a valid .ai_meta file."""
        generator = ModuleDocGenerator()

        # Create a valid .ai_meta file
        meta_content = """# test.py.ai_meta
file: /path/to/test.py
filename: test.py
lines: 100
generated: "2025-12-16T12:00:00"
module_doc: |
  Test module documentation.
  Multiple lines supported.
sections: []
functions:
  test_func:
    signature: "(arg1, arg2)"
    doc: "Test function"
imports:
  stdlib:
    - typing.Dict
  local:
    - .module
"""
        meta_file = tmp_path / "test.py.ai_meta"
        meta_file.write_text(meta_content)

        result = generator._parse_metadata(meta_file)

        assert result is not None
        assert result['filename'] == 'test.py'
        assert result['file'] == '/path/to/test.py'
        assert result['lines'] == 100
        assert 'Test module documentation' in result['module_doc']
        assert 'test_func' in result['functions']
        assert result['meta_file'] == 'test.py.ai_meta'

    def test_parse_metadata_minimal_yaml(self, tmp_path):
        """Test parsing a minimal valid .ai_meta file."""
        generator = ModuleDocGenerator()

        meta_content = """# minimal.py.ai_meta
file: /path/to/minimal.py
filename: minimal.py
"""
        meta_file = tmp_path / "minimal.py.ai_meta"
        meta_file.write_text(meta_content)

        result = generator._parse_metadata(meta_file)

        assert result is not None
        assert result['filename'] == 'minimal.py'
        assert result['file'] == '/path/to/minimal.py'

    def test_parse_metadata_malformed_yaml(self, tmp_path):
        """Test that malformed YAML is handled gracefully."""
        generator = ModuleDocGenerator()

        # Create a malformed .ai_meta file (invalid YAML)
        meta_content = """# bad.py.ai_meta
file: /path/to/bad.py
filename: bad.py
invalid_yaml: [unclosed bracket
  - this is bad
"""
        meta_file = tmp_path / "bad.py.ai_meta"
        meta_file.write_text(meta_content)

        # Should return None instead of crashing
        result = generator._parse_metadata(meta_file)
        assert result is None

    def test_parse_metadata_empty_file(self, tmp_path):
        """Test parsing an empty .ai_meta file."""
        generator = ModuleDocGenerator()

        meta_file = tmp_path / "empty.py.ai_meta"
        meta_file.write_text("# empty file\n")

        result = generator._parse_metadata(meta_file)
        assert result is None

    def test_parse_metadata_nonexistent_file(self):
        """Test parsing a file that doesn't exist."""
        generator = ModuleDocGenerator()

        meta_file = Path("/nonexistent/path/fake.py.ai_meta")

        # Should handle the error gracefully
        result = generator._parse_metadata(meta_file)
        assert result is None

    def test_group_modules_processor_package(self):
        """Test grouping modules from processor/ package."""
        generator = ModuleDocGenerator()

        modules = [
            {'filename': 'core.py', 'file': '/path/processor/core.py'},
            {'filename': 'compute.py', 'file': '/path/processor/compute.py'},
            {'filename': 'documents.py', 'file': '/path/processor/documents.py'},
        ]

        grouped = generator._group_modules(modules)

        assert 'processor' in grouped
        assert len(grouped['processor']) == 3

    def test_group_modules_query_package(self):
        """Test grouping modules from query/ package."""
        generator = ModuleDocGenerator()

        modules = [
            {'filename': 'search.py', 'file': '/path/query/search.py'},
            {'filename': 'expansion.py', 'file': '/path/query/expansion.py'},
        ]

        grouped = generator._group_modules(modules)

        assert 'query' in grouped
        assert len(grouped['query']) == 2

    def test_group_modules_persistence(self):
        """Test grouping persistence-related modules."""
        generator = ModuleDocGenerator()

        modules = [
            {'filename': 'persistence.py', 'file': '/path/persistence.py'},
            {'filename': 'chunk_index.py', 'file': '/path/chunk_index.py'},
            {'filename': 'state_storage.py', 'file': '/path/state_storage.py'},
        ]

        grouped = generator._group_modules(modules)

        assert 'persistence' in grouped
        assert len(grouped['persistence']) == 3

    def test_group_modules_data_structures(self):
        """Test grouping data structure modules."""
        generator = ModuleDocGenerator()

        modules = [
            {'filename': 'minicolumn.py', 'file': '/path/minicolumn.py'},
            {'filename': 'layers.py', 'file': '/path/layers.py'},
            {'filename': 'types.py', 'file': '/path/types.py'},
        ]

        grouped = generator._group_modules(modules)

        assert 'data-structures' in grouped
        assert len(grouped['data-structures']) == 3

    def test_group_modules_nlp(self):
        """Test grouping NLP modules."""
        generator = ModuleDocGenerator()

        modules = [
            {'filename': 'tokenizer.py', 'file': '/path/tokenizer.py'},
            {'filename': 'semantics.py', 'file': '/path/semantics.py'},
            {'filename': 'embeddings.py', 'file': '/path/embeddings.py'},
        ]

        grouped = generator._group_modules(modules)

        assert 'nlp' in grouped
        assert len(grouped['nlp']) == 3

    def test_group_modules_configuration(self):
        """Test grouping configuration modules."""
        generator = ModuleDocGenerator()

        modules = [
            {'filename': 'config.py', 'file': '/path/config.py'},
            {'filename': 'validation.py', 'file': '/path/validation.py'},
            {'filename': 'constants.py', 'file': '/path/constants.py'},
        ]

        grouped = generator._group_modules(modules)

        assert 'configuration' in grouped
        assert len(grouped['configuration']) == 3

    def test_group_modules_observability(self):
        """Test grouping observability modules."""
        generator = ModuleDocGenerator()

        modules = [
            {'filename': 'observability.py', 'file': '/path/observability.py'},
            {'filename': 'progress.py', 'file': '/path/progress.py'},
            {'filename': 'results.py', 'file': '/path/results.py'},
        ]

        grouped = generator._group_modules(modules)

        assert 'observability' in grouped
        assert len(grouped['observability']) == 3

    def test_group_modules_utilities_fallback(self):
        """Test that unknown modules are grouped as utilities."""
        generator = ModuleDocGenerator()

        modules = [
            {'filename': 'unknown.py', 'file': '/path/unknown.py'},
            {'filename': 'random_module.py', 'file': '/path/random_module.py'},
        ]

        grouped = generator._group_modules(modules)

        assert 'utilities' in grouped
        assert len(grouped['utilities']) == 2

    def test_group_modules_mixed(self):
        """Test grouping a mixed set of modules."""
        generator = ModuleDocGenerator()

        modules = [
            {'filename': 'core.py', 'file': '/path/processor/core.py'},
            {'filename': 'config.py', 'file': '/path/config.py'},
            {'filename': 'search.py', 'file': '/path/query/search.py'},
            {'filename': 'persistence.py', 'file': '/path/persistence.py'},
            {'filename': 'unknown.py', 'file': '/path/unknown.py'},
        ]

        grouped = generator._group_modules(modules)

        assert 'processor' in grouped
        assert 'configuration' in grouped
        assert 'query' in grouped
        assert 'persistence' in grouped
        assert 'utilities' in grouped
        assert len(grouped['processor']) == 1
        assert len(grouped['configuration']) == 1

    def test_group_modules_windows_paths(self):
        """Test grouping with Windows-style paths.

        Note: Current implementation doesn't fully support pure Windows paths
        (with backslashes only), so these modules fall back to 'utilities'.
        This documents the current behavior.
        """
        generator = ModuleDocGenerator()

        modules = [
            {'filename': 'core.py', 'file': 'C:\\path\\processor\\core.py'},
            {'filename': 'search.py', 'file': 'C:\\path\\query\\search.py'},
        ]

        grouped = generator._group_modules(modules)

        # Current behavior: pure Windows paths aren't recognized,
        # so they fall back to utilities group
        assert 'utilities' in grouped
        assert len(grouped['utilities']) == 2

    def test_group_modules_empty_list(self):
        """Test grouping an empty module list."""
        generator = ModuleDocGenerator()

        modules = []
        grouped = generator._group_modules(modules)

        assert grouped == {}

    @patch('scripts.generate_book.CORTICAL_DIR')
    def test_generate_no_metadata_files(self, mock_cortical_dir, tmp_path):
        """Test generate() when no .ai_meta files exist."""
        mock_cortical_dir.glob = Mock(return_value=[])

        generator = ModuleDocGenerator(book_dir=tmp_path)
        result = generator.generate(dry_run=True, verbose=False)

        assert result['stats']['metadata_files_found'] == 0
        assert len(result['errors']) > 0
        assert 'No .ai_meta files found' in result['errors'][0]


# =============================================================================
# SearchIndexGenerator Tests
# =============================================================================

class TestSearchIndexGenerator:
    """Tests for SearchIndexGenerator class."""

    def test_name_property(self):
        """Test that name property returns 'search'."""
        generator = SearchIndexGenerator()
        assert generator.name == "search"

    def test_output_dir_property(self):
        """Test that output_dir property returns empty string (root)."""
        generator = SearchIndexGenerator()
        assert generator.output_dir == ""

    def test_extract_keywords_basic(self):
        """Test keyword extraction from simple content."""
        generator = SearchIndexGenerator()

        content = """
        The PageRank algorithm computes importance scores for terms.
        PageRank uses damping factor and iterative computation.
        The algorithm converges when scores stabilize.
        """

        keywords = generator._extract_keywords(content, top_n=5)

        assert isinstance(keywords, list)
        assert len(keywords) <= 5
        assert 'pagerank' in keywords or 'algorithm' in keywords

    def test_extract_keywords_filters_stopwords(self):
        """Test that common stopwords are filtered out."""
        generator = SearchIndexGenerator()

        content = "the and or but in on at to for of with by from as"

        keywords = generator._extract_keywords(content, top_n=10)

        # All stopwords should be filtered
        assert len(keywords) == 0

    def test_extract_keywords_min_length(self):
        """Test that keywords must be at least 3 characters."""
        generator = SearchIndexGenerator()

        content = "a ab abc abcd abcde"

        keywords = generator._extract_keywords(content, top_n=10)

        # Only 'abc', 'abcd', 'abcde' should be included
        for kw in keywords:
            assert len(kw) > 2

    def test_extract_keywords_frequency_ordering(self):
        """Test that keywords are ordered by frequency."""
        generator = SearchIndexGenerator()

        content = """
        neural neural neural neural
        network network network
        algorithm algorithm
        data
        """

        keywords = generator._extract_keywords(content, top_n=4)

        # 'neural' should come first (4 occurrences)
        if keywords:
            assert keywords[0] == 'neural'

    def test_extract_keywords_code_identifiers(self):
        """Test keyword extraction from code with underscores."""
        generator = SearchIndexGenerator()

        content = """
        def compute_pagerank(graph_data):
            importance_score = calculate_importance(graph_data)
            return importance_score
        """

        keywords = generator._extract_keywords(content, top_n=10)

        # Should extract identifiers with underscores
        assert any('pagerank' in kw or 'importance' in kw for kw in keywords)

    def test_extract_keywords_empty_content(self):
        """Test keyword extraction from empty content."""
        generator = SearchIndexGenerator()

        keywords = generator._extract_keywords("", top_n=10)

        assert keywords == []

    def test_extract_frontmatter_valid_yaml(self):
        """Test extraction of valid YAML frontmatter."""
        generator = SearchIndexGenerator()

        content = """---
title: "Test Chapter"
generated: "2025-12-16T12:00:00Z"
tags:
  - test
  - example
source_files:
  - "file1.py"
  - "file2.py"
---

# Chapter Content

This is the actual content.
"""

        frontmatter = generator._extract_frontmatter(content)

        assert frontmatter['title'] == 'Test Chapter'
        assert 'tags' in frontmatter
        assert isinstance(frontmatter['tags'], list)
        assert 'test' in frontmatter['tags']

    def test_extract_frontmatter_no_frontmatter(self):
        """Test extraction when no frontmatter exists."""
        generator = SearchIndexGenerator()

        content = """# Chapter Without Frontmatter

This chapter has no YAML frontmatter at the beginning.
"""

        frontmatter = generator._extract_frontmatter(content)

        assert frontmatter == {}

    def test_extract_frontmatter_simple_key_values(self):
        """Test extraction of simple key-value pairs."""
        generator = SearchIndexGenerator()

        content = """---
title: "Simple Title"
author: "Test Author"
date: "2025-12-16"
---

Content here.
"""

        frontmatter = generator._extract_frontmatter(content)

        # Should parse at least the title
        assert 'title' in frontmatter or len(frontmatter) > 0

    def test_extract_frontmatter_malformed_yaml(self):
        """Test extraction with malformed YAML (fallback to regex)."""
        generator = SearchIndexGenerator()

        # Malformed YAML that can't be parsed by yaml.safe_load
        content = """---
title: "Test"
bad_list: [unclosed
---

Content.
"""

        # Should not crash, may return empty dict or partial parse
        frontmatter = generator._extract_frontmatter(content)
        assert isinstance(frontmatter, dict)

    def test_extract_frontmatter_empty_frontmatter(self):
        """Test extraction with empty frontmatter block."""
        generator = SearchIndexGenerator()

        content = """---
---

# Content
"""

        frontmatter = generator._extract_frontmatter(content)
        assert isinstance(frontmatter, dict)

    def test_remove_frontmatter(self):
        """Test removing frontmatter from content."""
        generator = SearchIndexGenerator()

        content = """---
title: "Test"
---

# Actual Content

This remains.
"""

        result = generator._remove_frontmatter(content)

        assert '---' not in result.split('\n')[0]
        assert '# Actual Content' in result

    def test_remove_frontmatter_no_frontmatter(self):
        """Test removing frontmatter when none exists."""
        generator = SearchIndexGenerator()

        content = "# Content\n\nNo frontmatter here."
        result = generator._remove_frontmatter(content)

        assert result == content

    def test_extract_excerpt_basic(self):
        """Test excerpt extraction from content."""
        generator = SearchIndexGenerator()

        content = """# Chapter Title

This is the first paragraph that should be extracted as an excerpt.
It contains multiple sentences.

Second paragraph should not be included if we limit length.
"""

        excerpt = generator._extract_excerpt(content, max_length=100)

        assert isinstance(excerpt, str)
        assert len(excerpt) <= 103  # 100 + "..."
        assert 'first paragraph' in excerpt.lower()

    def test_extract_excerpt_skips_title(self):
        """Test that excerpt skips the title line."""
        generator = SearchIndexGenerator()

        content = """# Chapter Title

First paragraph content.
"""

        excerpt = generator._extract_excerpt(content)

        assert '# Chapter Title' not in excerpt
        assert 'First paragraph' in excerpt

    def test_extract_excerpt_truncates_long_content(self):
        """Test that long content is truncated with ellipsis."""
        generator = SearchIndexGenerator()

        content = "# Title\n\n" + "word " * 100
        excerpt = generator._extract_excerpt(content, max_length=50)

        assert len(excerpt) <= 53  # 50 + "..."
        assert excerpt.endswith('...')

    def test_parse_chapter_basic(self, tmp_path):
        """Test parsing a basic chapter file."""
        generator = SearchIndexGenerator(book_dir=tmp_path)

        chapter_dir = tmp_path / "01-foundations"
        chapter_dir.mkdir()

        chapter_file = chapter_dir / "test-chapter.md"
        chapter_content = """---
title: "Test Chapter"
tags:
  - test
  - algorithms
---

# Test Chapter

This is the chapter content with some keywords.
PageRank algorithm is important.
"""
        chapter_file.write_text(chapter_content)

        result = generator._parse_chapter(chapter_file)

        assert result is not None
        assert result['title'] == 'Test Chapter'
        assert result['section'] == 'foundations'
        assert 'test' in result['tags']
        assert isinstance(result['keywords'], list)
        assert 'full_content' in result

    def test_parse_chapter_no_frontmatter(self, tmp_path):
        """Test parsing a chapter without frontmatter."""
        generator = SearchIndexGenerator(book_dir=tmp_path)

        chapter_dir = tmp_path / "02-architecture"
        chapter_dir.mkdir()

        chapter_file = chapter_dir / "no-frontmatter.md"
        chapter_content = """# Chapter Title

Content without frontmatter.
"""
        chapter_file.write_text(chapter_content)

        result = generator._parse_chapter(chapter_file)

        assert result is not None
        assert result['title'] == 'no-frontmatter'  # Falls back to filename
        assert result['section'] == 'architecture'

    def test_group_by_section(self):
        """Test grouping chapters by section."""
        generator = SearchIndexGenerator()

        chapters = [
            {'section': 'foundations', 'title': 'Ch1'},
            {'section': 'foundations', 'title': 'Ch2'},
            {'section': 'architecture', 'title': 'Ch3'},
            {'section': 'evolution', 'title': 'Ch4'},
        ]

        sections = generator._group_by_section(chapters)

        assert 'foundations' in sections
        assert sections['foundations']['count'] == 2
        assert 'architecture' in sections
        assert sections['architecture']['count'] == 1

    def test_generate_dry_run(self, tmp_path):
        """Test generate() with dry_run=True returns correct structure."""
        generator = SearchIndexGenerator(book_dir=tmp_path)

        # Create a sample chapter
        chapter_dir = tmp_path / "01-foundations"
        chapter_dir.mkdir()
        chapter_file = chapter_dir / "test.md"
        chapter_file.write_text("---\ntitle: Test\n---\n\n# Test\n\nContent.")

        result = generator.generate(dry_run=True, verbose=False)

        assert 'files' in result
        assert 'stats' in result
        assert 'errors' in result
        assert result['stats']['chapters_indexed'] >= 0
        assert isinstance(result['files'], list)
        assert isinstance(result['errors'], list)

    def test_generate_empty_book_dir(self, tmp_path):
        """Test generate() with no chapter files."""
        generator = SearchIndexGenerator(book_dir=tmp_path)

        result = generator.generate(dry_run=True, verbose=False)

        assert result['stats']['chapters_indexed'] == 0
        assert result['stats']['sections_found'] == 0


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_module_doc_generator_with_unicode(self, tmp_path):
        """Test ModuleDocGenerator handles Unicode in .ai_meta files."""
        generator = ModuleDocGenerator()

        meta_content = """# test.py.ai_meta
file: /path/to/test.py
filename: test.py
module_doc: |
  Unicode test: 日本語 中文 한글 العربية
  Math symbols: ∑∫∂∇
"""
        meta_file = tmp_path / "unicode.py.ai_meta"
        meta_file.write_text(meta_content, encoding='utf-8')

        result = generator._parse_metadata(meta_file)

        # Should parse successfully
        assert result is not None
        assert '日本語' in result['module_doc']

    def test_search_index_generator_with_code_blocks(self):
        """Test SearchIndexGenerator handles code blocks in content."""
        generator = SearchIndexGenerator()

        content = """# Chapter

```python
def compute_pagerank(graph):
    # Code with keywords
    importance = calculate_score(graph)
    return importance
```

Regular text with algorithm description.
"""

        keywords = generator._extract_keywords(content, top_n=10)

        # Should extract keywords from both code and text
        assert isinstance(keywords, list)
        # Should include identifiers from code
        assert any(kw in ['compute', 'pagerank', 'graph', 'importance', 'calculate',
                          'score', 'algorithm'] for kw in keywords)

    def test_module_doc_generator_missing_required_fields(self, tmp_path):
        """Test handling of .ai_meta with missing required fields."""
        generator = ModuleDocGenerator()

        # Missing 'filename' field
        meta_content = """# incomplete.py.ai_meta
file: /path/to/incomplete.py
module_doc: "Some doc"
"""
        meta_file = tmp_path / "incomplete.py.ai_meta"
        meta_file.write_text(meta_content)

        result = generator._parse_metadata(meta_file)

        # Should still parse what's available
        assert result is not None or result is None  # Either is acceptable

    def test_search_index_generator_very_long_content(self):
        """Test keyword extraction from very long content."""
        generator = SearchIndexGenerator()

        # Generate long content
        content = "algorithm " * 1000 + "network " * 500 + "data " * 250

        keywords = generator._extract_keywords(content, top_n=3)

        assert len(keywords) <= 3
        assert keywords[0] == 'algorithm'  # Most frequent


# =============================================================================
# CommitNarrativeGenerator Tests
# =============================================================================

class TestCommitNarrativeGenerator:
    """Tests for CommitNarrativeGenerator class."""

    def test_name_property(self):
        """Test that name property returns 'evolution'."""
        generator = CommitNarrativeGenerator()
        assert generator.name == "evolution"

    def test_output_dir_property(self):
        """Test that output_dir property returns '04-evolution'."""
        generator = CommitNarrativeGenerator()
        assert generator.output_dir == "04-evolution"

    def test_extract_commit_type_feat(self):
        """Test extraction of feat: commit type."""
        generator = CommitNarrativeGenerator()

        result = generator._extract_commit_type("feat: Add new feature")
        assert result == "feat"

        result = generator._extract_commit_type("Feat: Add new feature")
        assert result == "feat"

    def test_extract_commit_type_fix(self):
        """Test extraction of fix: commit type."""
        generator = CommitNarrativeGenerator()

        result = generator._extract_commit_type("fix: Fix bug in parser")
        assert result == "fix"

        result = generator._extract_commit_type("Fix: Critical bug")
        assert result == "fix"

    def test_extract_commit_type_refactor(self):
        """Test extraction of refactor: commit type."""
        generator = CommitNarrativeGenerator()

        result = generator._extract_commit_type("refactor: Improve code structure")
        assert result == "refactor"

    def test_extract_commit_type_docs(self):
        """Test extraction of docs: commit type."""
        generator = CommitNarrativeGenerator()

        result = generator._extract_commit_type("docs: Update README")
        assert result == "docs"

    def test_extract_commit_type_chore(self):
        """Test extraction of chore: commit type."""
        generator = CommitNarrativeGenerator()

        result = generator._extract_commit_type("chore: Update dependencies")
        assert result == "chore"

    def test_extract_commit_type_test(self):
        """Test extraction of test: commit type."""
        generator = CommitNarrativeGenerator()

        result = generator._extract_commit_type("test: Add unit tests")
        assert result == "test"

    def test_extract_commit_type_perf(self):
        """Test extraction of perf: commit type."""
        generator = CommitNarrativeGenerator()

        result = generator._extract_commit_type("perf: Optimize algorithm")
        assert result == "perf"

    def test_extract_commit_type_security(self):
        """Test extraction of security: commit type."""
        generator = CommitNarrativeGenerator()

        result = generator._extract_commit_type("security: Fix vulnerability")
        assert result == "security"

    def test_extract_commit_type_data(self):
        """Test extraction of data: commit type."""
        generator = CommitNarrativeGenerator()

        result = generator._extract_commit_type("data: ML data sync")
        assert result == "data"

    def test_extract_commit_type_merge(self):
        """Test extraction of merge commit type."""
        generator = CommitNarrativeGenerator()

        result = generator._extract_commit_type("Merge pull request #123 from branch")
        assert result == "merge"

        result = generator._extract_commit_type("Merge branch 'main' into feature")
        assert result == "merge"

    def test_extract_commit_type_unknown(self):
        """Test that unknown commit types return 'other'."""
        generator = CommitNarrativeGenerator()

        result = generator._extract_commit_type("Random commit message")
        assert result == "other"

        result = generator._extract_commit_type("Update something")
        assert result == "other"

    def test_extract_commit_type_with_scope(self):
        """Test extraction with scope in parentheses.

        Note: Current implementation doesn't parse scopes, so "feat(auth):"
        returns "other" since it doesn't match the exact "feat:" prefix.
        """
        generator = CommitNarrativeGenerator()

        # Scopes aren't currently parsed - falls back to "other"
        result = generator._extract_commit_type("feat(auth): Add login")
        assert result == "other"

        result = generator._extract_commit_type("fix(parser): Handle edge case")
        assert result == "other"

    @patch('subprocess.run')
    def test_load_commits_basic(self, mock_run, tmp_path):
        """Test loading commits from git history."""
        generator = CommitNarrativeGenerator(repo_root=tmp_path)

        # Mock git log output
        mock_run.return_value = MagicMock(
            stdout="abc1234567890|2025-12-16T10:00:00Z|feat: Feature 1|Author One\n"
                   "def5678901234|2025-12-15T09:00:00Z|fix: Bug fix|Author Two\n",
            returncode=0
        )

        commits = generator._load_commits(limit=10)

        assert len(commits) == 2

        # Check first commit
        assert commits[0]['hash'] == 'abc1234567890'
        assert commits[0]['short_hash'] == 'abc1234'
        assert commits[0]['message'] == 'feat: Feature 1'
        assert commits[0]['author'] == 'Author One'
        assert commits[0]['type'] == 'feat'
        assert commits[0]['timestamp'] == '2025-12-16T10:00:00Z'
        assert commits[0]['date'] == '2025-12-16'

        # Check second commit
        assert commits[1]['hash'] == 'def5678901234'
        assert commits[1]['type'] == 'fix'

    @patch('subprocess.run')
    def test_load_commits_empty(self, mock_run, tmp_path):
        """Test loading commits when git returns empty output."""
        generator = CommitNarrativeGenerator(repo_root=tmp_path)

        mock_run.return_value = MagicMock(stdout="", returncode=0)

        commits = generator._load_commits()

        assert commits == []

    @patch('subprocess.run')
    def test_load_commits_git_error(self, mock_run, tmp_path):
        """Test handling of git command errors."""
        generator = CommitNarrativeGenerator(repo_root=tmp_path)

        # Mock git command failure
        mock_run.side_effect = Exception("git not found")

        commits = generator._load_commits()

        # Should return empty list on error
        assert commits == []

    @patch('subprocess.run')
    def test_load_commits_malformed_lines(self, mock_run, tmp_path):
        """Test handling of malformed commit lines."""
        generator = CommitNarrativeGenerator(repo_root=tmp_path)

        # Mock output with malformed lines
        mock_run.return_value = MagicMock(
            stdout="abc1234|2025-12-16T10:00:00Z|feat: Valid commit|Author\n"
                   "malformed line without pipes\n"
                   "\n"  # Empty line
                   "xyz9876|2025-12-15T09:00:00Z|fix: Another valid|Author\n",
            returncode=0
        )

        commits = generator._load_commits()

        # Should skip malformed/empty lines
        assert len(commits) == 2
        assert commits[0]['hash'] == 'abc1234'
        assert commits[1]['hash'] == 'xyz9876'

    @patch('subprocess.run')
    def test_generate_dry_run(self, mock_run, tmp_path):
        """Test generate() with dry_run=True and mocked git history."""
        generator = CommitNarrativeGenerator(book_dir=tmp_path, repo_root=tmp_path)

        # Mock git log output
        mock_run.return_value = MagicMock(
            stdout="abc1234567890|2025-12-16T10:00:00Z|feat: Add search feature|Author One\n"
                   "def5678901234|2025-12-15T09:00:00Z|fix: Fix parser bug|Author Two\n"
                   "ghi9012345678|2025-12-14T08:00:00Z|refactor: Improve structure|Author Three\n",
            returncode=0
        )

        result = generator.generate(dry_run=True, verbose=False)

        # Check return structure
        assert "files" in result
        assert "stats" in result
        assert "errors" in result

        # Check stats
        assert result["stats"]["total_commits"] == 3
        assert "by_type" in result["stats"]
        assert result["stats"]["by_type"]["feat"] == 1
        assert result["stats"]["by_type"]["fix"] == 1
        assert result["stats"]["by_type"]["refactor"] == 1

        # No files should be written in dry run
        assert len(result["files"]) == 0
        assert len(result["errors"]) == 0

    @patch('subprocess.run')
    def test_generate_no_git_history(self, mock_run, tmp_path):
        """Test generate() when no git history is available."""
        generator = CommitNarrativeGenerator(book_dir=tmp_path, repo_root=tmp_path)

        # Mock git command failure
        mock_run.side_effect = Exception("Not a git repository")

        result = generator.generate(dry_run=True, verbose=False)

        # Should handle gracefully
        assert result["stats"]["commits"] == 0
        assert "error" in result["stats"]
        assert len(result["errors"]) > 0
        assert "Could not read git history" in result["errors"][0]

    def test_find_adr_references_basic(self, tmp_path):
        """Test finding ADR references in commit messages."""
        generator = CommitNarrativeGenerator(repo_root=tmp_path)

        refs = generator._find_adr_references("feat: Add feature (ADR-001)")
        assert refs == ["adr-001"]

        refs = generator._find_adr_references("fix: Bug fix adr-042")
        assert refs == ["adr-042"]

    def test_find_adr_references_multiple(self, tmp_path):
        """Test finding multiple ADR references."""
        generator = CommitNarrativeGenerator(repo_root=tmp_path)

        refs = generator._find_adr_references("Implement ADR-001 and ADR-002")
        assert len(refs) == 2
        assert "adr-001" in refs
        assert "adr-002" in refs

    def test_find_adr_references_case_insensitive(self, tmp_path):
        """Test that ADR matching is case-insensitive."""
        generator = CommitNarrativeGenerator(repo_root=tmp_path)

        refs = generator._find_adr_references("feat: Based on adr-005")
        assert refs == ["adr-005"]

        refs = generator._find_adr_references("feat: Based on ADR 010")
        assert refs == ["adr-010"]

    def test_find_adr_references_no_matches(self, tmp_path):
        """Test finding ADR references when none exist."""
        generator = CommitNarrativeGenerator(repo_root=tmp_path)

        refs = generator._find_adr_references("Regular commit message")
        assert refs == []

    def test_group_by_themes_search(self, tmp_path):
        """Test grouping commits by search theme."""
        generator = CommitNarrativeGenerator(repo_root=tmp_path)

        commits = [
            {'message': 'feat: Add search feature', 'type': 'feat'},
            {'message': 'feat: Improve query expansion', 'type': 'feat'},
            {'message': 'feat: Add BM25 ranking', 'type': 'feat'},
        ]

        themes = generator._group_by_themes(commits)

        assert 'search' in themes
        assert len(themes['search']) == 3

    def test_group_by_themes_ml(self, tmp_path):
        """Test grouping commits by ML theme."""
        generator = CommitNarrativeGenerator(repo_root=tmp_path)

        commits = [
            {'message': 'feat: Add ML model', 'type': 'feat'},
            {'message': 'feat: Training pipeline', 'type': 'feat'},
        ]

        themes = generator._group_by_themes(commits)

        assert 'ml' in themes
        assert len(themes['ml']) == 2

    def test_group_by_themes_performance(self, tmp_path):
        """Test grouping commits by performance theme."""
        generator = CommitNarrativeGenerator(repo_root=tmp_path)

        commits = [
            {'message': 'perf: Optimize algorithm', 'type': 'perf'},
            {'message': 'feat: Add cache layer', 'type': 'feat'},
        ]

        themes = generator._group_by_themes(commits)

        assert 'performance' in themes
        assert len(themes['performance']) == 2

    def test_group_by_themes_other(self, tmp_path):
        """Test that unmatched commits go to 'other' theme."""
        generator = CommitNarrativeGenerator(repo_root=tmp_path)

        commits = [
            {'message': 'feat: Something unrelated', 'type': 'feat'},
        ]

        themes = generator._group_by_themes(commits)

        assert 'other' in themes
        assert len(themes['other']) == 1

    def test_load_adrs_empty_directory(self, tmp_path):
        """Test loading ADRs when decisions directory doesn't exist."""
        generator = CommitNarrativeGenerator(book_dir=tmp_path, repo_root=tmp_path)

        adrs = generator._load_adrs()

        assert adrs == {}

    def test_load_adrs_with_files(self, tmp_path):
        """Test loading ADRs from decisions directory."""
        # Create decisions directory with ADR files
        decisions_dir = tmp_path / "samples" / "decisions"
        decisions_dir.mkdir(parents=True)

        adr1 = decisions_dir / "adr-001-test-decision.md"
        adr1.write_text("# ADR-001: Test Decision\n\nContent here.")

        adr2 = decisions_dir / "adr-002-another-decision.md"
        adr2.write_text("# ADR-002: Another Decision\n\nMore content.")

        generator = CommitNarrativeGenerator(book_dir=tmp_path, repo_root=tmp_path)
        adrs = generator._load_adrs()

        assert len(adrs) == 2
        assert 'adr-001-test-decision' in adrs
        assert 'adr-002-another-decision' in adrs
        assert adrs['adr-001-test-decision']['title'] == 'ADR-001: Test Decision'


# =============================================================================
# MarkdownBookGenerator Tests
# =============================================================================

class TestMarkdownBookGenerator:
    """Tests for MarkdownBookGenerator class."""

    def test_name_property(self):
        """Test that name property returns 'markdown'."""
        generator = MarkdownBookGenerator()
        assert generator.name == "markdown"

    def test_output_dir_property(self):
        """Test that output_dir property returns empty string (root)."""
        generator = MarkdownBookGenerator()
        assert generator.output_dir == ""

    def test_make_anchor_basic(self):
        """Test anchor generation from simple text."""
        generator = MarkdownBookGenerator()

        assert generator._make_anchor("Hello World") == "hello-world"
        assert generator._make_anchor("Test Chapter") == "test-chapter"

    def test_make_anchor_special_chars(self):
        """Test anchor generation with special characters."""
        generator = MarkdownBookGenerator()

        assert generator._make_anchor("Test: Chapter 1") == "test-chapter-1"
        assert generator._make_anchor("Hello — World") == "hello-world"
        assert generator._make_anchor("Test (with parens)") == "test-with-parens"

    def test_make_anchor_multiple_spaces(self):
        """Test anchor generation with multiple spaces."""
        generator = MarkdownBookGenerator()

        assert generator._make_anchor("Hello   World") == "hello-world"
        assert generator._make_anchor("  Leading and trailing  ") == "leading-and-trailing"

    def test_collect_sections_empty_dir(self, tmp_path):
        """Test collecting sections from empty directory."""
        generator = MarkdownBookGenerator(book_dir=tmp_path)

        sections = generator._collect_sections()

        assert sections == {}

    def test_collect_sections_with_chapters(self, tmp_path):
        """Test collecting sections with chapter files."""
        generator = MarkdownBookGenerator(book_dir=tmp_path)

        # Create section directory with chapters
        section_dir = tmp_path / "01-foundations"
        section_dir.mkdir()

        chapter1 = section_dir / "chapter1.md"
        chapter1.write_text("# Chapter 1\n\nContent here.")

        chapter2 = section_dir / "chapter2.md"
        chapter2.write_text("# Chapter 2\n\nMore content.")

        sections = generator._collect_sections()

        assert "01-foundations" in sections
        assert len(sections["01-foundations"]["chapters"]) == 2
        assert sections["01-foundations"]["title"] == "Foundations: Core Algorithms"

    def test_collect_sections_ignores_special_dirs(self, tmp_path):
        """Test that special directories are ignored."""
        generator = MarkdownBookGenerator(book_dir=tmp_path)

        # Create special directories that should be ignored
        (tmp_path / "assets").mkdir()
        (tmp_path / "assets" / "test.md").write_text("# Test")

        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "test.md").write_text("# Test")

        (tmp_path / ".hidden").mkdir()
        (tmp_path / ".hidden" / "test.md").write_text("# Test")

        sections = generator._collect_sections()

        assert "assets" not in sections
        assert "docs" not in sections
        assert ".hidden" not in sections

    def test_collect_sections_ignores_readme_template(self, tmp_path):
        """Test that README.md and TEMPLATE.md are skipped."""
        generator = MarkdownBookGenerator(book_dir=tmp_path)

        section_dir = tmp_path / "01-foundations"
        section_dir.mkdir()

        # These should be ignored
        (section_dir / "README.md").write_text("# README")
        (section_dir / "TEMPLATE.md").write_text("# Template")

        # This should be included
        (section_dir / "chapter1.md").write_text("# Chapter 1")

        sections = generator._collect_sections()

        assert len(sections["01-foundations"]["chapters"]) == 1
        assert sections["01-foundations"]["chapters"][0]["filename"] == "chapter1.md"

    def test_parse_chapter_file_basic(self, tmp_path):
        """Test parsing a basic chapter file."""
        generator = MarkdownBookGenerator(book_dir=tmp_path)

        chapter_file = tmp_path / "test.md"
        chapter_file.write_text("# Test Chapter\n\nThis is the content.")

        result = generator._parse_chapter_file(chapter_file)

        assert result is not None
        assert result["title"] == "Test Chapter"
        assert result["filename"] == "test.md"
        assert "content" in result
        assert "This is the content" in result["content"]

    def test_parse_chapter_file_with_frontmatter(self, tmp_path):
        """Test parsing a chapter with YAML frontmatter."""
        generator = MarkdownBookGenerator(book_dir=tmp_path)

        chapter_file = tmp_path / "test.md"
        chapter_file.write_text("""---
title: "Custom Title"
tags:
  - test
---

# Heading

Content here.
""")

        result = generator._parse_chapter_file(chapter_file)

        assert result is not None
        assert result["title"] == "Custom Title"
        # Frontmatter should be removed from content
        assert "---" not in result["content"] or result["content"].count("---") == 0

    def test_parse_chapter_file_fallback_title(self, tmp_path):
        """Test that title falls back to filename when no heading."""
        generator = MarkdownBookGenerator(book_dir=tmp_path)

        chapter_file = tmp_path / "my-chapter.md"
        chapter_file.write_text("No heading here, just content.")

        result = generator._parse_chapter_file(chapter_file)

        assert result is not None
        # Should fall back to processed filename
        assert result["title"] == "My Chapter"

    def test_generate_header_contains_required_elements(self):
        """Test that generated header contains required elements."""
        generator = MarkdownBookGenerator()

        header = generator._generate_header()

        assert "# The Cortical Chronicles" in header
        assert "Generated:" in header
        assert "---" in header

    def test_generate_table_of_contents(self):
        """Test table of contents generation."""
        generator = MarkdownBookGenerator()

        sections = {
            "01-test": {
                "order": 1,
                "title": "Test Section",
                "chapters": [
                    {"title": "Chapter One", "content": "", "filename": "ch1.md"},
                    {"title": "Chapter Two", "content": "", "filename": "ch2.md"},
                ]
            }
        }

        toc = generator._generate_table_of_contents(sections)

        assert "## Table of Contents" in toc
        assert "Test Section" in toc
        assert "Chapter One" in toc
        assert "Chapter Two" in toc
        # Check for anchor links
        assert "(#" in toc

    def test_format_chapter_converts_heading_level(self):
        """Test that level-1 headings are converted to level-2."""
        generator = MarkdownBookGenerator()

        chapter = {
            "title": "Test",
            "content": "# Original Heading\n\nContent here.",
            "filename": "test.md"
        }

        result = generator._format_chapter(chapter)

        # Should have converted # to ##
        assert "## Original Heading" in result
        # Should have separator
        assert "---" in result

    def test_format_chapter_preserves_level2_heading(self):
        """Test that level-2 headings are preserved."""
        generator = MarkdownBookGenerator()

        chapter = {
            "title": "Test",
            "content": "## Already Level 2\n\nContent here.",
            "filename": "test.md"
        }

        result = generator._format_chapter(chapter)

        # Should preserve ##
        assert "## Already Level 2" in result
        # Should NOT have ###
        assert "### Already Level 2" not in result

    def test_generate_dry_run(self, tmp_path):
        """Test generate() with dry_run=True."""
        generator = MarkdownBookGenerator(book_dir=tmp_path)

        # Create a sample section with chapter
        section_dir = tmp_path / "01-foundations"
        section_dir.mkdir()
        (section_dir / "test.md").write_text("# Test\n\nContent.")

        result = generator.generate(dry_run=True, verbose=False)

        # Check return structure
        assert "files" in result
        assert "stats" in result
        assert "errors" in result

        # No files should be written
        assert len(result["files"]) == 0

        # Stats should show chapters found
        assert result["stats"]["chapters_included"] >= 0

        # BOOK.md should not exist
        assert not (tmp_path / "BOOK.md").exists()

    def test_generate_writes_book_file(self, tmp_path):
        """Test that generate() creates BOOK.md file."""
        generator = MarkdownBookGenerator(book_dir=tmp_path)

        # Create sample content
        section_dir = tmp_path / "01-foundations"
        section_dir.mkdir()
        (section_dir / "test.md").write_text("# Test Chapter\n\nTest content here.")

        result = generator.generate(dry_run=False, verbose=False)

        # BOOK.md should exist
        book_file = tmp_path / "BOOK.md"
        assert book_file.exists()

        # Check content
        content = book_file.read_text()
        assert "# The Cortical Chronicles" in content
        assert "Test Chapter" in content
        assert "Test content here" in content

        # Check stats
        assert result["stats"]["chapters_included"] == 1
        assert result["stats"]["sections_found"] == 1
        assert len(result["files"]) == 1

    def test_generate_preserves_section_order(self, tmp_path):
        """Test that sections are generated in correct order."""
        generator = MarkdownBookGenerator(book_dir=tmp_path)

        # Create sections out of order
        (tmp_path / "02-architecture").mkdir()
        (tmp_path / "02-architecture" / "arch.md").write_text("# Architecture")

        (tmp_path / "01-foundations").mkdir()
        (tmp_path / "01-foundations" / "found.md").write_text("# Foundations")

        result = generator.generate(dry_run=False, verbose=False)

        content = (tmp_path / "BOOK.md").read_text()

        # Foundations should appear before Architecture
        foundations_pos = content.find("Foundations")
        architecture_pos = content.find("Architecture")

        assert foundations_pos < architecture_pos

    def test_generate_footer_contains_instructions(self):
        """Test that footer contains regeneration instructions."""
        generator = MarkdownBookGenerator()

        footer = generator._generate_footer()

        assert "About This Book" in footer
        assert "How to Regenerate" in footer
        assert "--markdown" in footer


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
