# Contributing to The Cortical Chronicles

> **Welcome!** This guide shows you how to add new generators and extend "The Cortical Chronicles."

---

## Table of Contents

- [Quick Start](#quick-start)
- [Generator Architecture](#generator-architecture)
- [Step-by-Step: Adding a Generator](#step-by-step-adding-a-generator)
- [Best Practices](#best-practices)
- [Testing Your Generator](#testing-your-generator)
- [CI Integration](#ci-integration)
- [Examples](#examples)

---

## Quick Start

### Prerequisites

```bash
# Ensure you have dependencies
pip install -e ".[dev]"

# Verify PyYAML is installed
python -c "import yaml; print('PyYAML OK')"

# Test existing generators
python scripts/generate_book.py --list
python scripts/generate_book.py --dry-run
```

### Template for New Generator

```python
from scripts.generate_book import ChapterGenerator
from pathlib import Path
from typing import Dict, Any, List

class MyChapterGenerator(ChapterGenerator):
    """Generate chapters from <your data source>."""

    @property
    def name(self) -> str:
        """Generator name for CLI and logging."""
        return "mychapter"

    @property
    def output_dir(self) -> str:
        """Subdirectory in book/ for output."""
        return "03-decisions"  # Choose: 00-05 based on content type

    def generate(self, dry_run: bool = False, verbose: bool = False) -> Dict[str, Any]:
        """
        Generate chapter content.

        Args:
            dry_run: If True, don't write files (just log)
            verbose: If True, print detailed progress

        Returns:
            Dict with:
                - files: List of generated file paths
                - stats: Generation statistics
                - errors: Any errors encountered
        """
        errors = []
        stats = {
            "items_processed": 0,
            "chapters_written": 0
        }

        if verbose:
            print("  Processing data...")

        # 1. Load your data source
        try:
            data = self._load_data()
        except Exception as e:
            errors.append(f"Failed to load data: {e}")
            return {"files": [], "stats": stats, "errors": errors}

        # 2. Process each item
        for item in data:
            content = self._generate_chapter_content(item)
            filename = self._generate_filename(item)

            if verbose:
                print(f"  Generating: {filename}")

            self.write_chapter(filename, content, dry_run=dry_run)
            stats["chapters_written"] += 1

        stats["items_processed"] = len(data)

        return {
            "files": [str(f) for f in self.generated_files],
            "stats": stats,
            "errors": errors
        }

    def _load_data(self) -> List[Any]:
        """Load your data source."""
        # Implement data loading
        pass

    def _generate_chapter_content(self, item: Any) -> str:
        """Generate markdown content for one chapter."""
        # Generate frontmatter
        frontmatter = self.generate_frontmatter(
            title=item['title'],
            tags=['tag1', 'tag2'],
            source_files=['source.py']
        )

        # Build content
        content = frontmatter
        content += f"# {item['title']}\n\n"
        content += item['body']
        content += "\n\n---\n\n"
        content += "*This chapter is part of [The Cortical Chronicles](../README.md).*\n"

        return content

    def _generate_filename(self, item: Any) -> str:
        """Generate filename from item."""
        # Slugify the title
        slug = item['title'].lower().replace(' ', '-')
        return f"{slug}.md"
```

---

## Generator Architecture

### Base Class: `ChapterGenerator`

All generators inherit from `ChapterGenerator` (defined in `scripts/generate_book.py`).

**Required Methods:**

| Method | Returns | Purpose |
|--------|---------|---------|
| `name` (property) | `str` | Generator identifier for CLI |
| `output_dir` (property) | `str` | Subdirectory in book/ |
| `generate(dry_run, verbose)` | `Dict[str, Any]` | Main generation logic |

**Provided Helper Methods:**

| Method | Purpose |
|--------|---------|
| `write_chapter(filename, content, dry_run)` | Write chapter with standard handling |
| `generate_frontmatter(title, tags, sources)` | Generate YAML frontmatter |

**Instance Variables:**

- `self.book_dir`: Path to book/ directory
- `self.generated_files`: List of written file paths (auto-tracked by `write_chapter`)

### Output Directories

Choose the appropriate section for your content:

| Directory | Purpose | Example Generators |
|-----------|---------|-------------------|
| `00-preface` | Book introduction | (manual) |
| `01-foundations` | Algorithm theory | AlgorithmChapterGenerator |
| `02-architecture` | Module documentation | ModuleDocGenerator |
| `03-decisions` | ADRs, design decisions | DecisionRecordGenerator |
| `04-evolution` | Commit narratives | CommitNarrativeGenerator |
| `05-future` | Roadmap, vision | (placeholder) |

### Return Value Format

```python
{
    "files": [
        "book/01-foundations/alg-pagerank.md",
        "book/01-foundations/alg-bm25.md"
    ],
    "stats": {
        "algorithms_found": 6,
        "chapters_written": 6,
        "custom_metric": 42
    },
    "errors": [
        "Warning: Optional data not found"
    ]
}
```

---

## Step-by-Step: Adding a Generator

### Step 1: Create the Generator Class

Add to `scripts/generate_book.py` or create a new file:

```python
class MyChapterGenerator(ChapterGenerator):
    """One-line description."""

    @property
    def name(self) -> str:
        return "mychapter"

    @property
    def output_dir(self) -> str:
        return "03-decisions"  # Choose appropriate section

    def generate(self, dry_run: bool = False, verbose: bool = False) -> Dict[str, Any]:
        # Implementation
        pass
```

### Step 2: Register in BookBuilder

In `scripts/generate_book.py`, find the `main()` function and add:

```python
def main():
    # ...existing code...

    # Register real generators
    builder.register_generator(AlgorithmChapterGenerator(book_dir=args.output))
    builder.register_generator(ModuleDocGenerator(book_dir=args.output))
    builder.register_generator(CommitNarrativeGenerator(book_dir=args.output))
    builder.register_generator(MyChapterGenerator(book_dir=args.output))  # <-- ADD THIS
    builder.register_generator(SearchIndexGenerator(book_dir=args.output))

    # ...rest of main()...
```

**Important:** Register `SearchIndexGenerator` LAST so it indexes all other chapters.

### Step 3: Test the Generator

```bash
# List all generators (verify yours appears)
python scripts/generate_book.py --list

# Test with dry-run
python scripts/generate_book.py --chapter mychapter --dry-run --verbose

# Generate for real
python scripts/generate_book.py --chapter mychapter --verbose

# Verify output
ls -la book/03-decisions/
```

### Step 4: Regenerate Search Index

After adding new chapters:

```bash
python scripts/generate_book.py --chapter search
```

### Step 5: Test Full Build

```bash
# Full regeneration
python scripts/generate_book.py --verbose

# Verify all chapters
find book/ -name "*.md" | sort

# Validate JSON outputs
python -c "import json; json.load(open('book/index.json'))"
python -c "import json; json.load(open('book/search.json'))"
```

---

## Best Practices

### 1. Output File Naming

**Convention:**

- Algorithm docs: `alg-<name>.md` (e.g., `alg-pagerank.md`)
- Module docs: `mod-<category>.md` (e.g., `mod-processor.md`)
- Narrative docs: `<topic>.md` (e.g., `timeline.md`, `features.md`)
- Decision records: `adr-<num>-<slug>.md` (e.g., `adr-001-architecture.md`)

**Rules:**

- Use lowercase
- Use hyphens, not underscores
- Be descriptive but concise
- Avoid special characters

### 2. Frontmatter Requirements

All chapters must have YAML frontmatter:

```yaml
---
title: "Chapter Title"
generated: "2025-12-16T10:30:00Z"
generator: "mychapter"
source_files:
  - "path/to/source.py"
  - "path/to/data.json"
tags:
  - tag1
  - tag2
  - tag3
---
```

**Required Fields:**

- `title`: Human-readable chapter title
- `generated`: ISO 8601 timestamp (use `datetime.utcnow().isoformat() + "Z"`)
- `generator`: Your generator's `name` property
- `source_files`: List of source files used
- `tags`: List of tags for categorization

**Use the helper:**

```python
frontmatter = self.generate_frontmatter(
    title="My Chapter",
    tags=['algorithms', 'foundations'],
    source_files=['docs/VISION.md', 'cortical/analysis.py']
)
```

### 3. Cross-Reference Patterns

**Link to other chapters:**

```markdown
See [Algorithm Documentation](alg-pagerank.md) for details.
See [Architecture Overview](../02-architecture/index.md).
```

**Link to source code:**

```markdown
**Implementation:** `cortical/analysis.py:compute_pagerank()`
```

**Link to ADRs:**

```markdown
**Related Decision:** [ADR-001: Architecture](../../samples/decisions/adr-001-*.md)
```

**Link to git commits:**

```markdown
**Commit:** `a1b2c3d`
```

### 4. Error Handling

**Always handle exceptions gracefully:**

```python
def generate(self, dry_run: bool = False, verbose: bool = False) -> Dict[str, Any]:
    errors = []
    stats = {}

    try:
        data = self._load_data()
    except FileNotFoundError as e:
        errors.append(f"Source file not found: {e}")
        return {"files": [], "stats": stats, "errors": errors}
    except Exception as e:
        errors.append(f"Unexpected error loading data: {e}")
        return {"files": [], "stats": stats, "errors": errors}

    # Continue processing...
```

**Error message guidelines:**

- Be specific (include file names, line numbers)
- Suggest fixes when possible
- Use warnings for non-critical issues
- Return partial results if some items succeed

**Example:**

```python
# Good
errors.append("Failed to parse VISION.md line 42: Missing '###' prefix")

# Bad
errors.append("Error in file")
```

### 5. Verbose Output

Provide helpful progress indicators:

```python
def generate(self, dry_run: bool = False, verbose: bool = False) -> Dict[str, Any]:
    if verbose:
        print("  Loading source data...")

    data = self._load_data()

    if verbose:
        print(f"  Found {len(data)} items to process")

    for item in data:
        if verbose:
            print(f"  Generating: {item['title']}")

        # Process item...
```

**Guidelines:**

- Use 2-space indentation for generator output
- Log counts and progress
- Don't log every detail (avoid spam)
- Use dry-run mode for testing without writes

### 6. Dry-Run Support

Always respect the `dry_run` parameter:

```python
# Good: Use write_chapter helper (handles dry_run automatically)
self.write_chapter(filename, content, dry_run=dry_run)

# If you need custom file writing:
if dry_run:
    print(f"  Would write: {output_path}")
else:
    output_path.write_text(content)
    self.generated_files.append(output_path)
```

### 7. Statistics Reporting

Return meaningful statistics:

```python
stats = {
    "items_found": len(all_items),
    "items_processed": len(processed_items),
    "chapters_written": len(self.generated_files),
    "items_skipped": len(skipped_items),
    "warnings": len(warnings)
}
```

---

## Testing Your Generator

### Unit Testing

Create a test file `tests/test_book_generation.py`:

```python
import unittest
from pathlib import Path
from scripts.generate_book import MyChapterGenerator

class TestMyChapterGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = MyChapterGenerator()

    def test_name(self):
        self.assertEqual(self.generator.name, "mychapter")

    def test_output_dir(self):
        self.assertEqual(self.generator.output_dir, "03-decisions")

    def test_dry_run(self):
        """Test that dry-run doesn't write files."""
        result = self.generator.generate(dry_run=True, verbose=False)
        self.assertEqual(result['files'], [])
        self.assertGreaterEqual(result['stats']['items_processed'], 0)

    def test_generate(self):
        """Test actual generation."""
        result = self.generator.generate(dry_run=False, verbose=False)
        self.assertGreater(len(result['files']), 0)
        self.assertEqual(result['errors'], [])
```

### Integration Testing

```bash
# 1. Full dry-run test
python scripts/generate_book.py --dry-run --verbose

# 2. Generate to temporary directory
python scripts/generate_book.py --output /tmp/test-book --chapter mychapter

# 3. Verify outputs
ls -la /tmp/test-book/03-decisions/
cat /tmp/test-book/03-decisions/example.md

# 4. Validate frontmatter
python -c "
import yaml
from pathlib import Path
for f in Path('/tmp/test-book').glob('**/*.md'):
    content = f.read_text()
    if content.startswith('---'):
        fm = content.split('---', 2)[1]
        yaml.safe_load(fm)
        print(f'{f.name}: OK')
"

# 5. Clean up
rm -rf /tmp/test-book
```

### Common Test Cases

1. **Empty data source** - Should return gracefully
2. **Malformed data** - Should log errors, continue processing
3. **Missing dependencies** - Should fail gracefully with clear error
4. **Dry-run mode** - Should not write any files
5. **Verbose mode** - Should print progress
6. **Frontmatter validation** - YAML should parse correctly
7. **Cross-references** - Links should be valid

---

## CI Integration

### GitHub Actions Workflow

Add to `.github/workflows/book-generation.yml`:

```yaml
name: Generate Book

on:
  push:
    branches: [main]
    paths:
      - 'docs/VISION.md'
      - 'cortical/**/*.py'
      - 'scripts/generate_book.py'
  workflow_dispatch:

jobs:
  generate:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3
        with:
          fetch-depth: 0  # Full history for commit narratives

      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -e ".[dev]"

      - name: Generate book
        run: |
          python scripts/generate_book.py --verbose

      - name: Validate outputs
        run: |
          python -c "import json; json.load(open('book/index.json'))"
          python -c "import json; json.load(open('book/search.json'))"

      - name: Upload artifact
        uses: actions/upload-artifact@v3
        with:
          name: cortical-chronicles
          path: book/
```

### Pre-Commit Hook

Add to `.git/hooks/pre-commit`:

```bash
#!/bin/bash
# Regenerate book if source files changed

SOURCES="docs/VISION.md cortical/ scripts/generate_book.py"
CHANGED=$(git diff --cached --name-only $SOURCES)

if [ -n "$CHANGED" ]; then
    echo "📚 Regenerating book chapters..."
    python scripts/generate_book.py --verbose || exit 1

    # Stage generated files
    git add book/
fi
```

---

## Examples

### Example 1: Simple Data-Driven Generator

Generate chapters from JSON files:

```python
class DataChapterGenerator(ChapterGenerator):
    """Generate chapters from data/*.json files."""

    @property
    def name(self) -> str:
        return "data"

    @property
    def output_dir(self) -> str:
        return "05-future"

    def generate(self, dry_run: bool = False, verbose: bool = False) -> Dict[str, Any]:
        import json

        data_dir = Path(__file__).parent.parent / "data"
        json_files = list(data_dir.glob("*.json"))

        if verbose:
            print(f"  Found {len(json_files)} JSON files")

        for json_file in json_files:
            data = json.loads(json_file.read_text())

            content = self.generate_frontmatter(
                title=data['title'],
                tags=data.get('tags', []),
                source_files=[str(json_file)]
            )
            content += f"# {data['title']}\n\n"
            content += data['body'] + "\n"

            filename = json_file.stem + ".md"
            self.write_chapter(filename, content, dry_run)

        return {
            "files": [str(f) for f in self.generated_files],
            "stats": {"files_processed": len(json_files)},
            "errors": []
        }
```

### Example 2: Git History Generator

Generate from commit messages:

```python
class GitHistoryGenerator(ChapterGenerator):
    """Generate timeline from git history."""

    @property
    def name(self) -> str:
        return "timeline"

    @property
    def output_dir(self) -> str:
        return "04-evolution"

    def _run_git(self, *args) -> str:
        import subprocess
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()

    def generate(self, dry_run: bool = False, verbose: bool = False) -> Dict[str, Any]:
        try:
            log_output = self._run_git("log", "--format=%H|%aI|%s|%an", "-100")
        except Exception as e:
            return {
                "files": [],
                "stats": {},
                "errors": [f"Git error: {e}"]
            }

        commits = []
        for line in log_output.split('\n'):
            if not line.strip():
                continue
            hash_val, timestamp, message, author = line.split('|', 3)
            commits.append({
                'hash': hash_val[:7],
                'date': timestamp[:10],
                'message': message,
                'author': author
            })

        # Generate timeline content
        content = self.generate_frontmatter(
            title="Project Timeline",
            tags=['timeline', 'history'],
            source_files=['git log']
        )
        content += "# Project Timeline\n\n"

        for commit in commits:
            content += f"- **{commit['date']}** (`{commit['hash']}`): {commit['message']}\n"

        self.write_chapter("timeline.md", content, dry_run)

        return {
            "files": [str(f) for f in self.generated_files],
            "stats": {"commits_processed": len(commits)},
            "errors": []
        }
```

### Example 3: Multi-File Generator

Generate multiple chapters from one data source:

```python
class MultiChapterGenerator(ChapterGenerator):
    """Generate multiple chapters from single source."""

    @property
    def name(self) -> str:
        return "concepts"

    @property
    def output_dir(self) -> str:
        return "02-architecture"

    def generate(self, dry_run: bool = False, verbose: bool = False) -> Dict[str, Any]:
        # Load Louvain clusters
        clusters = self._load_clusters()

        if verbose:
            print(f"  Found {len(clusters)} concept clusters")

        # Generate one chapter per cluster
        for cluster_id, terms in clusters.items():
            content = self.generate_frontmatter(
                title=f"Concept Cluster {cluster_id}",
                tags=['concepts', 'clustering', 'louvain'],
                source_files=['corpus_dev.pkl']
            )

            content += f"# Concept Cluster {cluster_id}\n\n"
            content += f"**Terms:** {', '.join(terms[:20])}\n\n"

            filename = f"concept-{cluster_id}.md"
            self.write_chapter(filename, content, dry_run)

        # Generate index
        index_content = self._generate_index(clusters)
        self.write_chapter("concepts-index.md", index_content, dry_run)

        return {
            "files": [str(f) for f in self.generated_files],
            "stats": {
                "clusters_found": len(clusters),
                "chapters_written": len(clusters) + 1
            },
            "errors": []
        }

    def _load_clusters(self) -> Dict[int, List[str]]:
        # Load from processor
        pass

    def _generate_index(self, clusters: Dict) -> str:
        # Generate index page
        pass
```

---

## Common Pitfalls

### ❌ Don't: Hardcode Paths

```python
# Bad
output_path = Path("/home/user/book/01-foundations/chapter.md")

# Good
output_path = self.book_dir / self.output_dir / "chapter.md"
```

### ❌ Don't: Ignore Dry-Run

```python
# Bad
with open(output_path, 'w') as f:
    f.write(content)

# Good
self.write_chapter(filename, content, dry_run=dry_run)
```

### ❌ Don't: Swallow Errors

```python
# Bad
try:
    data = self._load_data()
except:
    pass  # Silent failure!

# Good
try:
    data = self._load_data()
except Exception as e:
    errors.append(f"Failed to load data: {e}")
    return {"files": [], "stats": {}, "errors": errors}
```

### ❌ Don't: Generate Unsafe Filenames

```python
# Bad
filename = f"{user_input}.md"  # Could be "../../../etc/passwd.md"

# Good
filename = self._sanitize_filename(user_input) + ".md"

def _sanitize_filename(self, text: str) -> str:
    # Remove path separators
    text = text.replace('/', '-').replace('\\', '-')
    # Remove special chars
    text = re.sub(r'[^\w\s-]', '', text)
    # Normalize whitespace
    text = re.sub(r'[-\s]+', '-', text)
    return text.lower().strip('-')
```

---

## Next Steps

1. **Study existing generators** in `scripts/generate_book.py`
2. **Create your generator** following the template
3. **Test with dry-run** before writing files
4. **Validate frontmatter** with YAML parser
5. **Regenerate search index** after adding chapters
6. **Document your generator** in this guide

---

## Questions?

- Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues
- Review existing generators for patterns
- Test with `--dry-run --verbose` for debugging

---

*This guide is part of [The Cortical Chronicles](../README.md) documentation.*
