# Refactoring Plan: generate_book.py

**Status:** Proposed
**Date:** 2025-12-17
**Updated:** 2025-12-17 (after merge from main)
**Target:** `scripts/generate_book.py` (4,970 lines)

## Problem Statement

The `scripts/generate_book.py` file is excessively large at 4,970 lines, containing 16 generator classes plus supporting utilities. This causes:

- **Maintainability issues** - Hard to navigate and understand
- **Code duplication** - Multiple classes implement identical methods
- **Testing difficulties** - Tight coupling to file I/O
- **Merge conflicts** - Multiple contributors modifying same file

## Current Structure Analysis

### Generator Classes (16 total)

| Class | Lines | Purpose |
|-------|-------|---------|
| `ChapterGenerator` (ABC) | ~57 | Base class |
| `BookBuilder` | ~54 | Orchestrator |
| `PlaceholderGenerator` | ~34 | Placeholder |
| `AlgorithmChapterGenerator` | ~148 | Algorithms from VISION.md |
| `ModuleDocGenerator` | ~301 | Module docs from .ai_meta |
| `SearchIndexGenerator` | ~261 | Search index (search.json) |
| `MarkdownBookGenerator` | ~333 | Combined BOOK.md |
| `DecisionStoryGenerator` | ~374 | ADR stories |
| `CaseStudyGenerator` | ~416 | Case studies from sessions |
| `CommitNarrativeSynthesizer` | ~532 | Commit sequence stories |
| `CommitNarrativeGenerator` | ~404 | Timeline/features chapters |
| `LessonExtractor` | ~341 | Lessons from commits |
| `ConceptEvolutionGenerator` | ~404 | Concept evolution tracking |
| `ReaderJourneyGenerator` | ~470 | Learning paths |
| `ExerciseGenerator` | ~399 | Exercises from tests |
| `MLIntelligenceGenerator` | ~254 | **NEW** ML MoE + Attention Marketplace |

### Duplicated Code Patterns

The following methods are duplicated across multiple classes:

| Method | Occurrences | Classes |
|--------|-------------|---------|
| `_run_git()` | 5+ | CommitNarrativeGenerator, LessonExtractor, ConceptEvolutionGenerator, ReaderJourneyGenerator, etc. |
| `_load_commits()` | 5+ | Multiple commit-related generators |
| `_load_ml_commits()` | 4 | CommitNarrativeGenerator, LessonExtractor, etc. |
| `_extract_commit_type()` | 3 | CommitNarrativeGenerator, CommitNarrativeSynthesizer, LessonExtractor |
| `_find_adr_references()` | 2 | CommitNarrativeGenerator, DecisionStoryGenerator |
| `_generate_index()` | 6+ | Most generators |

## Proposed Solution

### New Package Structure

```
scripts/book_generation/
├── __init__.py              # Re-exports public API
├── base.py                  # ChapterGenerator, BookBuilder (~100 lines)
├── loaders.py               # Data loading (git, ML data, ADRs) (~200 lines)
├── formatters.py            # Markdown formatting, frontmatter (~150 lines)
├── utils.py                 # Date utils, slugify, text extraction (~100 lines)
│
└── generators/              # One file per generator
    ├── __init__.py          # Re-exports all generators
    ├── algorithm.py         # AlgorithmChapterGenerator (~150 lines)
    ├── modules.py           # ModuleDocGenerator (~250 lines)
    ├── search.py            # SearchIndexGenerator (~200 lines)
    ├── markdown.py          # MarkdownBookGenerator (~250 lines)
    ├── decisions.py         # DecisionStoryGenerator (~300 lines)
    ├── case_studies.py      # CaseStudyGenerator (~350 lines)
    ├── commits.py           # CommitNarrativeGenerator + Synthesizer (~400 lines)
    ├── lessons.py           # LessonExtractor (~250 lines)
    ├── concepts.py          # ConceptEvolutionGenerator (~300 lines)
    ├── journey.py           # ReaderJourneyGenerator (~350 lines)
    ├── exercises.py         # ExerciseGenerator (~300 lines)
    └── ml_intelligence.py   # MLIntelligenceGenerator (~200 lines) **NEW**

scripts/generate_book.py     # Thin CLI wrapper (~50 lines)
```

### Shared Utilities

**`loaders.py`** - Consolidate all data loading:

```python
class DataLoader:
    """Centralized data loading for book generation."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root

    def run_git(self, *args) -> str:
        """Run a git command and return output."""
        ...

    def load_commits(self, limit: int = 200) -> List[Dict]:
        """Load commits from git history."""
        ...

    def load_ml_commits(self) -> Dict[str, Dict]:
        """Load enriched commits from ML data store."""
        ...

    def load_ml_sessions(self) -> List[Dict]:
        """Load sessions from ML data store."""
        ...

    def load_ml_chats(self) -> List[Dict]:
        """Load chat transcripts from ML data store."""
        ...

    def load_adrs(self) -> Dict[str, Dict]:
        """Load ADR files from samples/decisions/."""
        ...
```

**`formatters.py`** - Consolidate markdown generation:

```python
def generate_frontmatter(title: str, tags: List[str],
                         source_files: List[str], generator_name: str) -> str:
    """Generate YAML frontmatter for a chapter."""
    ...

def extract_frontmatter(content: str) -> Dict[str, Any]:
    """Extract YAML frontmatter from markdown content."""
    ...

def remove_frontmatter(content: str) -> str:
    """Remove YAML frontmatter from markdown content."""
    ...

def make_anchor(text: str) -> str:
    """Convert text to a valid markdown anchor."""
    ...

def slugify(text: str) -> str:
    """Convert text to a URL-safe slug."""
    ...
```

**`utils.py`** - Shared helper functions:

```python
def extract_commit_type(message: str) -> str:
    """Extract commit type (feat, fix, etc.) from commit message."""
    ...

def find_adr_references(message: str) -> List[str]:
    """Find ADR references (ADR-001, etc.) in text."""
    ...

def days_between(date1: str, date2: str) -> int:
    """Calculate days between two ISO date strings."""
    ...

def extract_keywords(content: str, top_n: int = 10) -> List[str]:
    """Extract top keywords from content."""
    ...

def extract_excerpt(content: str, max_length: int = 200) -> str:
    """Extract a summary excerpt from content."""
    ...
```

## Migration Strategy

### Phase 1: Extract Utilities (Day 1)

1. Create `scripts/book_generation/` package
2. Extract shared utilities to `loaders.py`, `formatters.py`, `utils.py`
3. Add imports to original file to verify no breakage
4. Run existing tests

### Phase 2: Move Base Classes (Day 1)

1. Move `ChapterGenerator` and `BookBuilder` to `base.py`
2. Update imports in original file
3. Run tests

### Phase 3: Migrate Generators (Days 2-3)

Migrate one generator at a time, in order of dependency:

1. `AlgorithmChapterGenerator` → `generators/algorithm.py`
2. `ModuleDocGenerator` → `generators/modules.py`
3. `SearchIndexGenerator` → `generators/search.py`
4. `MarkdownBookGenerator` → `generators/markdown.py`
5. `DecisionStoryGenerator` → `generators/decisions.py`
6. `CaseStudyGenerator` → `generators/case_studies.py`
7. `CommitNarrativeGenerator` + `CommitNarrativeSynthesizer` → `generators/commits.py`
8. `LessonExtractor` → `generators/lessons.py`
9. `ConceptEvolutionGenerator` → `generators/concepts.py`
10. `ReaderJourneyGenerator` → `generators/journey.py`
11. `ExerciseGenerator` → `generators/exercises.py`
12. `MLIntelligenceGenerator` → `generators/ml_intelligence.py` **NEW**

### Phase 4: Update CLI (Day 3)

1. Make `scripts/generate_book.py` a thin wrapper
2. Import all generators from package
3. Keep CLI interface unchanged

### Phase 5: Update Tests (Day 4)

1. Update test imports
2. Add unit tests for new utility modules
3. Verify all existing tests pass

## Expected Benefits

| Metric | Before | After |
|--------|--------|-------|
| Main file size | 4,970 lines | ~50 lines (CLI only) |
| Largest module | 4,970 lines | ~400 lines |
| Code duplication | ~600 lines | ~50 lines |
| Files | 1 | 17 |
| Generator count | 16 in 1 file | 1 per file (12 files) |
| Testability | Poor | Good |

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Breaking existing functionality | Run full test suite after each phase |
| Import errors | Keep backward-compatible imports in `__init__.py` |
| CI/CD breakage | Test in branch before merging |
| Missing edge cases | Maintain identical behavior through extraction |

## Success Criteria

- [ ] All existing tests pass
- [ ] `python scripts/generate_book.py --help` works unchanged
- [ ] `python scripts/generate_book.py --dry-run` produces identical output
- [ ] Each module is < 500 lines
- [ ] No duplicated utility functions across modules
- [ ] Test coverage maintained at current level

## Next Steps

1. Review and approve this plan
2. Create feature branch (already on `claude/refactor-book-generation-vvG8g`)
3. Begin Phase 1 implementation
