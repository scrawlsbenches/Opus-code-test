# Contributing to Cortical Text Processor

Thank you for your interest in contributing! This guide will help you get started.

## Quick Start

```bash
# 1. Fork the repository on GitHub

# 2. Clone your fork
git clone https://github.com/YOUR-USERNAME/Opus-code-test.git
cd Opus-code-test

# 3. Install in development mode
pip install -e ".[dev]"

# 4. Run tests to verify setup
python -m unittest discover -s tests -v

# 5. Make your changes on a feature branch
git checkout -b feature/your-feature-name

# 6. Run tests again
python -m unittest discover -s tests -v

# 7. Commit and push
git add .
git commit -m "Add your feature"
git push origin feature/your-feature-name

# 8. Create a Pull Request on GitHub
```

## Development Setup

### Requirements

- Python 3.9 or higher
- No external dependencies for the library itself
- `coverage` package for running tests with coverage (installed with `.[dev]`)

### Installation

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Verify installation
python -c "from cortical import CorticalTextProcessor; print('OK')"
```

### Running Tests

```bash
# Run all tests
python -m unittest discover -s tests -v

# Run with coverage
coverage run -m unittest discover -s tests -v
coverage report -m

# Run a specific test file
python -m unittest tests/test_processor.py -v

# Run a specific test
python -m unittest tests.test_processor.TestProcessor.test_process_document -v
```

## Code Style

### Python Style

- Follow [PEP 8](https://pep8.org/) conventions
- Use **type hints** on all public functions
- Use **Google-style docstrings** with Args/Returns sections

### Example

```python
from typing import Dict, List, Optional

def find_documents(
    query: str,
    layers: Dict[str, Any],
    top_n: int = 5
) -> List[Tuple[str, float]]:
    """
    Find documents matching a query.

    Args:
        query: Search query string
        layers: Dictionary of hierarchical layers
        top_n: Number of results to return

    Returns:
        List of (doc_id, score) tuples sorted by relevance
    """
    # Implementation
```

### Naming Conventions

- `snake_case` for functions and variables
- `PascalCase` for classes
- `UPPER_CASE` for constants
- Prefix private methods with `_`

## Project Structure

```
cortical/
├── processor.py      # Main API - CorticalTextProcessor class
├── analysis.py       # Graph algorithms (PageRank, TF-IDF, clustering)
├── query.py          # Search, retrieval, query expansion
├── semantics.py      # Relation extraction, inheritance
├── minicolumn.py     # Core data structures (Minicolumn, Edge)
├── layers.py         # HierarchicalLayer with O(1) lookups
├── embeddings.py     # Graph embeddings
├── gaps.py           # Knowledge gap detection
├── persistence.py    # Save/load functionality
├── tokenizer.py      # Text tokenization
└── config.py         # Configuration dataclass

tests/                # Unit tests (run with unittest)
samples/              # Example documents for testing
docs/                 # Extended documentation
scripts/              # Utility scripts
```

## Making Changes

### Before You Start

1. Check existing tasks using `python scripts/task_utils.py list` for planned work
2. Read [CLAUDE.md](CLAUDE.md) for detailed developer documentation
3. Look at existing code to understand patterns

### When Implementing

1. **Read before writing** - Understand existing code before modifying
2. **Follow existing patterns** - The codebase is consistent
3. **Add tests** - All new functionality needs tests
4. **Update docs** - Keep documentation in sync

### Commit Messages

Write clear, concise commit messages:

```
Add query expansion for code patterns

- Implement programming synonym expansion
- Add code-aware tokenization option
- Update tests for new functionality
```

## Pull Request Process

1. Ensure all tests pass
2. Update documentation if needed
3. Create a task using `python scripts/new_task.py "your description"` if relevant
4. Request review from maintainers

### PR Checklist

- [ ] Tests pass (`python -m unittest discover -s tests -v`)
- [ ] Code follows style guidelines
- [ ] Documentation updated (if applicable)
- [ ] No unrelated changes included

## Quality Standards

We follow rigorous standards documented in:

- **[docs/code-of-ethics.md](docs/code-of-ethics.md)** - Scientific rigor and documentation standards
- **[docs/definition-of-done.md](docs/definition-of-done.md)** - When is a task truly complete?

Key principles:

1. **Verify claims** - Test assumptions, check edge cases
2. **Document findings** - Create tasks using `python scripts/new_task.py` for discovered issues
3. **Test thoroughly** - Empty corpus, single doc, multiple docs, edge cases
4. **Be skeptical** - Question the obvious

## Getting Help

- Open an issue for bugs or feature requests
- Check [CLAUDE.md](CLAUDE.md) for developer documentation
- Review existing tests for usage patterns

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
