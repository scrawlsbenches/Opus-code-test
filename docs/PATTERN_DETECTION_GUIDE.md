# Code Pattern Detection Guide

## Overview

The Cortical Text Processor now includes comprehensive code pattern detection capabilities. This feature can identify 32+ common programming patterns across 9 categories, making it easier to understand and analyze codebases.

## Supported Patterns

### Creational Patterns
- **Singleton**: Single instance control patterns
- **Factory**: Object creation patterns
- **Builder**: Fluent construction patterns

### Structural Patterns
- **Decorator**: Wrapping behavior patterns
- **Adapter**: Interface conversion patterns
- **Proxy**: Access control patterns

### Behavioral Patterns
- **Context Manager**: Resource management (with/as, __enter__/__exit__)
- **Generator**: Lazy iteration (yield)
- **Iterator**: Custom iteration (__iter__/__next__)
- **Observer**: Event notification patterns
- **Strategy**: Algorithm selection patterns

### Concurrency Patterns
- **Async/Await**: Asynchronous code patterns
- **Thread Safety**: Locks and synchronization
- **Concurrent Futures**: Thread/process pools

### Error Handling
- **Error Handling**: Try/except blocks
- **Custom Exception**: Custom exception classes
- **Assertion**: Runtime checks

### Python Idioms
- **Property Decorator**: Computed attributes (@property)
- **Dataclass**: Structured data (@dataclass)
- **Slots**: Memory optimization (__slots__)
- **Magic Methods**: Operator overloading (__repr__, __eq__, etc.)
- **Comprehension**: List/dict/set comprehensions
- **Unpacking**: *args, **kwargs patterns

### Testing Patterns
- **Unittest Class**: unittest.TestCase classes
- **Pytest Test**: pytest test functions
- **Mock Usage**: unittest.mock patterns
- **Fixture**: pytest fixtures

### Functional Programming
- **Lambda**: Anonymous functions
- **Map/Filter/Reduce**: Functional operations
- **Partial Application**: Currying patterns

### Type Annotations
- **Type Hints**: Static typing annotations
- **TYPE_CHECKING**: Import-time type guards

## Quick Start

```python
from cortical.processor import CorticalTextProcessor

# Create processor and add code
processor = CorticalTextProcessor()
processor.process_document('mycode.py', """
async def fetch_users():
    try:
        users = await api.get_users()
        for user in users:
            yield user
    except Exception as e:
        raise FetchError(e)
""")

# Detect patterns in the document
patterns = processor.detect_patterns('mycode.py')
print(patterns)
# Output: {
#     'async_await': [2, 4],
#     'error_handling': [3, 7],
#     'generator': [5],
#     'custom_exception': [7]
# }
```

## Usage Examples

### 1. Detect Patterns in a Single Document

```python
patterns = processor.detect_patterns('file.py')

for pattern_name, line_numbers in patterns.items():
    print(f"{pattern_name}: found on lines {line_numbers}")
```

### 2. Get Pattern Summary

```python
summary = processor.get_pattern_summary('file.py')
# Returns: {'async_await': 3, 'generator': 2, ...}

for pattern, count in summary.items():
    print(f"{pattern}: {count} occurrences")
```

### 3. Generate Human-Readable Report

```python
# Without line numbers
report = processor.format_pattern_report('file.py')
print(report)

# With line numbers
report = processor.format_pattern_report('file.py', show_lines=True)
print(report)
```

### 4. Corpus-Wide Pattern Analysis

```python
# Detect patterns across all documents
corpus_patterns = processor.detect_patterns_in_corpus()

for doc_id, patterns in corpus_patterns.items():
    print(f"{doc_id}: {list(patterns.keys())}")

# Get corpus statistics
stats = processor.get_corpus_pattern_statistics()
print(f"Total documents: {stats['total_documents']}")
print(f"Patterns found: {stats['patterns_found']}")
print(f"Most common: {stats['most_common_pattern']}")
```

### 5. Filter for Specific Patterns

```python
# Only detect async patterns
async_patterns = processor.detect_patterns(
    'file.py',
    patterns=['async_await', 'concurrent_futures']
)

# Find all files with a specific pattern
corpus_patterns = processor.detect_patterns_in_corpus(
    patterns=['singleton']
)
singleton_files = [
    doc_id for doc_id, patterns in corpus_patterns.items()
    if 'singleton' in patterns
]
```

### 6. List Available Patterns

```python
# List all pattern names
patterns = processor.list_available_patterns()
print(f"Total patterns: {len(patterns)}")

# List all categories
categories = processor.list_pattern_categories()
print(f"Categories: {', '.join(categories)}")
```

### 7. Pattern Metadata

```python
from cortical.patterns import (
    get_pattern_description,
    get_pattern_category,
    list_patterns_by_category
)

# Get pattern info
desc = get_pattern_description('singleton')
cat = get_pattern_category('singleton')

# List patterns in a category
creational = list_patterns_by_category('creational')
# Returns: ['builder', 'factory', 'singleton']
```

## Advanced Usage

### Finding Similar Code Patterns

```python
# Find all files using async patterns
corpus_patterns = processor.detect_patterns_in_corpus(
    patterns=['async_await', 'concurrent_futures']
)

async_files = {}
for doc_id, patterns in corpus_patterns.items():
    if any(p in patterns for p in ['async_await', 'concurrent_futures']):
        async_files[doc_id] = patterns

print(f"Found {len(async_files)} files using async patterns")
```

### Pattern-Based Code Search

```python
# Find all factory implementations
factories = processor.detect_patterns_in_corpus(patterns=['factory'])

for doc_id, patterns in factories.items():
    if 'factory' in patterns:
        print(f"{doc_id}: Factory pattern on lines {patterns['factory']}")
```

### Code Quality Analysis

```python
stats = processor.get_corpus_pattern_statistics()

# Find documents with good test coverage
corpus_patterns = processor.detect_patterns_in_corpus(
    patterns=['pytest_test', 'unittest_class', 'mock_usage']
)

test_files = [
    doc_id for doc_id, patterns in corpus_patterns.items()
    if any(p in patterns for p in ['pytest_test', 'unittest_class'])
]

print(f"Test files: {len(test_files)}")
```

## Integration with Search

Pattern detection works seamlessly with the existing search capabilities:

```python
# Index your codebase
for filepath in get_python_files():
    with open(filepath) as f:
        processor.process_document(filepath, f.read())

# Find files and their patterns
results = processor.find_documents_for_query("authentication")

for doc_id, score in results:
    patterns = processor.detect_patterns(doc_id)
    print(f"{doc_id} (score: {score:.2f})")
    if patterns:
        print(f"  Patterns: {', '.join(patterns.keys())}")
```

## Pattern Categories

The 9 pattern categories are:

1. **behavioral** - Behavioral design patterns
2. **concurrency** - Async/threading patterns
3. **creational** - Object creation patterns
4. **error_handling** - Error handling patterns
5. **functional** - Functional programming patterns
6. **idiom** - Python-specific idioms
7. **structural** - Structural design patterns
8. **testing** - Testing patterns
9. **typing** - Type annotation patterns

## Performance Notes

- Pattern detection uses regex-based matching (no AST parsing)
- Very fast: can analyze thousands of lines per second
- Line numbers are accurately tracked for all matches
- Multi-line patterns are supported

## Demo Script

Run the included demo script to see pattern detection in action:

```bash
python demo_pattern_detection.py
```

This will demonstrate:
- Pattern detection across multiple files
- Report generation
- Corpus-wide statistics
- Pattern filtering
- Category grouping

## Module Reference

### Main Functions

- `detect_patterns(doc_id, patterns=None)` - Detect patterns in a document
- `detect_patterns_in_corpus(patterns=None)` - Detect in all documents
- `get_pattern_summary(doc_id)` - Count occurrences per pattern
- `get_corpus_pattern_statistics()` - Corpus-wide statistics
- `format_pattern_report(doc_id, show_lines=False)` - Human-readable report
- `list_available_patterns()` - List all pattern names
- `list_pattern_categories()` - List all categories

### Direct Module Access

```python
from cortical.patterns import (
    detect_patterns_in_text,
    detect_patterns_in_documents,
    get_pattern_description,
    get_pattern_category,
    format_pattern_report,
    list_all_patterns,
    list_all_categories,
)

# Analyze text directly without a processor
code = "async def fetch(): pass"
patterns = detect_patterns_in_text(code)
```

## Implementation Notes

The pattern detection is implemented in:
- `cortical/patterns.py` - Core pattern detection logic (32 patterns)
- `cortical/processor/introspection.py` - Processor integration
- `tests/unit/test_patterns.py` - Comprehensive test suite

All patterns are defined with:
- Regex pattern (for matching)
- Description (human-readable)
- Category (for grouping)
