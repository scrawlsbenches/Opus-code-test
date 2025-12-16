# Troubleshooting Guide

> **Quick Recovery**: For most errors, run `python scripts/generate_book.py --dry-run --verbose` to diagnose without writing files.

This guide helps diagnose and fix common issues when generating "The Cortical Chronicles."

---

## Table of Contents

- [Common Errors](#common-errors)
- [Diagnostic Commands](#diagnostic-commands)
- [Recovery Procedures](#recovery-procedures)
- [Error Reference](#error-reference)
- [Getting Help](#getting-help)

---

## Common Errors

### 1. YAML Parsing Errors in .ai_meta Files

**Symptoms:**
```
Warning: Failed to parse analysis.py.ai_meta: ...
YAML parsing error
```

**Causes:**
- Malformed YAML frontmatter
- Unescaped special characters in docstrings
- Missing comment header stripping
- Mixed tabs and spaces

**Solutions:**

```bash
# Check if .ai_meta files exist
ls -la cortical/*.ai_meta

# Regenerate metadata files
python scripts/generate_ai_metadata.py --force

# Test parsing a specific file
python -c "import yaml; yaml.safe_load(open('cortical/analysis.py.ai_meta').read())"
```

**Prevention:**
- Run `generate_ai_metadata.py` after major docstring changes
- Avoid special YAML characters (`:`, `{`, `}`, `[`, `]`) in docstrings without quoting

### 2. Missing VISION.md Sections

**Symptoms:**
```
algorithms_found: 0
No algorithm sections extracted
```

**Causes:**
- VISION.md missing "Deep Algorithm Analysis" section
- Section header format changed
- Regex pattern mismatch

**Solutions:**

```bash
# Verify VISION.md structure
grep -n "## Deep Algorithm Analysis" docs/VISION.md

# Check for algorithm sections
grep -n "### Algorithm" docs/VISION.md

# Regenerate with verbose logging
python scripts/generate_book.py --chapter foundations --verbose
```

**Expected Structure:**
```markdown
## Deep Algorithm Analysis

### Algorithm 1: PageRank — Importance Discovery

**Implementation:** `cortical/analysis.py:compute_pagerank()`
...

### Algorithm 2: BM25/TF-IDF — Distinctiveness Scoring
...
```

### 3. Git History Access Issues

**Symptoms:**
```
Warning: Failed to read git history: ...
No git history found
Could not read git history
```

**Causes:**
- Not in a git repository
- Insufficient permissions
- Git not installed
- Detached HEAD state

**Solutions:**

```bash
# Check git availability
which git
git --version

# Verify repository
git status

# Test git log access
git log -5 --format="%H|%aI|%s|%an"

# Check permissions
ls -la .git/
```

**Workarounds:**
- Skip evolution chapters: `python scripts/generate_book.py --chapter foundations`
- Initialize git if missing: `git init && git add . && git commit -m "Initial commit"`

### 4. Missing Dependencies

**Symptoms:**
```
ModuleNotFoundError: No module named 'yaml'
ImportError: cannot import name 'yaml'
```

**Solution:**

```bash
# Install required dependencies
pip install pyyaml

# Or install all dev dependencies
pip install -e ".[dev]"

# Verify installation
python -c "import yaml; print('PyYAML OK')"
```

### 5. Permission Errors Writing to book/

**Symptoms:**
```
PermissionError: [Errno 13] Permission denied: 'book/01-foundations/...'
OSError: [Errno 30] Read-only file system
```

**Solutions:**

```bash
# Check directory permissions
ls -ld book/
ls -la book/01-foundations/

# Fix permissions
chmod -R u+w book/

# Try dry-run first
python scripts/generate_book.py --dry-run

# Check disk space
df -h .
```

### 6. Empty or Missing ML Data

**Symptoms:**
```
with_ml_data: 0
Warning: Failed to load ML data
```

**Causes:**
- `.git-ml/tracked/commits.jsonl` doesn't exist
- ML collection not started
- File is empty

**Solutions:**

```bash
# Check ML data file
ls -lh .git-ml/tracked/commits.jsonl

# Backfill ML data
python scripts/ml_data_collector.py backfill -n 100

# Verify data
python scripts/ml_data_collector.py stats
```

**Note:** ML data is optional. Chapters will generate without it (just with fewer details).

### 7. Search Index Generation Failures

**Symptoms:**
```
Failed to parse index.md: ...
JSONDecodeError
```

**Causes:**
- Malformed frontmatter in generated chapters
- Invalid JSON structure
- Missing chapter files

**Solutions:**

```bash
# Regenerate chapters first
python scripts/generate_book.py --chapter foundations
python scripts/generate_book.py --chapter architecture

# Then regenerate search index
python scripts/generate_book.py --chapter search

# Validate generated JSON
python -c "import json; json.load(open('book/index.json'))"
python -c "import json; json.load(open('book/search.json'))"
```

---

## Diagnostic Commands

### Health Check

```bash
# Full diagnostic run (no writes)
python scripts/generate_book.py --dry-run --verbose
```

Expected output:
```
Registered generator: foundations
Registered generator: architecture
...
Generating: foundations
  Found 6 algorithms in VISION.md
  Generating: alg-pagerank.md
...
Total files: 25
```

### Check Individual Generators

```bash
# List all generators
python scripts/generate_book.py --list

# Test specific generator
python scripts/generate_book.py --chapter foundations --dry-run --verbose
python scripts/generate_book.py --chapter architecture --dry-run --verbose
python scripts/generate_book.py --chapter evolution --dry-run --verbose
```

### Verify Dependencies

```bash
# Check Python version (3.8+ required)
python --version

# Check required packages
python -c "import yaml; print('PyYAML:', yaml.__version__)"
python -c "import json; print('json: OK')"
python -c "import re; print('re: OK')"

# Check git
git --version
git status
```

### Verify Source Files

```bash
# Check VISION.md
test -f docs/VISION.md && echo "VISION.md exists" || echo "VISION.md MISSING"
grep -c "### Algorithm" docs/VISION.md

# Check .ai_meta files
find cortical -name "*.ai_meta" | wc -l
ls -lh cortical/*.ai_meta | head -5

# Check git history
git log -5 --oneline
```

### Verify Output Structure

```bash
# Check generated files
find book/ -name "*.md" | sort
find book/ -name "*.json"

# Check chapter completeness
for dir in book/0*-*/; do
  echo "$dir: $(ls "$dir" | wc -l) files"
done

# Validate JSON outputs
python -c "import json; json.load(open('book/index.json')); print('index.json: OK')"
python -c "import json; json.load(open('book/search.json')); print('search.json: OK')"
```

---

## Recovery Procedures

### Complete Rebuild

When all else fails, regenerate from scratch:

```bash
# 1. Backup existing book (if needed)
cp -r book/ book.backup/

# 2. Clear generated files (keep docs and assets)
rm -rf book/0*-*/
rm -f book/index.json book/search.json

# 3. Regenerate metadata (if needed)
python scripts/generate_ai_metadata.py --force

# 4. Full regeneration
python scripts/generate_book.py --verbose

# 5. Verify
ls -lR book/
python -c "import json; json.load(open('book/index.json'))"
```

### Regenerate Single Chapter

If one chapter is corrupted:

```bash
# 1. Remove the chapter
rm -rf book/01-foundations/

# 2. Regenerate just that chapter
python scripts/generate_book.py --chapter foundations --verbose

# 3. Rebuild search index
python scripts/generate_book.py --chapter search

# 4. Verify
ls -la book/01-foundations/
```

### Fix Malformed Frontmatter

If chapters have malformed YAML frontmatter:

```bash
# 1. Identify the problem file
python -c "
import yaml
from pathlib import Path
for f in Path('book').glob('**/*.md'):
    try:
        content = f.read_text()
        if content.startswith('---'):
            fm = content.split('---', 2)[1]
            yaml.safe_load(fm)
    except Exception as e:
        print(f'ERROR: {f}: {e}')
"

# 2. Remove the problematic chapter
rm book/XX-section/problematic.md

# 3. Regenerate the parent chapter
python scripts/generate_book.py --chapter <generator-name>
```

### Restore from Git

If the book is tracked in git:

```bash
# Check what changed
git status book/
git diff book/

# Restore specific file
git restore book/01-foundations/alg-pagerank.md

# Restore entire book
git restore book/

# Or reset to last good state
git log --oneline -- book/
git restore --source=<commit-hash> book/
```

### Partial Failure Recovery

If some generators fail but others succeed:

```bash
# 1. Check which generators failed
python scripts/generate_book.py --verbose 2>&1 | grep -A 3 "ERROR:"

# 2. Regenerate only failed chapters
python scripts/generate_book.py --chapter <failed-generator> --verbose

# 3. Rebuild search index
python scripts/generate_book.py --chapter search
```

---

## Error Reference

### Generator-Specific Errors

#### AlgorithmChapterGenerator

| Error | Cause | Fix |
|-------|-------|-----|
| `algorithms_found: 0` | VISION.md missing section | Check docs/VISION.md structure |
| `Source file not found` | VISION.md doesn't exist | Create docs/VISION.md |
| `chapters_written: 0` | Regex pattern mismatch | Update `_extract_algorithms()` |

#### ModuleDocGenerator

| Error | Cause | Fix |
|-------|-------|-----|
| `No .ai_meta files found` | Metadata not generated | Run `generate_ai_metadata.py` |
| `Failed to parse <file>.ai_meta` | Malformed YAML | Regenerate metadata with `--force` |
| `modules_documented: 0` | All parsing failed | Check YAML structure |

#### CommitNarrativeGenerator

| Error | Cause | Fix |
|-------|-------|-----|
| `No git history found` | Not in git repo | Initialize git or skip chapter |
| `Failed to read git history` | Git not available | Install git |
| `with_ml_data: 0` | ML data missing | Run backfill (optional) |

#### SearchIndexGenerator

| Error | Cause | Fix |
|-------|-------|-----|
| `chapters_indexed: 0` | No chapter files | Generate chapters first |
| `Failed to parse <file>` | Malformed frontmatter | Regenerate source chapter |
| `JSONDecodeError` | Invalid JSON structure | Check chapter YAML |

### System-Level Errors

| Error | Typical Cause | Solution |
|-------|--------------|----------|
| `PermissionError` | Read-only filesystem | Check permissions with `ls -ld book/` |
| `FileNotFoundError` | Missing source file | Verify file exists with `ls -la` |
| `ModuleNotFoundError: yaml` | Missing dependency | Install with `pip install pyyaml` |
| `JSONDecodeError` | Corrupted output | Delete and regenerate file |
| `UnicodeDecodeError` | Binary file read as text | Check file encoding |
| `subprocess.CalledProcessError` | Git command failed | Verify git with `git status` |

---

## Prevention Tips

### Before Generating

1. **Verify prerequisites:**
   ```bash
   python --version  # 3.8+
   git --version
   python -c "import yaml; print('OK')"
   ```

2. **Check source files:**
   ```bash
   test -f docs/VISION.md || echo "WARNING: VISION.md missing"
   ls cortical/*.ai_meta | wc -l  # Should be >10
   ```

3. **Test with dry-run:**
   ```bash
   python scripts/generate_book.py --dry-run
   ```

### After Making Changes

1. **Regenerate affected chapters:**
   - Changed VISION.md → `--chapter foundations`
   - Changed docstrings → regenerate metadata, then `--chapter architecture`
   - New commits → `--chapter evolution`

2. **Always rebuild search index:**
   ```bash
   python scripts/generate_book.py --chapter search
   ```

3. **Validate outputs:**
   ```bash
   python -c "import json; json.load(open('book/index.json'))"
   ```

---

## Getting Help

### Debug Checklist

- [ ] Run with `--dry-run --verbose`
- [ ] Check error message in this guide
- [ ] Verify dependencies installed
- [ ] Test with single chapter generation
- [ ] Check source file structure
- [ ] Review recent git history
- [ ] Try complete rebuild

### Logging

Generate detailed logs for debugging:

```bash
# Full verbose output to file
python scripts/generate_book.py --verbose 2>&1 | tee generation.log

# Check for errors
grep -i "error\|warning\|failed" generation.log

# Check generator stats
grep "stats" generation.log
```

### Still Stuck?

1. **Check recent changes:**
   ```bash
   git log --oneline -10
   git diff HEAD~5 -- docs/ cortical/
   ```

2. **Isolate the problem:**
   - Test each generator individually
   - Compare with known-good state
   - Check file permissions

3. **Report issue with:**
   - Full error message
   - Output of `--dry-run --verbose`
   - Output of diagnostic commands
   - Recent changes to source files

---

## Appendix: File Locations

### Source Files

| File | Purpose | Generator |
|------|---------|-----------|
| `docs/VISION.md` | Algorithm descriptions | foundations |
| `cortical/*.ai_meta` | Module metadata | architecture |
| `.git/logs/` | Git history | evolution |
| `.git-ml/tracked/commits.jsonl` | ML commit data | evolution |
| `samples/decisions/adr-*.md` | ADRs | decisions |

### Output Files

| File | Generator | Can Delete? |
|------|-----------|-------------|
| `book/*/index.md` | Various | Yes (regenerates) |
| `book/01-foundations/*.md` | foundations | Yes |
| `book/02-architecture/*.md` | architecture | Yes |
| `book/04-evolution/*.md` | evolution | Yes |
| `book/index.json` | search | Yes |
| `book/search.json` | search | Yes |
| `book/README.md` | Manual | **No** (manual) |
| `book/docs/` | Manual | **No** (manual) |
| `book/assets/` | Manual | **No** (manual) |

---

*This troubleshooting guide is part of [The Cortical Chronicles](../README.md) documentation.*
