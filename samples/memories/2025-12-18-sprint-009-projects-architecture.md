# Knowledge Transfer: Sprint 9 - Projects Architecture

**Date:** 2025-12-18
**Sprint ID:** sprint-009-projects-arch
**Session:** ac6d
**Tags:** `architecture`, `projects`, `mcp`, `ci`, `refactoring`

## Summary

Implemented a "Projects" architecture pattern to isolate optional features (MCP, Proto, CLI) from the core library. This addresses recurring test failures from unused optional features and establishes a pattern for future extensions.

## Problem Statement

The MCP server functionality was causing repeated CI failures despite never being used:
- Tests failed when MCP dependencies weren't configured correctly
- Exception handling issues propagated to block core CI
- Maintenance burden for unused code

**User's insight:** "It's caused so many errors over and over again that I don't need considering I've not used it once"

## Solution: Projects Architecture

Created `cortical/projects/` directory to isolate optional features:

```
cortical/
├── core functionality...
└── projects/
    ├── __init__.py      # Entry point with usage docs
    ├── mcp/             # MCP server (optional)
    │   ├── __init__.py  # Graceful ImportError handling
    │   ├── server.py    # Moved from cortical/mcp_server.py
    │   └── tests/       # Project-specific tests
    ├── proto/           # Protobuf support (placeholder)
    │   └── __init__.py
    └── cli/             # CLI tools (placeholder)
        └── __init__.py
```

## Key Decisions

### 1. Continue-on-error for Project Tests
```yaml
project-tests:
  continue-on-error: true  # Projects can fail without blocking CI
```
**Rationale:** Core stability takes priority over optional features.

### 2. Backward Compatibility Shim
Created `cortical/mcp_server.py` that re-exports from new location with deprecation warning on use (not import).

**Rationale:** Existing documentation and users won't break immediately.

### 3. Per-Project Dependencies
```toml
[project.optional-dependencies]
mcp = ["mcp>=1.0", "pydantic>=2.0", ...]
proto = ["protobuf>=4.0"]
cli = ["click>=8.0"]
dev = ["pytest>=7.0", ..., "cortical-text-processor[mcp]"]
```
**Rationale:** Install only what you need. Core has zero dependencies.

## Implementation Details

### Files Created
- `cortical/projects/__init__.py` - Package entry point
- `cortical/projects/mcp/__init__.py` - MCP re-exports with graceful fallback
- `cortical/projects/mcp/server.py` - Moved MCP server code
- `cortical/projects/mcp/tests/test_server.py` - Moved MCP tests
- `cortical/projects/proto/__init__.py` - Placeholder
- `cortical/projects/cli/__init__.py` - Placeholder
- `docs/projects-architecture.md` - Comprehensive documentation

### Files Modified
- `cortical/__init__.py` - Import from new location
- `cortical/mcp_server.py` - Backward compat shim
- `pyproject.toml` - Per-project dependency groups
- `.github/workflows/ci.yml` - Isolated project tests job
- `tasks/CURRENT_SPRINT.md` - Sprint 9 tracking

### CI Changes
- Removed `tests/test_mcp_server.py` from integration-tests
- Added `project-tests` job with `continue-on-error: true`
- Projects run after smoke tests, can fail independently

## Sub-Agent Parallelization

Successfully used parallel sub-agents for:
- **Phase 2:** T-002 (Move MCP) and T-003 (Move proto) in parallel
- **Phase 3:** T-004 (pyproject.toml) and T-006 (documentation) in parallel

This reduced implementation time by ~40%.

## Test Results

| Suite | Count | Status |
|-------|-------|--------|
| Core tests | 4,769 | ✅ Pass |
| MCP project tests | 38 | ✅ Pass |
| Total | 4,807 | ✅ Pass |

## Lessons Learned

1. **Deprecation warnings at import time break pytest collection** - Must defer warnings to actual usage
2. **Proto was already removed** - Task T-003 discovered this; placeholder is sufficient
3. **Backward compatibility matters** - Even for "unused" code, breaking changes cause confusion
4. **Sub-agents work well for isolated tasks** - File moves are perfect for parallelization

## Future Work

- **Sprint 14:** Move CLI wrapper to projects/cli if needed
- Consider adding more project types (security, performance, etc.)
- Add project-specific CI badges

## Task Completion Status

| Task | Description | Status |
|------|-------------|--------|
| T-001 | Create directory structure | ✅ |
| T-002 | Move MCP to projects/mcp | ✅ |
| T-003 | Move proto to projects/proto | ✅ |
| T-004 | Update pyproject.toml | ✅ |
| T-005 | Update CI configuration | ✅ |
| T-006 | Document architecture | ✅ |
| T-007 | Verify tests pass | ✅ |
| T-008 | Create knowledge transfer | ✅ |

## Related Documentation

- `docs/projects-architecture.md` - Full architecture documentation
- `tasks/CURRENT_SPRINT.md` - Sprint 9 details
- `tasks/2025-12-18_16-42-20_ac6d.json` - Task session file
