# Patterns

> "In this codebase, we do X this way"

Add your patterns here. These teach the AI your conventions.

## Code Patterns

- **error handling**: We prefer explicit error types over exceptions where possible
- **testing**: Tests live in `tests/` mirroring `cortical/` structure
- **dependencies**: Zero runtime dependencies - build everything ourselves
- **naming**: snake_case for functions, PascalCase for classes

## Process Patterns

- **commits**: Descriptive messages following conventional commits (feat:, fix:, docs:)
- **tasks**: Merge-friendly task system in `tasks/` directory
- **memories**: Daily learnings captured in `samples/memories/`
- **decisions**: ADRs in `samples/decisions/` for architectural choices

## Documentation Patterns

- **CLAUDE.md**: Central guide for AI agents working on this repo
- **docstrings**: Google style with Args/Returns sections
- **type hints**: On all public functions

## Your Patterns

<!-- Add your own patterns below -->

