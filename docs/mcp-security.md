# MCP Server Security Model

This document describes the security considerations for the Cortical Text Processor MCP (Model Context Protocol) server.

## Overview

The MCP server (`cortical/mcp_server.py`) provides an interface for AI agents to interact with the Cortical Text Processor. It exposes five tools for semantic search, passage retrieval, query expansion, corpus statistics, and document indexing.

## Exposed Capabilities

| Tool | Operation | Risk Level | Notes |
|------|-----------|------------|-------|
| `search` | Read-only query | Low | Returns document IDs and scores |
| `passages` | Read-only query | Low | Returns text passages from corpus |
| `expand_query` | Read-only query | Low | Returns expansion terms |
| `corpus_stats` | Read-only stats | Low | Returns aggregate statistics |
| `add_document` | Write operation | Medium | Adds documents to corpus |

### Tool Details

#### `search(query, top_n)`
- **Input validation**: Query must be non-empty string; `top_n >= 1`
- **Output**: Document IDs and relevance scores
- **Risk**: Information disclosure if corpus contains sensitive data

#### `passages(query, top_n, chunk_size)`
- **Input validation**: Query must be non-empty string; `top_n >= 1`
- **Output**: Actual text content from documents
- **Risk**: Higher information disclosure risk than `search`

#### `expand_query(query, max_expansions)`
- **Input validation**: Query must be non-empty string; `max_expansions >= 1`
- **Output**: Related terms and weights
- **Risk**: Low - only exposes term relationships

#### `corpus_stats()`
- **Input validation**: None (no parameters)
- **Output**: Document counts, layer statistics
- **Risk**: Metadata disclosure only

#### `add_document(doc_id, content, recompute)`
- **Input validation**: `doc_id` non-empty string; `content` must be string; `recompute` in {'none', 'tfidf', 'full'}
- **Output**: Processing statistics
- **Risk**:
  - Corpus pollution (malicious content injection)
  - Resource exhaustion (large documents, frequent additions)
  - Overwriting existing documents (no duplicate protection)

## Trust Model

### Who Can Call the Server

The MCP server uses stdio transport by default, meaning:

1. **Local Execution**: The server runs as a local process
2. **No Network Exposure**: stdio transport doesn't listen on network ports
3. **Caller is Trusted**: The calling process (typically an AI assistant) has full access

### Trust Assumptions

| Assumption | Implication |
|------------|-------------|
| Caller is authorized | No authentication mechanism |
| Corpus content is not sensitive | Results returned without filtering |
| Input is well-formed | Basic validation only |
| Single-tenant usage | No multi-user isolation |

### Security Boundary

```
┌─────────────────────────────────────────────────────────┐
│                    AI Assistant Process                  │
│  ┌─────────────────┐      ┌─────────────────────────┐   │
│  │   MCP Client    │──────│   Cortical MCP Server   │   │
│  └─────────────────┘ stdio└─────────────────────────┘   │
│                                      │                   │
│                            ┌─────────▼─────────┐        │
│                            │ CorticalTextProc  │        │
│                            │    (in-memory)    │        │
│                            └───────────────────┘        │
└─────────────────────────────────────────────────────────┘
```

The security boundary is at the process level. Anyone who can communicate with the stdio interface has full access.

## Input Validation

### Current Validation

All tools perform basic input validation:

```python
# String validation
if not query or not query.strip():
    return {"error": "Query must be a non-empty string", ...}

# Numeric validation
if top_n < 1:
    return {"error": "top_n must be at least 1", ...}

# Enum validation
valid_recompute = {'none', 'tfidf', 'full'}
if recompute not in valid_recompute:
    return {"error": f"recompute must be one of {valid_recompute}", ...}
```

### Validation Gaps

| Gap | Risk | Mitigation |
|-----|------|------------|
| No max length for `content` | Memory exhaustion | Add content size limit |
| No max length for `query` | Performance impact | Add query length limit |
| No rate limiting | Resource exhaustion | Implement rate limiting |
| No `doc_id` format validation | Potential path issues | Validate `doc_id` characters |

## Resource Considerations

### Memory Usage

The processor stores all documents in memory:

- **Per document**: ~10-100KB depending on content
- **Index structures**: ~5x document size
- **No memory limits**: Can exhaust system memory

### CPU Usage

Resource-intensive operations:

| Operation | CPU Impact | Notes |
|-----------|------------|-------|
| `add_document` with `recompute='full'` | High | Triggers full recomputation |
| `search` with large corpus | Medium | Iterates all documents |
| `passages` | Medium | Text chunking and scoring |

## Rate Limiting Considerations

The MCP server does not implement rate limiting. For production deployments, consider:

### Recommended Limits

| Metric | Suggested Limit | Rationale |
|--------|-----------------|-----------|
| Requests per minute | 60 | Prevent runaway queries |
| Document additions per hour | 100 | Limit corpus growth |
| Max document size | 1 MB | Prevent memory exhaustion |
| Max query length | 1000 chars | Prevent complex queries |
| Max concurrent requests | 5 | Limit resource contention |

### Implementation Options

1. **Wrapper Rate Limiter**: Add rate limiting wrapper around tools
2. **External Proxy**: Use API gateway with rate limiting
3. **Token Bucket**: Implement in-memory token bucket

## Recommended Deployment Configurations

### Development / Local Use

```bash
# Default configuration - suitable for local development
python -m cortical.mcp_server
```

- Trust model: Single user, full trust
- Network: stdio only (no network exposure)
- Corpus: Ephemeral or local file

### Production Considerations

If deploying the MCP server in a production context:

1. **Do not expose to untrusted networks**
   - MCP over stdio is designed for local use
   - No authentication or authorization

2. **Corpus sensitivity**
   - Assume all corpus content can be returned to callers
   - Do not index sensitive/confidential documents unless callers are authorized

3. **Resource isolation**
   - Run in container with memory limits
   - Set CPU quotas
   - Monitor resource usage

4. **Logging and auditing**
   - Enable `CORTICAL_LOG_LEVEL=DEBUG` for request logging
   - Log all `add_document` calls for audit trail

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `CORTICAL_CORPUS_PATH` | Path to saved corpus | None (empty corpus) |
| `CORTICAL_LOG_LEVEL` | Logging verbosity | INFO |

## Security Checklist

Before deploying the MCP server:

- [ ] Verify corpus content is appropriate for callers
- [ ] Set memory limits on container/process
- [ ] Enable logging for audit trail
- [ ] Consider document size limits for `add_document`
- [ ] Review if `add_document` capability is needed
- [ ] Test with representative workloads

## Related Documentation

- [Security Knowledge Transfer](security-knowledge-transfer.md) - Full security review
- [README Security Section](../README.md#security-considerations) - Pickle warnings
- [Architecture](architecture.md) - System design
