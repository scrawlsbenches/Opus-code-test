# Architectural Roadmap

This document provides a visual roadmap of the Cortical Text Processor architecture, showing current state, planned improvements, and future capabilities.

**Related:** See [architecture.md](architecture.md) for detailed module documentation.

---

## Current Architecture (v2.0)

The system is organized into five architectural layers, with processor.py as the central orchestrator.

```mermaid
graph TB
    subgraph External["External Interfaces"]
        CLI["CLI<br/>scripts/*.py"]
        MCP["MCP Server<br/>mcp_server.py"]
        SDK["Python SDK<br/>CorticalTextProcessor"]
        Fluent["Fluent API<br/>FluentProcessor"]
    end

    subgraph Orchestration["Orchestration Layer"]
        Processor["processor.py<br/>2,301 lines<br/>60+ methods"]
    end

    subgraph Algorithm["Algorithm Layer"]
        Analysis["analysis.py<br/>PageRank, TF-IDF<br/>Louvain clustering"]
        Semantics["semantics.py<br/>Relation extraction<br/>Retrofitting"]
        Embeddings["embeddings.py<br/>Graph embeddings"]
        Gaps["gaps.py<br/>Gap detection"]
        Fingerprint["fingerprint.py<br/>Code similarity"]
    end

    subgraph Query["Query Layer (8 modules)"]
        Expansion["expansion.py"]
        Search["search.py"]
        Passages["passages.py"]
        Ranking["ranking.py"]
        Intent["intent.py"]
        Analogy["analogy.py"]
        Defs["definitions.py"]
        Chunking["chunking.py"]
    end

    subgraph Persistence["Persistence Layer"]
        Persist["persistence.py<br/>Pickle format"]
        ChunkIdx["chunk_index.py<br/>Git-friendly JSON"]
    end

    subgraph Foundation["Foundation Layer"]
        Minicolumn["minicolumn.py<br/>Minicolumn, Edge"]
        Layers["layers.py<br/>HierarchicalLayer"]
        Config["config.py<br/>CorticalConfig"]
        Tokenizer["tokenizer.py"]
        CodeConcepts["code_concepts.py"]
        Validation["validation.py"]
    end

    CLI --> Processor
    MCP --> Processor
    SDK --> Processor
    Fluent --> Processor

    Processor --> Analysis
    Processor --> Semantics
    Processor --> Embeddings
    Processor --> Gaps
    Processor --> Fingerprint
    Processor --> Expansion
    Processor --> Search
    Processor --> Passages
    Processor --> Persist
    Processor --> ChunkIdx

    Analysis --> Layers
    Semantics --> Layers
    Query --> Layers
    Persist --> Layers
    Layers --> Minicolumn

    classDef external fill:#e8f4f8,stroke:#0277bd
    classDef orchestrate fill:#ffebee,stroke:#c62828
    classDef algorithm fill:#fff3e0,stroke:#ef6c00
    classDef query fill:#f3e5f5,stroke:#7b1fa2
    classDef persist fill:#e8f5e9,stroke:#2e7d32
    classDef foundation fill:#e3f2fd,stroke:#1565c0

    class CLI,MCP,SDK,Fluent external
    class Processor orchestrate
    class Analysis,Semantics,Embeddings,Gaps,Fingerprint algorithm
    class Expansion,Search,Passages,Ranking,Intent,Analogy,Defs,Chunking query
    class Persist,ChunkIdx persist
    class Minicolumn,Layers,Config,Tokenizer,CodeConcepts,Validation foundation
```

---

## Data Layer Hierarchy

The 4-layer neocortex-inspired data model:

```mermaid
graph BT
    subgraph L0["Layer 0: TOKENS"]
        T1["neural"]
        T2["networks"]
        T3["learning"]
        T4["data"]
    end

    subgraph L1["Layer 1: BIGRAMS"]
        B1["neural networks"]
        B2["networks learning"]
        B3["learning data"]
    end

    subgraph L2["Layer 2: CONCEPTS"]
        C1["neural/networks/learning"]
        C2["data/processing/analysis"]
    end

    subgraph L3["Layer 3: DOCUMENTS"]
        D1["doc_ml_intro"]
        D2["doc_neural_nets"]
    end

    T1 <-.->|lateral| T2
    T2 <-.->|lateral| T3
    T3 <-.->|lateral| T4

    B1 -->|feedforward| T1
    B1 -->|feedforward| T2
    B2 -->|feedforward| T2
    B2 -->|feedforward| T3

    C1 -->|feedforward| T1
    C1 -->|feedforward| T2
    C1 -->|feedforward| T3

    D1 -->|feedforward| T1
    D1 -->|feedforward| T2
    D1 -->|feedforward| T3

    T1 -.->|feedback| B1
    T1 -.->|feedback| C1
    T1 -.->|feedback| D1

    style L0 fill:#e3f2fd
    style L1 fill:#e8f5e9
    style L2 fill:#fff3e0
    style L3 fill:#fce4ec
```

---

## Planned Improvements

### Phase 1: Modularity (Task #95)

Split the 2,301-line `processor.py` into focused modules:

```mermaid
graph LR
    subgraph Current["Current: processor.py (2,301 lines)"]
        All["CorticalTextProcessor<br/>60+ methods<br/>All responsibilities"]
    end

    subgraph Planned["Planned: processor/ package"]
        Core["core.py<br/>Main class<br/>Public API"]
        Docs["documents.py<br/>process_document()<br/>add_incremental()"]
        Compute["computation.py<br/>compute_all()<br/>Staleness tracking"]
        QueryW["query.py<br/>Search wrappers<br/>Facade methods"]
        Export["export.py<br/>Visualization<br/>JSON export"]
    end

    All -->|refactor| Core
    All -->|refactor| Docs
    All -->|refactor| Compute
    All -->|refactor| QueryW
    All -->|refactor| Export

    Core --> Docs
    Core --> Compute
    Core --> QueryW
    Core --> Export

    style Current fill:#ffcdd2
    style Planned fill:#c8e6c9
```

### Phase 2: Persistence Evolution (Tasks #133, #134)

Improve persistence with WAL and cross-language support:

```mermaid
graph TB
    subgraph Current["Current State"]
        Pickle["pickle.dump()<br/>Full state save<br/>Python-only"]
        Chunks["JSON chunks<br/>Git-friendly<br/>Append-only"]
    end

    subgraph WAL["Task #133: WAL + Snapshots"]
        WALFile["Write-Ahead Log<br/>Incremental operations"]
        Snapshot["Periodic snapshots<br/>Checkpoint recovery"]
        Replay["Crash recovery<br/>Replay from WAL"]
    end

    subgraph Proto["Task #134: Protobuf"]
        Schema["schema.proto<br/>Minicolumn, Layer, Edge"]
        ToProto["to_proto() methods"]
        FromProto["from_proto() methods"]
        CrossLang["Cross-language<br/>Python, Go, Rust"]
    end

    Pickle --> WAL
    Chunks --> WAL
    WAL --> Proto

    style Current fill:#fff3e0
    style WAL fill:#e3f2fd
    style Proto fill:#e8f5e9
```

### Phase 3: Performance (Task #135)

Parallelize computation for large corpora:

```mermaid
graph TB
    subgraph Serial["Current: Serial Processing"]
        S1["Document 1"] --> S2["Document 2"]
        S2 --> S3["Document 3"]
        S3 --> S4["Document N"]
        S4 --> SAll["Compute All"]
    end

    subgraph Parallel["Planned: Parallel Processing"]
        subgraph Workers["Worker Pool"]
            W1["Worker 1<br/>Docs 1-250"]
            W2["Worker 2<br/>Docs 251-500"]
            W3["Worker 3<br/>Docs 501-750"]
            W4["Worker 4<br/>Docs 751-1000"]
        end
        Merge["Merge Results"]
        PAll["Final compute_all()"]

        W1 --> Merge
        W2 --> Merge
        W3 --> Merge
        W4 --> Merge
        Merge --> PAll
    end

    Serial -->|"Task #135"| Parallel

    style Serial fill:#ffcdd2
    style Parallel fill:#c8e6c9
```

---

## Future Features Roadmap

### Async/Advanced Features (Tasks #187-191)

```mermaid
graph TB
    subgraph Current["Current: Synchronous API"]
        Sync["CorticalTextProcessor<br/>Blocking calls<br/>Single-threaded"]
    end

    subgraph Async["Task #187: Async API"]
        AsyncProc["AsyncCorticalTextProcessor<br/>async/await support<br/>Non-blocking I/O"]
    end

    subgraph Stream["Task #188: Streaming"]
        StreamResults["Streaming query results<br/>Yield as found<br/>Memory efficient"]
    end

    subgraph Observe["Task #189: Observability"]
        Timing["Operation timing"]
        Traces["Distributed traces"]
        Metrics["OpenTelemetry metrics"]
    end

    subgraph REST["Task #190: REST API"]
        FastAPI["FastAPI wrapper<br/>HTTP endpoints<br/>OpenAPI schema"]
    end

    subgraph REPL["Task #191: Interactive"]
        Interactive["REPL mode<br/>python -m cortical --interactive<br/>Exploration tools"]
    end

    Current --> Async
    Async --> Stream
    Async --> REST
    Current --> Observe
    Current --> REPL

    REST --> Stream

    style Current fill:#e0e0e0
    style Async fill:#bbdefb
    style Stream fill:#c8e6c9
    style Observe fill:#fff9c4
    style REST fill:#ffccbc
    style REPL fill:#e1bee7
```

---

## Integration Architecture

### Current Integration Points

```mermaid
graph TB
    subgraph Users["Users & Agents"]
        Human["Human Developer"]
        Claude["Claude Desktop"]
        Scripts["Automation Scripts"]
    end

    subgraph Integration["Integration Layer"]
        CLI["CLI Scripts<br/>search_codebase.py<br/>index_codebase.py"]
        MCPServer["MCP Server<br/>5 tools"]
        PythonAPI["Python API<br/>CorticalTextProcessor"]
        FluentAPI["Fluent API<br/>FluentProcessor"]
    end

    subgraph Skills["Claude Skills"]
        Search["codebase-search"]
        Indexer["corpus-indexer"]
        Metadata["ai-metadata"]
    end

    subgraph Hooks["Claude Code Hooks"]
        SessionStart["session_start.sh<br/>Auto-index on start"]
        PreCommit["pre-commit (planned)<br/>Reindex on changes"]
    end

    Human --> CLI
    Human --> PythonAPI
    Claude --> MCPServer
    Claude --> Skills
    Scripts --> PythonAPI

    Skills --> CLI
    Hooks --> CLI

    MCPServer --> PythonAPI
    CLI --> PythonAPI
    FluentAPI --> PythonAPI

    style Users fill:#e3f2fd
    style Integration fill:#fff3e0
    style Skills fill:#e8f5e9
    style Hooks fill:#fce4ec
```

### Future Integration Vision

```mermaid
graph TB
    subgraph Clients["Client Applications"]
        IDE["IDE Plugins<br/>VSCode, JetBrains"]
        Web["Web Dashboard<br/>Corpus explorer"]
        Mobile["Mobile Apps<br/>Documentation search"]
        Agents["AI Agents<br/>Claude, GPT, etc."]
    end

    subgraph Gateway["API Gateway"]
        REST["REST API<br/>(Task #190)"]
        GraphQL["GraphQL<br/>(Future)"]
        gRPC["gRPC<br/>(Future)"]
        MCP["MCP Protocol"]
    end

    subgraph Core["Core Engine"]
        Async["AsyncCorticalTextProcessor<br/>(Task #187)"]
        Stream["Streaming Results<br/>(Task #188)"]
        Observe["Observability<br/>(Task #189)"]
    end

    subgraph Storage["Storage Backends"]
        Pickle["Pickle (default)"]
        Proto["Protobuf (Task #134)"]
        SQLite["SQLite (future)"]
        Remote["Remote storage (future)"]
    end

    IDE --> REST
    Web --> REST
    Mobile --> REST
    Agents --> MCP
    Agents --> REST

    REST --> Async
    GraphQL --> Async
    gRPC --> Async
    MCP --> Async

    Async --> Stream
    Async --> Observe
    Async --> Pickle
    Async --> Proto
    Async --> SQLite
    Async --> Remote

    style Clients fill:#e3f2fd
    style Gateway fill:#fff3e0
    style Core fill:#e8f5e9
    style Storage fill:#f3e5f5
```

---

## Development Phases

```mermaid
gantt
    title Development Roadmap (Timeline-Agnostic Phases)
    dateFormat X
    axisFormat %s

    section Foundation
    Current v2.0 Complete        :done, 0, 1

    section Phase 1 - Modularity
    Split processor.py (#95)     :active, 1, 2
    Input validation (#99)       :1, 2

    section Phase 2 - Persistence
    WAL + Snapshots (#133)       :2, 3
    Protobuf serialization (#134):2, 3

    section Phase 3 - Performance
    Parallel processing (#135)   :3, 4

    section Phase 4 - Async
    Async API (#187)             :4, 5
    Streaming results (#188)     :5, 6
    Observability (#189)         :4, 5

    section Phase 5 - Integration
    REST API wrapper (#190)      :5, 6
    Interactive REPL (#191)      :5, 6
```

---

## Module Size Analysis

Current module sizes and planned evolution:

```mermaid
pie showData
    title Current Module Sizes (Lines of Code)
    "processor.py" : 2301
    "analysis.py" : 1123
    "semantics.py" : 915
    "persistence.py" : 606
    "chunk_index.py" : 574
    "tokenizer.py" : 398
    "minicolumn.py" : 357
    "config.py" : 352
    "fingerprint.py" : 315
    "layers.py" : 294
    "query/* (8 files)" : 1600
    "Other modules" : 800
```

After Task #95 (processor split):

```mermaid
pie showData
    title Projected Module Sizes After Refactoring
    "processor/core.py" : 500
    "processor/documents.py" : 400
    "processor/computation.py" : 600
    "processor/query.py" : 400
    "processor/export.py" : 400
    "analysis.py" : 1123
    "semantics.py" : 915
    "persistence.py" : 606
    "query/* (8 files)" : 1600
    "Other modules" : 2300
```

---

## Quick Reference: Task Dependencies

```mermaid
graph LR
    subgraph Core["Core Tasks"]
        T95["#95 Split processor.py"]
        T99["#99 Input validation"]
    end

    subgraph Persist["Persistence"]
        T133["#133 WAL + Snapshots"]
        T134["#134 Protobuf"]
    end

    subgraph Perf["Performance"]
        T135["#135 Parallel processing"]
    end

    subgraph Async["Async/Advanced"]
        T187["#187 Async API"]
        T188["#188 Streaming"]
        T189["#189 Observability"]
        T190["#190 REST API"]
        T191["#191 Interactive REPL"]
    end

    T95 --> T135
    T133 --> T134
    T187 --> T188
    T187 --> T190

    style Core fill:#e3f2fd
    style Persist fill:#e8f5e9
    style Perf fill:#fff3e0
    style Async fill:#f3e5f5
```

---

## Summary

| Phase | Focus | Key Tasks | Impact |
|-------|-------|-----------|--------|
| **Current** | v2.0 Complete | MCP Server, Skills, Hooks | AI agent integration |
| **Phase 1** | Modularity | #95, #99 | Maintainability |
| **Phase 2** | Persistence | #133, #134 | Reliability, cross-language |
| **Phase 3** | Performance | #135 | Scale to 10K+ docs |
| **Phase 4** | Async | #187, #188, #189 | Framework integration |
| **Phase 5** | Integration | #190, #191 | Broader adoption |

---

*Last updated: 2025-12-13*
