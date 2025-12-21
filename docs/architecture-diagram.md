# Cortical Text Processor - System Architecture

## High-Level Class Diagram

```mermaid
classDiagram
    direction TB

    %% ============================================
    %% CORE TEXT PROCESSING
    %% ============================================

    class CorticalTextProcessor {
        <<Main API>>
        +layers: Dict~CorticalLayer, HierarchicalLayer~
        +config: CorticalConfig
        +process_document(doc_id, text)
        +compute_all()
        +find_documents_for_query(query)
        +find_passages_for_query(query)
        +graph_boosted_search(query)
        +expand_query(query)
        +save(path)
        +load(path)$ CorticalTextProcessor
    }

    class HierarchicalLayer {
        <<Container>>
        +minicolumns: Dict~str, Minicolumn~
        +_id_index: Dict~str, str~
        +get_minicolumn(content)
        +get_by_id(col_id) Minicolumn
        +column_count() int
    }

    class Minicolumn {
        <<Core Data Unit>>
        +id: str
        +content: str
        +layer: int
        +activation: float
        +pagerank: float
        +tfidf: float
        +document_ids: Set~str~
        +lateral_connections: Dict
        +typed_connections: List~Edge~
    }

    class Edge {
        <<Connection>>
        +target: str
        +relation_type: str
        +weight: float
        +confidence: float
        +source: str
    }

    class CorticalConfig {
        <<Configuration>>
        +pagerank_damping: float
        +scoring_algorithm: str
        +bm25_k1: float
        +bm25_b: float
        +louvain_resolution: float
    }

    %% ============================================
    %% LAYER HIERARCHY
    %% ============================================

    class CorticalLayer {
        <<Enum>>
        TOKENS = 0
        BIGRAMS = 1
        CONCEPTS = 2
        DOCUMENTS = 3
    }

    %% ============================================
    %% REASONING FRAMEWORK (GoT)
    %% ============================================

    class ThoughtGraph {
        <<Graph Structure>>
        +nodes: Dict~str, ThoughtNode~
        +edges: List~ThoughtEdge~
        +clusters: Dict
        +add_node(id, type, content)
        +add_edge(from_id, to_id, type)
        +bfs(start) List
        +dfs(start) List
    }

    class ThoughtNode {
        <<Graph Node>>
        +id: str
        +node_type: NodeType
        +content: str
        +properties: Dict
        +metadata: Dict
    }

    class ThoughtEdge {
        <<Graph Edge>>
        +source_id: str
        +target_id: str
        +edge_type: EdgeType
        +weight: float
        +confidence: float
    }

    class NodeType {
        <<Enum>>
        TASK
        DECISION
        QUESTION
        HYPOTHESIS
        CONTEXT
    }

    class EdgeType {
        <<Enum>>
        DEPENDS_ON
        BLOCKS
        SIMILAR
        IMPLEMENTS
        MOTIVATES
        +23 more...
    }

    %% ============================================
    %% REASONING WORKFLOW
    %% ============================================

    class CognitiveLoop {
        <<QAPV Cycle>>
        +id: str
        +goal: str
        +status: LoopStatus
        +current_phase: LoopPhase
        +transitions: List
        -QUESTION phase
        -ANSWER phase
        -PRODUCE phase
        -VERIFY phase
    }

    class ReasoningWorkflow {
        <<Orchestrator>>
        +loop: CognitiveLoop
        +graph: ThoughtGraph
        +begin_question_phase()
        +begin_answer_phase()
        +begin_production_phase()
        +begin_verify_phase()
    }

    %% ============================================
    %% PERSISTENCE
    %% ============================================

    class GraphWAL {
        <<Write-Ahead Log>>
        +log_add_node()
        +log_add_edge()
        +create_snapshot()
        +load_snapshot()
    }

    class GoTProjectManager {
        <<Task Management>>
        +graph: ThoughtGraph
        +wal: GraphWAL
        +event_log: EventLog
        +create_task(title, priority)
        +complete_task(task_id)
        +sync_to_git()
    }

    %% ============================================
    %% RELATIONSHIPS
    %% ============================================

    CorticalTextProcessor --> HierarchicalLayer : contains 4
    CorticalTextProcessor --> CorticalConfig : configured by
    HierarchicalLayer --> Minicolumn : contains many
    Minicolumn --> Edge : has connections
    HierarchicalLayer --> CorticalLayer : identified by

    ThoughtGraph --> ThoughtNode : contains
    ThoughtGraph --> ThoughtEdge : contains
    ThoughtNode --> NodeType : typed by
    ThoughtEdge --> EdgeType : typed by

    ReasoningWorkflow --> CognitiveLoop : manages
    ReasoningWorkflow --> ThoughtGraph : uses

    GoTProjectManager --> ThoughtGraph : manages
    GoTProjectManager --> GraphWAL : persists via
```

## Data Flow

```mermaid
flowchart LR
    subgraph Input
        TEXT[Raw Text]
        QUERY[User Query]
    end

    subgraph Processing
        TOK[Tokenizer]
        L0[Layer 0: Tokens]
        L1[Layer 1: Bigrams]
        L2[Layer 2: Concepts]
        L3[Layer 3: Documents]
    end

    subgraph Algorithms
        PR[PageRank]
        TFIDF[TF-IDF/BM25]
        LOU[Louvain Clustering]
    end

    subgraph Output
        DOCS[Ranked Documents]
        PASS[Passages]
        EXP[Query Expansion]
    end

    TEXT --> TOK --> L0
    L0 --> L1 --> L2 --> L3

    L0 & L1 --> PR
    L0 & L1 --> TFIDF
    L0 & L1 --> LOU --> L2

    QUERY --> EXP
    EXP --> TFIDF
    TFIDF --> DOCS
    DOCS --> PASS
```

## Module Dependencies

```
cortical/
├── processor/          ←── Main API (uses all below)
│   ├── core.py         ←── Initialization, staleness
│   ├── documents.py    ←── Document processing
│   ├── compute.py      ←── PageRank, TF-IDF, clustering
│   ├── query_api.py    ←── Search methods
│   └── persistence_api.py
│
├── layers.py           ←── HierarchicalLayer container
├── minicolumn.py       ←── Core data structure
├── analysis.py         ←── Graph algorithms
├── semantics.py        ←── Relation extraction
├── tokenizer.py        ←── Text tokenization
│
└── reasoning/          ←── GoT Framework
    ├── thought_graph.py
    ├── cognitive_loop.py
    ├── workflow.py
    └── graph_persistence.py
```

## Key Metrics

| Component | Lines | Purpose |
|-----------|-------|---------|
| processor/ | ~2,200 | Main API |
| analysis.py | ~1,100 | Algorithms |
| reasoning/ | ~4,000 | GoT Framework |
| Total | ~11,100 | Core library |

---
*Generated: 2025-12-21*
