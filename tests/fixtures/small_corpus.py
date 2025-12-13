"""
Small Synthetic Corpus for Fast Tests
======================================

A 25-document synthetic corpus designed for:
- Fast test execution (< 2s to process and compute_all)
- Covering multiple domains for search relevance testing
- Predictable content for deterministic test assertions
- Testing clustering, PageRank, TF-IDF without real file I/O

Usage:
    from tests.fixtures.small_corpus import get_small_processor, SMALL_CORPUS_DOCS

    processor = get_small_processor()  # Already has compute_all() called
    docs = SMALL_CORPUS_DOCS           # Raw document dict
"""

import sys
import os

# Ensure cortical is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from cortical import CorticalTextProcessor
from cortical.tokenizer import Tokenizer


# Synthetic documents covering multiple domains
# Each ~50-100 words for fast processing
SMALL_CORPUS_DOCS = {
    # Machine Learning domain (5 docs)
    "ml_basics": """
        Machine learning is a subset of artificial intelligence that enables
        systems to learn from data. Supervised learning uses labeled examples
        to train models. Unsupervised learning finds patterns without labels.
        Neural networks are inspired by biological neurons and can learn
        complex representations from raw data.
    """,
    "deep_learning": """
        Deep learning uses neural networks with many layers to learn
        hierarchical representations. Convolutional networks excel at image
        recognition. Recurrent networks process sequential data like text.
        Training deep networks requires large datasets and significant
        computational resources like GPUs.
    """,
    "ml_optimization": """
        Training neural networks involves minimizing a loss function through
        gradient descent. The learning rate controls step size during optimization.
        Batch normalization and dropout help prevent overfitting. Adam optimizer
        adapts learning rates for each parameter automatically.
    """,
    "ml_evaluation": """
        Model evaluation requires splitting data into training and test sets.
        Cross-validation provides more robust performance estimates. Metrics
        like accuracy, precision, recall, and F1 score measure different aspects
        of model performance. Confusion matrices visualize classification errors.
    """,
    "ml_applications": """
        Machine learning powers recommendation systems, spam filters, and voice
        assistants. Computer vision enables autonomous vehicles and medical imaging.
        Natural language processing drives translation and chatbots. Predictive
        analytics helps businesses forecast demand and detect fraud.
    """,

    # Database domain (5 docs)
    "db_fundamentals": """
        Databases store and organize data for efficient retrieval. Relational
        databases use tables with rows and columns. SQL provides a standard
        language for querying and manipulating data. Primary keys uniquely
        identify records while foreign keys establish relationships.
    """,
    "db_indexing": """
        Database indexes speed up query performance by creating sorted data
        structures. B-tree indexes support range queries efficiently. Hash
        indexes provide constant-time lookups for equality comparisons.
        Index maintenance adds overhead to write operations.
    """,
    "db_transactions": """
        Database transactions ensure data consistency through ACID properties.
        Atomicity means all operations complete or none do. Isolation prevents
        concurrent transactions from interfering. Durability guarantees
        committed changes survive system failures.
    """,
    "db_nosql": """
        NoSQL databases handle unstructured and semi-structured data. Document
        stores like MongoDB store JSON-like objects. Key-value stores provide
        simple but fast access patterns. Graph databases model relationships
        between entities efficiently.
    """,
    "db_scaling": """
        Database scaling handles growing data volumes and query loads. Vertical
        scaling adds resources to a single server. Horizontal scaling distributes
        data across multiple nodes through sharding. Replication provides
        redundancy and read scalability.
    """,

    # Distributed Systems domain (5 docs)
    "dist_basics": """
        Distributed systems span multiple networked computers working together.
        Network partitions and node failures are inevitable challenges.
        The CAP theorem states that systems cannot simultaneously guarantee
        consistency, availability, and partition tolerance.
    """,
    "dist_consensus": """
        Consensus protocols help distributed nodes agree on shared state.
        Paxos and Raft are widely used consensus algorithms. Leader election
        selects a coordinator node for decision making. Quorum-based approaches
        require majority agreement for operations.
    """,
    "dist_messaging": """
        Message queues decouple distributed system components. Producers publish
        messages while consumers process them asynchronously. Message brokers
        like Kafka provide durable, ordered message delivery. Event sourcing
        captures all state changes as an immutable log.
    """,
    "dist_microservices": """
        Microservices architecture breaks applications into small, independent
        services. Each service owns its data and communicates via APIs.
        Service discovery helps locate service instances dynamically.
        Circuit breakers prevent cascade failures across services.
    """,
    "dist_caching": """
        Distributed caches reduce database load and improve response times.
        Cache invalidation ensures stale data is refreshed appropriately.
        Consistent hashing distributes cache entries across nodes evenly.
        Write-through and write-behind strategies handle cache updates.
    """,

    # Algorithms domain (5 docs)
    "algo_sorting": """
        Sorting algorithms arrange elements in order. Quicksort uses divide
        and conquer with average O(n log n) complexity. Merge sort guarantees
        O(n log n) but requires extra space. Insertion sort is efficient
        for small or nearly sorted arrays.
    """,
    "algo_searching": """
        Search algorithms find elements in data structures. Binary search
        achieves O(log n) on sorted arrays. Hash tables provide O(1) average
        lookup time. Breadth-first and depth-first search traverse graphs
        systematically.
    """,
    "algo_graphs": """
        Graph algorithms solve problems on networked structures. Dijkstra's
        algorithm finds shortest paths in weighted graphs. PageRank measures
        node importance based on link structure. Minimum spanning trees
        connect all nodes with minimum total edge weight.
    """,
    "algo_dynamic": """
        Dynamic programming solves problems by combining subproblem solutions.
        Memoization caches results to avoid redundant computation. The
        knapsack problem and longest common subsequence are classic examples.
        Bottom-up tabulation builds solutions iteratively.
    """,
    "algo_complexity": """
        Algorithm complexity measures resource usage as input grows. Time
        complexity counts operations while space complexity measures memory.
        Big O notation describes worst-case asymptotic behavior. Amortized
        analysis averages cost over operation sequences.
    """,

    # Software Engineering domain (5 docs)
    "se_testing": """
        Software testing verifies code behaves correctly. Unit tests check
        individual functions in isolation. Integration tests verify component
        interactions. Test-driven development writes tests before implementation.
        Code coverage measures which lines tests execute.
    """,
    "se_design_patterns": """
        Design patterns are reusable solutions to common problems. Factory
        pattern creates objects without specifying concrete classes. Observer
        pattern notifies dependents of state changes. Strategy pattern
        encapsulates interchangeable algorithms.
    """,
    "se_version_control": """
        Version control tracks changes to code over time. Git uses distributed
        repositories with branches for parallel development. Commits capture
        snapshots of project state. Merge and rebase integrate changes
        from different branches.
    """,
    "se_ci_cd": """
        Continuous integration automatically builds and tests code changes.
        Automated pipelines run tests on every commit. Continuous deployment
        releases validated changes to production automatically. Feature
        flags enable gradual rollouts and quick rollbacks.
    """,
    "se_code_quality": """
        Code quality practices improve maintainability and reliability.
        Code reviews catch bugs and share knowledge. Static analysis tools
        detect potential issues automatically. Refactoring improves code
        structure without changing behavior.
    """,
}


# Module-level singleton for shared small processor
_SMALL_PROCESSOR = None
_SMALL_PROCESSOR_INITIALIZED = False


def get_small_corpus() -> dict:
    """
    Get the raw small corpus documents.

    Returns:
        Dict mapping doc_id to content string
    """
    return SMALL_CORPUS_DOCS.copy()


def get_small_processor(recompute: bool = False) -> CorticalTextProcessor:
    """
    Get a processor initialized with the small corpus.

    This is a singleton - the processor is created once and reused.
    compute_all() has already been called.

    Args:
        recompute: If True, force recreation of the processor

    Returns:
        CorticalTextProcessor with small corpus loaded and computed
    """
    global _SMALL_PROCESSOR, _SMALL_PROCESSOR_INITIALIZED

    if _SMALL_PROCESSOR_INITIALIZED and not recompute:
        return _SMALL_PROCESSOR

    # Create fresh processor with code noise filtering
    tokenizer = Tokenizer(filter_code_noise=True)
    processor = CorticalTextProcessor(tokenizer=tokenizer)

    # Load all documents
    for doc_id, content in SMALL_CORPUS_DOCS.items():
        processor.process_document(doc_id, content)

    # Compute all network properties
    processor.compute_all(verbose=False)

    _SMALL_PROCESSOR = processor
    _SMALL_PROCESSOR_INITIALIZED = True

    return processor


def reset_small_processor():
    """Reset the singleton so next get_small_processor() creates fresh instance."""
    global _SMALL_PROCESSOR, _SMALL_PROCESSOR_INITIALIZED
    _SMALL_PROCESSOR = None
    _SMALL_PROCESSOR_INITIALIZED = False
