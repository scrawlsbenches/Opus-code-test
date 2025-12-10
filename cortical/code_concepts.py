"""
Code Concepts Module
====================

Programming concept groups for semantic code search.

Maps common programming synonyms and related terms to enable
intent-based code retrieval. When a developer searches for "get user",
the system can also find "fetch user", "load user", "retrieve user".
"""

from typing import Dict, List, Set, FrozenSet


# Programming concept groups - terms that are often interchangeable in code
CODE_CONCEPT_GROUPS: Dict[str, FrozenSet[str]] = {
    # Data retrieval operations
    'retrieval': frozenset([
        'get', 'fetch', 'load', 'retrieve', 'read', 'query', 'find',
        'lookup', 'obtain', 'acquire', 'pull', 'select'
    ]),

    # Data storage operations
    'storage': frozenset([
        'save', 'store', 'write', 'persist', 'cache', 'put', 'set',
        'insert', 'add', 'create', 'commit', 'push', 'update'
    ]),

    # Deletion operations
    'deletion': frozenset([
        'delete', 'remove', 'drop', 'clear', 'destroy', 'purge',
        'erase', 'clean', 'reset', 'dispose', 'unset'
    ]),

    # Authentication and security
    'auth': frozenset([
        'auth', 'authentication', 'login', 'logout', 'credentials',
        'token', 'session', 'password', 'user', 'permission', 'role',
        'access', 'authorize', 'verify', 'validate', 'identity'
    ]),

    # Error handling
    'error': frozenset([
        'error', 'exception', 'fail', 'failure', 'catch', 'handle',
        'throw', 'raise', 'try', 'recover', 'retry', 'fallback',
        'invalid', 'warning', 'fault', 'crash'
    ]),

    # Validation and checking
    'validation': frozenset([
        'validate', 'check', 'verify', 'assert', 'ensure', 'confirm',
        'test', 'inspect', 'examine', 'sanitize', 'filter', 'guard'
    ]),

    # Transformation operations
    'transform': frozenset([
        'transform', 'convert', 'parse', 'format', 'serialize',
        'deserialize', 'encode', 'decode', 'map', 'reduce', 'filter',
        'normalize', 'process', 'translate', 'render'
    ]),

    # Network and API
    'network': frozenset([
        'request', 'response', 'api', 'endpoint', 'http', 'rest',
        'client', 'server', 'socket', 'connection', 'send', 'receive',
        'url', 'route', 'handler', 'middleware'
    ]),

    # Database operations
    'database': frozenset([
        'database', 'db', 'sql', 'query', 'table', 'record', 'row',
        'column', 'index', 'schema', 'migration', 'model', 'entity',
        'repository', 'orm', 'transaction'
    ]),

    # Async and concurrency
    'async': frozenset([
        'async', 'await', 'promise', 'future', 'callback', 'thread',
        'concurrent', 'parallel', 'worker', 'queue', 'task', 'job',
        'schedule', 'spawn', 'sync', 'lock', 'mutex'
    ]),

    # Configuration and settings
    'config': frozenset([
        'config', 'configuration', 'settings', 'options', 'preferences',
        'env', 'environment', 'property', 'parameter', 'argument',
        'flag', 'constant', 'default', 'override'
    ]),

    # Logging and monitoring
    'logging': frozenset([
        'log', 'logger', 'logging', 'debug', 'info', 'warn', 'trace',
        'monitor', 'metric', 'telemetry', 'track', 'audit', 'record',
        'print', 'output', 'verbose'
    ]),

    # Testing
    'testing': frozenset([
        'test', 'spec', 'mock', 'stub', 'fake', 'fixture', 'assert',
        'expect', 'verify', 'unit', 'integration', 'coverage', 'suite',
        'setup', 'teardown', 'before', 'after'
    ]),

    # File operations
    'file': frozenset([
        'file', 'path', 'directory', 'folder', 'read', 'write', 'open',
        'close', 'stream', 'buffer', 'io', 'filesystem', 'upload',
        'download', 'copy', 'move', 'rename'
    ]),

    # Iteration and collections
    'iteration': frozenset([
        'iterate', 'loop', 'each', 'map', 'filter', 'reduce', 'fold',
        'list', 'array', 'collection', 'set', 'dict', 'hash', 'tree',
        'queue', 'stack', 'sort', 'search', 'find'
    ]),

    # Initialization and lifecycle
    'lifecycle': frozenset([
        'init', 'initialize', 'setup', 'start', 'stop', 'shutdown',
        'bootstrap', 'create', 'destroy', 'build', 'configure',
        'register', 'unregister', 'connect', 'disconnect', 'close'
    ]),

    # Events and messaging
    'events': frozenset([
        'event', 'emit', 'listen', 'subscribe', 'publish', 'dispatch',
        'handler', 'callback', 'hook', 'trigger', 'notify', 'observe',
        'broadcast', 'signal', 'message', 'channel'
    ]),
}

# Build reverse index: term -> list of concept groups it belongs to
_TERM_TO_CONCEPTS: Dict[str, List[str]] = {}
for concept, terms in CODE_CONCEPT_GROUPS.items():
    for term in terms:
        if term not in _TERM_TO_CONCEPTS:
            _TERM_TO_CONCEPTS[term] = []
        _TERM_TO_CONCEPTS[term].append(concept)


def get_related_terms(term: str, max_terms: int = 5) -> List[str]:
    """
    Get programming terms related to the given term.

    Args:
        term: A programming term (e.g., "fetch", "authenticate")
        max_terms: Maximum number of related terms to return

    Returns:
        List of related terms, excluding the input term

    Example:
        >>> get_related_terms("fetch")
        ['get', 'load', 'retrieve', 'read', 'query']
    """
    term_lower = term.lower()
    related: Set[str] = set()

    # Find all concept groups this term belongs to
    concepts = _TERM_TO_CONCEPTS.get(term_lower, [])

    for concept in concepts:
        terms = CODE_CONCEPT_GROUPS.get(concept, frozenset())
        related.update(terms)

    # Remove the original term
    related.discard(term_lower)

    # Return top terms sorted alphabetically for consistent results
    return sorted(related)[:max_terms]


def expand_code_concepts(
    terms: List[str],
    max_expansions_per_term: int = 3,
    weight: float = 0.6
) -> Dict[str, float]:
    """
    Expand a list of terms using code concept groups.

    Args:
        terms: List of query terms to expand
        max_expansions_per_term: Max related terms to add per input term
        weight: Weight to assign to expanded terms (0.0-1.0)

    Returns:
        Dict mapping expanded terms to weights

    Example:
        >>> expand_code_concepts(["fetch", "user"])
        {'get': 0.6, 'load': 0.6, 'retrieve': 0.6, ...}
    """
    expanded: Dict[str, float] = {}
    input_terms = set(t.lower() for t in terms)

    for term in terms:
        related = get_related_terms(term, max_terms=max_expansions_per_term)
        for related_term in related:
            # Don't add terms that were in the original query
            if related_term not in input_terms:
                # Keep highest weight if term appears multiple times
                if related_term not in expanded or expanded[related_term] < weight:
                    expanded[related_term] = weight

    return expanded


def get_concept_group(term: str) -> List[str]:
    """
    Get the concept group names a term belongs to.

    Args:
        term: A programming term

    Returns:
        List of concept group names

    Example:
        >>> get_concept_group("fetch")
        ['retrieval']
        >>> get_concept_group("validate")
        ['validation', 'testing']
    """
    return _TERM_TO_CONCEPTS.get(term.lower(), [])


def list_concept_groups() -> List[str]:
    """
    List all available concept group names.

    Returns:
        Sorted list of concept group names
    """
    return sorted(CODE_CONCEPT_GROUPS.keys())


def get_group_terms(group_name: str) -> List[str]:
    """
    Get all terms in a concept group.

    Args:
        group_name: Name of the concept group

    Returns:
        Sorted list of terms in the group, or empty list if group not found
    """
    terms = CODE_CONCEPT_GROUPS.get(group_name, frozenset())
    return sorted(terms)
