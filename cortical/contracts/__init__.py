"""
Behavioral Contracts - Executable Intent for Cortical.

This module provides Design-by-Contract (DbC) decorators that:
1. Define preconditions (@requires), postconditions (@ensures), and invariants (@invariant)
2. Emit CEL events when contracts are checked or violated
3. Materialize contract state from event history
4. Enable temporal queries: "Was this contract ever violated?"

Integration with CEL:
    - Contract checks → Observation events (passive recording)
    - Contract violations → MetaCognition events (system self-awareness)
    - Contract state → Materialized from events via ContractReducer

Example:
    from cortical.contracts import requires, ensures, invariant, ContractRegistry

    registry = ContractRegistry()

    class TransactionManager:
        @requires(lambda self: not self._in_transaction, "Cannot nest transactions")
        @ensures(lambda self, result: self._in_transaction, "Must be in transaction after begin")
        @registry.track
        def begin(self):
            self._in_transaction = True
            return self

        @invariant("wal_entries >= committed_entities")
        @registry.track
        def commit(self):
            ...

Philosophy:
    Contracts are executable documentation. They capture INTENT, not just behavior.
    Unlike tests (which verify specific cases), contracts express universal truths
    that must hold across all executions.

    When a contract is violated, it means we misunderstood the system's invariants,
    not that we found a bug. This distinction matters for how we respond.
"""

from .decorators import requires, ensures, invariant, ContractViolation
from .registry import ContractRegistry, Contract, ContractType
from .cel_integration import ContractEventEmitter, contract_reducer
from .materializer import ContractState, ContractMaterializer

__all__ = [
    # Decorators
    'requires',
    'ensures',
    'invariant',
    'ContractViolation',
    # Registry
    'ContractRegistry',
    'Contract',
    'ContractType',
    # CEL Integration
    'ContractEventEmitter',
    'contract_reducer',
    # Materialization
    'ContractState',
    'ContractMaterializer',
]
