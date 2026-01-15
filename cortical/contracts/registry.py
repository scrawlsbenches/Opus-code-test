"""
Contract Registry - Centralized tracking of all contracts.

The registry provides:
1. Automatic discovery of contracts from decorated methods
2. Queryable contract state (all contracts, by class, by type)
3. Integration point for CEL event emission
4. Contract metadata for documentation generation
"""

from __future__ import annotations

import inspect
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar

from .decorators import ContractSpec

F = TypeVar('F', bound=Callable[..., Any])


class ContractType(Enum):
    """Types of contracts."""
    REQUIRES = auto()     # Precondition
    ENSURES = auto()      # Postcondition
    INVARIANT = auto()    # Class invariant


@dataclass
class Contract:
    """
    A registered contract with full metadata.

    This is the queryable representation of a contract,
    containing everything needed for documentation and CEL events.
    """

    id: str  # Unique identifier: "ClassName.method_name.contract_type.N"
    contract_type: ContractType
    description: str
    class_name: str
    method_name: str
    source_file: Optional[str]
    source_line: Optional[int]
    registered_at: datetime = field(default_factory=datetime.now)

    # Runtime statistics (updated by event emitter)
    check_count: int = 0
    violation_count: int = 0
    last_checked: Optional[datetime] = None
    last_violation: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for CEL event content."""
        return {
            'id': self.id,
            'contract_type': self.contract_type.name.lower(),
            'description': self.description,
            'class_name': self.class_name,
            'method_name': self.method_name,
            'source_file': self.source_file,
            'source_line': self.source_line,
            'registered_at': self.registered_at.isoformat(),
            'check_count': self.check_count,
            'violation_count': self.violation_count,
        }


class ContractRegistry:
    """
    Centralized registry for all contracts.

    Usage:
        registry = ContractRegistry()

        class MyClass:
            @requires(lambda self: self.ready, "Must be ready")
            @registry.track
            def do_work(self):
                ...

        # Query contracts
        all_contracts = registry.all()
        class_contracts = registry.for_class("MyClass")
        violations = registry.contracts_with_violations()
    """

    def __init__(self, emitter: Optional[Any] = None):
        """
        Initialize the registry.

        Args:
            emitter: Optional ContractEventEmitter for CEL integration
        """
        self._contracts: Dict[str, Contract] = {}
        self._by_class: Dict[str, Set[str]] = {}
        self._by_method: Dict[str, Set[str]] = {}
        self._by_type: Dict[ContractType, Set[str]] = {t: set() for t in ContractType}
        self._emitter = emitter
        self._lock = threading.Lock()

    def track(self, func: F) -> F:
        """
        Decorator to register all contracts on a method.

        This should be the innermost decorator (closest to the function).

        Example:
            @requires(lambda self: self.ready)
            @ensures(lambda self, r: r is not None)
            @registry.track  # <-- Registers both contracts
            def my_method(self):
                ...
        """
        specs = getattr(func, '_contract_specs', [])

        for i, spec in enumerate(specs):
            contract = self._spec_to_contract(spec, i)

            with self._lock:
                self._contracts[contract.id] = contract

                # Index by class
                if contract.class_name:
                    if contract.class_name not in self._by_class:
                        self._by_class[contract.class_name] = set()
                    self._by_class[contract.class_name].add(contract.id)

                # Index by method
                method_key = f"{contract.class_name}.{contract.method_name}"
                if method_key not in self._by_method:
                    self._by_method[method_key] = set()
                self._by_method[method_key].add(contract.id)

                # Index by type
                self._by_type[contract.contract_type].add(contract.id)

            # Inject emitter into spec for CEL event emission
            if self._emitter is not None:
                spec._emitter = self._emitter

        return func

    def _spec_to_contract(self, spec: ContractSpec, index: int) -> Contract:
        """Convert a ContractSpec to a registered Contract."""
        contract_type = ContractType[spec.contract_type.upper()]
        class_name = spec.class_name or 'global'
        method_name = spec.method_name or 'unknown'

        contract_id = f"{class_name}.{method_name}.{spec.contract_type}.{index}"

        return Contract(
            id=contract_id,
            contract_type=contract_type,
            description=spec.description,
            class_name=class_name,
            method_name=method_name,
            source_file=spec.source_file,
            source_line=spec.source_line,
        )

    def set_emitter(self, emitter: Any) -> None:
        """Set the CEL event emitter (can be done after construction)."""
        self._emitter = emitter
        # Update existing specs
        with self._lock:
            for contract_id in self._contracts:
                # Re-inject emitter - this is a bit awkward but necessary
                # for late binding
                pass

    # =========================================================================
    # Query Methods
    # =========================================================================

    def all(self) -> List[Contract]:
        """Get all registered contracts."""
        with self._lock:
            return list(self._contracts.values())

    def get(self, contract_id: str) -> Optional[Contract]:
        """Get a specific contract by ID."""
        with self._lock:
            return self._contracts.get(contract_id)

    def for_class(self, class_name: str) -> List[Contract]:
        """Get all contracts for a class."""
        with self._lock:
            contract_ids = self._by_class.get(class_name, set())
            return [self._contracts[cid] for cid in contract_ids]

    def for_method(self, class_name: str, method_name: str) -> List[Contract]:
        """Get all contracts for a specific method."""
        method_key = f"{class_name}.{method_name}"
        with self._lock:
            contract_ids = self._by_method.get(method_key, set())
            return [self._contracts[cid] for cid in contract_ids]

    def by_type(self, contract_type: ContractType) -> List[Contract]:
        """Get all contracts of a specific type."""
        with self._lock:
            contract_ids = self._by_type.get(contract_type, set())
            return [self._contracts[cid] for cid in contract_ids]

    def contracts_with_violations(self) -> List[Contract]:
        """Get contracts that have had violations."""
        with self._lock:
            return [c for c in self._contracts.values() if c.violation_count > 0]

    def contracts_never_checked(self) -> List[Contract]:
        """Get contracts that have never been checked (dead code?)."""
        with self._lock:
            return [c for c in self._contracts.values() if c.check_count == 0]

    # =========================================================================
    # Statistics
    # =========================================================================

    def stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        with self._lock:
            total = len(self._contracts)
            violations = sum(1 for c in self._contracts.values() if c.violation_count > 0)
            never_checked = sum(1 for c in self._contracts.values() if c.check_count == 0)
            total_checks = sum(c.check_count for c in self._contracts.values())
            total_violations = sum(c.violation_count for c in self._contracts.values())

            return {
                'total_contracts': total,
                'contracts_with_violations': violations,
                'contracts_never_checked': never_checked,
                'total_checks': total_checks,
                'total_violations': total_violations,
                'violation_rate': total_violations / total_checks if total_checks > 0 else 0.0,
                'by_type': {
                    t.name.lower(): len(ids)
                    for t, ids in self._by_type.items()
                },
                'classes_covered': len(self._by_class),
            }

    def update_stats(
        self,
        contract_id: str,
        passed: bool,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Update statistics for a contract (called by emitter)."""
        timestamp = timestamp or datetime.now()
        with self._lock:
            contract = self._contracts.get(contract_id)
            if contract:
                contract.check_count += 1
                contract.last_checked = timestamp
                if not passed:
                    contract.violation_count += 1
                    contract.last_violation = timestamp

    # =========================================================================
    # Discovery
    # =========================================================================

    def discover(self, obj: Any) -> int:
        """
        Discover and register contracts from an object's methods.

        This walks the object's class hierarchy and registers any
        contracts found on methods.

        Args:
            obj: Object instance or class to scan

        Returns:
            Number of contracts discovered
        """
        cls = obj if isinstance(obj, type) else type(obj)
        discovered = 0

        for name in dir(cls):
            if name.startswith('_'):
                continue

            try:
                attr = getattr(cls, name)
            except AttributeError:
                continue

            if callable(attr):
                specs = getattr(attr, '_contract_specs', [])
                for i, spec in enumerate(specs):
                    contract = self._spec_to_contract(spec, i)
                    with self._lock:
                        if contract.id not in self._contracts:
                            self._contracts[contract.id] = contract
                            discovered += 1

                            # Update indexes
                            if contract.class_name not in self._by_class:
                                self._by_class[contract.class_name] = set()
                            self._by_class[contract.class_name].add(contract.id)

                            method_key = f"{contract.class_name}.{contract.method_name}"
                            if method_key not in self._by_method:
                                self._by_method[method_key] = set()
                            self._by_method[method_key].add(contract.id)

                            self._by_type[contract.contract_type].add(contract.id)

        return discovered

    def export_documentation(self) -> str:
        """
        Export all contracts as documentation.

        Returns markdown-formatted documentation of all registered contracts.
        """
        lines = ["# Contract Documentation\n"]
        lines.append(f"Generated: {datetime.now().isoformat()}\n")
        lines.append(f"Total contracts: {len(self._contracts)}\n\n")

        # Group by class
        for class_name in sorted(self._by_class.keys()):
            lines.append(f"## {class_name}\n\n")

            contracts = self.for_class(class_name)
            contracts.sort(key=lambda c: (c.method_name, c.contract_type.value))

            current_method = None
            for contract in contracts:
                if contract.method_name != current_method:
                    current_method = contract.method_name
                    lines.append(f"### {contract.method_name}()\n\n")

                type_label = contract.contract_type.name.lower()
                lines.append(f"- **{type_label}**: {contract.description}\n")

                if contract.source_file and contract.source_line:
                    lines.append(f"  - Source: `{contract.source_file}:{contract.source_line}`\n")

            lines.append("\n")

        return "".join(lines)
