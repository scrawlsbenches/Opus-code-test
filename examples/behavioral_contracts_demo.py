#!/usr/bin/env python3
"""
Behavioral Contracts - End-to-End Demonstration

This example demonstrates the full contract lifecycle:
1. Define contracts with @requires, @ensures, @invariant
2. Register contracts with ContractRegistry
3. Execute code and emit CEL events
4. Materialize contract state from events
5. Query violations and generate health reports

Run with:
    python examples/behavioral_contracts_demo.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cortical.contracts import (
    requires,
    ensures,
    invariant,
    ContractViolation,
    ContractRegistry,
    ContractEventEmitter,
    ContractMaterializer,
    ContractState,
)


# =============================================================================
# STEP 1: Create Registry and Emitter
# =============================================================================

print("=" * 70)
print("BEHAVIORAL CONTRACTS - END-TO-END DEMONSTRATION")
print("=" * 70)
print()

# Create event emitter (standalone mode - no CEL store)
# In production, you'd pass a CEL EventStore here
emitter = ContractEventEmitter(emit_all_checks=True)

# Create registry with emitter
registry = ContractRegistry(emitter=emitter)

print("[1] Created ContractRegistry and ContractEventEmitter")
print(f"    - emit_all_checks: True (for demonstration)")
print(f"    - CEL store: None (using in-memory buffer)")
print()


# =============================================================================
# STEP 2: Define a Class with Contracts
# =============================================================================

class BankAccount:
    """
    Example class demonstrating behavioral contracts.

    This models a simple bank account with contracts that express
    the INTENT of each operation.

    NOTE: @registry.track MUST be the OUTERMOST decorator (first applied).
    This allows it to capture all contracts defined by inner decorators.
    """

    def __init__(self, owner: str, initial_balance: float = 0.0):
        self._owner = owner
        self._balance = initial_balance
        self._transaction_count = 0
        self._is_frozen = False

    @registry.track  # OUTERMOST - captures all contracts below
    @requires(lambda self: not self._is_frozen, "Account must not be frozen")
    @requires(lambda self, amount: amount > 0, "Deposit amount must be positive")
    @ensures(lambda self, result: self._balance >= 0, "Balance must remain non-negative")
    def deposit(self, amount: float) -> float:
        """Deposit funds into the account."""
        self._balance += amount
        self._transaction_count += 1
        return self._balance

    @registry.track  # OUTERMOST
    @requires(lambda self: not self._is_frozen, "Account must not be frozen")
    @requires(lambda self, amount: amount > 0, "Withdrawal amount must be positive")
    @requires(lambda self, amount: self._balance >= amount, "Insufficient funds")
    @ensures(lambda self, result: self._balance >= 0, "Balance must remain non-negative")
    @invariant(lambda self: self._transaction_count >= 0, "Transaction count must be non-negative")
    def withdraw(self, amount: float) -> float:
        """Withdraw funds from the account."""
        self._balance -= amount
        self._transaction_count += 1
        return self._balance

    @registry.track  # OUTERMOST
    @requires(lambda self, target: target is not self, "Cannot transfer to self")
    @requires(lambda self, target, amount: amount > 0, "Transfer amount must be positive")
    @ensures(lambda self, result: result['success'], "Transfer must succeed")
    def transfer(self, target: 'BankAccount', amount: float) -> dict:
        """Transfer funds to another account."""
        self.withdraw(amount)
        target.deposit(amount)
        return {'success': True, 'amount': amount}

    @registry.track  # OUTERMOST
    @invariant(lambda self: self._balance >= 0 or self._is_frozen,
               "Unfrozen account must have non-negative balance")
    def freeze(self) -> None:
        """Freeze the account (no more transactions)."""
        self._is_frozen = True

    @property
    def balance(self) -> float:
        return self._balance


print("[2] Defined BankAccount class with contracts:")
print("    - deposit(): @requires not frozen, amount > 0")
print("                 @ensures balance >= 0")
print("    - withdraw(): @requires not frozen, amount > 0, sufficient funds")
print("                  @ensures balance >= 0")
print("                  @invariant transaction_count >= 0")
print("    - transfer(): @requires target != self, amount > 0")
print("                  @ensures success")
print("    - freeze():   @invariant balance >= 0 or frozen")
print()


# =============================================================================
# STEP 3: Execute Operations (Some Will Pass, Some Will Fail)
# =============================================================================

print("[3] Executing operations...")
print()

# Create accounts
alice = BankAccount("Alice", 1000.0)
bob = BankAccount("Bob", 500.0)

# Successful operations
print("    [OK] Alice deposits $200")
alice.deposit(200)

print("    [OK] Alice withdraws $300")
alice.withdraw(300)

print("    [OK] Alice transfers $100 to Bob")
alice.transfer(bob, 100)

print(f"    Alice balance: ${alice.balance}")
print(f"    Bob balance: ${bob.balance}")
print()

# Now trigger some violations
print("    Attempting operations that will violate contracts...")
print()

# Violation 1: Negative deposit
try:
    print("    [FAIL] Alice deposits $-50 (negative amount)")
    alice.deposit(-50)
except ContractViolation as e:
    print(f"           Caught: {e.contract_type} - {e.description}")
print()

# Violation 2: Insufficient funds
try:
    print("    [FAIL] Bob withdraws $10000 (insufficient funds)")
    bob.withdraw(10000)
except ContractViolation as e:
    print(f"           Caught: {e.contract_type} - {e.description}")
print()

# Violation 3: Transfer to self
try:
    print("    [FAIL] Alice transfers to herself")
    alice.transfer(alice, 50)
except ContractViolation as e:
    print(f"           Caught: {e.contract_type} - {e.description}")
print()

# Violation 4: Frozen account
alice.freeze()
try:
    print("    [FAIL] Alice deposits after freeze")
    alice.deposit(100)
except ContractViolation as e:
    print(f"           Caught: {e.contract_type} - {e.description}")
print()


# =============================================================================
# STEP 4: Query Contract State via Materializer
# =============================================================================

print("[4] Materializing contract state from events...")
print()

materializer = ContractMaterializer(emitter=emitter)
state = materializer.current_state()

print(f"    {state}")
print(f"    - Total checks: {state.total_checks}")
print(f"    - Total violations: {state.total_violations}")
print(f"    - Violation rate: {state.violation_rate:.2f}%")
print(f"    - Is healthy: {state.is_healthy}")
print()

if state.violations_by_method:
    print("    Violations by method:")
    for method, count in state.violations_by_method.items():
        print(f"      - {method}: {count}")
    print()


# =============================================================================
# STEP 5: View CEL Events (Buffered)
# =============================================================================

print("[5] CEL Events emitted (in-memory buffer):")
print()

events = emitter.buffered_events
print(f"    Total events: {len(events)}")
print()

# Show last 5 events
print("    Last 5 events:")
for event in events[-5:]:
    event_type = event.get('event_type', 'UNKNOWN')
    content = event.get('content', {})

    if event_type == 'OBSERVATION':
        check_type = content.get('contract_type', 'unknown')
        method = content.get('method', 'unknown')
        passed = content.get('passed', False)
        status = "PASS" if passed else "FAIL"
        print(f"      [{event_type}] {status} {check_type} on {method}")
    elif event_type == 'METACOGNITION':
        obs_type = content.get('observation_type', 'unknown')
        conclusions = content.get('conclusions', [])
        print(f"      [{event_type}] {obs_type}: {conclusions[0] if conclusions else 'N/A'}")
print()


# =============================================================================
# STEP 6: Generate Health Report
# =============================================================================

print("[6] Health Report:")
print()

report = materializer.health_report()

print(f"    Status: {report['status']}")
print(f"    Recent violations (24h): {report['recent_violations_24h']}")
print()

if report['recommendations']:
    print("    Recommendations:")
    for rec in report['recommendations']:
        print(f"      - {rec}")
print()


# =============================================================================
# STEP 7: Query Recent Violations
# =============================================================================

print("[7] Recent Violations:")
print()

violations = materializer.violations_since(hours=1)
print(f"    Found {len(violations)} violations in last hour:")
print()

for v in violations:
    print(f"      {v.timestamp.strftime('%H:%M:%S')} | {v.method}")
    print(f"        {v.description}")
print()


# =============================================================================
# STEP 8: Registry Statistics
# =============================================================================

print("[8] Contract Registry Statistics:")
print()

stats = registry.stats()
print(f"    Total contracts: {stats['total_contracts']}")
print(f"    Contracts with violations: {stats['contracts_with_violations']}")
print(f"    Total checks: {stats['total_checks']}")
print(f"    Total violations: {stats['total_violations']}")
print(f"    Violation rate: {stats['violation_rate']:.2%}")
print()

print("    By type:")
for type_name, count in stats['by_type'].items():
    print(f"      - {type_name}: {count}")
print()


# =============================================================================
# STEP 9: Export Documentation
# =============================================================================

print("[9] Exported Contract Documentation:")
print()

docs = registry.export_documentation()
# Show first 30 lines
lines = docs.split('\n')[:30]
for line in lines:
    print(f"    {line}")
print("    ...")
print()


# =============================================================================
# SUMMARY
# =============================================================================

print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()
print("This demonstration showed the full contract lifecycle:")
print()
print("  1. DEFINE: @requires, @ensures, @invariant decorators")
print("     - Express INTENT, not just behavior")
print("     - Contracts are executable documentation")
print()
print("  2. REGISTER: ContractRegistry.track decorator")
print("     - Centralized contract tracking")
print("     - Queryable metadata")
print()
print("  3. EMIT: ContractEventEmitter")
print("     - Contract checks → CEL Observation events")
print("     - Violations → CEL MetaCognition events")
print()
print("  4. MATERIALIZE: ContractMaterializer")
print("     - Fold events into ContractState")
print("     - Temporal queries supported")
print()
print("  5. QUERY: Health reports, violation lists, statistics")
print("     - \"What was the contract state at time T?\"")
print("     - \"Which methods have the most violations?\"")
print()
print("Integration with CEL enables:")
print("  - Event compaction: Summarize contract history")
print("  - Temporal queries: State at any point in time")
print("  - Audit trails: Full history of contract checks")
print("  - Self-healing: MetaCognition events trigger actions")
print()
print("=" * 70)
