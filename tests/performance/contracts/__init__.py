"""
Performance Contracts - Sacred Promises We Keep

This module contains performance contracts that are enforced on every CI run.
Breaking a contract blocks the build. There are no exceptions.

Contracts are different from benchmarks:
- Benchmarks measure current performance
- Contracts GUARANTEE minimum performance levels

To renegotiate a contract, you must:
1. Document why the contract must change
2. Assess impact on users
3. Get team review
4. Update the contract values based on measurements
"""
