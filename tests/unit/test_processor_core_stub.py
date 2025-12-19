"""
Import Stub for test_processor_core.py

The original test_processor_core.py (3534 lines) has been refactored into 8 focused test files:
- test_processor_init.py         (40 tests)  - Initialization & Configuration
- test_processor_documents.py    (56 tests)  - Document Management
- test_processor_staleness.py    (26 tests)  - Staleness Tracking
- test_processor_compute.py      (50 tests)  - Compute Methods
- test_processor_wrappers.py     (37 tests)  - Wrapper Methods
- test_processor_query.py        (31 tests)  - Query Expansion & Search
- test_processor_search.py       (23 tests)  - Search Facade Methods
- test_processor_coverage.py     (51 tests)  - Coverage Gap Tests

Total: 314 tests across 27 test classes

This stub file imports all test classes for backward compatibility.
To run all tests: python -m pytest tests/unit/test_processor_*.py
"""

# Import all test classes from split files
from tests.unit.test_processor_init import (
    TestProcessorInitialization,
    TestLayerAccess,
    TestConfiguration,
    TestBasicValidation,
)

from tests.unit.test_processor_documents import (
    TestDocumentManagement,
    TestIncrementalDocumentAddition,
    TestBatchDocumentOperations,
    TestMetadataManagement,
    TestAdditionalBatchOperations,
    TestEdgeCasesAndErrors,
)

from tests.unit.test_processor_staleness import (
    TestStalenessTracking,
    TestRecompute,
    TestErrorHandling,
)

from tests.unit.test_processor_compute import (
    TestComputeWrapperMethods,
    TestComputeAllParameters,
    TestComputeAllVerbose,
    TestVerbosePathCoverage,
)

from tests.unit.test_processor_wrappers import (
    TestAdditionalWrapperMethods,
    TestSemanticImportance,
    TestSimpleWrapperMethods,
    TestWrapperEdgeCases,
)

from tests.unit.test_processor_query import (
    TestQueryExpansion,
    TestFindDocumentsMethods,
)

from tests.unit.test_processor_search import (
    TestQuickSearch,
    TestRagRetrieve,
    TestExplore,
)

from tests.unit.test_processor_coverage import (
    TestAdditionalCoverage,
)

__all__ = [
    # Initialization (4 classes, 40 tests)
    'TestProcessorInitialization',
    'TestLayerAccess',
    'TestConfiguration',
    'TestBasicValidation',

    # Documents (6 classes, 56 tests)
    'TestDocumentManagement',
    'TestIncrementalDocumentAddition',
    'TestBatchDocumentOperations',
    'TestMetadataManagement',
    'TestAdditionalBatchOperations',
    'TestEdgeCasesAndErrors',

    # Staleness (3 classes, 26 tests)
    'TestStalenessTracking',
    'TestRecompute',
    'TestErrorHandling',

    # Compute (4 classes, 50 tests)
    'TestComputeWrapperMethods',
    'TestComputeAllParameters',
    'TestComputeAllVerbose',
    'TestVerbosePathCoverage',

    # Wrappers (4 classes, 37 tests)
    'TestAdditionalWrapperMethods',
    'TestSemanticImportance',
    'TestSimpleWrapperMethods',
    'TestWrapperEdgeCases',

    # Query (2 classes, 31 tests)
    'TestQueryExpansion',
    'TestFindDocumentsMethods',

    # Search (3 classes, 23 tests)
    'TestQuickSearch',
    'TestRagRetrieve',
    'TestExplore',

    # Coverage (1 class, 51 tests)
    'TestAdditionalCoverage',
]

if __name__ == '__main__':
    import unittest
    unittest.main()
