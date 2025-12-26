#!/usr/bin/env python3
"""
Cognitive Demo v3: AST-Enhanced Code Intelligence

Combines semantic analysis (what concepts co-occur) with structural
analysis (actual code relationships) for deeper insights.

This hybrid approach reveals things neither approach finds alone:
- Semantic: "task" and "transaction" appear together
- Structural: TaskManager.create() calls TransactionManager.begin()
- Combined: Semantic coupling CONFIRMED by structural dependency

Usage:
    python scripts/cognitive_demo_ast.py
"""

import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cortical import CorticalTextProcessor, CorticalLayer
from cortical.spark.ast_index import ASTIndex


def section(title: str, char: str = "─"):
    """Print section header."""
    print(f"\n{char * 70}")
    print(f" {title}")
    print(char * 70 + "\n")


def load_both_indices(root: Path) -> Tuple[CorticalTextProcessor, ASTIndex]:
    """Load both semantic and structural indices."""
    print("Building semantic index (text-based)...")
    processor = CorticalTextProcessor()
    for py_file in root.rglob("*.py"):
        if "__pycache__" in py_file.parts or "test" in py_file.parts:
            continue
        try:
            content = py_file.read_text()
            doc_id = str(py_file.relative_to(root))
            processor.process_document(doc_id, content)
        except Exception:
            pass
    processor.compute_all()
    print(f"  {len(processor.documents)} documents indexed")

    print("Building structural index (AST-based)...")
    ast_index = ASTIndex()
    ast_index.index_directory(root)
    stats = ast_index.get_stats()
    print(f"  {stats['classes']} classes, {stats['functions']} functions, {stats['call_edges']} call edges")

    return processor, ast_index


def insight_architecture_layers(ast_index: ASTIndex):
    """Discover architectural layers through inheritance hierarchies."""
    section("INSIGHT 1: Architectural Layers (Inheritance Analysis)")

    print("Major class hierarchies (inheritance depth ≥ 2):\n")

    # Find base classes (classes that are inherited but don't inherit)
    all_children = set()
    for children in ast_index.inheritance.values():
        all_children.update(children)

    base_classes = set(ast_index.inheritance.keys()) - all_children

    for base in sorted(base_classes):
        tree = ast_index.get_inheritance_tree(base)
        if tree['children']:  # Has at least one level of inheritance
            _print_tree(tree, 0)
            print()

def _print_tree(tree: dict, indent: int):
    """Recursively print inheritance tree."""
    prefix = "  " * indent + ("└── " if indent > 0 else "")
    print(f"{prefix}{tree['name']}")
    for child in tree['children']:
        _print_tree(child, indent + 1)


def insight_call_hotspots(ast_index: ASTIndex):
    """Find the most-called functions (architectural hotspots)."""
    section("INSIGHT 2: Call Hotspots (Most Called Functions)")

    print("Functions called from many places (potential API surface):\n")

    # Count incoming calls for each function
    call_counts = {}
    for callee, callers in ast_index.reverse_call_graph.items():
        call_counts[callee] = len(callers)

    # Sort by call count
    sorted_calls = sorted(call_counts.items(), key=lambda x: -x[1])[:20]

    print("  Function                              Called By")
    print("  " + "-" * 55)
    for func, count in sorted_calls:
        if count >= 3:  # Only show functions called 3+ times
            # Find the file it's defined in
            if func in ast_index.functions:
                file_info = ast_index.functions[func].file_path
                short_file = file_info.split("/")[-1] if "/" in file_info else file_info
            else:
                short_file = "?"
            print(f"  {func:40} {count:3} callers ({short_file})")


def insight_module_coupling(ast_index: ASTIndex):
    """Analyze coupling between modules through calls."""
    section("INSIGHT 3: Module Coupling (Cross-Module Calls)")

    print("Which modules call into which others:\n")

    # Build module-to-module call matrix
    module_calls = defaultdict(lambda: defaultdict(int))

    for func_name, func_info in ast_index.functions.items():
        caller_module = func_info.file_path.split("/")[0] if "/" in func_info.file_path else "root"

        for callee in func_info.calls:
            # Try to find which module the callee is in
            if callee in ast_index.functions:
                callee_info = ast_index.functions[callee]
                callee_module = callee_info.file_path.split("/")[0] if "/" in callee_info.file_path else "root"

                if caller_module != callee_module:  # Cross-module call
                    module_calls[caller_module][callee_module] += 1

    # Print as matrix
    modules = sorted(set(module_calls.keys()) | set(
        m for calls in module_calls.values() for m in calls.keys()
    ))

    if module_calls:
        print("  Caller       → Called Module (# of calls)")
        print("  " + "-" * 50)
        for caller in sorted(module_calls.keys()):
            targets = sorted(module_calls[caller].items(), key=lambda x: -x[1])[:3]
            target_str = ", ".join(f"{t}({c})" for t, c in targets)
            print(f"  {caller:12} → {target_str}")
    else:
        print("  No significant cross-module coupling detected")


def insight_semantic_vs_structural(processor: CorticalTextProcessor, ast_index: ASTIndex):
    """Compare semantic coupling with structural dependencies."""
    section("INSIGHT 4: Semantic vs Structural Coupling")

    print("""Comparing what the text suggests (semantic) vs what the code does (structural):

    ✓ CONFIRMED = Semantic coupling backed by actual calls
    ? SEMANTIC  = Concepts appear together but no direct calls
    ! STRUCTURAL = Direct calls but concepts not co-located in text
""")

    # Find semantic pairs (classes that share vocabulary)
    token_layer = processor.layers[CorticalLayer.TOKENS]

    # Map classes to their files
    class_files = {name: info.file_path for name, info in ast_index.classes.items()}

    # Find pairs of classes that appear in similar semantic contexts
    semantic_pairs = []
    classes = list(ast_index.classes.keys())

    for i, c1 in enumerate(classes[:30]):  # Limit for performance
        for c2 in classes[i+1:30]:
            f1 = class_files.get(c1)
            f2 = class_files.get(c2)
            if not f1 or not f2 or f1 == f2:
                continue

            # Check semantic overlap
            col1 = token_layer.get_by_id(f"L0_{c1.lower()}")
            col2 = token_layer.get_by_id(f"L0_{c2.lower()}")

            if col1 and col2:
                # They share document context
                shared_docs = col1.document_ids & col2.document_ids
                if len(shared_docs) >= 2:
                    semantic_pairs.append((c1, c2, len(shared_docs)))

    # Check structural relationships for these pairs
    print("  Class Pairs with Semantic Affinity:\n")
    for c1, c2, overlap in sorted(semantic_pairs, key=lambda x: -x[2])[:10]:
        # Check if c1 calls c2 or vice versa
        c1_methods = [f"{c1}.{m}" for m in ast_index.classes.get(c1, ClassStub()).methods]
        c2_methods = [f"{c2}.{m}" for m in ast_index.classes.get(c2, ClassStub()).methods]

        structural_link = False
        for m1 in c1_methods:
            if m1 in ast_index.functions:
                calls = ast_index.functions[m1].calls
                if any(c2.lower() in call.lower() for call in calls):
                    structural_link = True
                    break

        if not structural_link:
            for m2 in c2_methods:
                if m2 in ast_index.functions:
                    calls = ast_index.functions[m2].calls
                    if any(c1.lower() in call.lower() for call in calls):
                        structural_link = True
                        break

        status = "✓ CONFIRMED" if structural_link else "? SEMANTIC "
        print(f"  {status}  {c1} <-> {c2} (overlap: {overlap} docs)")


class ClassStub:
    """Stub for missing classes."""
    methods = []


def insight_hidden_apis(ast_index: ASTIndex):
    """Find functions that are called but not obviously part of the public API."""
    section("INSIGHT 5: Hidden APIs (Internal Functions with Wide Use)")

    print("Functions called widely but defined in implementation modules:\n")

    # Implementation files (not __init__, not top-level)
    impl_files = set()
    for func_info in ast_index.functions.values():
        if "__init__" not in func_info.file_path and "/" in func_info.file_path:
            impl_files.add(func_info.file_path)

    # Functions in impl files that are called from outside that file
    hidden_apis = []
    for func_name, func_info in ast_index.functions.items():
        if func_info.file_path in impl_files:
            # Check if called from other files
            callers = ast_index.reverse_call_graph.get(func_name, set())
            external_callers = [c for c in callers if c in ast_index.functions
                               and ast_index.functions[c].file_path != func_info.file_path]

            if len(external_callers) >= 2:
                hidden_apis.append((func_name, len(external_callers), func_info.file_path))

    hidden_apis.sort(key=lambda x: -x[1])

    print("  Function                         External Callers  Defined In")
    print("  " + "-" * 65)
    for func, callers, file_path in hidden_apis[:15]:
        short_file = file_path.split("/")[-1]
        print(f"  {func:35} {callers:5}           {short_file}")

    if hidden_apis:
        print("\n  Consider: Should these be promoted to explicit public APIs?")


def insight_unused_classes(ast_index: ASTIndex):
    """Find classes that are defined but never instantiated or inherited."""
    section("INSIGHT 6: Potentially Unused Classes")

    print("Classes with no callers and no children (may be entry points or unused):\n")

    used_classes = set()

    # Classes that are inherited from
    for children in ast_index.inheritance.values():
        used_classes.update(children)
    for parent in ast_index.inheritance.keys():
        used_classes.add(parent)

    # Classes whose methods are called
    for callee in ast_index.reverse_call_graph.keys():
        if "." in callee:
            class_name = callee.split(".")[0]
            used_classes.add(class_name)

    # Find unused
    all_classes = set(ast_index.classes.keys())
    potentially_unused = all_classes - used_classes

    for cls in sorted(potentially_unused)[:15]:
        info = ast_index.classes[cls]
        short_file = info.file_path.split("/")[-1]
        method_count = len(info.methods)
        print(f"  {cls:30} ({method_count} methods) - {short_file}")

    if potentially_unused:
        print("\n  Note: Some may be entry points, factories, or used via reflection")


def insight_complexity_indicators(ast_index: ASTIndex):
    """Find complexity hotspots through structural metrics."""
    section("INSIGHT 7: Complexity Indicators")

    print("Classes with many methods (potential god objects):\n")

    large_classes = [(name, len(info.methods), info.file_path)
                     for name, info in ast_index.classes.items()
                     if len(info.methods) >= 10]
    large_classes.sort(key=lambda x: -x[1])

    for name, method_count, file_path in large_classes[:10]:
        short_file = file_path.split("/")[-1]
        bar = "█" * (method_count // 2)
        print(f"  {name:30} {bar} ({method_count} methods)")

    print("\n\nFunctions with many outgoing calls (potential orchestrators):\n")

    heavy_callers = [(name, len(info.calls), info.file_path)
                     for name, info in ast_index.functions.items()
                     if len(info.calls) >= 10]
    heavy_callers.sort(key=lambda x: -x[1])

    for name, call_count, file_path in heavy_callers[:10]:
        short_file = file_path.split("/")[-1]
        bar = "█" * (call_count // 3)
        print(f"  {name:35} {bar} ({call_count} calls)")


def insight_decorator_patterns(ast_index: ASTIndex):
    """Analyze decorator usage patterns."""
    section("INSIGHT 8: Decorator Patterns (Cross-Cutting Concerns)")

    print("Decorators reveal cross-cutting concerns and aspects:\n")

    decorator_usage = defaultdict(list)
    for func_name, func_info in ast_index.functions.items():
        for dec in func_info.decorators:
            # Normalize decorator name
            dec_name = dec.split("(")[0]  # Remove arguments
            decorator_usage[dec_name].append((func_name, func_info.file_path))

    # Sort by frequency
    sorted_decorators = sorted(decorator_usage.items(), key=lambda x: -len(x[1]))

    print("  Decorator                   Usage  Sample Functions")
    print("  " + "-" * 60)
    for dec, usages in sorted_decorators[:15]:
        if len(usages) >= 2:  # Only show decorators used 2+ times
            sample = [u[0].split(".")[-1] for u in usages[:2]]
            print(f"  {dec:25} {len(usages):5}  {', '.join(sample)}...")


def insight_import_dependencies(ast_index: ASTIndex):
    """Analyze import patterns to find dependency structure."""
    section("INSIGHT 9: Import Dependency Analysis")

    print("Most imported modules (core dependencies):\n")

    # Count how many files import each module
    import_counts = defaultdict(int)
    for imp in ast_index.imports:
        import_counts[imp.module] += 1

    # Filter to internal modules
    internal_imports = {m: c for m, c in import_counts.items()
                       if "cortical" in m or not "." in m}

    sorted_imports = sorted(internal_imports.items(), key=lambda x: -x[1])[:15]

    for module, count in sorted_imports:
        bar = "█" * (count // 2)
        print(f"  {module:40} {bar} ({count} files)")


def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║      COGNITIVE DEMO v3: AST-Enhanced Code Intelligence              ║
║                                                                      ║
║  Combining semantic analysis with structural (AST) analysis         ║
╚══════════════════════════════════════════════════════════════════════╝
""")

    cortical_root = project_root / "cortical"
    processor, ast_index = load_both_indices(cortical_root)

    # Run insights
    insight_architecture_layers(ast_index)
    insight_call_hotspots(ast_index)
    insight_module_coupling(ast_index)
    insight_semantic_vs_structural(processor, ast_index)
    insight_hidden_apis(ast_index)
    insight_unused_classes(ast_index)
    insight_complexity_indicators(ast_index)
    insight_decorator_patterns(ast_index)
    insight_import_dependencies(ast_index)

    # Summary
    print("\n" + "═" * 70)
    print(" SYNTHESIS: Structural Insights")
    print("═" * 70 + "\n")

    stats = ast_index.get_stats()
    print(f"""
Structural Analysis Summary:
  • {stats['classes']} classes across {stats['files']} files
  • {stats['functions']} functions/methods
  • {stats['call_edges']} call relationships
  • {stats['inheritance_edges']} inheritance relationships

Key Findings:

1. LAYERED ARCHITECTURE: Clear inheritance hierarchies show
   intentional abstraction layers (Mixin pattern prominent)

2. CALL HOTSPOTS: Certain functions are called from many places,
   indicating stable APIs that many components depend on

3. MODULE COUPLING: Cross-module calls reveal actual (not just
   intended) dependencies between subsystems

4. SEMANTIC-STRUCTURAL CORRELATION: When concepts appear together
   in text AND have call relationships, the coupling is real

5. HIDDEN APIs: Functions widely used but buried in implementation
   files may deserve promotion to explicit public APIs

6. COMPLEXITY HOTSPOTS: Large classes and functions with many calls
   are candidates for refactoring

The AST analysis complements semantic analysis by showing actual
code structure vs. conceptual proximity.
""")

    return processor, ast_index


if __name__ == "__main__":
    processor, ast_index = main()
