"""
Audit Executor: PLN-based audit reasoning.

Wraps AuditReasoner to execute audit queries:
- "risky files in cortical/" -> list files with risk scores
- "why is prism_pln.py flagged" -> explain inference chain
- "files with high_churn" -> filter by trait
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from .protocol import BaseExecutor, ExecutionResult

# Import audit infrastructure
from cortical.audits.reasoning import (
    AuditQuery,
    AuditReasoner,
    translate_audit_query,
)


class AuditExecutor(BaseExecutor):
    """
    Executes audit queries using PLN-based reasoning.

    Capabilities:
    - List risky files with scores
    - Explain why files are flagged
    - Filter by traits (high_churn, todo, etc.)
    - Track file importance over time
    """

    def __init__(
        self,
        reasoner: Optional[AuditReasoner] = None,
        scan_directory: Optional[Path] = None,
    ):
        """
        Initialize audit executor.

        Args:
            reasoner: Existing AuditReasoner (created if not provided)
            scan_directory: Directory to scan for audit data
        """
        self._reasoner = reasoner
        self._scan_directory = scan_directory or Path(".")
        self._initialized = False

    @property
    def name(self) -> str:
        return "audit"

    @property
    def reasoner(self) -> AuditReasoner:
        """Get or create the audit reasoner."""
        if self._reasoner is None:
            self._reasoner = AuditReasoner(use_persistence=True)
            self._reasoner.add_default_rules()
            self._reasoner.load_rules_from_woven_mind()
        return self._reasoner

    def execute(self, query: AuditQuery) -> ExecutionResult:
        """
        Execute an audit query.

        Args:
            query: AuditQuery from translate_audit_query()

        Returns:
            ExecutionResult with risky files or explanation
        """
        # Handle explain intent
        if query.intent == "explain" and query.target_file:
            return self._execute_explain(query)

        # Handle list/filter intent
        return self._execute_list(query)

    def _execute_explain(self, query: AuditQuery) -> ExecutionResult:
        """Execute an explanation query."""
        file_id = query.target_file
        if not file_id:
            return ExecutionResult(
                items=[],
                confidence=0.0,
                source=self.name,
                explanation="No file specified for explanation.",
            )

        explanation = self.reasoner.explain_file_risk(file_id)

        return ExecutionResult(
            items=[explanation],
            confidence=0.8 if explanation.get("risk_level") else 0.3,
            source=self.name,
            explanation=explanation.get("summary", ""),
            metadata={
                "intent": "explain",
                "file_id": file_id,
                "facts_count": len(explanation.get("facts", [])),
                "suggestions_count": len(explanation.get("suggestions", [])),
            }
        )

    def _execute_list(self, query: AuditQuery) -> ExecutionResult:
        """Execute a list/filter query."""
        # Get priority files (already sorted by risk + importance)
        max_results = query.max_results or 20
        priority_files = self.reasoner.get_priority_files(top_n=max_results * 2)

        # Check if we have any data
        if not priority_files and not self.reasoner.file_importance:
            return ExecutionResult(
                items=[],
                confidence=0.1,
                source=self.name,
                explanation="No audit data loaded. Run 'audit analyze' first to scan files.",
                metadata={"intent": "list", "error": "no_data"}
            )

        # Filter by min_risk if specified
        if query.min_risk > 0:
            priority_files = [
                (f, score) for f, score in priority_files
                if score >= query.min_risk
            ]

        # Apply directory filter if specified
        if query.directory:
            dir_name = query.directory.rstrip("/")
            # Filter files that match directory (in facts or by name)
            priority_files = [
                (f, score) for f, score in priority_files
                if dir_name in f or self._file_in_directory(f, dir_name)
            ]

        # Apply trait filters - files must have ALL specified traits
        if query.include_traits:
            priority_files = [
                (f, score) for f, score in priority_files
                if self._file_has_traits(f, query.include_traits)
            ]

        # Apply negations
        for negation in query.negations:
            priority_files = [
                (f, score) for f, score in priority_files
                if negation.lower() not in f.lower()
            ]

        # Format results
        items = []
        for file_id, score in priority_files[:max_results]:
            risk_info = self.reasoner.query_file_risk(file_id)
            items.append({
                "file": file_id,
                "risk_score": score,
                "details": risk_info,
            })

        explanation = None
        if items:
            avg_risk = sum(item["risk_score"] for item in items) / len(items)
            explanation = f"Found {len(items)} files with average risk score {avg_risk:.2f}"
        else:
            explanation = "No files matched the query criteria."

        return ExecutionResult(
            items=items,
            confidence=0.8 if items else 0.3,
            source=self.name,
            explanation=explanation,
            metadata={
                "intent": "list",
                "directory": query.directory,
                "min_risk": query.min_risk,
                "negations": query.negations,
                "traits_filter": query.include_traits,
            }
        )

    def _file_in_directory(self, file_id: str, dir_name: str) -> bool:
        """Check if a file_id is associated with a directory."""
        # File IDs are normalized (e.g., prism_pln_py)
        # Check if the directory name appears in facts
        for atom_name in self.reasoner.pln.graph._atoms.keys():
            if file_id in atom_name and f"has_dir({file_id}, {dir_name})" in atom_name:
                return True
        return False

    def _file_has_traits(self, file_id: str, traits: List[str]) -> bool:
        """Check if a file has all specified traits."""
        # Check PLN graph for has_trait facts
        for trait in traits:
            trait_fact = f"has_trait({file_id}, {trait})"
            found = False
            for atom_name in self.reasoner.pln.graph._atoms.keys():
                if trait_fact in atom_name:
                    found = True
                    break
            if not found:
                return False
        return True

    def format_result(self, result: ExecutionResult) -> str:
        """Format audit results as natural language."""
        if result.is_empty:
            return result.explanation or "No risky files found."

        intent = result.metadata.get("intent", "list")

        if intent == "explain":
            return self._format_explanation(result)
        else:
            return self._format_list(result)

    def _format_explanation(self, result: ExecutionResult) -> str:
        """Format an explanation result."""
        if not result.items:
            return "No explanation available."

        explanation = result.items[0]
        lines = [f"Risk Analysis for {explanation.get('file_id', 'unknown')}:"]

        # Risk level
        risk_level = explanation.get("risk_level")
        if risk_level:
            lines.append(f"  Risk: {risk_level.get('mean', 0):.2f} "
                        f"(strength={risk_level.get('strength', 0):.2f}, "
                        f"confidence={risk_level.get('confidence', 0):.2f})")

        # Facts
        facts = explanation.get("facts", [])
        if facts:
            lines.append(f"\n  Evidence ({len(facts)} facts):")
            for fact in facts[:5]:
                lines.append(f"    - {fact.get('atom', '')}")

        # Suggestions
        suggestions = explanation.get("suggestions", [])
        if suggestions:
            lines.append("\n  Recommendations:")
            for suggestion in suggestions:
                lines.append(f"    - {suggestion}")

        return "\n".join(lines)

    def _format_list(self, result: ExecutionResult) -> str:
        """Format a list result."""
        lines = []

        if result.explanation:
            lines.append(result.explanation)
            lines.append("")

        for i, item in enumerate(result.items[:10], 1):
            file_id = item.get("file", "unknown")
            score = item.get("risk_score", 0)
            lines.append(f"  {i}. {file_id} (risk: {score:.2f})")

        if len(result.items) > 10:
            lines.append(f"  ... and {len(result.items) - 10} more files")

        return "\n".join(lines)
