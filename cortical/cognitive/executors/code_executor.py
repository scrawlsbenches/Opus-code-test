"""
Code Executor: Code structure queries via CodeBridge.

Handles queries about code structure:
- "who calls the validate function" -> callers_of
- "what are the subclasses of BaseClass" -> subclasses_of
- "what methods does X have" -> methods_of
- "what's defined in file.py" -> defined_in
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from .protocol import BaseExecutor, ExecutionResult


class CodeExecutor(BaseExecutor):
    """
    Executes code structure queries using CodeBridge.

    Capabilities:
    - Find callers of a function
    - Find subclasses of a class
    - Find methods of a class
    - Find entities defined in a file
    - Bridge between vocabulary and code (code_for_word, words_for_code)
    """

    def __init__(self, code_bridge: Optional[Any] = None):
        """
        Initialize code executor.

        Args:
            code_bridge: Existing CodeBridge instance (lazy-loaded if not provided)
        """
        self._code_bridge = code_bridge
        self._initialized = False

    @property
    def name(self) -> str:
        return "code"

    @property
    def code_bridge(self) -> Any:
        """Get or create the code bridge."""
        if self._code_bridge is None:
            # Lazy load - try to get from cognitive agent
            try:
                from cortical.cognitive.graph import CognitiveGraph
                from cortical.cognitive.code_bridge import CodeBridge

                graph = CognitiveGraph()
                self._code_bridge = CodeBridge(graph)
            except ImportError:
                self._code_bridge = None
        return self._code_bridge

    def execute(self, query: Dict[str, Any]) -> ExecutionResult:
        """
        Execute a code structure query.

        Args:
            query: Dict with action, subject, intent, question_word

        Returns:
            ExecutionResult with code entities
        """
        if self.code_bridge is None:
            return ExecutionResult(
                items=[],
                confidence=0.0,
                source=self.name,
                explanation="CodeBridge not available. Index codebase first.",
            )

        action = query.get("action", "")
        subject = query.get("subject", "")
        intent = query.get("intent", "")
        question_word = query.get("question_word", "")

        # Route to appropriate query method
        if action == "call" or "call" in str(query).lower():
            return self._execute_callers_query(subject)

        if action == "implement" or "subclass" in str(query).lower():
            return self._execute_subclasses_query(subject)

        if "method" in str(query).lower():
            return self._execute_methods_query(subject)

        if intent == "location" or "defined" in str(query).lower():
            return self._execute_defined_in_query(subject)

        # Default: try to find related code entities
        return self._execute_general_code_query(query)

    def _execute_callers_query(self, function_name: str) -> ExecutionResult:
        """Find callers of a function."""
        if not function_name:
            return ExecutionResult(
                items=[],
                confidence=0.0,
                source=self.name,
                explanation="No function name specified.",
            )

        callers = self.code_bridge.query_callers_of(function_name)

        items = []
        for caller in callers:
            items.append({
                "name": caller.name,
                "type": "function",
                "file_path": caller.metadata.get("file_path", ""),
                "lineno": caller.metadata.get("lineno", 0),
            })

        return ExecutionResult(
            items=items,
            confidence=0.8 if items else 0.3,
            source=self.name,
            explanation=f"Found {len(items)} callers of '{function_name}'",
            metadata={"query_type": "callers_of", "target": function_name}
        )

    def _execute_subclasses_query(self, class_name: str) -> ExecutionResult:
        """Find subclasses of a class."""
        if not class_name:
            return ExecutionResult(
                items=[],
                confidence=0.0,
                source=self.name,
                explanation="No class name specified.",
            )

        subclasses = self.code_bridge.query_subclasses_of(class_name)

        items = []
        for subclass in subclasses:
            items.append({
                "name": subclass.name,
                "type": "class",
                "file_path": subclass.metadata.get("file_path", ""),
                "lineno": subclass.metadata.get("lineno", 0),
                "bases": subclass.metadata.get("bases", []),
            })

        return ExecutionResult(
            items=items,
            confidence=0.8 if items else 0.3,
            source=self.name,
            explanation=f"Found {len(items)} subclasses of '{class_name}'",
            metadata={"query_type": "subclasses_of", "target": class_name}
        )

    def _execute_methods_query(self, class_name: str) -> ExecutionResult:
        """Find methods of a class."""
        if not class_name:
            return ExecutionResult(
                items=[],
                confidence=0.0,
                source=self.name,
                explanation="No class name specified.",
            )

        methods = self.code_bridge.query_methods_of(class_name)

        items = []
        for method in methods:
            items.append({
                "name": method.name,
                "type": "method",
                "args": method.metadata.get("args", []),
                "lineno": method.metadata.get("lineno", 0),
                "docstring": method.metadata.get("docstring", ""),
            })

        return ExecutionResult(
            items=items,
            confidence=0.8 if items else 0.3,
            source=self.name,
            explanation=f"Found {len(items)} methods in '{class_name}'",
            metadata={"query_type": "methods_of", "target": class_name}
        )

    def _execute_defined_in_query(self, file_path: str) -> ExecutionResult:
        """Find entities defined in a file."""
        if not file_path:
            return ExecutionResult(
                items=[],
                confidence=0.0,
                source=self.name,
                explanation="No file path specified.",
            )

        entities = self.code_bridge.query_defined_in(file_path)

        items = []
        for entity in entities:
            items.append({
                "name": entity.name,
                "type": entity.atom_type.name.lower(),
                "lineno": entity.metadata.get("lineno", 0),
            })

        return ExecutionResult(
            items=items,
            confidence=0.8 if items else 0.3,
            source=self.name,
            explanation=f"Found {len(items)} entities defined in '{file_path}'",
            metadata={"query_type": "defined_in", "target": file_path}
        )

    def _execute_general_code_query(self, query: Dict[str, Any]) -> ExecutionResult:
        """Execute a general code query by trying to find relevant entities."""
        subject = query.get("subject", "")
        action = query.get("action", "")

        # Try code_for_word if we have a subject
        if subject and hasattr(self.code_bridge, "query_code_for_word"):
            code_entities = self.code_bridge.query_code_for_word(subject)

            if code_entities:
                items = []
                for entity in code_entities:
                    items.append({
                        "name": entity.name,
                        "type": entity.atom_type.name.lower(),
                        "file_path": entity.metadata.get("file_path", ""),
                    })

                return ExecutionResult(
                    items=items,
                    confidence=0.6,
                    source=self.name,
                    explanation=f"Found {len(items)} code entities related to '{subject}'",
                    metadata={"query_type": "code_for_word", "target": subject}
                )

        return ExecutionResult(
            items=[],
            confidence=0.2,
            source=self.name,
            explanation=f"No code entities found for query about '{subject or action}'",
        )

    def format_result(self, result: ExecutionResult) -> str:
        """Format code query results as natural language."""
        if result.is_empty:
            return result.explanation or "No code entities found."

        lines = []
        if result.explanation:
            lines.append(result.explanation)
            lines.append("")

        for i, item in enumerate(result.items[:10], 1):
            name = item.get("name", "unknown")
            item_type = item.get("type", "entity")
            file_path = item.get("file_path", "")
            lineno = item.get("lineno", 0)

            location = ""
            if file_path:
                location = f" ({file_path}"
                if lineno:
                    location += f":{lineno}"
                location += ")"

            lines.append(f"  {i}. {name} [{item_type}]{location}")

            # Add extra details for certain types
            if item_type == "method":
                args = item.get("args", [])
                if args:
                    lines.append(f"      Args: {', '.join(args)}")

        if len(result.items) > 10:
            lines.append(f"  ... and {len(result.items) - 10} more")

        return "\n".join(lines)
