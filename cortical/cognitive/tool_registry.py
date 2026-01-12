"""
Tool Registry for Natural Language Query System.

Allows registering and retrieving tools that can be used to answer questions.
Extensible: add new capabilities by registering tools, not modifying core code.

Usage:
    registry = ToolRegistry()
    registry.register("callers_of", my_handler, "Find function callers", category="cognitive")
    handler = registry.get("callers_of")
    results = handler("my_function")
"""

from dataclasses import dataclass, field
from typing import Callable, List, Dict, Any, Optional


@dataclass
class Tool:
    """A registered tool."""
    name: str
    handler: Callable[[str], List[Any]]
    description: str
    category: str = "general"


class ToolRegistry:
    """
    Registry of tools the query system can use.

    Tools are functions that take a target string and return results.
    They can be organized by category for different use cases.

    Categories:
        - cognitive: Word associations, code queries (built-in)
        - cdg: CDG entity queries (future)
        - got: GoT task/sprint queries (future)
        - custom: User-defined tools
    """

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(
        self,
        name: str,
        handler: Callable[[str], List[Any]],
        description: str,
        category: str = "general"
    ) -> None:
        """
        Register a tool.

        Args:
            name: Tool identifier (e.g., "callers_of")
            handler: Function to execute. Takes target string, returns list of results.
            description: What this tool does (used for intent matching)
            category: Tool category (cognitive, cdg, got, custom)
        """
        self._tools[name] = Tool(
            name=name,
            handler=handler,
            description=description,
            category=category
        )

    def get(self, name: str) -> Optional[Callable[[str], List[Any]]]:
        """
        Get a registered tool's handler by name.

        Args:
            name: Tool identifier

        Returns:
            Handler function, or None if not found
        """
        tool = self._tools.get(name)
        return tool.handler if tool else None

    def has(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get the full Tool object by name."""
        return self._tools.get(name)

    def find_by_category(self, category: str) -> List[Tool]:
        """
        Get all tools in a category.

        Args:
            category: Category name

        Returns:
            List of Tools in that category
        """
        return [t for t in self._tools.values() if t.category == category]

    def list_all(self) -> List[Tool]:
        """Get all registered tools."""
        return list(self._tools.values())

    def match_intent(self, intent: str) -> List[Tool]:
        """
        Find tools that might match an intent description.

        Simple keyword matching - can be enhanced later.

        Args:
            intent: Description of what user wants

        Returns:
            List of potentially matching tools
        """
        intent_lower = intent.lower()
        matches = []

        for tool in self._tools.values():
            # Check if any word in intent matches tool name or description
            if tool.name in intent_lower:
                matches.append(tool)
            elif any(word in tool.description.lower() for word in intent_lower.split()):
                matches.append(tool)

        return matches

    def __repr__(self) -> str:
        return f"ToolRegistry({len(self._tools)} tools)"
