#!/usr/bin/env python3
"""
Thought Chain - Unified Schema for Cognitive Pipeline
======================================================

Provides a standardized JSON schema for preserving thought context
across pipeline stages, enabling loop-based reanalysis and middleware insertion.

The ThoughtChain schema:
{
    "chain_id": "unique-uuid",
    "iteration": 0,
    "created_at": "ISO timestamp",
    "updated_at": "ISO timestamp",
    "stages": ["stage1", "stage2", ...],
    "current_stage": "stage_name",
    "context": {
        "query": "original query",
        "focus_terms": ["term1", "term2"],
        "depth": 3,
        "exploration_mode": "breadth|depth|hybrid"
    },
    "results": {
        "stage1": {...},
        "stage2": {...}
    },
    "insights": [
        {"iteration": 0, "stage": "stage1", "insight": "..."},
        ...
    ],
    "parameters": {
        "top_k": 10,
        "min_confidence": 0.5,
        ...
    },
    "hooks": {
        "pre_stage": "script_path",
        "post_stage": "script_path"
    }
}

Usage:
    from thought_chain import ThoughtChain, ChainContext

    # Create new chain
    chain = ThoughtChain.create(query="prediction models", depth=3)

    # Add stage results
    chain.add_result("world_model_analysis", {...})

    # Iterate for reanalysis
    chain.next_iteration()

    # Export for piping
    print(chain.to_json())
"""

import json
import uuid
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field, asdict
from enum import Enum


class ExplorationMode(Enum):
    """Exploration strategy for knowledge graph traversal."""
    BREADTH = "breadth"   # Wide exploration, many shallow paths
    DEPTH = "depth"       # Deep exploration, fewer deep paths
    HYBRID = "hybrid"     # Balanced approach
    FOCUSED = "focused"   # Narrow focus on specific concepts


@dataclass
class ChainContext:
    """Context for the current thought chain exploration."""
    query: str = ""
    focus_terms: List[str] = field(default_factory=list)
    depth: int = 3
    exploration_mode: str = "hybrid"
    max_iterations: int = 5
    convergence_threshold: float = 0.1
    include_weak_links: bool = True
    bridge_priority: str = "strength"  # strength, novelty, coverage

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChainContext':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ChainParameters:
    """Configurable parameters for pipeline stages."""
    # Query expansion
    top_k: int = 10
    expand_terms: bool = True
    max_expansions: int = 20

    # Analysis
    min_confidence: float = 0.3
    cluster_threshold: float = 0.5
    pattern_types: List[str] = field(default_factory=lambda: ["hub", "bridge", "cluster"])

    # Bridge detection
    min_gap_distance: int = 2
    weak_link_threshold: float = 0.2
    max_bridges: int = 50

    # LLM generation
    template: str = "synthesis"
    max_tokens: int = 1000
    model: str = "claude-3-haiku-20240307"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChainParameters':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def update(self, **kwargs) -> 'ChainParameters':
        """Return new ChainParameters with updated values."""
        current = self.to_dict()
        current.update(kwargs)
        return ChainParameters.from_dict(current)


@dataclass
class Insight:
    """An insight discovered during analysis."""
    iteration: int
    stage: str
    insight: str
    confidence: float = 0.0
    related_terms: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Insight':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class ThoughtChain:
    """
    Manages the state of a cognitive exploration pipeline.

    Preserves context across stages, supports iteration for reanalysis,
    and provides hooks for middleware insertion.
    """

    # Standard pipeline stages
    STANDARD_STAGES = [
        "world_model_analysis",
        "question_connection",
        "knowledge_analysis",
        "knowledge_bridge",
        "llm_generate_response"
    ]

    def __init__(
        self,
        chain_id: Optional[str] = None,
        context: Optional[ChainContext] = None,
        parameters: Optional[ChainParameters] = None
    ):
        self.chain_id = chain_id or str(uuid.uuid4())[:8]
        self.iteration = 0
        self.created_at = datetime.utcnow().isoformat()
        self.updated_at = self.created_at

        self.stages: List[str] = []
        self.current_stage: Optional[str] = None

        self.context = context or ChainContext()
        self.parameters = parameters or ChainParameters()

        self.results: Dict[str, Any] = {}
        self.insights: List[Insight] = []

        self.hooks: Dict[str, str] = {}

        # Track convergence for loop termination
        self._previous_key_terms: Set[str] = set()
        self._convergence_scores: List[float] = []

    @classmethod
    def create(
        cls,
        query: str = "",
        depth: int = 3,
        exploration_mode: str = "hybrid",
        **kwargs
    ) -> 'ThoughtChain':
        """Create a new thought chain with initial context."""
        context = ChainContext(
            query=query,
            depth=depth,
            exploration_mode=exploration_mode,
            **{k: v for k, v in kwargs.items() if hasattr(ChainContext, k)}
        )
        params = ChainParameters(
            **{k: v for k, v in kwargs.items() if hasattr(ChainParameters, k)}
        )
        return cls(context=context, parameters=params)

    @classmethod
    def from_json(cls, json_str: str) -> 'ThoughtChain':
        """Load chain from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ThoughtChain':
        """Load chain from dictionary."""
        chain = cls()

        chain.chain_id = data.get("chain_id", chain.chain_id)
        chain.iteration = data.get("iteration", 0)
        chain.created_at = data.get("created_at", chain.created_at)
        chain.updated_at = data.get("updated_at", chain.updated_at)

        chain.stages = data.get("stages", [])
        chain.current_stage = data.get("current_stage")

        if "context" in data:
            chain.context = ChainContext.from_dict(data["context"])
        if "parameters" in data:
            chain.parameters = ChainParameters.from_dict(data["parameters"])

        chain.results = data.get("results", {})
        chain.insights = [
            Insight.from_dict(i) for i in data.get("insights", [])
        ]
        chain.hooks = data.get("hooks", {})

        return chain

    @classmethod
    def from_stdin(cls) -> 'ThoughtChain':
        """Load chain from stdin, auto-detecting format."""
        if sys.stdin.isatty():
            return cls.create()

        try:
            data = json.load(sys.stdin)
            # Check if this is already a thought chain
            if "chain_id" in data and "context" in data:
                return cls.from_dict(data)
            else:
                # Wrap existing analysis output in a new chain
                chain = cls.create()
                chain.add_result("input", data)
                return chain
        except json.JSONDecodeError:
            return cls.create()

    def to_dict(self) -> Dict[str, Any]:
        """Export chain as dictionary."""
        return {
            "chain_id": self.chain_id,
            "iteration": self.iteration,
            "created_at": self.created_at,
            "updated_at": datetime.utcnow().isoformat(),
            "stages": self.stages,
            "current_stage": self.current_stage,
            "context": self.context.to_dict(),
            "parameters": self.parameters.to_dict(),
            "results": self.results,
            "insights": [i.to_dict() for i in self.insights],
            "hooks": self.hooks
        }

    def to_json(self, indent: int = 2) -> str:
        """Export chain as JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def add_result(self, stage: str, result: Dict[str, Any]) -> None:
        """Add results from a pipeline stage."""
        if stage not in self.stages:
            self.stages.append(stage)

        # Store with iteration key for history
        key = f"{stage}_iter{self.iteration}"
        self.results[key] = result

        # Also store as current result
        self.results[stage] = result

        self.current_stage = stage
        self.updated_at = datetime.utcnow().isoformat()

        # Extract key terms for convergence tracking
        self._extract_key_terms(result)

    def add_insight(
        self,
        insight: str,
        stage: Optional[str] = None,
        confidence: float = 0.5,
        related_terms: Optional[List[str]] = None
    ) -> None:
        """Add an insight discovered during analysis."""
        self.insights.append(Insight(
            iteration=self.iteration,
            stage=stage or self.current_stage or "unknown",
            insight=insight,
            confidence=confidence,
            related_terms=related_terms or []
        ))

    def get_result(self, stage: str, iteration: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Get results from a specific stage and iteration."""
        if iteration is not None:
            key = f"{stage}_iter{iteration}"
            return self.results.get(key)
        return self.results.get(stage)

    def next_iteration(self) -> int:
        """Advance to next iteration for reanalysis."""
        self.iteration += 1
        self._calculate_convergence()
        return self.iteration

    def should_continue(self) -> bool:
        """Check if we should continue iterating."""
        if self.iteration >= self.context.max_iterations:
            return False

        if len(self._convergence_scores) >= 2:
            recent_change = abs(
                self._convergence_scores[-1] - self._convergence_scores[-2]
            )
            if recent_change < self.context.convergence_threshold:
                return False

        return True

    def set_hook(self, hook_type: str, script_path: str) -> None:
        """Set a hook script to run at specific points."""
        valid_hooks = ["pre_stage", "post_stage", "pre_iteration", "post_iteration"]
        if hook_type not in valid_hooks:
            raise ValueError(f"Invalid hook type. Must be one of: {valid_hooks}")
        self.hooks[hook_type] = script_path

    def get_focus_terms(self) -> List[str]:
        """Get current focus terms based on context and results."""
        terms = list(self.context.focus_terms)

        # Add terms from query
        if self.context.query:
            terms.extend(self.context.query.lower().split())

        # Add high-importance terms from results
        if "world_model_analysis" in self.results:
            wm = self.results["world_model_analysis"]
            if "concepts" in wm:
                for concept in wm["concepts"][:10]:
                    if isinstance(concept, dict):
                        terms.append(concept.get("term", ""))

        return list(set(t for t in terms if t))

    def get_iteration_summary(self) -> Dict[str, Any]:
        """Get summary of current iteration state."""
        return {
            "chain_id": self.chain_id,
            "iteration": self.iteration,
            "stages_completed": len(self.stages),
            "insights_count": len(self.insights),
            "convergence_score": self._convergence_scores[-1] if self._convergence_scores else None,
            "should_continue": self.should_continue(),
            "focus_terms": self.get_focus_terms()[:10]
        }

    def _extract_key_terms(self, result: Dict[str, Any]) -> None:
        """Extract key terms from results for convergence tracking."""
        terms = set()

        def extract_terms(obj, depth=0):
            if depth > 3:
                return
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k in ["term", "name", "concept"]:
                        if isinstance(v, str):
                            terms.add(v.lower())
                    else:
                        extract_terms(v, depth + 1)
            elif isinstance(obj, list):
                for item in obj[:20]:  # Limit to avoid explosion
                    extract_terms(item, depth + 1)

        extract_terms(result)
        self._previous_key_terms = terms

    def _calculate_convergence(self) -> float:
        """Calculate convergence score based on term stability."""
        if not self._previous_key_terms:
            self._convergence_scores.append(0.0)
            return 0.0

        # Get current terms from latest result
        current_terms = set()
        for stage_result in self.results.values():
            if isinstance(stage_result, dict):
                def extract(obj, depth=0):
                    if depth > 2:
                        return
                    if isinstance(obj, dict):
                        for k, v in obj.items():
                            if k in ["term", "name", "concept"] and isinstance(v, str):
                                current_terms.add(v.lower())
                            else:
                                extract(v, depth + 1)
                    elif isinstance(obj, list):
                        for item in obj[:10]:
                            extract(item, depth + 1)
                extract(stage_result)

        if not current_terms:
            self._convergence_scores.append(0.0)
            return 0.0

        # Jaccard similarity as convergence measure
        intersection = len(current_terms & self._previous_key_terms)
        union = len(current_terms | self._previous_key_terms)

        score = intersection / union if union > 0 else 0.0
        self._convergence_scores.append(score)

        return score


def wrap_existing_output(data: Dict[str, Any], stage: str = "input") -> Dict[str, Any]:
    """
    Wrap existing pipeline output in thought chain format.

    This allows existing scripts to be chain-aware without modification.
    """
    chain = ThoughtChain.create()
    chain.add_result(stage, data)
    return chain.to_dict()


def extract_from_chain(chain_data: Dict[str, Any], stage: str) -> Dict[str, Any]:
    """
    Extract stage results from chain format.

    This allows chain-aware scripts to read from chain format.
    """
    if "results" in chain_data and stage in chain_data["results"]:
        return chain_data["results"][stage]
    return chain_data


def is_chain_format(data: Dict[str, Any]) -> bool:
    """Check if data is in thought chain format."""
    return "chain_id" in data and "context" in data and "results" in data


# CLI for testing and manual chain manipulation
def main():
    """CLI for thought chain manipulation."""
    import argparse

    parser = argparse.ArgumentParser(description="Thought Chain Management")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Create command
    create_parser = subparsers.add_parser("create", help="Create new chain")
    create_parser.add_argument("--query", "-q", help="Initial query")
    create_parser.add_argument("--depth", "-d", type=int, default=3, help="Exploration depth")
    create_parser.add_argument("--mode", "-m", default="hybrid",
                               choices=["breadth", "depth", "hybrid", "focused"])

    # Wrap command
    wrap_parser = subparsers.add_parser("wrap", help="Wrap existing JSON in chain")
    wrap_parser.add_argument("--input", "-i", help="Input file (default: stdin)")
    wrap_parser.add_argument("--stage", "-s", default="input", help="Stage name")

    # Extract command
    extract_parser = subparsers.add_parser("extract", help="Extract stage from chain")
    extract_parser.add_argument("--input", "-i", help="Input file (default: stdin)")
    extract_parser.add_argument("--stage", "-s", required=True, help="Stage to extract")

    # Summary command
    summary_parser = subparsers.add_parser("summary", help="Show chain summary")
    summary_parser.add_argument("--input", "-i", help="Input file (default: stdin)")

    args = parser.parse_args()

    if args.command == "create":
        chain = ThoughtChain.create(
            query=args.query or "",
            depth=args.depth,
            exploration_mode=args.mode
        )
        print(chain.to_json())

    elif args.command == "wrap":
        if args.input:
            with open(args.input) as f:
                data = json.load(f)
        else:
            data = json.load(sys.stdin)

        wrapped = wrap_existing_output(data, args.stage)
        print(json.dumps(wrapped, indent=2))

    elif args.command == "extract":
        if args.input:
            with open(args.input) as f:
                data = json.load(f)
        else:
            data = json.load(sys.stdin)

        extracted = extract_from_chain(data, args.stage)
        print(json.dumps(extracted, indent=2))

    elif args.command == "summary":
        if args.input:
            with open(args.input) as f:
                data = json.load(f)
        else:
            data = json.load(sys.stdin)

        chain = ThoughtChain.from_dict(data)
        print(json.dumps(chain.get_iteration_summary(), indent=2))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
