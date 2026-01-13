"""
Natural Language Query Interface for CognitiveAgent.

Takes natural language questions and generates complete answers
using trained knowledge and code structure.

Supports two modes:
- Legacy: Uses word associations and code bridge (original behavior)
- Unified: Uses QueryRouter → Executors → Aggregator → Formatter pipeline

Usage:
    from cortical.cognitive.nl_query import NLQuery

    # Legacy mode (default)
    nl = NLQuery(agent)
    response = nl.ask("How does code indexing work?")

    # Unified pipeline mode
    nl = NLQuery(agent, use_unified=True)
    response = nl.ask("risky files in cortical/")
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, TYPE_CHECKING

from cortical.cognitive.tool_registry import ToolRegistry

if TYPE_CHECKING:
    from cortical.cognitive.graph import CognitiveAgent, Atom


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class QueryIntent:
    """Parsed intent from a natural language question."""
    question_type: str  # how, what, where, who, why
    concepts: List[str]  # extracted key concepts
    query_strategy: List[str]  # which tools/queries to use
    raw_question: str = ""


@dataclass
class GatheredKnowledge:
    """Knowledge collected from various sources."""
    associations: List[str] = field(default_factory=list)
    code_entities: List["Atom"] = field(default_factory=list)
    related_files: List[str] = field(default_factory=list)
    methods: List["Atom"] = field(default_factory=list)
    callers: List["Atom"] = field(default_factory=list)


# =============================================================================
# Stop Words (filter these from concepts)
# =============================================================================

STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare",
    "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
    "from", "as", "into", "through", "during", "before", "after", "above",
    "below", "between", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "each", "few",
    "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "just", "and", "but",
    "if", "or", "because", "until", "while", "this", "that", "these",
    "those", "what", "which", "who", "whom", "it", "its", "i", "me", "my",
    "work", "works", "working", "does", "you", "your", "we", "our", "they",
    "their", "he", "she", "him", "her", "py", "txt", "md", "json", "also",
    "use", "using", "get", "set", "new", "like", "see", "one", "two", "first"
}


# =============================================================================
# NLQuery Class
# =============================================================================

class NLQuery:
    """
    Natural language query interface.

    Parses questions, gathers knowledge, generates responses.

    Supports two modes:
    - Legacy (default): Uses word associations and code bridge
    - Unified: Uses QueryRouter → Executors → Aggregator → Formatter
    """

    def __init__(
        self,
        agent: "CognitiveAgent",
        registry: Optional[ToolRegistry] = None,
        use_unified: bool = False,
    ):
        """
        Initialize NLQuery.

        Args:
            agent: CognitiveAgent with trained knowledge
            registry: Optional custom ToolRegistry (creates default if None)
            use_unified: If True, use unified query pipeline (Phase 5)
        """
        self.agent = agent
        self.registry = registry or self._create_default_registry()
        self.use_unified = use_unified

        # Initialize unified pipeline components lazily
        self._router = None
        self._executors = None
        self._aggregator = None
        self._formatter = None

    def _create_default_registry(self) -> ToolRegistry:
        """Create registry with default cognitive tools."""
        registry = ToolRegistry()

        # Register cognitive tools that wrap agent.query()
        registry.register(
            "similar_to",
            lambda target: self.agent.query("similar_to", target),
            "Find semantically related words",
            category="cognitive"
        )
        registry.register(
            "code_for_word",
            lambda target: self.agent.query("code_for_word", target),
            "Find code entities matching word",
            category="cognitive"
        )
        registry.register(
            "callers_of",
            lambda target: self.agent.query("callers_of", target),
            "Find functions that call target",
            category="cognitive"
        )
        registry.register(
            "methods_of",
            lambda target: self.agent.query("methods_of", target),
            "Find methods of a class",
            category="cognitive"
        )
        registry.register(
            "defined_in",
            lambda target: self.agent.query("defined_in", target),
            "Find entities defined in file",
            category="cognitive"
        )
        registry.register(
            "subclasses_of",
            lambda target: self.agent.query("subclasses_of", target),
            "Find classes that inherit from target",
            category="cognitive"
        )

        return registry

    # =========================================================================
    # Intent Parser
    # =========================================================================

    def parse_intent(self, question: str) -> QueryIntent:
        """
        Parse natural language question into structured intent.

        Args:
            question: Natural language question

        Returns:
            QueryIntent with question type, concepts, and query strategy
        """
        question_lower = question.lower().strip()

        # Determine question type
        question_type = self._detect_question_type(question_lower)

        # Extract concepts (meaningful words)
        concepts = self._extract_concepts(question)

        # Determine query strategy based on question type and concepts
        query_strategy = self._determine_strategy(question_type, question_lower)

        return QueryIntent(
            question_type=question_type,
            concepts=concepts,
            query_strategy=query_strategy,
            raw_question=question
        )

    def _detect_question_type(self, question: str) -> str:
        """Detect the type of question (how, what, where, etc.)."""
        if question.startswith("how"):
            return "how"
        elif question.startswith("where"):
            return "where"
        elif question.startswith("what"):
            return "what"
        elif question.startswith("who"):
            return "who"
        elif question.startswith("why"):
            return "why"
        elif question.startswith("which"):
            return "which"
        else:
            return "general"

    def _extract_concepts(self, question: str) -> List[str]:
        """Extract meaningful concepts from question.

        Prioritizes CamelCase/PascalCase terms (like WovenMind, CodeBridge)
        as they typically represent specific code entities.
        Also detects compound terms by combining adjacent non-stop-words.
        """
        # Tokenize: split on non-alphanumeric, keep underscores
        tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', question)

        # Filter stop words and short tokens, track original case
        filtered = []
        for t in tokens:
            if t.lower() not in STOP_WORDS and len(t) >= 2:
                # Check if CamelCase/PascalCase (has internal uppercase)
                is_camel = any(c.isupper() for c in t[1:])
                filtered.append((t.lower(), is_camel, t))

        # Remove duplicates while preserving order, track original index
        seen = set()
        unique = []
        for idx, (lower, is_camel, original) in enumerate(filtered):
            if lower not in seen:
                seen.add(lower)
                unique.append((lower, is_camel, idx))

        # Sort: CamelCase terms first, then preserve original order within groups
        unique.sort(key=lambda x: (not x[1], x[2]))

        concepts = [concept for concept, _, _ in unique]

        # Also try compound forms if we have adjacent concepts
        # e.g., "cognitive agent" -> also try "cognitiveagent", "cognitive_agent"
        compound_concepts = []
        for i in range(len(concepts) - 1):
            # Concatenated form (cognitiveagent)
            compound = concepts[i] + concepts[i + 1]
            compound_concepts.append(compound)
            # Underscore form (cognitive_agent)
            compound_underscore = concepts[i] + "_" + concepts[i + 1]
            compound_concepts.append(compound_underscore)

        # Add compounds to the front (higher priority)
        return compound_concepts + concepts

    def _determine_strategy(self, question_type: str, question: str) -> List[str]:
        """Determine which tools to use based on question type."""
        strategies = {
            "how": ["similar_to", "code_for_word", "methods_of"],
            "where": ["code_for_word", "defined_in"],
            "what": ["similar_to", "code_for_word"],  # Identity questions need associations
            "who": ["code_for_word"],
            "why": ["similar_to"],
            "which": ["code_for_word"],
            "general": ["similar_to", "code_for_word"]
        }

        strategy = strategies.get(question_type, ["similar_to"])

        # Enhance strategy based on keywords
        if "call" in question:
            strategy = ["callers_of"] + strategy
        if "inherit" in question or "subclass" in question:
            strategy = ["subclasses_of"] + strategy

        return strategy

    def _is_identity_question(self, question: str) -> bool:
        """Check if this is a 'what is X' style identity question."""
        question_lower = question.lower().strip()
        identity_patterns = [
            r"^what is (the |a |an )?",
            r"^what('s| is) (the |a |an )?",
            r"^define ",
            r"^explain (the |what |a |an )?",
            r"^describe (the |what |a |an )?",
        ]
        for pattern in identity_patterns:
            if re.match(pattern, question_lower):
                return True
        return False

    # =========================================================================
    # Knowledge Gatherer
    # =========================================================================

    def gather_knowledge(self, intent: QueryIntent) -> GatheredKnowledge:
        """
        Gather knowledge from multiple sources based on intent.

        Args:
            intent: Parsed query intent

        Returns:
            GatheredKnowledge with associations, code entities, etc.
        """
        knowledge = GatheredKnowledge()

        # Track which concepts we found results for (prioritize compound terms)
        found_compound = False

        for concept in intent.concepts:
            # Get word associations
            similar_tool = self.registry.get("similar_to")
            if similar_tool:
                try:
                    results = similar_tool(concept)
                    added_any = False
                    for atom in results:
                        if hasattr(atom, 'name'):
                            name = atom.name
                            # Filter out stop words and very short terms
                            if name.lower() in STOP_WORDS or len(name) < 3:
                                continue
                            # Skip single-letter words and common noise
                            if name not in knowledge.associations:
                                knowledge.associations.append(name)
                                added_any = True
                    # If we found results for a compound term, prioritize those
                    if added_any and ("_" in concept or len(concept) > 12):
                        found_compound = True
                except Exception:
                    pass

            # If we found good compound results, limit how many single-word concepts we process
            if found_compound and "_" not in concept and len(concept) <= 12:
                continue  # Skip individual words if compound found good results

            # Get code entities
            code_tool = self.registry.get("code_for_word")
            if code_tool:
                try:
                    results = code_tool(concept)
                    for atom in results:
                        if atom not in knowledge.code_entities:
                            knowledge.code_entities.append(atom)
                            # Extract file path
                            file_path = atom.metadata.get("file_path", "")
                            if file_path and file_path not in knowledge.related_files:
                                knowledge.related_files.append(file_path)
                except Exception:
                    pass

            # Get callers if in strategy
            if "callers_of" in intent.query_strategy:
                callers_tool = self.registry.get("callers_of")
                if callers_tool:
                    try:
                        results = callers_tool(concept)
                        for atom in results:
                            if atom not in knowledge.callers:
                                knowledge.callers.append(atom)
                    except Exception:
                        pass

            # Get methods if in strategy
            if "methods_of" in intent.query_strategy:
                methods_tool = self.registry.get("methods_of")
                if methods_tool:
                    try:
                        results = methods_tool(concept)
                        for atom in results:
                            if atom not in knowledge.methods:
                                knowledge.methods.append(atom)
                    except Exception:
                        pass

        return knowledge

    # =========================================================================
    # Response Generator
    # =========================================================================

    def generate_response(
        self,
        intent: QueryIntent,
        knowledge: GatheredKnowledge
    ) -> str:
        """
        Generate natural language response from gathered knowledge.

        Args:
            intent: Original query intent
            knowledge: Gathered knowledge

        Returns:
            Formatted response string
        """
        lines = []

        # Check if we found anything
        has_knowledge = (
            knowledge.associations or
            knowledge.code_entities or
            knowledge.methods or
            knowledge.callers
        )

        if not has_knowledge:
            # Honest "I don't know"
            concepts_str = ", ".join(intent.concepts) if intent.concepts else "that"
            return f"I don't have information about {concepts_str}. Try training on relevant documents or indexing code."

        # Generate summary based on question type
        if intent.question_type == "where":
            lines.append(self._generate_location_summary(intent, knowledge))
        elif intent.question_type == "how":
            lines.append(self._generate_mechanism_summary(intent, knowledge))
        else:
            lines.append(self._generate_general_summary(intent, knowledge))

        # Add code entities with locations
        if knowledge.code_entities:
            lines.append("")
            lines.append("Code locations:")
            for entity in knowledge.code_entities[:5]:  # Limit to 5
                file_path = entity.metadata.get("file_path", "unknown")
                lineno = entity.metadata.get("lineno", "?")
                lines.append(f"  {entity.name} - {file_path}:{lineno}")

        # Add related concepts
        if knowledge.associations:
            lines.append("")
            related = ", ".join(knowledge.associations[:10])
            lines.append(f"Related concepts: {related}")

        # Add callers if any
        if knowledge.callers:
            lines.append("")
            lines.append("Called by:")
            for caller in knowledge.callers[:5]:
                lines.append(f"  {caller.name}")

        # Add methods if any
        if knowledge.methods:
            lines.append("")
            lines.append("Methods:")
            for method in knowledge.methods[:5]:
                lines.append(f"  {method.name}")

        return "\n".join(lines)

    def _generate_location_summary(
        self,
        intent: QueryIntent,
        knowledge: GatheredKnowledge
    ) -> str:
        """Generate summary for 'where' questions."""
        if knowledge.code_entities:
            entity = knowledge.code_entities[0]
            file_path = entity.metadata.get("file_path", "unknown location")
            lineno = entity.metadata.get("lineno", "")
            line_str = f" (line {lineno})" if lineno else ""
            return f"{entity.name} is located in {file_path}{line_str}."
        elif knowledge.related_files:
            return f"Found in: {knowledge.related_files[0]}"
        else:
            return f"Location information for {', '.join(intent.concepts)}:"

    def _generate_mechanism_summary(
        self,
        intent: QueryIntent,
        knowledge: GatheredKnowledge
    ) -> str:
        """Generate summary for 'how' questions."""
        if knowledge.code_entities:
            entity = knowledge.code_entities[0]
            file_path = entity.metadata.get("file_path", "")
            return f"{entity.name} handles this in {file_path}."
        elif knowledge.associations:
            concepts = ", ".join(intent.concepts)
            related = ", ".join(knowledge.associations[:3])
            return f"{concepts.title()} involves: {related}."
        else:
            return f"Information about {', '.join(intent.concepts)}:"

    def _generate_general_summary(
        self,
        intent: QueryIntent,
        knowledge: GatheredKnowledge
    ) -> str:
        """Generate summary for general questions."""
        # Check if this is an identity question ("What is X?")
        if self._is_identity_question(intent.raw_question):
            return self._generate_identity_summary(intent, knowledge)

        if knowledge.code_entities:
            entity = knowledge.code_entities[0]
            return f"Found: {entity.name}"
        elif knowledge.associations:
            return f"Related to: {', '.join(knowledge.associations[:5])}"
        else:
            return f"Information about {', '.join(intent.concepts)}:"

    def _generate_identity_summary(
        self,
        intent: QueryIntent,
        knowledge: GatheredKnowledge
    ) -> str:
        """Generate a rich summary for identity questions ('What is X?')."""
        # Extract subject from original question by removing question words
        raw = intent.raw_question.lower().strip()
        # Remove common question prefixes
        for prefix in ["what is the ", "what is a ", "what is an ", "what is ",
                       "what's the ", "what's a ", "what's an ", "what's ",
                       "define ", "explain the ", "explain ", "describe the ", "describe "]:
            if raw.startswith(prefix):
                raw = raw[len(prefix):]
                break
        # Remove trailing punctuation
        subject = raw.rstrip("?.,!").strip()
        subject_title = subject.title()

        # Group associations by semantic categories to build a description
        if knowledge.associations:
            # Classify associations to build coherent description
            technical_terms = []
            functional_terms = []
            other_terms = []

            tech_keywords = {"graph", "model", "data", "storage", "index", "atom", "link",
                           "memory", "learning", "training", "neural", "semantic", "token",
                           "prediction", "query", "code", "cognitive", "structure"}
            func_keywords = {"process", "create", "build", "manage", "track", "compute",
                           "analyze", "extract", "generate", "transform", "load", "save",
                           "train", "predict", "search", "find", "index"}

            for assoc in knowledge.associations[:15]:
                assoc_lower = assoc.lower()
                if any(kw in assoc_lower for kw in tech_keywords) or assoc_lower in tech_keywords:
                    technical_terms.append(assoc)
                elif any(kw in assoc_lower for kw in func_keywords) or assoc_lower in func_keywords:
                    functional_terms.append(assoc)
                else:
                    other_terms.append(assoc)

            # Build description parts
            parts = []

            # Core identity
            parts.append(f"The {subject_title} is a component that")

            # What it works with (technical terms)
            if technical_terms:
                tech_str = ", ".join(technical_terms[:4])
                parts.append(f"works with {tech_str}")

            # What it does (functional terms)
            if functional_terms:
                func_str = ", ".join(functional_terms[:3])
                if technical_terms:
                    parts.append(f"and handles {func_str}")
                else:
                    parts.append(f"handles {func_str}")

            # Additional context
            if other_terms and len(parts) < 4:
                other_str = ", ".join(other_terms[:3])
                parts.append(f"({other_str})")

            # Join parts into readable sentence
            if len(parts) > 1:
                description = " ".join(parts) + "."
            else:
                description = f"The {subject_title} relates to: {', '.join(knowledge.associations[:5])}."

            return description

        elif knowledge.code_entities:
            entity = knowledge.code_entities[0]
            entity_type = entity.atom_type.name.lower() if hasattr(entity.atom_type, 'name') else "entity"
            file_path = entity.metadata.get("file_path", "")
            if file_path:
                return f"The {subject_title} is a {entity_type} defined in {file_path}."
            return f"The {subject_title} is a {entity_type} in the codebase."

        else:
            return f"The {subject_title} is a concept in this system."

    # =========================================================================
    # Unified Pipeline Support (Phase 5)
    # =========================================================================

    def _init_unified_pipeline(self) -> None:
        """Initialize unified pipeline components lazily."""
        if self._router is not None:
            return  # Already initialized

        from cortical.cognitive.unified_query import QueryRouter
        from cortical.cognitive.executors import (
            AuditExecutor,
            SemanticExecutor,
            CodeExecutor,
            CDGExecutor,
        )
        from cortical.cognitive.aggregator import ResultAggregator
        from cortical.cognitive.formatter import ResponseFormatter

        self._router = QueryRouter()
        self._executors = {
            "audit": AuditExecutor(),
            "semantic": SemanticExecutor(agent=self.agent),
            "code": CodeExecutor(),
            "cdg": CDGExecutor(),
        }
        self._aggregator = ResultAggregator(strategy="merge")
        self._formatter = ResponseFormatter()

    def _ask_unified(self, question: str) -> str:
        """
        Ask using unified pipeline (QueryRouter → Executor → Aggregator → Formatter).

        Args:
            question: Natural language question

        Returns:
            Formatted answer string
        """
        self._init_unified_pipeline()

        # 1. Route the question to appropriate backend
        unified_query = self._router.route(question)

        # 2. Get the executor for this query type
        executor = self._executors.get(unified_query.query_type)
        if not executor:
            return f"No executor available for query type: {unified_query.query_type}"

        # 3. Execute the query
        result = executor.execute(unified_query.parsed)

        # 4. Aggregate results (single source, but normalizes format)
        aggregated = self._aggregator.aggregate([result])

        # 5. Format the response
        response = self._formatter.format(unified_query, aggregated)

        return response

    # =========================================================================
    # Main Entry Point
    # =========================================================================

    def ask(self, question: str) -> str:
        """
        Ask a natural language question and get a complete answer.

        Uses unified pipeline if use_unified=True, otherwise uses legacy mode.

        Args:
            question: Natural language question

        Returns:
            Formatted answer string
        """
        if self.use_unified:
            return self._ask_unified(question)

        # Legacy mode: Parse intent, gather knowledge, generate response
        intent = self.parse_intent(question)
        knowledge = self.gather_knowledge(intent)
        response = self.generate_response(intent, knowledge)

        return response
