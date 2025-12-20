"""
Factory functions for common Graph of Thought patterns.

This module provides pre-structured ThoughtGraph instances for common reasoning
workflows like investigation, decision-making, debugging, feature planning, and
requirements analysis.

Each pattern creates a graph with appropriate node types and relationships,
ready to be populated with actual content during reasoning.
"""

from typing import List, Optional
import uuid

from .thought_graph import ThoughtGraph
from .graph_of_thought import NodeType, EdgeType


def _generate_id(prefix: str = "n") -> str:
    """Generate a unique node ID with optional prefix."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def create_investigation_graph(
    question: str,
    initial_hypotheses: Optional[List[str]] = None
) -> ThoughtGraph:
    """
    Create a graph structure for investigating a question.

    Structure:
        question (root)
        ├── hypothesis_1 (placeholder)
        ├── hypothesis_2 (placeholder)
        └── hypothesis_3 (placeholder)

    Args:
        question: The investigation question
        initial_hypotheses: Optional list of initial hypotheses

    Returns:
        ThoughtGraph with question and hypothesis placeholder nodes
    """
    graph = ThoughtGraph()

    # Create root question node
    question_id = _generate_id("q")
    graph.add_node(
        node_id=question_id,
        node_type=NodeType.QUESTION,
        content=question,
    )

    # Add initial hypotheses or placeholders
    hypotheses = initial_hypotheses or [
        "Hypothesis 1 (to be investigated)",
        "Hypothesis 2 (to be investigated)",
        "Hypothesis 3 (to be investigated)",
    ]

    for hypothesis in hypotheses:
        hyp_id = _generate_id("h")
        graph.add_node(
            node_id=hyp_id,
            node_type=NodeType.HYPOTHESIS,
            content=hypothesis,
            metadata={"placeholder": initial_hypotheses is None},
        )
        graph.add_edge(
            from_id=question_id,
            to_id=hyp_id,
            edge_type=EdgeType.EXPLORES,
            weight=1.0 if initial_hypotheses else 0.5,
        )

    return graph


def create_decision_graph(
    decision: str,
    options: List[str]
) -> ThoughtGraph:
    """
    Create a graph structure for decision-making with options.

    Structure:
        decision (root)
        ├── option_1
        │   ├── pro_1 (placeholder)
        │   └── con_1 (placeholder)
        ├── option_2
        │   ├── pro_1 (placeholder)
        │   └── con_1 (placeholder)
        └── ...

    Args:
        decision: The decision to be made
        options: List of decision options

    Returns:
        ThoughtGraph with decision and option nodes with pro/con structure
    """
    graph = ThoughtGraph()

    # Create root decision node
    decision_id = _generate_id("d")
    graph.add_node(
        node_id=decision_id,
        node_type=NodeType.DECISION,
        content=decision,
    )

    # Create option nodes with pros/cons
    for i, option in enumerate(options):
        option_id = _generate_id(f"opt{i}")
        graph.add_node(
            node_id=option_id,
            node_type=NodeType.OPTION,
            content=option,
        )
        graph.add_edge(
            from_id=decision_id,
            to_id=option_id,
            edge_type=EdgeType.HAS_OPTION,
            weight=1.0,
        )

        # Add pro placeholder
        pro_id = _generate_id(f"pro{i}")
        graph.add_node(
            node_id=pro_id,
            node_type=NodeType.EVIDENCE,
            content=f"Pros for: {option}",
            metadata={"valence": "positive", "placeholder": True},
        )
        graph.add_edge(
            from_id=option_id,
            to_id=pro_id,
            edge_type=EdgeType.SUPPORTS,
            weight=0.5,
        )

        # Add con placeholder
        con_id = _generate_id(f"con{i}")
        graph.add_node(
            node_id=con_id,
            node_type=NodeType.EVIDENCE,
            content=f"Cons for: {option}",
            metadata={"valence": "negative", "placeholder": True},
        )
        graph.add_edge(
            from_id=option_id,
            to_id=con_id,
            edge_type=EdgeType.CONTRADICTS,
            weight=0.5,
        )

    return graph


def create_debug_graph(symptom: str) -> ThoughtGraph:
    """
    Create a graph structure for debugging a problem.

    Structure:
        symptom (root)
        ├── observation_1 (placeholder)
        ├── observation_2 (placeholder)
        └── observation_3 (placeholder)
            ├── possible_cause_1 (placeholder)
            ├── possible_cause_2 (placeholder)
            └── possible_cause_3 (placeholder)

    Args:
        symptom: The observed symptom or problem

    Returns:
        ThoughtGraph with symptom -> observations -> causes structure
    """
    graph = ThoughtGraph()

    # Create root symptom node
    symptom_id = _generate_id("sym")
    graph.add_node(
        node_id=symptom_id,
        node_type=NodeType.OBSERVATION,
        content=symptom,
    )

    # Add observation placeholders
    observation_ids = []
    for i in range(3):
        obs_id = _generate_id(f"obs{i}")
        graph.add_node(
            node_id=obs_id,
            node_type=NodeType.OBSERVATION,
            content=f"Observation {i+1} (to be documented)",
            metadata={"placeholder": True},
        )
        graph.add_edge(
            from_id=symptom_id,
            to_id=obs_id,
            edge_type=EdgeType.OBSERVES,
            weight=0.5,
        )
        observation_ids.append(obs_id)

    # Add possible cause placeholders to the last observation
    if observation_ids:
        last_obs = observation_ids[-1]
        for i in range(3):
            cause_id = _generate_id(f"cause{i}")
            graph.add_node(
                node_id=cause_id,
                node_type=NodeType.HYPOTHESIS,
                content=f"Possible cause {i+1} (to be investigated)",
                metadata={"placeholder": True},
            )
            graph.add_edge(
                from_id=last_obs,
                to_id=cause_id,
                edge_type=EdgeType.SUGGESTS,
                weight=0.3,
            )

    return graph


def create_feature_graph(goal: str, user_story: str) -> ThoughtGraph:
    """
    Create a graph structure for feature planning.

    Structure:
        goal (root)
        └── user_story
            ├── task_1 (placeholder)
            ├── task_2 (placeholder)
            └── task_3 (placeholder)

    Args:
        goal: The feature goal
        user_story: The user story

    Returns:
        ThoughtGraph with goal -> story -> tasks structure
    """
    graph = ThoughtGraph()

    # Create root goal node
    goal_id = _generate_id("goal")
    graph.add_node(
        node_id=goal_id,
        node_type=NodeType.GOAL,
        content=goal,
    )

    # Create user story node
    story_id = _generate_id("story")
    graph.add_node(
        node_id=story_id,
        node_type=NodeType.CONTEXT,
        content=user_story,
    )
    graph.add_edge(
        from_id=goal_id,
        to_id=story_id,
        edge_type=EdgeType.MOTIVATES,
        weight=1.0,
    )

    # Add task placeholders
    for i in range(3):
        task_id = _generate_id(f"task{i}")
        graph.add_node(
            node_id=task_id,
            node_type=NodeType.ACTION,
            content=f"Task {i+1} (to be defined)",
            metadata={"placeholder": True},
        )
        graph.add_edge(
            from_id=story_id,
            to_id=task_id,
            edge_type=EdgeType.REQUIRES,
            weight=0.5,
        )

    return graph


def create_requirements_graph(user_need: str) -> ThoughtGraph:
    """
    Create a graph structure for requirements analysis.

    Structure:
        user_need (root)
        ├── requirement_1 (placeholder)
        │   └── specification_1 (placeholder)
        │       └── design_1 (placeholder)
        ├── requirement_2 (placeholder)
        │   └── specification_2 (placeholder)
        │       └── design_2 (placeholder)
        └── requirement_3 (placeholder)
            └── specification_3 (placeholder)
                └── design_3 (placeholder)

    Args:
        user_need: The user need to analyze

    Returns:
        ThoughtGraph with need -> requirements -> specs -> design structure
    """
    graph = ThoughtGraph()

    # Create root user need node
    need_id = _generate_id("need")
    graph.add_node(
        node_id=need_id,
        node_type=NodeType.GOAL,
        content=user_need,
    )

    # Create requirement chains
    for i in range(3):
        # Requirement node
        req_id = _generate_id(f"req{i}")
        graph.add_node(
            node_id=req_id,
            node_type=NodeType.CONSTRAINT,
            content=f"Requirement {i+1} (to be defined)",
            metadata={"placeholder": True},
        )
        graph.add_edge(
            from_id=need_id,
            to_id=req_id,
            edge_type=EdgeType.REQUIRES,
            weight=0.5,
        )

        # Specification node
        spec_id = _generate_id(f"spec{i}")
        graph.add_node(
            node_id=spec_id,
            node_type=NodeType.CONTEXT,
            content=f"Specification {i+1} (to be detailed)",
            metadata={"placeholder": True},
        )
        graph.add_edge(
            from_id=req_id,
            to_id=spec_id,
            edge_type=EdgeType.REFINES,
            weight=0.5,
        )

        # Design node
        design_id = _generate_id(f"design{i}")
        graph.add_node(
            node_id=design_id,
            node_type=NodeType.ACTION,
            content=f"Design {i+1} (to be created)",
            metadata={"placeholder": True},
        )
        graph.add_edge(
            from_id=spec_id,
            to_id=design_id,
            edge_type=EdgeType.IMPLEMENTS,
            weight=0.5,
        )

    return graph


def create_analysis_graph(topic: str, aspects: Optional[List[str]] = None) -> ThoughtGraph:
    """
    Create a graph structure for analyzing a topic.

    Structure:
        topic (root)
        ├── aspect_1
        │   ├── finding_1 (placeholder)
        │   └── finding_2 (placeholder)
        ├── aspect_2
        │   ├── finding_1 (placeholder)
        │   └── finding_2 (placeholder)
        └── aspect_3
            ├── finding_1 (placeholder)
            └── finding_2 (placeholder)

    Args:
        topic: The topic to analyze
        aspects: Optional list of specific aspects to analyze

    Returns:
        ThoughtGraph with topic -> aspects -> findings structure
    """
    graph = ThoughtGraph()

    # Create root topic node
    topic_id = _generate_id("topic")
    graph.add_node(
        node_id=topic_id,
        node_type=NodeType.CONTEXT,
        content=topic,
    )

    # Default aspects if none provided
    if aspects is None:
        aspects = [
            "Aspect 1 (to be defined)",
            "Aspect 2 (to be defined)",
            "Aspect 3 (to be defined)",
        ]

    # Create aspect nodes with finding placeholders
    for i, aspect in enumerate(aspects):
        aspect_id = _generate_id(f"aspect{i}")
        is_placeholder = "to be defined" in aspect.lower()
        graph.add_node(
            node_id=aspect_id,
            node_type=NodeType.CONTEXT,
            content=aspect,
            metadata={"placeholder": is_placeholder},
        )
        graph.add_edge(
            from_id=topic_id,
            to_id=aspect_id,
            edge_type=EdgeType.HAS_ASPECT,
            weight=1.0,
        )

        # Add finding placeholders
        for j in range(2):
            finding_id = _generate_id(f"find{i}_{j}")
            graph.add_node(
                node_id=finding_id,
                node_type=NodeType.EVIDENCE,
                content=f"Finding {j+1} for {aspect}",
                metadata={"placeholder": True},
            )
            graph.add_edge(
                from_id=aspect_id,
                to_id=finding_id,
                edge_type=EdgeType.OBSERVES,
                weight=0.5,
            )

    return graph


# Factory registry for dynamic pattern creation
PATTERN_REGISTRY = {
    "investigation": create_investigation_graph,
    "decision": create_decision_graph,
    "debug": create_debug_graph,
    "feature": create_feature_graph,
    "requirements": create_requirements_graph,
    "analysis": create_analysis_graph,
}


def create_pattern_graph(pattern_name: str, **kwargs) -> ThoughtGraph:
    """
    Create a thought graph using a named pattern.

    Args:
        pattern_name: Name of the pattern (investigation, decision, debug,
                     feature, requirements, analysis)
        **kwargs: Pattern-specific arguments

    Returns:
        ThoughtGraph instance for the specified pattern

    Raises:
        ValueError: If pattern_name is not recognized

    Example:
        >>> graph = create_pattern_graph("investigation", question="Why is the API slow?")
        >>> graph = create_pattern_graph("decision", decision="Auth method", options=["OAuth", "JWT"])
    """
    if pattern_name not in PATTERN_REGISTRY:
        valid_patterns = ", ".join(PATTERN_REGISTRY.keys())
        raise ValueError(f"Unknown pattern '{pattern_name}'. Valid patterns: {valid_patterns}")

    factory = PATTERN_REGISTRY[pattern_name]
    return factory(**kwargs)
