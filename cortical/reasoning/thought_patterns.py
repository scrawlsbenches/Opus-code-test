"""
Factory functions for common Graph of Thought patterns.

This module provides pre-structured ThoughtGraph instances for common reasoning
workflows like investigation, decision-making, debugging, feature planning, and
requirements analysis.
"""

from typing import List, Optional
from .thought_graph import ThoughtGraph
from .graph_of_thought import NodeType, EdgeType


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
    question_id = graph.add_node(
        content=question,
        node_type=NodeType.QUESTION
    )

    # Add initial hypotheses or placeholders
    if initial_hypotheses:
        for i, hypothesis in enumerate(initial_hypotheses):
            hyp_id = graph.add_node(
                content=hypothesis,
                node_type=NodeType.HYPOTHESIS
            )
            graph.add_edge(
                source=question_id,
                target=hyp_id,
                edge_type=EdgeType.EXPLORES,
                weight=1.0
            )
    else:
        # Create placeholder hypothesis nodes
        for i in range(3):
            hyp_id = graph.add_node(
                content=f"Hypothesis {i+1} (to be investigated)",
                node_type=NodeType.HYPOTHESIS,
                metadata={"placeholder": True}
            )
            graph.add_edge(
                source=question_id,
                target=hyp_id,
                edge_type=EdgeType.EXPLORES,
                weight=0.5
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
        │   ├── pro_1
        │   ├── pro_2
        │   ├── con_1
        │   └── con_2
        ├── option_2
        │   ├── pro_1
        │   └── con_1
        └── option_3
            └── ...

    Args:
        decision: The decision to be made
        options: List of decision options

    Returns:
        ThoughtGraph with decision and option nodes with pro/con structure
    """
    graph = ThoughtGraph()

    # Create root decision node
    decision_id = graph.add_node(
        content=decision,
        node_type=NodeType.DECISION
    )

    # Create option nodes
    for option in options:
        option_id = graph.add_node(
            content=option,
            node_type=NodeType.OPTION
        )
        graph.add_edge(
            source=decision_id,
            target=option_id,
            edge_type=EdgeType.HAS_OPTION,
            weight=1.0
        )

        # Add pro placeholder
        pro_id = graph.add_node(
            content=f"Pros for: {option}",
            node_type=NodeType.EVIDENCE,
            metadata={"valence": "positive", "placeholder": True}
        )
        graph.add_edge(
            source=option_id,
            target=pro_id,
            edge_type=EdgeType.SUPPORTS,
            weight=0.5
        )

        # Add con placeholder
        con_id = graph.add_node(
            content=f"Cons for: {option}",
            node_type=NodeType.EVIDENCE,
            metadata={"valence": "negative", "placeholder": True}
        )
        graph.add_edge(
            source=option_id,
            target=con_id,
            edge_type=EdgeType.CONTRADICTS,
            weight=0.5
        )

    return graph


def create_debug_graph(symptom: str) -> ThoughtGraph:
    """
    Create a graph structure for debugging a problem.

    Structure:
        symptom (root)
        ├── observation_1
        ├── observation_2
        └── observation_3
            ├── possible_cause_1
            ├── possible_cause_2
            └── possible_cause_3

    Args:
        symptom: The observed symptom or problem

    Returns:
        ThoughtGraph with symptom -> observations -> causes structure
    """
    graph = ThoughtGraph()

    # Create root symptom node
    symptom_id = graph.add_node(
        content=symptom,
        node_type=NodeType.OBSERVATION
    )

    # Add observation placeholders
    observation_ids = []
    for i in range(3):
        obs_id = graph.add_node(
            content=f"Observation {i+1} (to be documented)",
            node_type=NodeType.OBSERVATION,
            metadata={"placeholder": True}
        )
        graph.add_edge(
            source=symptom_id,
            target=obs_id,
            edge_type=EdgeType.OBSERVES,
            weight=0.5
        )
        observation_ids.append(obs_id)

    # Add possible cause placeholders to the last observation
    if observation_ids:
        last_obs = observation_ids[-1]
        for i in range(3):
            cause_id = graph.add_node(
                content=f"Possible cause {i+1} (to be investigated)",
                node_type=NodeType.HYPOTHESIS,
                metadata={"placeholder": True}
            )
            graph.add_edge(
                source=last_obs,
                target=cause_id,
                edge_type=EdgeType.SUGGESTS,
                weight=0.3
            )

    return graph


def create_feature_graph(goal: str, user_story: str) -> ThoughtGraph:
    """
    Create a graph structure for feature planning.

    Structure:
        goal (root)
        └── user_story
            ├── task_1
            ├── task_2
            └── task_3

    Args:
        goal: The feature goal
        user_story: The user story

    Returns:
        ThoughtGraph with goal -> story -> tasks structure
    """
    graph = ThoughtGraph()

    # Create root goal node
    goal_id = graph.add_node(
        content=goal,
        node_type=NodeType.GOAL
    )

    # Create user story node
    story_id = graph.add_node(
        content=user_story,
        node_type=NodeType.CONTEXT
    )
    graph.add_edge(
        source=goal_id,
        target=story_id,
        edge_type=EdgeType.MOTIVATES,
        weight=1.0
    )

    # Add task placeholders
    for i in range(3):
        task_id = graph.add_node(
            content=f"Task {i+1} (to be defined)",
            node_type=NodeType.ACTION,
            metadata={"placeholder": True}
        )
        graph.add_edge(
            source=story_id,
            target=task_id,
            edge_type=EdgeType.REQUIRES,
            weight=0.5
        )

    return graph


def create_requirements_graph(user_need: str) -> ThoughtGraph:
    """
    Create a graph structure for requirements analysis.

    Structure:
        user_need (root)
        ├── requirement_1
        │   └── specification_1
        │       └── design_1
        ├── requirement_2
        │   └── specification_2
        │       └── design_2
        └── requirement_3
            └── specification_3
                └── design_3

    Args:
        user_need: The user need to analyze

    Returns:
        ThoughtGraph with need -> requirements -> specs -> design structure
    """
    graph = ThoughtGraph()

    # Create root user need node
    need_id = graph.add_node(
        content=user_need,
        node_type=NodeType.GOAL
    )

    # Create requirement chains
    for i in range(3):
        # Requirement node
        req_id = graph.add_node(
            content=f"Requirement {i+1} (to be defined)",
            node_type=NodeType.CONSTRAINT,
            metadata={"placeholder": True}
        )
        graph.add_edge(
            source=need_id,
            target=req_id,
            edge_type=EdgeType.REQUIRES,
            weight=0.5
        )

        # Specification node
        spec_id = graph.add_node(
            content=f"Specification {i+1} (to be detailed)",
            node_type=NodeType.CONTEXT,
            metadata={"placeholder": True}
        )
        graph.add_edge(
            source=req_id,
            target=spec_id,
            edge_type=EdgeType.REFINES,
            weight=0.5
        )

        # Design node
        design_id = graph.add_node(
            content=f"Design {i+1} (to be created)",
            node_type=NodeType.ACTION,
            metadata={"placeholder": True}
        )
        graph.add_edge(
            source=spec_id,
            target=design_id,
            edge_type=EdgeType.IMPLEMENTS,
            weight=0.5
        )

    return graph


def create_analysis_graph(topic: str, aspects: Optional[List[str]] = None) -> ThoughtGraph:
    """
    Create a graph structure for analyzing a topic.

    Structure:
        topic (root)
        ├── aspect_1
        │   ├── finding_1
        │   └── finding_2
        ├── aspect_2
        │   ├── finding_1
        │   └── finding_2
        └── aspect_3
            ├── finding_1
            └── finding_2

    Args:
        topic: The topic to analyze
        aspects: Optional list of specific aspects to analyze

    Returns:
        ThoughtGraph with topic -> aspects -> findings structure
    """
    graph = ThoughtGraph()

    # Create root topic node
    topic_id = graph.add_node(
        content=topic,
        node_type=NodeType.CONTEXT
    )

    # Default aspects if none provided
    if aspects is None:
        aspects = ["Aspect 1 (to be defined)", "Aspect 2 (to be defined)", "Aspect 3 (to be defined)"]

    # Create aspect nodes with finding placeholders
    for aspect in aspects:
        aspect_id = graph.add_node(
            content=aspect,
            node_type=NodeType.CONTEXT,
            metadata={"placeholder": "to be defined" in aspect.lower()}
        )
        graph.add_edge(
            source=topic_id,
            target=aspect_id,
            edge_type=EdgeType.HAS_ASPECT,
            weight=1.0
        )

        # Add finding placeholders
        for i in range(2):
            finding_id = graph.add_node(
                content=f"Finding {i+1} for {aspect}",
                node_type=NodeType.EVIDENCE,
                metadata={"placeholder": True}
            )
            graph.add_edge(
                source=aspect_id,
                target=finding_id,
                edge_type=EdgeType.OBSERVES,
                weight=0.5
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
        pattern_name: Name of the pattern (investigation, decision, debug, feature, requirements, analysis)
        **kwargs: Pattern-specific arguments

    Returns:
        ThoughtGraph instance for the specified pattern

    Raises:
        ValueError: If pattern_name is not recognized
    """
    if pattern_name not in PATTERN_REGISTRY:
        valid_patterns = ", ".join(PATTERN_REGISTRY.keys())
        raise ValueError(f"Unknown pattern '{pattern_name}'. Valid patterns: {valid_patterns}")

    factory = PATTERN_REGISTRY[pattern_name]
    return factory(**kwargs)
