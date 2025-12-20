"""
Reasoning Workflow Orchestrator: Unified Framework for Complex Reasoning.

This module provides the main orchestrator that ties together all components
of the reasoning framework as defined in docs/complex-reasoning-workflow.md.

It provides a unified interface for:
- QAPV cognitive loops (cognitive_loop.py)
- Production state management (production_state.py)
- Crisis management (crisis_manager.py)
- Verification protocols (verification.py)
- Collaboration coordination (collaboration.py)
- Graph of thought representation (thought_graph.py)

The ReasoningWorkflow class is the primary entry point for using the framework.
It coordinates all subsystems and provides high-level operations for common
reasoning tasks.

Design Philosophy:
    "The quality of your output is bounded by the quality of your thinking
    process, which is bounded by your awareness of your thinking process."
    - Meta-cognition principle from the workflow document

    This orchestrator makes the thinking process explicit and observable,
    enabling better collaboration between humans and AI.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from .cognitive_loop import (
    CognitiveLoop,
    CognitiveLoopManager,
    LoopPhase,
    LoopStatus,
    TerminationReason,
)
from .production_state import (
    ProductionManager,
    ProductionTask,
    ProductionState,
    ProductionChunk,
    CommentMarker,
)
from .crisis_manager import (
    CrisisManager,
    CrisisEvent,
    CrisisLevel,
    RecoveryAction,
    RepeatedFailureTracker,
    ScopeCreepDetector,
)
from .verification import (
    VerificationManager,
    VerificationSuite,
    VerificationCheck,
    VerificationStatus,
    VerificationLevel,
    VerificationPhase,
)
from .collaboration import (
    CollaborationManager,
    CollaborationMode,
    StatusUpdate,
    Blocker,
    BlockerType,
    DisagreementRecord,
    ActiveWorkHandoff,
)
from .thought_graph import ThoughtGraph
from .graph_of_thought import NodeType, EdgeType, ThoughtNode


@dataclass
class WorkflowContext:
    """
    Context for a reasoning workflow session.

    Captures the state of an ongoing reasoning process, including
    all active components and their relationships.
    """
    session_id: str
    goal: str
    started_at: datetime = field(default_factory=datetime.now)

    # Active components
    current_loop_id: Optional[str] = None
    current_task_id: Optional[str] = None
    current_verification_suite: Optional[str] = None

    # Tracking
    decisions_made: List[Dict[str, Any]] = field(default_factory=list)
    questions_answered: List[Dict[str, Any]] = field(default_factory=list)
    artifacts_produced: List[str] = field(default_factory=list)
    lessons_learned: List[str] = field(default_factory=list)

    # Graph representation
    thought_graph: Optional[ThoughtGraph] = None


class ReasoningWorkflow:
    """
    Main orchestrator for the reasoning workflow framework.

    This class coordinates all subsystems and provides high-level operations
    for conducting complex reasoning tasks. It is the primary entry point
    for using the framework.

    Example:
        >>> workflow = ReasoningWorkflow()
        >>> ctx = workflow.start_session("Implement user authentication")
        >>> workflow.begin_question_phase(ctx)
        >>> workflow.record_decision(ctx, "Use OAuth", "Industry standard, user familiarity")
        >>> workflow.begin_production_phase(ctx)
        >>> workflow.verify(ctx)
        >>> workflow.complete_session(ctx)

    Components Orchestrated:
        - CognitiveLoopManager: QAPV loop lifecycle
        - ProductionManager: Artifact creation tracking
        - CrisisManager: Failure handling and recovery
        - VerificationManager: Testing and validation
        - CollaborationManager: Human-AI coordination
    """

    def __init__(self, collaboration_mode: CollaborationMode = CollaborationMode.SEMI_SYNCHRONOUS):
        """
        Initialize the reasoning workflow orchestrator.

        Args:
            collaboration_mode: How human-AI collaboration will work
        """
        # Initialize all subsystems
        self._loop_manager = CognitiveLoopManager()
        self._production_manager = ProductionManager()
        self._crisis_manager = CrisisManager()
        self._verification_manager = VerificationManager()
        self._collaboration_manager = CollaborationManager(mode=collaboration_mode)

        # Active contexts
        self._contexts: Dict[str, WorkflowContext] = {}

        # Wire up cross-system integration
        self._setup_integrations()

    def _setup_integrations(self) -> None:
        """Set up cross-system event handlers."""
        # When loop transitions, check for crisis indicators
        self._loop_manager.register_transition_handler(self._on_loop_transition)

        # When production state changes, update verification
        self._production_manager.register_state_change_handler(self._on_production_state_change)

        # When verification fails, notify crisis manager
        self._verification_manager.register_failure_handler(self._on_verification_failure)

        # When crisis is recorded, update collaboration status
        self._crisis_manager.register_crisis_handler(self._on_crisis_recorded)

    # =========================================================================
    # SESSION LIFECYCLE
    # =========================================================================

    def start_session(self, goal: str, session_id: str = None) -> WorkflowContext:
        """
        Start a new reasoning workflow session.

        Args:
            goal: What this session aims to achieve
            session_id: Optional session identifier (auto-generated if not provided)

        Returns:
            WorkflowContext for the new session
        """
        import uuid

        session_id = session_id or str(uuid.uuid4())[:8]

        # Create context
        context = WorkflowContext(
            session_id=session_id,
            goal=goal,
            thought_graph=ThoughtGraph(),
        )
        self._contexts[session_id] = context

        # Create initial loop
        loop = self._loop_manager.create_loop(goal)
        context.current_loop_id = loop.id

        # Add goal node to thought graph
        context.thought_graph.add_node(
            f"goal_{session_id}",
            NodeType.TASK,
            goal,
            properties={'type': 'session_goal'},
        )

        return context

    def complete_session(
        self,
        context: WorkflowContext,
        reason: TerminationReason = TerminationReason.SUCCESS
    ) -> Dict[str, Any]:
        """
        Complete a reasoning workflow session.

        Args:
            context: The session context
            reason: Why the session is ending

        Returns:
            Session summary
        """
        # Complete the current loop
        if context.current_loop_id:
            loop = self._loop_manager.get_loop(context.current_loop_id)
            if loop and loop.status == LoopStatus.ACTIVE:
                loop.complete(reason)

        # Generate summary
        summary = self._generate_session_summary(context)

        return summary

    def abandon_session(self, context: WorkflowContext, reason: str) -> Dict[str, Any]:
        """
        Abandon a session without completing it.

        Args:
            context: The session context
            reason: Why the session is being abandoned

        Returns:
            Abandonment record
        """
        if context.current_loop_id:
            loop = self._loop_manager.get_loop(context.current_loop_id)
            if loop:
                loop.abandon(reason)

        context.lessons_learned.append(f"Session abandoned: {reason}")

        return {
            'session_id': context.session_id,
            'reason': reason,
            'decisions_made': len(context.decisions_made),
            'artifacts_produced': len(context.artifacts_produced),
            'lessons_learned': context.lessons_learned,
        }

    # =========================================================================
    # QAPV PHASE OPERATIONS
    # =========================================================================

    def begin_question_phase(self, context: WorkflowContext) -> None:
        """
        Begin the QUESTION phase of the QAPV loop.

        In this phase:
        - Clarify ambiguity
        - Discover constraints
        - Map scope
        - Understand intent
        """
        loop = self._loop_manager.get_loop(context.current_loop_id)
        if loop.status == LoopStatus.NOT_STARTED:
            loop.start(LoopPhase.QUESTION)
        else:
            loop.transition(LoopPhase.QUESTION, "Entering question phase")

        # Add to thought graph
        context.thought_graph.add_node(
            f"phase_question_{loop.get_iteration_count(LoopPhase.QUESTION)}",
            NodeType.TASK,
            "Question Phase: Clarify requirements and constraints",
            properties={'loop_id': loop.id, 'phase': 'question'},
        )

    def begin_answer_phase(self, context: WorkflowContext) -> None:
        """
        Begin the ANSWER phase of the QAPV loop.

        In this phase:
        - Research existing solutions
        - Analyze options
        - Propose approaches
        - Evaluate trade-offs
        """
        loop = self._loop_manager.get_loop(context.current_loop_id)
        loop.transition(LoopPhase.ANSWER, "Requirements clarified, entering answer phase")

        context.thought_graph.add_node(
            f"phase_answer_{loop.get_iteration_count(LoopPhase.ANSWER)}",
            NodeType.TASK,
            "Answer Phase: Research and propose solutions",
            properties={'loop_id': loop.id, 'phase': 'answer'},
        )

    def begin_production_phase(self, context: WorkflowContext) -> None:
        """
        Begin the PRODUCE phase of the QAPV loop.

        In this phase:
        - Create artifacts (code, docs, tests)
        - Implement solutions
        - Track progress with production states
        """
        loop = self._loop_manager.get_loop(context.current_loop_id)
        loop.transition(LoopPhase.PRODUCE, "Approach approved, entering production phase")

        # Create production task
        task = self._production_manager.create_task(
            goal=context.goal,
            description=f"Production for session {context.session_id}",
        )
        context.current_task_id = task.id

        context.thought_graph.add_node(
            f"phase_produce_{loop.get_iteration_count(LoopPhase.PRODUCE)}",
            NodeType.TASK,
            "Produce Phase: Create artifacts and implement",
            properties={'loop_id': loop.id, 'phase': 'produce', 'task_id': task.id},
        )

    def begin_verify_phase(self, context: WorkflowContext) -> None:
        """
        Begin the VERIFY phase of the QAPV loop.

        In this phase:
        - Run verification suite
        - Confirm correctness
        - Validate against requirements
        """
        loop = self._loop_manager.get_loop(context.current_loop_id)
        loop.transition(LoopPhase.VERIFY, "Production complete, entering verification phase")

        # Create verification suite
        suite = self._verification_manager.create_standard_suite(
            f"verify_{context.session_id}",
            f"Verification for session {context.session_id}",
        )
        context.current_verification_suite = suite.name

        context.thought_graph.add_node(
            f"phase_verify_{loop.get_iteration_count(LoopPhase.VERIFY)}",
            NodeType.TASK,
            "Verify Phase: Confirm correctness and validity",
            properties={'loop_id': loop.id, 'phase': 'verify', 'suite': suite.name},
        )

    # =========================================================================
    # REASONING OPERATIONS
    # =========================================================================

    def record_question(self, context: WorkflowContext, question: str, category: str = "clarification") -> str:
        """
        Record a question being asked.

        Args:
            context: Session context
            question: The question text
            category: Type of question (clarification, exploration, validation, constraint, meta)

        Returns:
            Question node ID in the thought graph
        """
        node_id = f"q_{len(context.thought_graph.nodes) + 1}"

        context.thought_graph.add_node(
            node_id,
            NodeType.QUESTION,
            question,
            properties={'category': category, 'answered': False},
        )

        # Link to current phase
        loop = self._loop_manager.get_loop(context.current_loop_id)
        if loop:
            loop.current_context().questions_raised.append(question)

        return node_id

    def record_answer(self, context: WorkflowContext, question_id: str, answer: str, confidence: float = 0.8) -> str:
        """
        Record an answer to a question.

        Args:
            context: Session context
            question_id: ID of the question being answered
            answer: The answer text
            confidence: Confidence in the answer (0-1)

        Returns:
            Answer node ID
        """
        answer_id = f"a_{len(context.thought_graph.nodes) + 1}"

        context.thought_graph.add_node(
            answer_id,
            NodeType.FACT,
            answer,
            properties={'confidence': confidence},
        )

        # Link answer to question
        context.thought_graph.add_edge(
            answer_id,
            question_id,
            EdgeType.ANSWERS,
            weight=confidence,
        )

        # Mark question as answered
        question_node = context.thought_graph.get_node(question_id)
        if question_node:
            question_node.properties['answered'] = True

        context.questions_answered.append({
            'question_id': question_id,
            'answer_id': answer_id,
            'answer': answer,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
        })

        return answer_id

    def record_decision(self, context: WorkflowContext, decision: str, rationale: str, options_considered: List[str] = None) -> str:
        """
        Record a decision made during reasoning.

        Args:
            context: Session context
            decision: What was decided
            rationale: Why this decision was made
            options_considered: Other options that were evaluated

        Returns:
            Decision node ID
        """
        decision_id = f"d_{len(context.decisions_made) + 1}"

        context.thought_graph.add_node(
            decision_id,
            NodeType.DECISION,
            decision,
            properties={
                'rationale': rationale,
                'options_considered': options_considered or [],
                'timestamp': datetime.now().isoformat(),
            },
        )

        context.decisions_made.append({
            'id': decision_id,
            'decision': decision,
            'rationale': rationale,
            'options': options_considered or [],
            'timestamp': datetime.now().isoformat(),
        })

        # Update loop context
        loop = self._loop_manager.get_loop(context.current_loop_id)
        if loop:
            loop.current_context().decisions_made.append({
                'decision': decision,
                'rationale': rationale,
            })

        return decision_id

    def record_insight(self, context: WorkflowContext, insight: str, source: str = "analysis") -> str:
        """
        Record an insight discovered during reasoning.

        Args:
            context: Session context
            insight: The insight text
            source: How the insight was discovered

        Returns:
            Insight node ID
        """
        insight_id = f"i_{len(context.thought_graph.nodes) + 1}"

        context.thought_graph.add_node(
            insight_id,
            NodeType.INSIGHT,
            insight,
            properties={'source': source},
        )

        context.lessons_learned.append(insight)

        return insight_id

    # =========================================================================
    # PRODUCTION OPERATIONS
    # =========================================================================

    def create_production_chunk(
        self,
        context: WorkflowContext,
        name: str,
        goal: str,
        files: List[str] = None,
        estimate_minutes: int = 30
    ) -> ProductionChunk:
        """
        Create a production chunk for the current task.

        Args:
            context: Session context
            name: Chunk name
            goal: What this chunk accomplishes
            files: Files to be created/modified
            estimate_minutes: Time estimate

        Returns:
            ProductionChunk instance
        """
        task = self._production_manager.get_task(context.current_task_id)
        if not task:
            raise ValueError("No active production task")

        chunk = ProductionChunk(
            name=name,
            goal=goal,
            outputs=files or [],
            time_estimate_minutes=estimate_minutes,
        )
        task.add_chunk(chunk)

        return chunk

    def add_comment_marker(
        self,
        context: WorkflowContext,
        marker_type: str,
        content: str,
        file_path: str = None
    ) -> None:
        """
        Add a comment marker during production.

        Args:
            context: Session context
            marker_type: THINKING, TODO, QUESTION, NOTE, PERF, HACK
            content: Marker content
            file_path: File the marker relates to
        """
        task = self._production_manager.get_task(context.current_task_id)
        if task:
            marker = CommentMarker(
                marker_type=marker_type,
                content=content,
                file_path=file_path,
            )
            task.add_marker(marker)

    def record_artifact(self, context: WorkflowContext, artifact_path: str, artifact_type: str = "file") -> None:
        """
        Record an artifact produced during the session.

        Args:
            context: Session context
            artifact_path: Path to the artifact
            artifact_type: Type of artifact (file, doc, test, etc.)
        """
        context.artifacts_produced.append(artifact_path)

        # Update production task
        task = self._production_manager.get_task(context.current_task_id)
        if task:
            task.add_file(artifact_path)

        # Add to thought graph
        context.thought_graph.add_node(
            f"artifact_{len(context.artifacts_produced)}",
            NodeType.ARTIFACT,
            artifact_path,
            properties={'type': artifact_type},
        )

    # =========================================================================
    # CRISIS AND VERIFICATION
    # =========================================================================

    def report_crisis(
        self,
        context: WorkflowContext,
        level: CrisisLevel,
        description: str
    ) -> CrisisEvent:
        """
        Report a crisis event.

        Args:
            context: Session context
            level: Severity level
            description: What happened

        Returns:
            CrisisEvent instance
        """
        event = self._crisis_manager.record_crisis(
            level,
            description,
            context={'session_id': context.session_id, 'goal': context.goal},
        )

        # Update loop status if severe
        if level in (CrisisLevel.WALL, CrisisLevel.CRISIS):
            loop = self._loop_manager.get_loop(context.current_loop_id)
            if loop:
                loop.block(f"Crisis: {description}")

        return event

    def verify(self, context: WorkflowContext, phase: VerificationPhase = None) -> Dict[str, Any]:
        """
        Run verification checks.

        Args:
            context: Session context
            phase: Which verification phase (None = all applicable)

        Returns:
            Verification results
        """
        suite = self._verification_manager.get_suite(context.current_verification_suite)
        if not suite:
            return {'error': 'No verification suite'}

        checks = suite.get_checks_for_phase(phase) if phase else suite.checks
        results = {'passed': 0, 'failed': 0, 'pending': 0}

        for check in checks:
            # In a full implementation, we'd run the actual checks
            # For now, mark them as needing manual verification
            if check.status == VerificationStatus.PENDING:
                results['pending'] += 1
            elif check.status == VerificationStatus.PASSED:
                results['passed'] += 1
            elif check.status == VerificationStatus.FAILED:
                results['failed'] += 1

        return results

    # =========================================================================
    # COLLABORATION
    # =========================================================================

    def post_status(
        self,
        context: WorkflowContext,
        progress: int,
        current_activity: str
    ) -> StatusUpdate:
        """
        Post a status update for the session.

        Args:
            context: Session context
            progress: Progress percentage (0-100)
            current_activity: What's currently being done

        Returns:
            StatusUpdate instance
        """
        loop = self._loop_manager.get_loop(context.current_loop_id)
        phase = loop.current_phase.value if loop and loop.current_phase else "unknown"

        update = StatusUpdate(
            task_name=context.goal,
            progress_percent=progress,
            current_phase=phase,
            completed_items=[f"Phase: {p}" for p in ['question', 'answer', 'produce'] if p < phase],
            in_progress_items=[current_activity],
            blockers=[b.description for b in self._collaboration_manager.get_active_blockers()],
        )

        self._collaboration_manager.post_status(update)
        return update

    def raise_disagreement(
        self,
        context: WorkflowContext,
        instruction: str,
        concern: str,
        evidence: List[str],
        risk: str,
        alternative: str
    ) -> DisagreementRecord:
        """
        Raise a disagreement with current direction.

        Args:
            context: Session context
            instruction: What was instructed
            concern: Why there's disagreement
            evidence: Supporting evidence
            risk: Risk if proceeding
            alternative: Suggested alternative

        Returns:
            DisagreementRecord instance
        """
        return self._collaboration_manager.record_disagreement(
            instruction, concern, evidence, risk, alternative
        )

    def create_handoff(self, context: WorkflowContext) -> ActiveWorkHandoff:
        """
        Create a handoff document for the current session.

        Args:
            context: Session context

        Returns:
            ActiveWorkHandoff document
        """
        loop = self._loop_manager.get_loop(context.current_loop_id)
        phase = loop.current_phase.value if loop and loop.current_phase else "unknown"

        handoff = self._collaboration_manager.create_handoff(
            task=context.goal,
            status=f"In {phase} phase, {len(context.decisions_made)} decisions made",
            urgency="medium",
        )

        # Fill in from context
        handoff.files_working = context.artifacts_produced[:5]
        handoff.key_decisions = {d['decision']: d['rationale'] for d in context.decisions_made[-5:]}
        handoff.gotchas = context.lessons_learned[-5:]
        handoff.open_questions = [
            q['question'] for q in context.questions_answered
            if not q.get('answered', True)
        ][-5:]

        return handoff

    # =========================================================================
    # EVENT HANDLERS (Internal)
    # =========================================================================

    def _on_loop_transition(self, loop: CognitiveLoop, transition) -> None:
        """Handle loop transitions."""
        # Check for stuck loops (danger sign)
        for phase in LoopPhase:
            if loop.get_iteration_count(phase) >= 3:
                self._crisis_manager.record_crisis(
                    CrisisLevel.OBSTACLE,
                    f"Loop {loop.id} has iterated {phase.value} phase 3+ times",
                    context={'loop_id': loop.id, 'phase': phase.value},
                )
                break

    def _on_production_state_change(self, task: ProductionTask, old_state: ProductionState, new_state: ProductionState) -> None:
        """Handle production state changes."""
        # If moving to REWORK, that's a verification failure
        if new_state == ProductionState.REWORK:
            self._crisis_manager.record_crisis(
                CrisisLevel.HICCUP,
                f"Production task {task.id} entering rework",
                context={'task_id': task.id},
            )

    def _on_verification_failure(self, check: VerificationCheck, failure) -> None:
        """Handle verification failures."""
        self._crisis_manager.record_crisis(
            CrisisLevel.HICCUP,
            f"Verification failed: {check.name}",
            context={'check_name': check.name, 'level': check.level.name},
        )

    def _on_crisis_recorded(self, event: CrisisEvent) -> None:
        """Handle crisis events."""
        if event.level == CrisisLevel.CRISIS:
            # Post urgent status update
            for ctx in self._contexts.values():
                update = StatusUpdate(
                    task_name="CRISIS ALERT",
                    progress_percent=0,
                    current_phase="STOPPED",
                    blockers=[event.description],
                )
                self._collaboration_manager.post_status(update)

    # =========================================================================
    # REPORTING
    # =========================================================================

    def _generate_session_summary(self, context: WorkflowContext) -> Dict[str, Any]:
        """Generate a summary of the session."""
        loop = self._loop_manager.get_loop(context.current_loop_id)
        task = self._production_manager.get_task(context.current_task_id) if context.current_task_id else None

        return {
            'session_id': context.session_id,
            'goal': context.goal,
            'duration_minutes': (datetime.now() - context.started_at).total_seconds() / 60,
            'loop_status': loop.status.name if loop else None,
            'loop_iterations': sum(
                loop.get_iteration_count(phase) for phase in LoopPhase
            ) if loop else 0,
            'decisions_made': len(context.decisions_made),
            'questions_answered': len(context.questions_answered),
            'artifacts_produced': len(context.artifacts_produced),
            'lessons_learned': context.lessons_learned,
            'production_state': task.state.name if task else None,
            'thought_graph_nodes': context.thought_graph.node_count() if context.thought_graph else 0,
            'thought_graph_edges': context.thought_graph.edge_count() if context.thought_graph else 0,
        }

    def get_workflow_summary(self) -> Dict[str, Any]:
        """Get summary of all workflow components."""
        return {
            'active_sessions': len(self._contexts),
            'loops': self._loop_manager.get_summary(),
            'production': self._production_manager.get_summary(),
            'crises': self._crisis_manager.get_summary(),
            'verification': self._verification_manager.get_summary(),
            'collaboration': self._collaboration_manager.get_summary(),
        }
