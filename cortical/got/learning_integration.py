"""
Learning Integration: Bridge between GoT Task System and LearningCycle

This module connects the Graph of Thought (GoT) transactional task system
with the LearningCycle experience capture and pattern extraction system.

Key Features:
- Converts completed tasks into learning experiences
- Maps task metadata to learning contexts
- Captures retrospectives as reflection data
- Retrieves lessons to guide new tasks
- Auto-tags experiences based on task properties

Storage: .got/learning/ subdirectory for experiences, patterns, and lessons.
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Set

from llm_orchestration.learning import (
    LearningCycle,
    Experience,
    Context,
    Action,
    Outcome,
    OutcomeType,
    ExperienceType,
)

logger = logging.getLogger(__name__)


class GoTLearningBridge:
    """
    Bridge between GoT task system and LearningCycle.

    Provides bidirectional integration:
    - Task completion → Experience capture
    - Task planning → Lesson retrieval
    - Task failure → Failure pattern capture

    Usage:
        bridge = GoTLearningBridge(got_dir)

        # When task completes
        bridge.capture_task_completion(
            task_id="T-123",
            retrospective="Used TDD, tests passed first try",
            files_changed=["api.py", "test_api.py"],
            approach="test-first"
        )

        # When planning new task
        guidance = bridge.get_guidance_for_task(
            task_title="Implement user authentication",
            task_category="feature"
        )
    """

    def __init__(self, got_dir: Path):
        """
        Initialize the learning bridge.

        Args:
            got_dir: Base GoT directory (e.g., /path/to/.got)
        """
        self.got_dir = Path(got_dir)
        self.learning_dir = self.got_dir / "learning"
        self.learning_dir.mkdir(parents=True, exist_ok=True)

        # Initialize learning cycle with subdirectory
        self.cycle = LearningCycle(self.learning_dir)

        logger.info(f"GoTLearningBridge initialized at {self.learning_dir}")

    def capture_task_completion(
        self,
        task_id: str,
        retrospective: str = "",
        files_changed: Optional[List[str]] = None,
        approach: Optional[str] = None,
        task_title: str = "",
        task_category: str = "general",
        task_priority: str = "medium",
        duration_seconds: Optional[float] = None,
    ) -> Experience:
        """
        Capture a completed task as a learning experience.

        Converts task completion into an Experience with:
        - Context derived from task metadata
        - Actions inferred from files changed
        - Outcome as SUCCESS
        - Reflection from retrospective

        Args:
            task_id: Task identifier (e.g., "T-20260103-123456-abc123")
            retrospective: Task completion notes/lessons learned
            files_changed: List of files modified during task
            approach: Strategy/approach used (e.g., "test-first", "refactor")
            task_title: Task title for context
            task_category: Task category (feature, bugfix, refactor, docs, test)
            task_priority: Task priority (critical, high, medium, low)
            duration_seconds: How long the task took

        Returns:
            The created Experience object
        """
        # Map task category to goal type
        goal_type = self._map_category_to_goal_type(task_category)

        # Map task priority to complexity
        goal_complexity = self._map_priority_to_complexity(task_priority)

        # Build context
        context = Context(
            goal_type=goal_type,
            goal_complexity=goal_complexity,
            domain=self._infer_domain_from_files(files_changed or []),
            available_tools=self._infer_tools_from_files(files_changed or []),
            notes=f"Task: {task_id} - {task_title}"
        )

        # Start experience
        experience = self.cycle.start_experience(
            context=context,
            intent=task_title or f"Complete task {task_id}",
            experience_type=ExperienceType.TASK_EXECUTION,
            strategy=approach
        )

        # Add actions based on files changed
        if files_changed:
            for file_path in files_changed:
                action = Action(
                    action_type=self._infer_action_type(file_path),
                    description=f"Modified {file_path}",
                    target=file_path,
                    parameters={"task_id": task_id},
                    timestamp=datetime.now(),
                    duration_ms=None
                )
                experience.add_action(action)

        # Create successful outcome
        outcome = Outcome(
            outcome_type=OutcomeType.SUCCESS,
            description=f"Task {task_id} completed successfully",
            achieved=[task_title or task_id],
            quality_score=1.0,
            efficiency_score=self._compute_efficiency_score(duration_seconds)
        )

        # Parse retrospective into structured reflection
        reflection = self._parse_retrospective(retrospective)

        # Complete and save experience
        self.cycle.complete_experience(
            experience=experience,
            outcome=outcome,
            reflection=reflection
        )

        # Add task-specific tags
        experience.tags.add(f"task:{task_id}")
        experience.tags.add(f"category:{task_category}")
        experience.tags.add(f"priority:{task_priority}")
        if approach:
            experience.tags.add(f"approach:{approach}")

        # Re-save with additional tags
        self.cycle.store.save(experience)

        logger.info(f"Captured task completion: {task_id} -> {experience.id}")
        return experience

    def capture_task_failure(
        self,
        task_id: str,
        error_message: str,
        attempted_approach: Optional[str] = None,
        task_title: str = "",
        task_category: str = "general",
        task_priority: str = "medium",
        files_attempted: Optional[List[str]] = None,
        blockers: Optional[List[str]] = None,
    ) -> Experience:
        """
        Capture a failed task attempt as a learning experience.

        Records what went wrong and what was attempted, enabling
        future avoidance of similar failures.

        Args:
            task_id: Task identifier
            error_message: Description of what went wrong
            attempted_approach: Strategy that was tried
            task_title: Task title
            task_category: Task category
            task_priority: Task priority
            files_attempted: Files that were attempted to be modified
            blockers: List of blocking issues encountered

        Returns:
            The created Experience object
        """
        # Build context
        context = Context(
            goal_type=self._map_category_to_goal_type(task_category),
            goal_complexity=self._map_priority_to_complexity(task_priority),
            domain=self._infer_domain_from_files(files_attempted or []),
            prior_failures=1,
            constraints=blockers or [],
            notes=f"Task: {task_id} - {task_title} (FAILED)"
        )

        # Start experience
        experience = self.cycle.start_experience(
            context=context,
            intent=task_title or f"Complete task {task_id}",
            experience_type=ExperienceType.TASK_EXECUTION,
            strategy=attempted_approach
        )

        # Add attempted actions
        if files_attempted:
            for file_path in files_attempted:
                action = Action(
                    action_type=self._infer_action_type(file_path),
                    description=f"Attempted to modify {file_path}",
                    target=file_path,
                    parameters={"task_id": task_id, "failed": True}
                )
                experience.add_action(action)

        # Create failure outcome
        outcome = Outcome(
            outcome_type=OutcomeType.FAILURE,
            description=f"Task {task_id} failed: {error_message}",
            not_achieved=[task_title or task_id],
            error_message=error_message,
            quality_score=0.0,
            efficiency_score=0.0
        )

        # Add failure reflection
        reflection = {
            'worked': [],
            'didnt_work': [attempted_approach or "Unknown approach", error_message],
            'different': ["Need alternative approach", "Address blockers first"]
        }

        if blockers:
            reflection['didnt_work'].extend([f"Blocker: {b}" for b in blockers])

        # Complete and save
        self.cycle.complete_experience(
            experience=experience,
            outcome=outcome,
            reflection=reflection
        )

        # Add failure-specific tags
        experience.tags.add(f"task:{task_id}")
        experience.tags.add(f"category:{task_category}")
        experience.tags.add("failure")
        if attempted_approach:
            experience.tags.add(f"failed_approach:{attempted_approach}")

        # Re-save with tags
        self.cycle.store.save(experience)

        logger.warning(f"Captured task failure: {task_id} -> {experience.id}")
        return experience

    def get_guidance_for_task(
        self,
        task_title: str,
        task_category: str = "general",
        task_priority: str = "medium",
        files_to_modify: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve relevant lessons and experiences for planning a task.

        Queries the learning system for:
        - Applicable lessons from patterns
        - Successful similar tasks
        - Failed similar tasks (to avoid)
        - Recommendations and warnings

        Args:
            task_title: Title of task being planned
            task_category: Category (feature, bugfix, etc.)
            task_priority: Priority level
            files_to_modify: Files that will likely be modified

        Returns:
            Dictionary with keys:
            - lessons: List of Lesson objects
            - recommendations: List of recommendation strings
            - warnings: List of warning strings
            - relevant_successes: List of successful Experience objects
            - relevant_failures: List of failed Experience objects
        """
        # Build context for the upcoming task
        context = Context(
            goal_type=self._map_category_to_goal_type(task_category),
            goal_complexity=self._map_priority_to_complexity(task_priority),
            domain=self._infer_domain_from_files(files_to_modify or []),
            notes=f"Planning: {task_title}"
        )

        # Get guidance from learning cycle
        guidance = self.cycle.get_guidance(context, include_experiences=True)

        logger.info(
            f"Retrieved guidance for '{task_title}': "
            f"{len(guidance['lessons'])} lessons, "
            f"{len(guidance['relevant_successes'])} successes, "
            f"{len(guidance['relevant_failures'])} failures"
        )

        return guidance

    def link_task_to_experiences(
        self,
        task_id: str,
        task_category: str = "general",
        task_title: str = "",
    ) -> List[Experience]:
        """
        Find past experiences related to a task.

        Uses tags and context similarity to find relevant experiences.

        Args:
            task_id: Task identifier
            task_category: Task category for filtering
            task_title: Task title for context matching

        Returns:
            List of related Experience objects
        """
        related = []

        # Search by category tag
        category_tag = f"category:{task_category}"
        tagged_experiences = self.cycle.store.find_by_tags({category_tag})

        # If we have a title, also search by context similarity
        if task_title:
            context = Context(
                goal_type=self._map_category_to_goal_type(task_category),
                goal_complexity="moderate",
                notes=task_title
            )
            similar = self.cycle.store.find_similar_context(
                context,
                min_similarity=0.3,
                limit=10
            )
            # Combine results
            related_ids = {exp.id for exp in tagged_experiences}
            for exp, similarity in similar:
                if exp.id not in related_ids:
                    related.append(exp)
                    related_ids.add(exp.id)

            # Add tagged experiences
            related.extend(tagged_experiences)
        else:
            related = tagged_experiences

        logger.debug(f"Found {len(related)} related experiences for {task_id}")
        return related

    def extract_patterns_and_lessons(self) -> Dict[str, int]:
        """
        Run pattern extraction and lesson distillation.

        Should be called periodically (e.g., after every 10 task completions)
        to update the learning system.

        Returns:
            Dictionary with counts of patterns and lessons extracted
        """
        logger.info("Extracting patterns and distilling lessons...")
        results = self.cycle.extract_and_distill()
        logger.info(
            f"Extraction complete: "
            f"{results['sequence_patterns']} sequence patterns, "
            f"{results['strategy_patterns']} strategy patterns, "
            f"{results['antipatterns']} antipatterns, "
            f"{results['lessons']} lessons"
        )
        return results

    def get_learning_stats(self) -> Dict[str, Any]:
        """
        Get statistics about captured learning.

        Returns:
            Dictionary with experience counts, pattern counts, lesson counts
        """
        return self.cycle.get_stats()

    # ==================== Private Helper Methods ====================

    def _map_category_to_goal_type(self, category: str) -> str:
        """Map task category to learning goal type."""
        mapping = {
            "feature": "implementation",
            "bugfix": "debugging",
            "refactor": "refactoring",
            "docs": "documentation",
            "test": "testing",
            "chore": "maintenance",
        }
        return mapping.get(category.lower(), "general")

    def _map_priority_to_complexity(self, priority: str) -> str:
        """Map task priority to goal complexity."""
        mapping = {
            "critical": "complex",
            "high": "complex",
            "medium": "moderate",
            "low": "simple",
        }
        return mapping.get(priority.lower(), "moderate")

    def _infer_domain_from_files(self, files: List[str]) -> str:
        """Infer domain from file paths."""
        if not files:
            return "general"

        # Extract common directory prefixes
        dirs = set()
        for file_path in files:
            parts = Path(file_path).parts
            if len(parts) > 1:
                dirs.add(parts[0])

        if dirs:
            # Return most common top-level directory
            return sorted(dirs)[0]

        return "general"

    def _infer_tools_from_files(self, files: List[str]) -> List[str]:
        """Infer tools used from file extensions."""
        tools = set()

        for file_path in files:
            ext = Path(file_path).suffix.lower()

            if ext == ".py":
                tools.add("python")
            elif ext in [".js", ".ts", ".tsx", ".jsx"]:
                tools.add("javascript")
            elif ext in [".md", ".rst", ".txt"]:
                tools.add("documentation")
            elif ext in [".json", ".yaml", ".yml", ".toml"]:
                tools.add("configuration")
            elif ext in [".sh", ".bash"]:
                tools.add("shell")

        return list(tools)

    def _infer_action_type(self, file_path: str) -> str:
        """Infer action type from file path."""
        path = Path(file_path)

        if "test" in path.name.lower():
            return "write_test"
        elif path.suffix == ".md":
            return "write_documentation"
        elif "api" in path.name.lower():
            return "implement_api"
        elif path.suffix == ".py":
            return "write_code"
        else:
            return "modify_file"

    def _compute_efficiency_score(
        self,
        duration_seconds: Optional[float]
    ) -> Optional[float]:
        """
        Compute efficiency score based on duration.

        Uses heuristic thresholds:
        - < 1 hour: 1.0 (very efficient)
        - 1-4 hours: 0.8 (efficient)
        - 4-8 hours: 0.6 (moderate)
        - > 8 hours: 0.4 (slow)

        Args:
            duration_seconds: Task duration in seconds

        Returns:
            Efficiency score 0.0-1.0, or None if duration unknown
        """
        if duration_seconds is None:
            return None

        hours = duration_seconds / 3600

        if hours < 1:
            return 1.0
        elif hours < 4:
            return 0.8
        elif hours < 8:
            return 0.6
        else:
            return 0.4

    def _parse_retrospective(self, retrospective: str) -> Dict[str, List[str]]:
        """
        Parse retrospective text into structured reflection.

        Looks for keywords like "worked", "didn't work", "would do differently".
        Falls back to simple heuristics if not found.

        Args:
            retrospective: Free-form retrospective text

        Returns:
            Dictionary with keys: 'worked', 'didnt_work', 'different'
        """
        reflection = {
            'worked': [],
            'didnt_work': [],
            'different': []
        }

        if not retrospective:
            return reflection

        # Simple keyword-based parsing
        lower_retro = retrospective.lower()

        # Split by sentences
        sentences = [s.strip() for s in retrospective.split('.') if s.strip()]

        for sentence in sentences:
            lower_sent = sentence.lower()

            # Check for positive indicators
            if any(word in lower_sent for word in ['worked', 'successful', 'good', 'effective']):
                reflection['worked'].append(sentence)

            # Check for negative indicators
            elif any(word in lower_sent for word in ['failed', 'problem', 'issue', 'difficult', 'struggled']):
                reflection['didnt_work'].append(sentence)

            # Check for improvement indicators
            elif any(word in lower_sent for word in ['next time', 'should have', 'could have', 'would']):
                reflection['different'].append(sentence)

        # If no structured data extracted, put whole text in 'worked'
        if not any(reflection.values()):
            reflection['worked'] = [retrospective]

        return reflection
