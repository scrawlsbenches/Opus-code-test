#!/usr/bin/env python3
"""Unit tests for workflow template engine."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from workflow import (
    WorkflowVariable,
    WorkflowTask,
    Workflow,
    substitute_variables,
    run_workflow,
)


class TestWorkflowVariable(unittest.TestCase):
    """Tests for WorkflowVariable dataclass."""

    def test_variable_with_defaults(self):
        """WorkflowVariable should use default values."""
        var = WorkflowVariable(
            name="test_var",
            description="Test variable"
        )
        self.assertEqual(var.name, "test_var")
        self.assertEqual(var.description, "Test variable")
        self.assertTrue(var.required)
        self.assertIsNone(var.default)
        self.assertIsNone(var.choices)

    def test_variable_with_all_fields(self):
        """WorkflowVariable should accept all optional fields."""
        var = WorkflowVariable(
            name="priority",
            description="Task priority",
            required=False,
            default="medium",
            choices=["low", "medium", "high"]
        )
        self.assertEqual(var.name, "priority")
        self.assertEqual(var.description, "Task priority")
        self.assertFalse(var.required)
        self.assertEqual(var.default, "medium")
        self.assertEqual(var.choices, ["low", "medium", "high"])

    def test_variable_required_false(self):
        """WorkflowVariable should allow required=False."""
        var = WorkflowVariable(
            name="optional",
            description="Optional field",
            required=False
        )
        self.assertFalse(var.required)

    def test_variable_with_default_no_choices(self):
        """WorkflowVariable should allow default without choices."""
        var = WorkflowVariable(
            name="effort",
            description="Effort estimate",
            default="medium"
        )
        self.assertEqual(var.default, "medium")
        self.assertIsNone(var.choices)


class TestWorkflowTask(unittest.TestCase):
    """Tests for WorkflowTask dataclass."""

    def test_task_post_init_none_depends_on(self):
        """__post_init__ should initialize depends_on to empty list if None."""
        task = WorkflowTask(
            id="task1",
            title="Test Task",
            depends_on=None
        )
        self.assertEqual(task.depends_on, [])

    def test_task_post_init_preserves_depends_on(self):
        """__post_init__ should preserve non-None depends_on."""
        task = WorkflowTask(
            id="task1",
            title="Test Task",
            depends_on=["task0"]
        )
        self.assertEqual(task.depends_on, ["task0"])

    def test_task_with_defaults(self):
        """WorkflowTask should use default values."""
        task = WorkflowTask(
            id="task1",
            title="Test Task"
        )
        self.assertEqual(task.id, "task1")
        self.assertEqual(task.title, "Test Task")
        self.assertEqual(task.category, "general")
        self.assertEqual(task.priority, "medium")
        self.assertEqual(task.effort, "medium")
        self.assertEqual(task.description, "")
        self.assertEqual(task.depends_on, [])

    def test_task_with_all_fields(self):
        """WorkflowTask should accept all fields."""
        task = WorkflowTask(
            id="task1",
            title="Test Task",
            category="bugfix",
            priority="high",
            effort="large",
            description="Detailed description",
            depends_on=["task0", "task2"]
        )
        self.assertEqual(task.id, "task1")
        self.assertEqual(task.title, "Test Task")
        self.assertEqual(task.category, "bugfix")
        self.assertEqual(task.priority, "high")
        self.assertEqual(task.effort, "large")
        self.assertEqual(task.description, "Detailed description")
        self.assertEqual(task.depends_on, ["task0", "task2"])


class TestWorkflowFromDict(unittest.TestCase):
    """Tests for Workflow.from_dict() parsing."""

    def test_minimal_workflow(self):
        """from_dict should parse workflow with minimal fields."""
        data = {
            "name": "Test Workflow",
            "description": "A test workflow",
            "category": "test"
        }
        workflow = Workflow.from_dict(data)
        self.assertEqual(workflow.name, "Test Workflow")
        self.assertEqual(workflow.description, "A test workflow")
        self.assertEqual(workflow.category, "test")
        self.assertEqual(workflow.variables, [])
        self.assertEqual(workflow.tasks, [])

    def test_workflow_with_empty_variables(self):
        """from_dict should handle empty variables list."""
        data = {
            "name": "Test",
            "variables": []
        }
        workflow = Workflow.from_dict(data)
        self.assertEqual(workflow.variables, [])

    def test_workflow_with_empty_tasks(self):
        """from_dict should handle empty tasks list."""
        data = {
            "name": "Test",
            "tasks": []
        }
        workflow = Workflow.from_dict(data)
        self.assertEqual(workflow.tasks, [])

    def test_workflow_with_variables(self):
        """from_dict should parse variables with all optional fields."""
        data = {
            "name": "Test",
            "variables": [
                {
                    "name": "bug_title",
                    "description": "Bug title",
                    "required": True
                },
                {
                    "name": "priority",
                    "description": "Priority level",
                    "required": False,
                    "default": "medium",
                    "choices": ["low", "medium", "high"]
                }
            ]
        }
        workflow = Workflow.from_dict(data)
        self.assertEqual(len(workflow.variables), 2)

        var1 = workflow.variables[0]
        self.assertEqual(var1.name, "bug_title")
        self.assertEqual(var1.description, "Bug title")
        self.assertTrue(var1.required)
        self.assertIsNone(var1.default)
        self.assertIsNone(var1.choices)

        var2 = workflow.variables[1]
        self.assertEqual(var2.name, "priority")
        self.assertEqual(var2.description, "Priority level")
        self.assertFalse(var2.required)
        self.assertEqual(var2.default, "medium")
        self.assertEqual(var2.choices, ["low", "medium", "high"])

    def test_workflow_with_tasks(self):
        """from_dict should parse tasks with all fields."""
        data = {
            "name": "Test",
            "tasks": [
                {
                    "id": "task1",
                    "title": "Fix {bug_title}",
                    "category": "bugfix",
                    "priority": "high",
                    "effort": "small",
                    "description": "Fix the bug",
                    "depends_on": []
                },
                {
                    "id": "task2",
                    "title": "Test fix",
                    "depends_on": ["task1"]
                }
            ]
        }
        workflow = Workflow.from_dict(data)
        self.assertEqual(len(workflow.tasks), 2)

        task1 = workflow.tasks[0]
        self.assertEqual(task1.id, "task1")
        self.assertEqual(task1.title, "Fix {bug_title}")
        self.assertEqual(task1.category, "bugfix")
        self.assertEqual(task1.priority, "high")
        self.assertEqual(task1.effort, "small")
        self.assertEqual(task1.description, "Fix the bug")
        self.assertEqual(task1.depends_on, [])

        task2 = workflow.tasks[1]
        self.assertEqual(task2.id, "task2")
        self.assertEqual(task2.title, "Test fix")
        self.assertEqual(task2.category, "general")  # default
        self.assertEqual(task2.priority, "medium")  # default
        self.assertEqual(task2.depends_on, ["task1"])

    def test_workflow_default_category(self):
        """from_dict should use 'general' as default category."""
        data = {
            "name": "Test"
        }
        workflow = Workflow.from_dict(data)
        self.assertEqual(workflow.category, "general")

    def test_workflow_default_description(self):
        """from_dict should use empty string as default description."""
        data = {
            "name": "Test"
        }
        workflow = Workflow.from_dict(data)
        self.assertEqual(workflow.description, "")


class TestSubstituteVariables(unittest.TestCase):
    """Tests for substitute_variables() function."""

    def test_single_variable(self):
        """Should substitute single variable placeholder."""
        text = "Fix {bug_title}"
        variables = {"bug_title": "Login crash"}
        result = substitute_variables(text, variables)
        self.assertEqual(result, "Fix Login crash")

    def test_multiple_variables(self):
        """Should substitute multiple variable placeholders."""
        text = "Fix {bug_title} with {priority} priority"
        variables = {
            "bug_title": "Login crash",
            "priority": "high"
        }
        result = substitute_variables(text, variables)
        self.assertEqual(result, "Fix Login crash with high priority")

    def test_variable_not_in_dict(self):
        """Should leave placeholder unchanged if variable not in dict."""
        text = "Fix {bug_title} with {priority}"
        variables = {"bug_title": "Login crash"}
        result = substitute_variables(text, variables)
        self.assertEqual(result, "Fix Login crash with {priority}")

    def test_empty_variables_dict(self):
        """Should return original text if variables dict is empty."""
        text = "Fix {bug_title}"
        variables = {}
        result = substitute_variables(text, variables)
        self.assertEqual(result, "Fix {bug_title}")

    def test_no_placeholders(self):
        """Should return original text if no placeholders."""
        text = "Simple text without placeholders"
        variables = {"bug_title": "Something"}
        result = substitute_variables(text, variables)
        self.assertEqual(result, "Simple text without placeholders")

    def test_multiple_occurrences(self):
        """Should substitute all occurrences of same variable."""
        text = "{priority} bug: {bug_title} needs {priority} attention"
        variables = {
            "bug_title": "Login crash",
            "priority": "high"
        }
        result = substitute_variables(text, variables)
        self.assertEqual(result, "high bug: Login crash needs high attention")

    def test_nested_braces(self):
        """Should handle text with non-variable braces."""
        text = "Code: function() { return {value}; }"
        variables = {"value": "42"}
        result = substitute_variables(text, variables)
        self.assertEqual(result, "Code: function() { return 42; }")


class TestRunWorkflow(unittest.TestCase):
    """Tests for run_workflow() function."""

    def test_required_variable_missing_raises_error(self):
        """Should raise ValueError if required variable is missing."""
        workflow = Workflow(
            name="Test",
            description="Test workflow",
            category="test",
            variables=[
                WorkflowVariable(
                    name="bug_title",
                    description="Bug title",
                    required=True
                )
            ],
            tasks=[]
        )
        variables = {}

        with self.assertRaises(ValueError) as cm:
            run_workflow(workflow, variables, dry_run=True)

        self.assertIn("Missing required variable: bug_title", str(cm.exception))

    def test_required_variable_with_default_uses_default(self):
        """Should use default value if required variable is missing but has default."""
        workflow = Workflow(
            name="Test",
            description="Test workflow",
            category="test",
            variables=[
                WorkflowVariable(
                    name="priority",
                    description="Priority",
                    required=True,
                    default="medium"
                )
            ],
            tasks=[
                WorkflowTask(
                    id="task1",
                    title="Task with {priority} priority",
                    priority="{priority}"
                )
            ]
        )
        variables = {}

        tasks = run_workflow(workflow, variables, dry_run=True)

        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].title, "Task with medium priority")
        self.assertEqual(tasks[0].priority, "medium")

    def test_choice_validation_valid(self):
        """Should accept valid choice value."""
        workflow = Workflow(
            name="Test",
            description="Test workflow",
            category="test",
            variables=[
                WorkflowVariable(
                    name="priority",
                    description="Priority",
                    choices=["low", "medium", "high"]
                )
            ],
            tasks=[
                WorkflowTask(
                    id="task1",
                    title="Task"
                )
            ]
        )
        variables = {"priority": "high"}

        # Should not raise
        tasks = run_workflow(workflow, variables, dry_run=True)
        self.assertEqual(len(tasks), 1)

    def test_choice_validation_invalid_raises_error(self):
        """Should raise ValueError if choice value is invalid."""
        workflow = Workflow(
            name="Test",
            description="Test workflow",
            category="test",
            variables=[
                WorkflowVariable(
                    name="priority",
                    description="Priority",
                    choices=["low", "medium", "high"]
                )
            ],
            tasks=[]
        )
        variables = {"priority": "invalid"}

        with self.assertRaises(ValueError) as cm:
            run_workflow(workflow, variables, dry_run=True)

        self.assertIn("Invalid value for priority: invalid", str(cm.exception))
        self.assertIn("Must be one of:", str(cm.exception))

    @patch('workflow.TaskSession')
    def test_dry_run_mode_no_save(self, mock_session_class):
        """Dry run should create tasks but not save them."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        # Mock the task creation
        mock_task = MagicMock()
        mock_task.id = "T-test-001"
        mock_task.title = "Test Task"
        mock_task.priority = "medium"
        mock_task.description = ""
        mock_task.depends_on = []
        mock_session.create_task.return_value = mock_task

        workflow = Workflow(
            name="Test",
            description="Test workflow",
            category="test",
            variables=[],
            tasks=[
                WorkflowTask(
                    id="task1",
                    title="Test Task"
                )
            ]
        )
        variables = {}

        tasks = run_workflow(workflow, variables, dry_run=True)

        self.assertEqual(len(tasks), 1)
        # save() should NOT be called in dry run mode
        mock_session.save.assert_not_called()

    @patch('workflow.TaskSession')
    def test_normal_mode_saves(self, mock_session_class):
        """Normal mode should create and save tasks."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        # Mock the task creation
        mock_task = MagicMock()
        mock_task.id = "T-test-001"
        mock_task.title = "Test Task"
        mock_task.priority = "medium"
        mock_task.description = ""
        mock_task.depends_on = []
        mock_session.create_task.return_value = mock_task
        mock_session.save.return_value = Path("tasks/test.json")

        workflow = Workflow(
            name="Test",
            description="Test workflow",
            category="test",
            variables=[],
            tasks=[
                WorkflowTask(
                    id="task1",
                    title="Test Task"
                )
            ]
        )
        variables = {}

        tasks = run_workflow(workflow, variables, dry_run=False)

        self.assertEqual(len(tasks), 1)
        # save() SHOULD be called in normal mode
        mock_session.save.assert_called_once_with("tasks")

    @patch('workflow.TaskSession')
    def test_task_dependency_resolution(self, mock_session_class):
        """Should resolve workflow task IDs to actual task IDs."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        # Mock task creation to return different IDs
        mock_task1 = MagicMock()
        mock_task1.id = "T-actual-001"
        mock_task1.title = "Task 1"
        mock_task1.priority = "medium"
        mock_task1.description = ""
        mock_task1.depends_on = []

        mock_task2 = MagicMock()
        mock_task2.id = "T-actual-002"
        mock_task2.title = "Task 2"
        mock_task2.priority = "medium"
        mock_task2.description = ""
        mock_task2.depends_on = ["T-actual-001"]

        mock_session.create_task.side_effect = [mock_task1, mock_task2]

        workflow = Workflow(
            name="Test",
            description="Test workflow",
            category="test",
            variables=[],
            tasks=[
                WorkflowTask(
                    id="wf_task1",
                    title="Task 1"
                ),
                WorkflowTask(
                    id="wf_task2",
                    title="Task 2",
                    depends_on=["wf_task1"]
                )
            ]
        )
        variables = {}

        tasks = run_workflow(workflow, variables, dry_run=True)

        self.assertEqual(len(tasks), 2)
        # Second task should have actual ID of first task in depends_on
        second_call = mock_session.create_task.call_args_list[1]
        self.assertEqual(second_call[1]['depends_on'], ["T-actual-001"])

    @patch('workflow.TaskSession')
    def test_variable_substitution_in_all_fields(self, mock_session_class):
        """Should substitute variables in title, description, priority, effort."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        mock_task = MagicMock()
        mock_task.id = "T-test-001"
        mock_task.title = "Fix Login crash"
        mock_task.priority = "high"
        mock_task.effort = "large"
        mock_task.description = "Critical bug: Login crash"
        mock_task.depends_on = []
        mock_session.create_task.return_value = mock_task

        workflow = Workflow(
            name="Test",
            description="Test workflow",
            category="test",
            variables=[
                WorkflowVariable(name="bug_title", description="Bug title"),
                WorkflowVariable(name="priority", description="Priority"),
                WorkflowVariable(name="effort", description="Effort")
            ],
            tasks=[
                WorkflowTask(
                    id="task1",
                    title="Fix {bug_title}",
                    priority="{priority}",
                    effort="{effort}",
                    description="Critical bug: {bug_title}"
                )
            ]
        )
        variables = {
            "bug_title": "Login crash",
            "priority": "high",
            "effort": "large"
        }

        tasks = run_workflow(workflow, variables, dry_run=True)

        # Verify substitution happened in create_task call
        call_args = mock_session.create_task.call_args
        self.assertEqual(call_args[1]['title'], "Fix Login crash")
        self.assertEqual(call_args[1]['priority'], "high")
        self.assertEqual(call_args[1]['effort'], "large")
        self.assertEqual(call_args[1]['description'], "Critical bug: Login crash")

    @patch('workflow.TaskSession')
    def test_multiple_tasks_created(self, mock_session_class):
        """Should create all tasks in workflow."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        mock_tasks = []
        for i in range(3):
            mock_task = MagicMock()
            mock_task.id = f"T-test-{i:03d}"
            mock_task.title = f"Task {i+1}"
            mock_task.priority = "medium"
            mock_task.description = ""
            mock_task.depends_on = []
            mock_tasks.append(mock_task)

        mock_session.create_task.side_effect = mock_tasks

        workflow = Workflow(
            name="Test",
            description="Test workflow",
            category="test",
            variables=[],
            tasks=[
                WorkflowTask(id="task1", title="Task 1"),
                WorkflowTask(id="task2", title="Task 2"),
                WorkflowTask(id="task3", title="Task 3")
            ]
        )
        variables = {}

        tasks = run_workflow(workflow, variables, dry_run=True)

        self.assertEqual(len(tasks), 3)
        self.assertEqual(mock_session.create_task.call_count, 3)

    @patch('workflow.TaskSession')
    def test_custom_tasks_dir(self, mock_session_class):
        """Should use custom tasks directory when provided."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        mock_task = MagicMock()
        mock_task.id = "T-test-001"
        mock_task.title = "Test Task"
        mock_task.priority = "medium"
        mock_task.description = ""
        mock_task.depends_on = []
        mock_session.create_task.return_value = mock_task
        mock_session.save.return_value = Path("custom_dir/test.json")

        workflow = Workflow(
            name="Test",
            description="Test workflow",
            category="test",
            variables=[],
            tasks=[
                WorkflowTask(id="task1", title="Test Task")
            ]
        )
        variables = {}

        tasks = run_workflow(workflow, variables, tasks_dir="custom_dir", dry_run=False)

        mock_session.save.assert_called_once_with("custom_dir")

    @patch('workflow.TaskSession')
    def test_missing_dependency_ignored(self, mock_session_class):
        """Should ignore dependencies that haven't been created yet."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        mock_task = MagicMock()
        mock_task.id = "T-test-001"
        mock_task.title = "Task"
        mock_task.priority = "medium"
        mock_task.description = ""
        mock_task.depends_on = []
        mock_session.create_task.return_value = mock_task

        workflow = Workflow(
            name="Test",
            description="Test workflow",
            category="test",
            variables=[],
            tasks=[
                WorkflowTask(
                    id="task1",
                    title="Task",
                    depends_on=["nonexistent_task"]
                )
            ]
        )
        variables = {}

        tasks = run_workflow(workflow, variables, dry_run=True)

        # Should create task with empty depends_on (missing dependency ignored)
        call_args = mock_session.create_task.call_args
        self.assertEqual(call_args[1]['depends_on'], [])


if __name__ == "__main__":
    unittest.main()
