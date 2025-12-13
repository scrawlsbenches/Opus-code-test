#!/usr/bin/env python3
"""
Integration tests for workflow template engine.

These tests verify that the workflow system loads real templates,
executes them correctly, and creates valid task files with proper
dependency resolution.

Run with: pytest tests/integration/test_workflow_integration.py -v
"""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from workflow import (
    Workflow,
    WorkflowVariable,
    WorkflowTask,
    list_workflows,
    run_workflow,
    substitute_variables,
    WORKFLOWS_DIR,
)
from task_utils import Task, TaskSession, load_all_tasks


class TestWorkflowLoading(unittest.TestCase):
    """Test loading real workflow templates from .claude/workflows/."""

    def test_load_bugfix_workflow(self):
        """Verify bugfix.yaml loads correctly."""
        workflow_path = WORKFLOWS_DIR / "bugfix.yaml"
        self.assertTrue(workflow_path.exists(), "bugfix.yaml should exist")

        workflow = Workflow.load(workflow_path)

        self.assertEqual(workflow.name, "Bug Fix")
        self.assertEqual(workflow.category, "bugfix")
        self.assertEqual(len(workflow.tasks), 4, "Bugfix workflow should have 4 tasks")
        self.assertGreater(len(workflow.variables), 0, "Should have variables")

        # Verify task IDs
        task_ids = [t.id for t in workflow.tasks]
        self.assertIn("investigate", task_ids)
        self.assertIn("fix", task_ids)
        self.assertIn("test", task_ids)
        self.assertIn("document", task_ids)

        # Verify dependencies
        fix_task = next(t for t in workflow.tasks if t.id == "fix")
        self.assertIn("investigate", fix_task.depends_on)

    def test_load_feature_workflow(self):
        """Verify feature.yaml loads correctly."""
        workflow_path = WORKFLOWS_DIR / "feature.yaml"
        self.assertTrue(workflow_path.exists(), "feature.yaml should exist")

        workflow = Workflow.load(workflow_path)

        self.assertEqual(workflow.name, "Feature")
        self.assertEqual(workflow.category, "feature")
        self.assertEqual(len(workflow.tasks), 5, "Feature workflow should have 5 tasks")

        # Verify task IDs
        task_ids = [t.id for t in workflow.tasks]
        self.assertIn("design", task_ids)
        self.assertIn("implement", task_ids)
        self.assertIn("unit_tests", task_ids)
        self.assertIn("integration_tests", task_ids)
        self.assertIn("documentation", task_ids)

        # Verify complex dependencies
        doc_task = next(t for t in workflow.tasks if t.id == "documentation")
        self.assertIn("unit_tests", doc_task.depends_on)
        self.assertIn("integration_tests", doc_task.depends_on)

    def test_load_refactor_workflow(self):
        """Verify refactor.yaml loads correctly."""
        workflow_path = WORKFLOWS_DIR / "refactor.yaml"
        self.assertTrue(workflow_path.exists(), "refactor.yaml should exist")

        workflow = Workflow.load(workflow_path)

        self.assertEqual(workflow.name, "Refactor")
        self.assertEqual(workflow.category, "refactor")
        self.assertEqual(len(workflow.tasks), 4, "Refactor workflow should have 4 tasks")

        # Verify task IDs
        task_ids = [t.id for t in workflow.tasks]
        self.assertIn("analyze", task_ids)
        self.assertIn("refactor", task_ids)
        self.assertIn("verify", task_ids)
        self.assertIn("cleanup", task_ids)

    def test_list_workflows_returns_all(self):
        """Verify list_workflows finds all three templates."""
        workflows = list_workflows()

        self.assertGreaterEqual(len(workflows), 3, "Should find at least 3 workflows")

        workflow_names = [w.name for w in workflows]
        self.assertIn("Bug Fix", workflow_names)
        self.assertIn("Feature", workflow_names)
        self.assertIn("Refactor", workflow_names)

    def test_workflow_variables_have_correct_types(self):
        """Verify workflow variables are parsed correctly."""
        workflow = Workflow.load(WORKFLOWS_DIR / "bugfix.yaml")

        # Find bug_title variable
        bug_title_var = next(v for v in workflow.variables if v.name == "bug_title")
        self.assertTrue(bug_title_var.required)
        self.assertIsNone(bug_title_var.default)

        # Find priority variable with choices
        priority_var = next(v for v in workflow.variables if v.name == "priority")
        self.assertIsNotNone(priority_var.choices)
        self.assertIn("high", priority_var.choices)
        self.assertIn("medium", priority_var.choices)
        self.assertIn("low", priority_var.choices)


class TestWorkflowExecution(unittest.TestCase):
    """Test end-to-end workflow execution with real file creation."""

    def setUp(self):
        """Create temporary directory for tasks."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_bugfix_workflow_creates_four_tasks(self):
        """Verify bugfix workflow creates exactly 4 tasks."""
        workflow = Workflow.load(WORKFLOWS_DIR / "bugfix.yaml")
        variables = {
            "bug_title": "Login crashes on special chars",
            "priority": "high"
        }

        tasks = run_workflow(workflow, variables, tasks_dir=self.temp_dir, dry_run=True)

        self.assertEqual(len(tasks), 4, "Bugfix workflow should create 4 tasks")
        self.assertIsInstance(tasks[0], Task)

        # Verify titles contain substituted bug_title
        for task in tasks:
            self.assertIn("Login crashes on special chars", task.title)

    def test_bugfix_workflow_saves_to_json(self):
        """Verify workflow creates valid JSON file."""
        workflow = Workflow.load(WORKFLOWS_DIR / "bugfix.yaml")
        variables = {
            "bug_title": "Null pointer in auth module",
            "priority": "high"
        }

        tasks = run_workflow(workflow, variables, tasks_dir=self.temp_dir, dry_run=False)

        # Check that JSON file was created
        json_files = list(Path(self.temp_dir).glob("*.json"))
        self.assertEqual(len(json_files), 1, "Should create exactly one JSON file")

        # Verify JSON is valid
        with open(json_files[0]) as f:
            data = json.load(f)

        self.assertEqual(data["version"], 1)
        self.assertIn("tasks", data)
        self.assertEqual(len(data["tasks"]), 4)

        # Verify tasks have correct structure
        for task_data in data["tasks"]:
            self.assertIn("id", task_data)
            self.assertIn("title", task_data)
            self.assertIn("status", task_data)
            self.assertIn("depends_on", task_data)

    def test_feature_workflow_creates_five_tasks(self):
        """Verify feature workflow creates exactly 5 tasks."""
        workflow = Workflow.load(WORKFLOWS_DIR / "feature.yaml")
        variables = {
            "feature_name": "Dark mode toggle",
            "priority": "medium",
            "effort": "large"
        }

        tasks = run_workflow(workflow, variables, tasks_dir=self.temp_dir, dry_run=True)

        self.assertEqual(len(tasks), 5, "Feature workflow should create 5 tasks")

        # Verify task categories
        categories = [t.category for t in tasks]
        self.assertIn("arch", categories)  # design task
        self.assertIn("feature", categories)  # implement task
        self.assertIn("test", categories)  # test tasks
        self.assertIn("docs", categories)  # documentation task

    def test_refactor_workflow_creates_four_tasks(self):
        """Verify refactor workflow creates exactly 4 tasks."""
        workflow = Workflow.load(WORKFLOWS_DIR / "refactor.yaml")
        variables = {
            "refactor_target": "query module split",
            "priority": "medium"
        }

        tasks = run_workflow(workflow, variables, tasks_dir=self.temp_dir, dry_run=True)

        self.assertEqual(len(tasks), 4, "Refactor workflow should create 4 tasks")

        # Verify titles
        titles = [t.title for t in tasks]
        self.assertTrue(any("Analyze" in t for t in titles))
        self.assertTrue(any("Refactor" in t for t in titles))
        self.assertTrue(any("Verify" in t for t in titles))
        self.assertTrue(any("Cleanup" in t for t in titles))


class TestDependencyResolution(unittest.TestCase):
    """Test that workflow dependencies are resolved to actual task IDs."""

    def setUp(self):
        """Create temporary directory for tasks."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_bugfix_dependency_chain(self):
        """Verify bugfix workflow dependency chain is correct."""
        workflow = Workflow.load(WORKFLOWS_DIR / "bugfix.yaml")
        variables = {"bug_title": "Test bug", "priority": "high"}

        tasks = run_workflow(workflow, variables, tasks_dir=self.temp_dir, dry_run=True)

        # Build task map by title pattern
        investigate = next(t for t in tasks if "Investigate" in t.title)
        fix = next(t for t in tasks if "Fix:" in t.title)
        test = next(t for t in tasks if "regression test" in t.title)
        document = next(t for t in tasks if "Document" in t.title)

        # Verify dependencies use actual task IDs, not template IDs
        self.assertEqual(len(investigate.depends_on), 0, "Investigate has no dependencies")
        self.assertIn(investigate.id, fix.depends_on, "Fix depends on investigate")
        self.assertIn(fix.id, test.depends_on, "Test depends on fix")
        self.assertIn(fix.id, document.depends_on, "Document depends on fix")

        # Verify IDs are not template IDs
        self.assertNotIn("investigate", fix.depends_on)
        self.assertNotIn("fix", test.depends_on)

    def test_feature_complex_dependencies(self):
        """Verify feature workflow has correct dependency graph."""
        workflow = Workflow.load(WORKFLOWS_DIR / "feature.yaml")
        variables = {
            "feature_name": "Test feature",
            "priority": "medium",
            "effort": "large"
        }

        tasks = run_workflow(workflow, variables, tasks_dir=self.temp_dir, dry_run=True)

        # Map tasks by title pattern
        design = next(t for t in tasks if "Design:" in t.title)
        implement = next(t for t in tasks if "Implement:" in t.title)
        unit_tests = next(t for t in tasks if "Unit tests" in t.title)
        integration_tests = next(t for t in tasks if "Integration tests" in t.title)
        documentation = next(t for t in tasks if "Documentation" in t.title)

        # Verify dependency chain
        self.assertEqual(len(design.depends_on), 0, "Design has no dependencies")
        self.assertIn(design.id, implement.depends_on, "Implement depends on design")
        self.assertIn(implement.id, unit_tests.depends_on, "Unit tests depend on implement")
        self.assertIn(implement.id, integration_tests.depends_on, "Integration tests depend on implement")

        # Documentation depends on BOTH test tasks
        self.assertIn(unit_tests.id, documentation.depends_on)
        self.assertIn(integration_tests.id, documentation.depends_on)

        # Verify no template IDs leak through
        for task in tasks:
            for dep_id in task.depends_on:
                self.assertTrue(dep_id.startswith("T-"), f"Dependency {dep_id} should be a real task ID")

    def test_dependencies_point_to_existing_tasks(self):
        """Verify all dependency IDs reference actual tasks in the session."""
        workflow = Workflow.load(WORKFLOWS_DIR / "feature.yaml")
        variables = {
            "feature_name": "Test",
            "priority": "high",
            "effort": "medium"
        }

        tasks = run_workflow(workflow, variables, tasks_dir=self.temp_dir, dry_run=False)

        # Load tasks from file
        loaded_tasks = load_all_tasks(self.temp_dir)
        task_ids = {t.id for t in loaded_tasks}

        # Verify every dependency points to an existing task
        for task in loaded_tasks:
            for dep_id in task.depends_on:
                self.assertIn(dep_id, task_ids, f"Dependency {dep_id} should exist in task set")


class TestVariableSubstitution(unittest.TestCase):
    """Test variable substitution in task titles and descriptions."""

    def setUp(self):
        """Create temporary directory for tasks."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_bug_title_appears_in_all_tasks(self):
        """Verify bug_title variable is substituted in all task titles."""
        workflow = Workflow.load(WORKFLOWS_DIR / "bugfix.yaml")
        bug_title = "Timeout in database connection"
        variables = {"bug_title": bug_title, "priority": "high"}

        tasks = run_workflow(workflow, variables, tasks_dir=self.temp_dir, dry_run=True)

        # All tasks should contain the bug title
        for task in tasks:
            self.assertIn(bug_title, task.title,
                         f"Task title '{task.title}' should contain '{bug_title}'")

    def test_description_substitution(self):
        """Verify variable substitution works in task descriptions."""
        workflow = Workflow.load(WORKFLOWS_DIR / "bugfix.yaml")
        bug_title = "Memory leak in cache"
        variables = {"bug_title": bug_title, "priority": "high"}

        tasks = run_workflow(workflow, variables, tasks_dir=self.temp_dir, dry_run=True)

        # At least one description should contain the bug title
        descriptions = [t.description for t in tasks if t.description]
        self.assertTrue(any(bug_title in desc for desc in descriptions),
                       "At least one description should contain bug_title")

    def test_priority_substitution(self):
        """Verify priority variable is substituted correctly."""
        workflow = Workflow.load(WORKFLOWS_DIR / "bugfix.yaml")
        variables = {"bug_title": "Test", "priority": "low"}

        tasks = run_workflow(workflow, variables, tasks_dir=self.temp_dir, dry_run=True)

        # First task (investigate) should have priority from variable
        investigate_task = tasks[0]
        self.assertEqual(investigate_task.priority, "low")

    def test_effort_substitution_in_feature_workflow(self):
        """Verify effort variable is substituted in feature workflow."""
        workflow = Workflow.load(WORKFLOWS_DIR / "feature.yaml")
        variables = {
            "feature_name": "Test",
            "priority": "high",
            "effort": "small"  # Override default
        }

        tasks = run_workflow(workflow, variables, tasks_dir=self.temp_dir, dry_run=True)

        # Implementation task should have small effort
        implement_task = next(t for t in tasks if "Implement:" in t.title)
        self.assertEqual(implement_task.effort, "small")

    def test_multiple_variable_substitution(self):
        """Verify multiple variables work together."""
        workflow = Workflow.load(WORKFLOWS_DIR / "feature.yaml")
        variables = {
            "feature_name": "Semantic search",
            "priority": "high",
            "effort": "large"
        }

        tasks = run_workflow(workflow, variables, tasks_dir=self.temp_dir, dry_run=True)

        # Check design task has both substitutions
        design_task = tasks[0]
        self.assertIn("Semantic search", design_task.title)
        self.assertEqual(design_task.priority, "high")
        self.assertEqual(design_task.effort, "medium")  # Design has hardcoded medium

        # Check implementation task
        impl_task = next(t for t in tasks if "Implement:" in t.title)
        self.assertIn("Semantic search", impl_task.title)
        self.assertEqual(impl_task.priority, "high")
        self.assertEqual(impl_task.effort, "large")

    def test_substitute_variables_function(self):
        """Test the substitute_variables helper function directly."""
        text = "Fix: {bug_title} (Priority: {priority})"
        variables = {"bug_title": "Auth error", "priority": "high"}

        result = substitute_variables(text, variables)

        self.assertEqual(result, "Fix: Auth error (Priority: high)")
        self.assertNotIn("{", result)
        self.assertNotIn("}", result)


class TestWorkflowValidation(unittest.TestCase):
    """Test workflow variable validation and error handling."""

    def setUp(self):
        """Create temporary directory for tasks."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_missing_required_variable_raises_error(self):
        """Verify missing required variable raises ValueError."""
        workflow = Workflow.load(WORKFLOWS_DIR / "bugfix.yaml")
        variables = {"priority": "high"}  # Missing bug_title

        with self.assertRaises(ValueError) as ctx:
            run_workflow(workflow, variables, tasks_dir=self.temp_dir, dry_run=True)

        self.assertIn("bug_title", str(ctx.exception))

    def test_invalid_choice_raises_error(self):
        """Verify invalid choice value raises ValueError."""
        workflow = Workflow.load(WORKFLOWS_DIR / "bugfix.yaml")
        variables = {
            "bug_title": "Test",
            "priority": "urgent"  # Invalid - not in choices
        }

        with self.assertRaises(ValueError) as ctx:
            run_workflow(workflow, variables, tasks_dir=self.temp_dir, dry_run=True)

        self.assertIn("priority", str(ctx.exception))
        self.assertIn("urgent", str(ctx.exception))

    def test_default_value_used_when_missing(self):
        """Verify default values are used for missing optional variables."""
        workflow = Workflow.load(WORKFLOWS_DIR / "bugfix.yaml")
        variables = {"bug_title": "Test"}  # Missing priority, should use default

        tasks = run_workflow(workflow, variables, tasks_dir=self.temp_dir, dry_run=True)

        # Should use default priority "high"
        self.assertTrue(any(t.priority == "high" for t in tasks))


class TestDryRunMode(unittest.TestCase):
    """Test dry run mode doesn't create files."""

    def setUp(self):
        """Create temporary directory for tasks."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_dry_run_does_not_create_files(self):
        """Verify dry_run=True doesn't create JSON files."""
        workflow = Workflow.load(WORKFLOWS_DIR / "bugfix.yaml")
        variables = {"bug_title": "Test", "priority": "high"}

        tasks = run_workflow(workflow, variables, tasks_dir=self.temp_dir, dry_run=True)

        # Should return tasks
        self.assertEqual(len(tasks), 4)

        # But should not create files
        json_files = list(Path(self.temp_dir).glob("*.json"))
        self.assertEqual(len(json_files), 0, "Dry run should not create files")

    def test_normal_run_creates_files(self):
        """Verify dry_run=False creates files."""
        workflow = Workflow.load(WORKFLOWS_DIR / "bugfix.yaml")
        variables = {"bug_title": "Test", "priority": "high"}

        tasks = run_workflow(workflow, variables, tasks_dir=self.temp_dir, dry_run=False)

        # Should create exactly one JSON file
        json_files = list(Path(self.temp_dir).glob("*.json"))
        self.assertEqual(len(json_files), 1, "Should create one JSON file")


if __name__ == "__main__":
    unittest.main()
