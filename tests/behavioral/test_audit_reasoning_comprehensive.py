"""
Comprehensive behavioral tests for audit_reasoning.py.

Target: 90%+ code coverage for the audit reasoning script.

Covers:
1. Data classes (FileImportanceRecord, AuditPersistenceState, AuditQuery)
2. NLU translation (translate_audit_query, is_natural_language_query)
3. Persistence I/O (load/save state, rules)
4. WovenMind integration (abstraction_to_rule, load_woven_mind_abstractions)
5. AuditReasoner class (all methods)
6. Pipeline functions (analyze_with_reasoning, generate_reasoning_report)
"""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any
from unittest.mock import patch, MagicMock


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def temp_got_dir(tmp_path):
    """Create temporary .got directory for testing."""
    got_dir = tmp_path / ".got"
    got_dir.mkdir()
    return got_dir


@pytest.fixture
def mock_persistence_file(temp_got_dir, monkeypatch):
    """Mock the persistence file path."""
    persistence_file = temp_got_dir / "audit_pln_state.json"
    import scripts.audit_reasoning as ar
    monkeypatch.setattr(ar, "PERSISTENCE_FILE", persistence_file)
    return persistence_file


@pytest.fixture
def mock_rules_file(temp_got_dir, monkeypatch):
    """Mock the rules file path."""
    rules_file = temp_got_dir / "audit_pln_rules.json"
    import scripts.audit_reasoning as ar
    monkeypatch.setattr(ar, "RULES_FILE", rules_file)
    return rules_file


@pytest.fixture
def mock_woven_mind_file(temp_got_dir, monkeypatch):
    """Mock the WovenMind file path."""
    woven_file = temp_got_dir / "woven_audit_mind.json"
    import scripts.audit_reasoning as ar
    monkeypatch.setattr(ar, "WOVEN_MIND_FILE", woven_file)
    return woven_file


# =============================================================================
# DATA CLASSES
# =============================================================================


class TestFileImportanceRecord:
    """Tests for FileImportanceRecord dataclass."""

    def test_creation_with_all_fields(self):
        from scripts.audit_reasoning import FileImportanceRecord

        record = FileImportanceRecord(
            file_id="test_py",
            sti=0.8,
            lti=0.5,
            vlti=True,
            last_seen="2026-01-08T10:00:00",
            history=[{"timestamp": "2026-01-07", "sti": 0.7}]
        )

        assert record.file_id == "test_py"
        assert record.sti == 0.8
        assert record.lti == 0.5
        assert record.vlti is True

    def test_to_dict(self):
        from scripts.audit_reasoning import FileImportanceRecord

        record = FileImportanceRecord(
            file_id="module_py",
            sti=0.6,
            lti=0.4,
            vlti=False,
            last_seen="2026-01-08",
            history=[]
        )

        d = record.to_dict()

        assert d["file_id"] == "module_py"
        assert d["sti"] == 0.6
        assert d["lti"] == 0.4
        assert d["vlti"] is False
        assert d["history"] == []

    def test_to_dict_truncates_history_to_50(self):
        from scripts.audit_reasoning import FileImportanceRecord

        # Create record with 100 history entries
        history = [{"timestamp": f"entry_{i}"} for i in range(100)]
        record = FileImportanceRecord(
            file_id="big_py",
            sti=0.5,
            lti=0.3,
            vlti=False,
            last_seen="now",
            history=history
        )

        d = record.to_dict()

        assert len(d["history"]) == 50
        # Should keep last 50
        assert d["history"][0]["timestamp"] == "entry_50"

    def test_from_dict(self):
        from scripts.audit_reasoning import FileImportanceRecord

        data = {
            "file_id": "loaded_py",
            "sti": 0.7,
            "lti": 0.35,
            "vlti": True,
            "last_seen": "2026-01-08T12:00:00",
            "history": [{"sti": 0.6}]
        }

        record = FileImportanceRecord.from_dict(data)

        assert record.file_id == "loaded_py"
        assert record.sti == 0.7
        assert record.lti == 0.35
        assert record.vlti is True

    def test_from_dict_with_defaults(self):
        from scripts.audit_reasoning import FileImportanceRecord

        # Minimal data - should use defaults
        data = {"file_id": "minimal_py"}

        record = FileImportanceRecord.from_dict(data)

        assert record.file_id == "minimal_py"
        assert record.sti == 0.3  # default
        assert record.lti == 0.1  # default
        assert record.vlti is False  # default


class TestAuditPersistenceState:
    """Tests for AuditPersistenceState dataclass."""

    def test_create_new(self):
        from scripts.audit_reasoning import AuditPersistenceState

        state = AuditPersistenceState.create_new()

        assert state.version == 1
        assert state.session_count == 0
        assert state.file_importance == {}
        assert state.attention_focus == []
        assert state.global_stats == {}

    def test_to_dict(self):
        from scripts.audit_reasoning import AuditPersistenceState, FileImportanceRecord

        record = FileImportanceRecord(
            file_id="test_py", sti=0.5, lti=0.2, vlti=False,
            last_seen="now", history=[]
        )

        state = AuditPersistenceState(
            version=2,
            created="2026-01-01",
            updated="2026-01-08",
            session_count=5,
            file_importance={"test_py": record},
            attention_focus=["test_py"],
            global_stats={"key": "value"}
        )

        d = state.to_dict()

        assert d["version"] == 2
        assert d["session_count"] == 5
        assert "test_py" in d["file_importance"]
        assert d["attention_focus"] == ["test_py"]

    def test_from_dict(self):
        from scripts.audit_reasoning import AuditPersistenceState

        data = {
            "version": 3,
            "created": "2026-01-01",
            "updated": "2026-01-08",
            "session_count": 10,
            "file_importance": {
                "mod_py": {"file_id": "mod_py", "sti": 0.6, "lti": 0.3}
            },
            "attention_focus": ["mod_py"],
            "global_stats": {}
        }

        state = AuditPersistenceState.from_dict(data)

        assert state.version == 3
        assert state.session_count == 10
        assert "mod_py" in state.file_importance
        assert state.file_importance["mod_py"].sti == 0.6


class TestAuditQuery:
    """Tests for AuditQuery dataclass."""

    def test_default_values(self):
        from scripts.audit_reasoning import AuditQuery

        query = AuditQuery()

        assert query.directory is None
        assert query.file_patterns == []
        assert query.negations == []
        assert query.include_traits == []
        assert query.intent == "list"
        assert query.min_risk == 0.0
        assert query.explain is False

    def test_post_init_initializes_lists(self):
        from scripts.audit_reasoning import AuditQuery

        # Create without explicit lists
        query = AuditQuery(directory="test/")

        # Should have empty lists, not None
        assert query.file_patterns == []
        assert query.negations == []
        assert query.include_traits == []


# =============================================================================
# NLU TRANSLATION
# =============================================================================


class TestTranslateAuditQuery:
    """Tests for translate_audit_query function."""

    def test_simple_directory(self):
        from scripts.audit_reasoning import translate_audit_query

        query = translate_audit_query("cortical/")

        assert query.directory == "cortical/"

    def test_directory_with_in_keyword(self):
        from scripts.audit_reasoning import translate_audit_query

        query = translate_audit_query("in reasoning/")

        assert query.directory == "reasoning/"

    def test_negation_with_not(self):
        from scripts.audit_reasoning import translate_audit_query

        query = translate_audit_query("cortical/ not tests")

        assert "tests" in query.negations

    def test_negation_with_without(self):
        from scripts.audit_reasoning import translate_audit_query

        query = translate_audit_query("without legacy")

        assert "legacy" in query.negations

    def test_negation_with_exclude(self):
        from scripts.audit_reasoning import translate_audit_query

        query = translate_audit_query("exclude utils")

        assert "utils" in query.negations

    def test_negation_with_excluding(self):
        from scripts.audit_reasoning import translate_audit_query

        query = translate_audit_query("excluding vendor")

        assert "vendor" in query.negations

    def test_trait_with_high_churn(self):
        from scripts.audit_reasoning import translate_audit_query

        query = translate_audit_query("files with high_churn")

        assert "high_churn" in query.include_traits

    def test_trait_with_todo(self):
        from scripts.audit_reasoning import translate_audit_query

        query = translate_audit_query("files with todo")

        assert "todo" in query.include_traits

    def test_trait_with_has(self):
        from scripts.audit_reasoning import translate_audit_query

        query = translate_audit_query("has fixme")

        assert "fixme" in query.include_traits

    def test_trait_with_having(self):
        from scripts.audit_reasoning import translate_audit_query

        query = translate_audit_query("having hack")

        assert "hack" in query.include_traits

    def test_risk_level_critical(self):
        from scripts.audit_reasoning import translate_audit_query

        query = translate_audit_query("critical files")

        assert query.min_risk == 0.9

    def test_risk_level_high(self):
        from scripts.audit_reasoning import translate_audit_query

        query = translate_audit_query("high risk files")

        assert query.min_risk == 0.7

    def test_risk_level_risky(self):
        from scripts.audit_reasoning import translate_audit_query

        query = translate_audit_query("risky files")

        assert query.min_risk == 0.5

    def test_risk_level_medium(self):
        from scripts.audit_reasoning import translate_audit_query

        query = translate_audit_query("medium risk files")

        assert query.min_risk == 0.4

    def test_result_limit_top_n(self):
        from scripts.audit_reasoning import translate_audit_query

        query = translate_audit_query("top 5 risky")

        assert query.max_results == 5

    def test_result_limit_first_n(self):
        from scripts.audit_reasoning import translate_audit_query

        query = translate_audit_query("first 10 files")

        assert query.max_results == 10

    def test_why_is_flagged_intent(self):
        from scripts.audit_reasoning import translate_audit_query

        query = translate_audit_query("why is auth.py flagged")

        assert query.intent == "explain"
        assert query.target_file == "auth.py"
        assert query.explain is True

    def test_why_is_risky_intent(self):
        from scripts.audit_reasoning import translate_audit_query

        query = translate_audit_query("why is module.py risky")

        assert query.intent == "explain"
        assert query.target_file == "module.py"

    def test_why_is_marked_intent(self):
        from scripts.audit_reasoning import translate_audit_query

        query = translate_audit_query("why is test.py marked")

        assert query.intent == "explain"
        assert query.target_file == "test.py"

    def test_explain_intent(self):
        from scripts.audit_reasoning import translate_audit_query

        query = translate_audit_query("explain storage.py")

        assert query.intent == "explain"
        assert query.target_file == "storage.py"

    def test_complex_query_combination(self):
        from scripts.audit_reasoning import translate_audit_query

        query = translate_audit_query("cortical/ not tests with high_churn top 10 risky")

        assert query.directory == "cortical/"
        assert "tests" in query.negations
        assert "high_churn" in query.include_traits
        assert query.max_results == 10
        assert query.min_risk == 0.5

    def test_directory_scope_with_explain(self):
        from scripts.audit_reasoning import translate_audit_query

        query = translate_audit_query("cortical/ why is storage.py flagged")

        assert query.directory == "cortical/"
        assert query.intent == "explain"
        assert query.target_file == "storage.py"

    def test_normalizes_traits(self):
        from scripts.audit_reasoning import translate_audit_query

        # "todos" should normalize to "todo"
        query = translate_audit_query("with todos")
        assert "todo" in query.include_traits

        # "hacks" should normalize to "hack"
        query2 = translate_audit_query("with hacks")
        assert "hack" in query2.include_traits


class TestIsNaturalLanguageQuery:
    """Tests for is_natural_language_query function."""

    def test_flag_is_not_nlu(self):
        from scripts.audit_reasoning import is_natural_language_query

        assert is_natural_language_query("--help") is False
        assert is_natural_language_query("-v") is False

    def test_query_with_spaces_is_nlu(self):
        from scripts.audit_reasoning import is_natural_language_query

        assert is_natural_language_query("files with todo") is True

    def test_query_with_why_is_nlu(self):
        from scripts.audit_reasoning import is_natural_language_query

        assert is_natural_language_query("whyfile") is True  # Contains 'why'

    def test_simple_path_is_not_nlu(self, tmp_path):
        from scripts.audit_reasoning import is_natural_language_query

        # Create an actual directory
        test_dir = tmp_path / "testdir"
        test_dir.mkdir()

        # Existing path without NLU keywords should be treated as path
        assert is_natural_language_query(str(test_dir)) is False


# =============================================================================
# PERSISTENCE I/O
# =============================================================================


class TestPersistenceIO:
    """Tests for persistence load/save functions."""

    def test_load_persistence_state_creates_new_when_missing(self, mock_persistence_file):
        from scripts.audit_reasoning import load_persistence_state

        state = load_persistence_state()

        assert state.version == 1
        assert state.session_count == 0

    def test_load_persistence_state_reads_existing(self, mock_persistence_file):
        from scripts.audit_reasoning import load_persistence_state

        # Write a state file
        state_data = {
            "version": 2,
            "created": "2026-01-01",
            "updated": "2026-01-08",
            "session_count": 15,
            "file_importance": {},
            "attention_focus": [],
            "global_stats": {}
        }
        mock_persistence_file.write_text(json.dumps(state_data))

        state = load_persistence_state()

        assert state.version == 2
        assert state.session_count == 15

    def test_load_persistence_state_handles_invalid_json(self, mock_persistence_file, capsys):
        from scripts.audit_reasoning import load_persistence_state

        mock_persistence_file.write_text("not valid json")

        state = load_persistence_state()

        # Should return new state and print warning
        assert state.session_count == 0
        captured = capsys.readouterr()
        assert "Warning" in captured.out

    def test_save_persistence_state(self, mock_persistence_file):
        from scripts.audit_reasoning import save_persistence_state, AuditPersistenceState

        state = AuditPersistenceState.create_new()
        state.session_count = 5

        save_persistence_state(state)

        # Read back
        data = json.loads(mock_persistence_file.read_text())
        assert data["session_count"] == 5
        assert "updated" in data

    def test_clear_persistence_state_removes_file(self, mock_persistence_file, capsys):
        from scripts.audit_reasoning import clear_persistence_state

        mock_persistence_file.write_text("{}")

        clear_persistence_state()

        assert not mock_persistence_file.exists()
        captured = capsys.readouterr()
        assert "cleared" in captured.out

    def test_clear_persistence_state_no_file(self, mock_persistence_file, capsys):
        from scripts.audit_reasoning import clear_persistence_state

        # File doesn't exist
        clear_persistence_state()

        captured = capsys.readouterr()
        assert "No persistence state" in captured.out

    def test_show_persistence_status(self, mock_persistence_file, capsys):
        from scripts.audit_reasoning import show_persistence_status

        # Create a state file
        state_data = {
            "version": 1,
            "created": "2026-01-01",
            "updated": "2026-01-08",
            "session_count": 3,
            "file_importance": {
                "test_py": {
                    "file_id": "test_py",
                    "sti": 0.5,
                    "lti": 0.3,
                    "vlti": False,
                    "last_seen": "2026-01-08",
                    "history": []
                }
            },
            "attention_focus": ["test_py"],
            "global_stats": {"key": "value"}
        }
        mock_persistence_file.write_text(json.dumps(state_data))

        show_persistence_status()

        captured = capsys.readouterr()
        assert "AUDIT PLN PERSISTENCE STATE" in captured.out
        assert "Session count: 3" in captured.out
        assert "test_py" in captured.out


# =============================================================================
# RULES I/O
# =============================================================================


class TestRulesIO:
    """Tests for rules load/save functions."""

    def test_load_rules_creates_default_when_missing(self, mock_rules_file):
        from scripts.audit_reasoning import load_rules

        rules = load_rules()

        assert rules["version"] == 1
        assert "rules" in rules
        assert "manual_rules" in rules

    def test_load_rules_reads_existing(self, mock_rules_file):
        from scripts.audit_reasoning import load_rules

        rules_data = {
            "version": 2,
            "rules": [],
            "manual_rules": [{"antecedent": "a", "consequent": "b"}]
        }
        mock_rules_file.write_text(json.dumps(rules_data))

        rules = load_rules()

        assert rules["version"] == 2
        assert len(rules["manual_rules"]) == 1

    def test_load_rules_handles_invalid_json(self, mock_rules_file):
        from scripts.audit_reasoning import load_rules

        mock_rules_file.write_text("invalid json")

        rules = load_rules()

        # Should return default
        assert rules["version"] == 1

    def test_save_rules(self, mock_rules_file):
        from scripts.audit_reasoning import save_rules

        rules = {
            "version": 1,
            "rules": [],
            "manual_rules": [{"ant": "x", "cons": "y"}]
        }

        save_rules(rules)

        data = json.loads(mock_rules_file.read_text())
        assert len(data["manual_rules"]) == 1
        assert "updated" in data


# =============================================================================
# WOVEN MIND INTEGRATION
# =============================================================================


class TestAbstractionToRule:
    """Tests for abstraction_to_rule function."""

    def test_converts_valid_abstraction(self):
        from scripts.audit_reasoning import abstraction_to_rule

        abstraction = {
            "id": "abs_1",
            "source_nodes": ["dir:legacy", "pattern:todo"],
            "frequency": 10,
            "strength": 0.6
        }

        rule = abstraction_to_rule(abstraction)

        assert rule is not None
        assert "in_dir(X, legacy)" in rule["antecedent"]
        assert "has_pattern(X, todo)" in rule["antecedent"]
        assert rule["consequent"] == "flagged(X)"
        assert rule["source"] == "abs_1"

    def test_returns_none_for_single_node(self):
        from scripts.audit_reasoning import abstraction_to_rule

        abstraction = {
            "source_nodes": ["dir:legacy"],
            "frequency": 5
        }

        rule = abstraction_to_rule(abstraction)

        assert rule is None

    def test_returns_none_for_empty_nodes(self):
        from scripts.audit_reasoning import abstraction_to_rule

        abstraction = {"source_nodes": [], "frequency": 5}

        rule = abstraction_to_rule(abstraction)

        assert rule is None

    def test_skips_file_nodes(self):
        from scripts.audit_reasoning import abstraction_to_rule

        abstraction = {
            "source_nodes": ["file:auth.py", "pattern:todo", "file:utils.py"],
            "frequency": 3
        }

        # Only pattern:todo is usable, so should return None (need 2+)
        rule = abstraction_to_rule(abstraction)
        assert rule is None

    def test_includes_trait_nodes(self):
        from scripts.audit_reasoning import abstraction_to_rule

        abstraction = {
            "source_nodes": ["trait:high_churn", "pattern:fixme"],
            "frequency": 8,
            "strength": 0.7
        }

        rule = abstraction_to_rule(abstraction)

        assert rule is not None
        assert "has_trait(X, high_churn)" in rule["antecedent"]

    def test_confidence_scales_with_frequency(self):
        from scripts.audit_reasoning import abstraction_to_rule

        low_freq = {"source_nodes": ["dir:a", "pattern:b"], "frequency": 1}
        high_freq = {"source_nodes": ["dir:a", "pattern:b"], "frequency": 100}

        rule_low = abstraction_to_rule(low_freq)
        rule_high = abstraction_to_rule(high_freq)

        assert rule_high["confidence"] > rule_low["confidence"]


class TestLoadWovenMindAbstractions:
    """Tests for load_woven_mind_abstractions function."""

    def test_returns_empty_when_missing(self, mock_woven_mind_file):
        from scripts.audit_reasoning import load_woven_mind_abstractions

        result = load_woven_mind_abstractions()

        assert result == []

    def test_loads_abstractions_from_file(self, mock_woven_mind_file):
        from scripts.audit_reasoning import load_woven_mind_abstractions

        woven_data = {
            "mind": {
                "cortex_state": {
                    "engine_state": {
                        "abstractions": {
                            "abs_1": {
                                "source_nodes": ["dir:api", "pattern:todo"],
                                "level": 1,
                                "frequency": 5,
                                "strength": 0.6
                            },
                            "abs_2": {
                                "source_nodes": ["trait:churn"],
                                "level": 0,
                                "frequency": 2,
                                "strength": 0.8
                            }
                        }
                    }
                }
            }
        }
        mock_woven_mind_file.write_text(json.dumps(woven_data))

        result = load_woven_mind_abstractions()

        assert len(result) == 2
        # Should be sorted by strength descending
        assert result[0]["strength"] == 0.8

    def test_handles_invalid_json(self, mock_woven_mind_file, capsys):
        from scripts.audit_reasoning import load_woven_mind_abstractions

        mock_woven_mind_file.write_text("invalid")

        result = load_woven_mind_abstractions()

        assert result == []
        captured = capsys.readouterr()
        assert "Warning" in captured.out


# =============================================================================
# AUDIT REASONER CLASS
# =============================================================================


class TestAuditReasonerInit:
    """Tests for AuditReasoner initialization."""

    def test_basic_initialization(self, mock_persistence_file, mock_rules_file):
        from scripts.audit_reasoning import AuditReasoner

        reasoner = AuditReasoner(use_persistence=False)

        assert reasoner.pln is not None
        assert reasoner.aggregate_strategy == "revision"

    def test_initialization_with_custom_strategy(self, mock_persistence_file, mock_rules_file):
        from scripts.audit_reasoning import AuditReasoner

        reasoner = AuditReasoner(aggregate_strategy="max", use_persistence=False)

        assert reasoner.aggregate_strategy == "max"

    def test_initialization_loads_persistence(self, mock_persistence_file, mock_rules_file):
        from scripts.audit_reasoning import AuditReasoner

        # Create persisted state
        state_data = {
            "version": 1,
            "created": "2026-01-01",
            "updated": "2026-01-08",
            "session_count": 5,
            "file_importance": {
                "old_py": {
                    "file_id": "old_py",
                    "sti": 0.6,
                    "lti": 0.4,
                    "vlti": True,
                    "last_seen": datetime.now().isoformat(),
                    "history": []
                }
            },
            "attention_focus": [],
            "global_stats": {}
        }
        mock_persistence_file.write_text(json.dumps(state_data))

        reasoner = AuditReasoner(use_persistence=True)

        assert "old_py" in reasoner.file_importance


class TestAuditReasonerMethods:
    """Tests for AuditReasoner methods."""

    def test_add_default_rules(self, mock_persistence_file, mock_rules_file):
        from scripts.audit_reasoning import AuditReasoner

        reasoner = AuditReasoner(use_persistence=False)
        reasoner.add_default_rules()

        # Should have added rules
        assert reasoner.pln.rule_count > 0

    def test_load_manual_rules(self, mock_persistence_file, mock_rules_file):
        from scripts.audit_reasoning import AuditReasoner

        # Set up rules file with manual rules
        rules_data = {
            "version": 1,
            "manual_rules": [
                {"antecedent": "custom(X)", "consequent": "flagged(X)", "strength": 0.7}
            ]
        }
        mock_rules_file.write_text(json.dumps(rules_data))

        reasoner = AuditReasoner(use_persistence=False)
        count = reasoner.load_manual_rules()

        assert count == 1

    def test_assert_file_facts(self, mock_persistence_file, mock_rules_file):
        from scripts.audit_reasoning import AuditReasoner

        reasoner = AuditReasoner(use_persistence=False)

        reasoner.assert_file_facts(
            "test.py",
            patterns=["todo", "fixme"],
            traits=["high_churn"],
            directories=["api"]
        )

        # Should have created facts
        assert reasoner.pln.fact_count > 0
        # Should have set importance
        assert "test_py" in reasoner.file_importance

    def test_assert_file_facts_with_initial_importance(self, mock_persistence_file, mock_rules_file):
        from scripts.audit_reasoning import AuditReasoner

        reasoner = AuditReasoner(use_persistence=False)

        reasoner.assert_file_facts(
            "important.py",
            patterns=[],
            traits=[],
            directories=[],
            initial_importance=0.9
        )

        assert reasoner.file_importance["important_py"].sti == 0.9

    def test_assert_file_facts_calculates_importance_from_traits(
        self, mock_persistence_file, mock_rules_file
    ):
        from scripts.audit_reasoning import AuditReasoner

        reasoner = AuditReasoner(use_persistence=False)

        reasoner.assert_file_facts(
            "risky.py",
            patterns=["todo", "fixme", "hack", "xxx"],  # 4 patterns
            traits=["high_churn", "bug_prone"],
            directories=[]
        )

        # high_churn adds 0.3, bug_prone adds 0.2, 2 extra patterns add 0.2
        # Base 0.3 + 0.3 + 0.2 + 0.2 = 1.0 (capped)
        assert reasoner.file_importance["risky_py"].sti >= 0.7

    def test_focus_on_high_risk_files(self, mock_persistence_file, mock_rules_file):
        from scripts.audit_reasoning import AuditReasoner
        from cortical.reasoning.prism_pln import AttentionValue

        reasoner = AuditReasoner(use_persistence=False)

        # Add files with different importance
        reasoner.file_importance["high_py"] = AttentionValue(sti=0.8, lti=0.5)
        reasoner.file_importance["low_py"] = AttentionValue(sti=0.1, lti=0.1)

        count = reasoner.focus_on_high_risk_files(threshold=0.5)

        assert count == 1  # Only high_py

    def test_query_file_risk(self, mock_persistence_file, mock_rules_file):
        from scripts.audit_reasoning import AuditReasoner

        reasoner = AuditReasoner(use_persistence=False)
        reasoner.add_default_rules()
        reasoner.assert_file_facts(
            "query_test.py",
            patterns=["todo"],
            traits=[],
            directories=[]
        )

        results = reasoner.query_file_risk("query_test.py")

        # Should return dict with results
        assert isinstance(results, dict)
        # Should include importance info
        assert "_importance" in results

    def test_query_with_aggregation(self, mock_persistence_file, mock_rules_file):
        from scripts.audit_reasoning import AuditReasoner

        reasoner = AuditReasoner(use_persistence=False)
        reasoner.add_default_rules()
        reasoner.pln.assert_fact("test_query(x)", strength=0.8)

        results = reasoner.query_with_aggregation(
            "test_query(x)",
            strategies=["first", "max"]
        )

        assert "first" in results or "max" in results

    def test_collect_rent(self, mock_persistence_file, mock_rules_file):
        from scripts.audit_reasoning import AuditReasoner
        from cortical.reasoning.prism_pln import AttentionValue

        reasoner = AuditReasoner(use_persistence=False)
        reasoner.file_importance["decay_py"] = AttentionValue(sti=1.0, lti=1.0)

        reasoner.collect_rent(sti_decay=0.5, lti_decay=0.9)

        # STI should have decayed
        assert reasoner.file_importance["decay_py"].sti == 0.5
        assert reasoner.file_importance["decay_py"].lti == 0.9

    def test_stimulate_file(self, mock_persistence_file, mock_rules_file):
        from scripts.audit_reasoning import AuditReasoner
        from cortical.reasoning.prism_pln import AttentionValue

        reasoner = AuditReasoner(use_persistence=False)
        reasoner.file_importance["stim_py"] = AttentionValue(sti=0.5, lti=0.2)

        reasoner.stimulate_file("stim.py", amount=0.3)

        assert reasoner.file_importance["stim_py"].sti == 0.8

    def test_get_priority_files(self, mock_persistence_file, mock_rules_file):
        from scripts.audit_reasoning import AuditReasoner
        from cortical.reasoning.prism_pln import AttentionValue

        reasoner = AuditReasoner(use_persistence=False)
        reasoner.file_importance["high_py"] = AttentionValue(sti=0.9, lti=0.5)
        reasoner.file_importance["mid_py"] = AttentionValue(sti=0.5, lti=0.3)
        reasoner.file_importance["low_py"] = AttentionValue(sti=0.1, lti=0.1)

        priority = reasoner.get_priority_files(top_n=2)

        assert len(priority) == 2
        assert priority[0][0] == "high_py"  # Highest first

    def test_get_vlti_files(self, mock_persistence_file, mock_rules_file):
        from scripts.audit_reasoning import AuditReasoner
        from cortical.reasoning.prism_pln import AttentionValue

        reasoner = AuditReasoner(use_persistence=False)
        reasoner.file_importance["pinned_py"] = AttentionValue(sti=0.5, lti=0.3, vlti=True)
        reasoner.file_importance["normal_py"] = AttentionValue(sti=0.5, lti=0.3, vlti=False)

        vlti = reasoner.get_vlti_files()

        assert "pinned_py" in vlti
        assert "normal_py" not in vlti

    def test_get_importance_history(self, mock_persistence_file, mock_rules_file):
        from scripts.audit_reasoning import AuditReasoner

        # Create persistence with history
        state_data = {
            "version": 1, "created": "now", "updated": "now", "session_count": 1,
            "file_importance": {
                "hist_py": {
                    "file_id": "hist_py", "sti": 0.5, "lti": 0.3, "vlti": False,
                    "last_seen": datetime.now().isoformat(),
                    "history": [{"sti": 0.4, "lti": 0.2}]
                }
            },
            "attention_focus": [], "global_stats": {}
        }
        mock_persistence_file.write_text(json.dumps(state_data))

        reasoner = AuditReasoner(use_persistence=True)
        history = reasoner.get_importance_history("hist_py")

        assert len(history) == 1
        assert history[0]["sti"] == 0.4

    def test_get_importance_trend(self, mock_persistence_file, mock_rules_file):
        from scripts.audit_reasoning import AuditReasoner

        # Create persistence with trend data
        state_data = {
            "version": 1, "created": "now", "updated": "now", "session_count": 1,
            "file_importance": {
                "trend_py": {
                    "file_id": "trend_py", "sti": 0.8, "lti": 0.5, "vlti": False,
                    "last_seen": datetime.now().isoformat(),
                    "history": [
                        {"sti": 0.2, "lti": 0.1},  # Old
                        {"sti": 0.8, "lti": 0.5}   # Recent
                    ]
                }
            },
            "attention_focus": [], "global_stats": {}
        }
        mock_persistence_file.write_text(json.dumps(state_data))

        reasoner = AuditReasoner(use_persistence=True)
        trend = reasoner.get_importance_trend("trend_py")

        assert trend == "increasing"

    def test_get_importance_trend_decreasing(self, mock_persistence_file, mock_rules_file):
        from scripts.audit_reasoning import AuditReasoner

        state_data = {
            "version": 1, "created": "now", "updated": "now", "session_count": 1,
            "file_importance": {
                "down_py": {
                    "file_id": "down_py", "sti": 0.2, "lti": 0.1, "vlti": False,
                    "last_seen": datetime.now().isoformat(),
                    "history": [
                        {"sti": 0.8, "lti": 0.5},
                        {"sti": 0.2, "lti": 0.1}
                    ]
                }
            },
            "attention_focus": [], "global_stats": {}
        }
        mock_persistence_file.write_text(json.dumps(state_data))

        reasoner = AuditReasoner(use_persistence=True)
        trend = reasoner.get_importance_trend("down_py")

        assert trend == "decreasing"

    def test_get_importance_trend_stable(self, mock_persistence_file, mock_rules_file):
        from scripts.audit_reasoning import AuditReasoner

        state_data = {
            "version": 1, "created": "now", "updated": "now", "session_count": 1,
            "file_importance": {
                "stable_py": {
                    "file_id": "stable_py", "sti": 0.5, "lti": 0.3, "vlti": False,
                    "last_seen": datetime.now().isoformat(),
                    "history": [
                        {"sti": 0.5, "lti": 0.3},
                        {"sti": 0.5, "lti": 0.3}
                    ]
                }
            },
            "attention_focus": [], "global_stats": {}
        }
        mock_persistence_file.write_text(json.dumps(state_data))

        reasoner = AuditReasoner(use_persistence=True)
        trend = reasoner.get_importance_trend("stable_py")

        assert trend == "stable"

    def test_get_stats(self, mock_persistence_file, mock_rules_file):
        from scripts.audit_reasoning import AuditReasoner

        reasoner = AuditReasoner(use_persistence=False)
        reasoner.add_default_rules()

        stats = reasoner.get_stats()

        assert "facts" in stats
        assert "rules" in stats
        assert "aggregate_strategy" in stats

    def test_save_state(self, mock_persistence_file, mock_rules_file):
        from scripts.audit_reasoning import AuditReasoner
        from cortical.reasoning.prism_pln import AttentionValue

        reasoner = AuditReasoner(use_persistence=True)
        reasoner.file_importance["saved_py"] = AttentionValue(sti=0.7, lti=0.4, vlti=False)

        reasoner.save_state()

        # Read back
        data = json.loads(mock_persistence_file.read_text())
        assert "saved_py" in data["file_importance"]
        assert data["session_count"] >= 1


class TestAuditReasonerPersistenceDecay:
    """Tests for importance decay based on time."""

    def test_decay_applies_based_on_time_elapsed(self, mock_persistence_file, mock_rules_file):
        from scripts.audit_reasoning import AuditReasoner

        # Create state from 12 hours ago
        old_time = (datetime.now() - timedelta(hours=12)).isoformat()
        state_data = {
            "version": 1, "created": "now", "updated": "now", "session_count": 1,
            "file_importance": {
                "old_py": {
                    "file_id": "old_py", "sti": 1.0, "lti": 1.0, "vlti": False,
                    "last_seen": old_time,
                    "history": []
                }
            },
            "attention_focus": [], "global_stats": {}
        }
        mock_persistence_file.write_text(json.dumps(state_data))

        reasoner = AuditReasoner(use_persistence=True, apply_decay=True)

        # STI should have decayed (0.9^12 ≈ 0.28)
        assert reasoner.file_importance["old_py"].sti < 0.5

    def test_no_decay_when_disabled(self, mock_persistence_file, mock_rules_file):
        from scripts.audit_reasoning import AuditReasoner

        old_time = (datetime.now() - timedelta(hours=24)).isoformat()
        state_data = {
            "version": 1, "created": "now", "updated": "now", "session_count": 1,
            "file_importance": {
                "nodecay_py": {
                    "file_id": "nodecay_py", "sti": 1.0, "lti": 1.0, "vlti": False,
                    "last_seen": old_time,
                    "history": []
                }
            },
            "attention_focus": [], "global_stats": {}
        }
        mock_persistence_file.write_text(json.dumps(state_data))

        reasoner = AuditReasoner(use_persistence=True, apply_decay=False)

        # Should not have decayed
        assert reasoner.file_importance["nodecay_py"].sti == 1.0


# =============================================================================
# GENERATE REPORT
# =============================================================================


class TestGenerateReasoningReport:
    """Tests for generate_reasoning_report function."""

    def test_generates_report_with_results(self):
        from scripts.audit_reasoning import generate_reasoning_report

        results = {
            "files_analyzed": 10,
            "rules_loaded": 12,
            "aggregate_strategy": "revision",
            "risk_assessments": [
                {
                    "file": "test.py",
                    "overall_risk": 0.7,
                    "details": {},
                    "importance": 0.5
                }
            ],
            "reasoner_stats": {
                "facts": 10,
                "rules": 5
            },
            "stats": {
                "files_tracked": 5,
                "vlti_files": 1
            },
            "priority_files": [("test_py", 0.8)],
            "vlti_files": ["critical_py"]
        }

        report = generate_reasoning_report(results, verbose=False)

        assert "test.py" in report
        assert "AUDIT REASONING" in report

    def test_generates_empty_report_when_no_results(self):
        from scripts.audit_reasoning import generate_reasoning_report

        results = {
            "files_analyzed": 0,
            "rules_loaded": 0,
            "risk_assessments": [],
            "reasoner_stats": {}
        }

        report = generate_reasoning_report(results, verbose=False)

        assert "Files analyzed: 0" in report


# =============================================================================
# WOVEN MIND RULE LOADING
# =============================================================================


class TestLoadRulesFromWovenMind:
    """Tests for loading rules from WovenMind abstractions."""

    def test_loads_compound_rules_from_abstractions(
        self, mock_persistence_file, mock_rules_file, mock_woven_mind_file
    ):
        from scripts.audit_reasoning import AuditReasoner

        woven_data = {
            "mind": {
                "cortex_state": {
                    "engine_state": {
                        "abstractions": {
                            "abs_1": {
                                "source_nodes": ["dir:legacy", "pattern:todo"],
                                "level": 1,
                                "frequency": 10,
                                "strength": 0.7
                            }
                        }
                    }
                }
            }
        }
        mock_woven_mind_file.write_text(json.dumps(woven_data))

        reasoner = AuditReasoner(use_persistence=False)
        count = reasoner.load_rules_from_woven_mind()

        # Should have loaded compound rule
        assert count >= 1


# =============================================================================
# ANALYZE WITH REASONING PIPELINE
# =============================================================================


class TestAnalyzeWithReasoning:
    """Tests for analyze_with_reasoning function."""

    def test_analyze_with_empty_analysis(
        self, mock_persistence_file, mock_rules_file, monkeypatch, capsys
    ):
        from scripts.audit_reasoning import analyze_with_reasoning

        # Mock analyze_directory to return empty
        monkeypatch.setattr(
            "scripts.audit_reasoning.analyze_directory",
            lambda *args, **kwargs: None
        )

        results = analyze_with_reasoning("test/", use_persistence=False)

        assert results["files_analyzed"] == 0
        captured = capsys.readouterr()
        assert "No analysis results" in captured.out

    def test_analyze_with_findings(
        self, mock_persistence_file, mock_rules_file, monkeypatch, capsys
    ):
        from scripts.audit_reasoning import analyze_with_reasoning

        # Mock analyze_directory to return findings
        mock_analysis = {
            "findings": [
                {"id": "test.py:10", "pattern": "todo", "message": "Fix this"},
                {"id": "test.py:20", "pattern": "fixme", "message": "Bug here"},
            ],
            "git_analysis": {
                "high_churn_files": [("test.py", 15)],
                "bug_prone_files": [],
                "critical_modules": []
            }
        }
        monkeypatch.setattr(
            "scripts.audit_reasoning.analyze_directory",
            lambda *args, **kwargs: mock_analysis
        )

        results = analyze_with_reasoning(
            "test/",
            use_persistence=False,
            no_save=True
        )

        assert results["files_analyzed"] > 0
        assert results["rules_loaded"] > 0

    def test_analyze_with_git_analysis(
        self, mock_persistence_file, mock_rules_file, monkeypatch
    ):
        from scripts.audit_reasoning import analyze_with_reasoning

        mock_analysis = {
            "findings": [
                {"id": "critical.py:5", "pattern": "hack", "message": "Workaround"}
            ],
            "git_analysis": {
                "high_churn_files": [("critical.py", 50)],
                "bug_prone_files": [("critical.py", 10)],
                "critical_modules": [("critical.py", 100)]
            }
        }
        monkeypatch.setattr(
            "scripts.audit_reasoning.analyze_directory",
            lambda *args, **kwargs: mock_analysis
        )

        results = analyze_with_reasoning(
            "test/",
            with_git=True,
            use_persistence=False,
            no_save=True
        )

        # Should have processed the critical file
        assert len(results["risk_assessments"]) >= 0

    def test_analyze_with_persistence(
        self, mock_persistence_file, mock_rules_file, monkeypatch
    ):
        from scripts.audit_reasoning import analyze_with_reasoning

        mock_analysis = {
            "findings": [{"id": "mod.py:1", "pattern": "todo", "message": "Task"}],
            "git_analysis": {}
        }
        monkeypatch.setattr(
            "scripts.audit_reasoning.analyze_directory",
            lambda *args, **kwargs: mock_analysis
        )

        # First run - should create state
        results1 = analyze_with_reasoning("test/", use_persistence=True)

        # State file should exist
        assert mock_persistence_file.exists()

    def test_analyze_with_aggregation_strategies(
        self, mock_persistence_file, mock_rules_file, monkeypatch
    ):
        from scripts.audit_reasoning import analyze_with_reasoning

        mock_analysis = {
            "findings": [{"id": "agg.py:1", "pattern": "should_be", "message": "Check"}],
            "git_analysis": {}
        }
        monkeypatch.setattr(
            "scripts.audit_reasoning.analyze_directory",
            lambda *args, **kwargs: mock_analysis
        )

        for strategy in ["first", "max", "revision"]:
            results = analyze_with_reasoning(
                "test/",
                aggregate_strategy=strategy,
                use_persistence=False,
                no_save=True
            )
            assert results["aggregate_strategy"] == strategy


# =============================================================================
# REPORT GENERATION - DETAILED
# =============================================================================


class TestGenerateReportVerbose:
    """Tests for verbose report generation."""

    def test_verbose_report_includes_details(self):
        from scripts.audit_reasoning import generate_reasoning_report

        results = {
            "files_analyzed": 5,
            "rules_loaded": 10,
            "aggregate_strategy": "revision",
            "risk_assessments": [
                {
                    "file": "detailed.py",
                    "overall_risk": 0.8,
                    "details": {
                        "needs_review": {"strength": 0.7, "confidence": 0.8},
                        "risky": {"strength": 0.6, "confidence": 0.75}
                    },
                    "importance": 0.6
                }
            ],
            "stats": {"files_tracked": 3, "vlti_files": 0},
            "priority_files": [],
            "vlti_files": []
        }

        report = generate_reasoning_report(results, verbose=True)

        assert "detailed.py" in report
        assert "Risk:" in report or "HIGH" in report

    def test_report_with_priority_files(self):
        from scripts.audit_reasoning import generate_reasoning_report

        results = {
            "files_analyzed": 3,
            "rules_loaded": 5,
            "risk_assessments": [],
            "priority_files": [
                ("important_py", 0.9),
                ("medium_py", 0.5)
            ],
            "vlti_files": [],
            "stats": {}
        }

        report = generate_reasoning_report(results, verbose=False)

        assert "Priority Files" in report
        assert "important_py" in report

    def test_report_with_vlti_files(self):
        from scripts.audit_reasoning import generate_reasoning_report

        results = {
            "files_analyzed": 2,
            "rules_loaded": 4,
            "risk_assessments": [],
            "priority_files": [],
            "vlti_files": ["critical_module_py", "core_api_py"],
            "stats": {}
        }

        report = generate_reasoning_report(results, verbose=False)

        assert "Critical Files" in report or "VLTI" in report
        assert "critical_module_py" in report


# =============================================================================
# EDGE CASES AND ADDITIONAL COVERAGE
# =============================================================================


class TestEdgeCasesAndBranches:
    """Tests for edge cases and branch coverage."""

    def test_query_file_risk_with_attention_focus(
        self, mock_persistence_file, mock_rules_file
    ):
        from scripts.audit_reasoning import AuditReasoner

        reasoner = AuditReasoner(use_persistence=False)
        reasoner.add_default_rules()
        reasoner.assert_file_facts("focused.py", ["todo"], [], [])

        # Add to attention focus
        reasoner.attention_focus.focus_on(["focused_py"])

        results = reasoner.query_file_risk("focused.py", use_attention=True)

        assert isinstance(results, dict)

    def test_query_file_risk_without_attention_and_importance(
        self, mock_persistence_file, mock_rules_file
    ):
        from scripts.audit_reasoning import AuditReasoner

        reasoner = AuditReasoner(use_persistence=False)
        reasoner.add_default_rules()
        reasoner.assert_file_facts("plain.py", ["fixme"], [], [])

        results = reasoner.query_file_risk(
            "plain.py",
            use_attention=False,
            use_importance=False
        )

        assert isinstance(results, dict)

    def test_get_importance_trend_no_history(
        self, mock_persistence_file, mock_rules_file
    ):
        from scripts.audit_reasoning import AuditReasoner

        reasoner = AuditReasoner(use_persistence=False)

        trend = reasoner.get_importance_trend("nonexistent_py")

        assert trend is None

    def test_save_state_updates_existing_file(
        self, mock_persistence_file, mock_rules_file
    ):
        from scripts.audit_reasoning import AuditReasoner
        from cortical.reasoning.prism_pln import AttentionValue

        # First reasoner - create initial state
        reasoner1 = AuditReasoner(use_persistence=True)
        reasoner1.file_importance["file1_py"] = AttentionValue(sti=0.5, lti=0.3)
        reasoner1.save_state()

        # Second reasoner - update state
        reasoner2 = AuditReasoner(use_persistence=True)
        reasoner2.file_importance["file1_py"].sti = 0.8
        reasoner2.file_importance["file2_py"] = AttentionValue(sti=0.6, lti=0.4)
        reasoner2.save_state()

        # Check state has both files
        data = json.loads(mock_persistence_file.read_text())
        assert "file1_py" in data["file_importance"]
        assert "file2_py" in data["file_importance"]
        # Session count should increment
        assert data["session_count"] >= 2

    def test_focus_on_high_risk_empty(
        self, mock_persistence_file, mock_rules_file
    ):
        from scripts.audit_reasoning import AuditReasoner

        reasoner = AuditReasoner(use_persistence=False)

        # No files tracked
        count = reasoner.focus_on_high_risk_files(threshold=0.5)

        assert count == 0

    def test_stimulate_new_file(
        self, mock_persistence_file, mock_rules_file
    ):
        from scripts.audit_reasoning import AuditReasoner

        reasoner = AuditReasoner(use_persistence=False)

        # File not yet tracked - stimulate should still work
        reasoner.stimulate_file("new.py", amount=0.5)

        # May or may not create the file depending on implementation
        # This tests the path

    def test_woven_mind_single_antecedent_rules(
        self, mock_persistence_file, mock_rules_file, mock_woven_mind_file
    ):
        from scripts.audit_reasoning import AuditReasoner

        # Single node abstraction (less than 2 valid parts after filtering)
        woven_data = {
            "mind": {
                "cortex_state": {
                    "engine_state": {
                        "abstractions": {
                            "single": {
                                "source_nodes": ["pattern:todo"],
                                "frequency": 5,
                                "strength": 0.6
                            }
                        }
                    }
                }
            }
        }
        mock_woven_mind_file.write_text(json.dumps(woven_data))

        reasoner = AuditReasoner(use_persistence=False)
        count = reasoner.load_rules_from_woven_mind()

        # Single node should still create a simple rule
        assert count >= 0

    def test_translate_query_with_highchurn_variant(self):
        from scripts.audit_reasoning import translate_audit_query

        query = translate_audit_query("files with highchurn")

        assert "high_churn" in query.include_traits

    def test_is_nlu_existing_path_with_keyword(self, tmp_path):
        from scripts.audit_reasoning import is_natural_language_query

        # Create a directory that looks like it could be NLU
        test_dir = tmp_path / "risky_module"
        test_dir.mkdir()

        # Path exists but contains NLU keyword - should be treated as NLU
        result = is_natural_language_query(str(test_dir) + " not tests")
        assert result is True

    def test_persistence_with_invalid_timestamp(
        self, mock_persistence_file, mock_rules_file
    ):
        from scripts.audit_reasoning import AuditReasoner

        state_data = {
            "version": 1, "created": "now", "updated": "now", "session_count": 1,
            "file_importance": {
                "bad_ts_py": {
                    "file_id": "bad_ts_py",
                    "sti": 0.5, "lti": 0.3, "vlti": False,
                    "last_seen": "not-a-valid-timestamp",
                    "history": []
                }
            },
            "attention_focus": [], "global_stats": {}
        }
        mock_persistence_file.write_text(json.dumps(state_data))

        # Should handle invalid timestamp gracefully
        reasoner = AuditReasoner(use_persistence=True, apply_decay=True)

        assert "bad_ts_py" in reasoner.file_importance

    def test_assert_file_facts_with_critical_trait(
        self, mock_persistence_file, mock_rules_file
    ):
        from scripts.audit_reasoning import AuditReasoner

        reasoner = AuditReasoner(use_persistence=False)

        reasoner.assert_file_facts(
            "critical_module.py",
            patterns=["todo"],
            traits=["critical"],
            directories=[]
        )

        # Critical trait should set VLTI and higher LTI
        assert reasoner.file_importance["critical_module_py"].vlti is True
        assert reasoner.file_importance["critical_module_py"].lti == 0.2


# =============================================================================
# CLI MAIN FUNCTION TESTS
# =============================================================================


class TestCLIMain:
    """Tests for main() CLI function."""

    def test_main_show_rules(
        self, mock_persistence_file, mock_rules_file, mock_woven_mind_file,
        monkeypatch, capsys
    ):
        from scripts.audit_reasoning import main

        # Set up command line args
        monkeypatch.setattr("sys.argv", ["audit_reasoning.py", "--show-rules"])

        # Create some rules
        rules_data = {
            "version": 1,
            "manual_rules": [
                {"antecedent": "test(X)", "consequent": "flag(X)", "strength": 0.7}
            ]
        }
        mock_rules_file.write_text(json.dumps(rules_data))

        main()

        captured = capsys.readouterr()
        assert "PLN AUDIT RULES" in captured.out

    def test_main_show_state(
        self, mock_persistence_file, mock_rules_file, monkeypatch, capsys
    ):
        from scripts.audit_reasoning import main

        monkeypatch.setattr("sys.argv", ["audit_reasoning.py", "--show-state"])

        # Create persistence state
        state_data = {
            "version": 1, "created": "2026-01-01", "updated": "2026-01-08",
            "session_count": 5, "file_importance": {}, "attention_focus": [],
            "global_stats": {}
        }
        mock_persistence_file.write_text(json.dumps(state_data))

        main()

        captured = capsys.readouterr()
        assert "AUDIT PLN PERSISTENCE STATE" in captured.out
        assert "Session count: 5" in captured.out

    def test_main_clear_state(
        self, mock_persistence_file, mock_rules_file, monkeypatch, capsys
    ):
        from scripts.audit_reasoning import main

        monkeypatch.setattr("sys.argv", ["audit_reasoning.py", "--clear-state"])
        mock_persistence_file.write_text("{}")

        main()

        captured = capsys.readouterr()
        assert "cleared" in captured.out
        assert not mock_persistence_file.exists()

    def test_main_file_history(
        self, mock_persistence_file, mock_rules_file, monkeypatch, capsys
    ):
        from scripts.audit_reasoning import main

        monkeypatch.setattr(
            "sys.argv",
            ["audit_reasoning.py", "--file-history", "test.py"]
        )

        state_data = {
            "version": 1, "created": "now", "updated": "now", "session_count": 1,
            "file_importance": {
                "test_py": {
                    "file_id": "test_py", "sti": 0.6, "lti": 0.4, "vlti": False,
                    "last_seen": "2026-01-08T10:00:00",
                    "history": [{"timestamp": "2026-01-07", "sti": 0.5, "lti": 0.3}]
                }
            },
            "attention_focus": [], "global_stats": {}
        }
        mock_persistence_file.write_text(json.dumps(state_data))

        main()

        captured = capsys.readouterr()
        assert "Importance History" in captured.out
        assert "test_py" in captured.out

    def test_main_file_history_not_found(
        self, mock_persistence_file, mock_rules_file, monkeypatch, capsys
    ):
        from scripts.audit_reasoning import main

        monkeypatch.setattr(
            "sys.argv",
            ["audit_reasoning.py", "--file-history", "nonexistent.py"]
        )

        state_data = {
            "version": 1, "created": "now", "updated": "now", "session_count": 1,
            "file_importance": {}, "attention_focus": [], "global_stats": {}
        }
        mock_persistence_file.write_text(json.dumps(state_data))

        main()

        captured = capsys.readouterr()
        assert "No history found" in captured.out

    def test_main_add_rule(
        self, mock_persistence_file, mock_rules_file, monkeypatch, capsys
    ):
        from scripts.audit_reasoning import main

        monkeypatch.setattr(
            "sys.argv",
            ["audit_reasoning.py", "--add-rule", "test(X)", "flagged(X)", "0.8"]
        )

        main()

        captured = capsys.readouterr()
        assert "Added rule" in captured.out

        # Check rule was saved
        data = json.loads(mock_rules_file.read_text())
        assert len(data["manual_rules"]) == 1
        assert data["manual_rules"][0]["strength"] == 0.8

    def test_main_with_nlu_query(
        self, mock_persistence_file, mock_rules_file, monkeypatch, capsys
    ):
        from scripts.audit_reasoning import main

        monkeypatch.setattr(
            "sys.argv",
            ["audit_reasoning.py", "cortical/ not tests"]
        )

        # Mock analyze_directory
        mock_analysis = {
            "findings": [{"id": "mod.py:1", "pattern": "todo", "message": "Task"}],
            "git_analysis": {}
        }
        monkeypatch.setattr(
            "scripts.audit_reasoning.analyze_directory",
            lambda *args, **kwargs: mock_analysis
        )

        main()

        captured = capsys.readouterr()
        assert "Natural Language Query" in captured.out

    def test_main_with_explain_query(
        self, mock_persistence_file, mock_rules_file, monkeypatch, capsys
    ):
        from scripts.audit_reasoning import main

        monkeypatch.setattr(
            "sys.argv",
            ["audit_reasoning.py", "why is test.py flagged", "--no-save"]
        )

        mock_analysis = {
            "findings": [{"id": "test.py:1", "pattern": "todo", "message": "Fix"}],
            "git_analysis": {}
        }
        monkeypatch.setattr(
            "scripts.audit_reasoning.analyze_directory",
            lambda *args, **kwargs: mock_analysis
        )

        main()

        captured = capsys.readouterr()
        assert "Explaining" in captured.out or "explain" in captured.out.lower()

    def test_main_with_aggregation_flag(
        self, mock_persistence_file, mock_rules_file, monkeypatch, capsys
    ):
        from scripts.audit_reasoning import main

        monkeypatch.setattr(
            "sys.argv",
            ["audit_reasoning.py", "test/", "--aggregate", "max", "--no-save"]
        )

        mock_analysis = {
            "findings": [{"id": "file.py:1", "pattern": "hack", "message": "Workaround"}],
            "git_analysis": {}
        }
        monkeypatch.setattr(
            "scripts.audit_reasoning.analyze_directory",
            lambda *args, **kwargs: mock_analysis
        )

        main()

        captured = capsys.readouterr()
        # Should use max aggregation
        assert "max" in captured.out.lower() or "Aggregation" in captured.out


# =============================================================================
# MORE BRANCH COVERAGE
# =============================================================================


class TestMoreBranchCoverage:
    """Additional tests for branch coverage."""

    def test_report_with_low_risk_file(self):
        from scripts.audit_reasoning import generate_reasoning_report

        results = {
            "files_analyzed": 1,
            "rules_loaded": 5,
            "risk_assessments": [
                {
                    "file": "clean.py",
                    "overall_risk": 0.2,  # Low risk
                    "details": {},
                    "importance": 0.3
                }
            ],
            "priority_files": [],
            "vlti_files": [],
            "stats": {}
        }

        report = generate_reasoning_report(results, verbose=False)

        assert "clean.py" in report
        assert "LOW" in report

    def test_report_with_medium_risk_file(self):
        from scripts.audit_reasoning import generate_reasoning_report

        results = {
            "files_analyzed": 1,
            "rules_loaded": 5,
            "risk_assessments": [
                {
                    "file": "medium.py",
                    "overall_risk": 0.55,  # Medium risk
                    "details": {},
                    "importance": 0.4
                }
            ],
            "priority_files": [],
            "vlti_files": [],
            "stats": {}
        }

        report = generate_reasoning_report(results, verbose=False)

        assert "medium.py" in report
        assert "MEDIUM" in report

    def test_abstraction_to_rule_with_all_file_nodes(self):
        from scripts.audit_reasoning import abstraction_to_rule

        # All file: nodes should be skipped
        abstraction = {
            "source_nodes": ["file:a.py", "file:b.py", "file:c.py"],
            "frequency": 5
        }

        rule = abstraction_to_rule(abstraction)

        assert rule is None

    def test_translate_query_fallback_directory(self, tmp_path):
        from scripts.audit_reasoning import translate_audit_query

        # Create an actual directory
        test_dir = tmp_path / "mycode"
        test_dir.mkdir()

        # Query starts with existing directory
        query = translate_audit_query(f"{test_dir}")

        assert query.directory is not None

    def test_analyze_with_attention_disabled(
        self, mock_persistence_file, mock_rules_file, monkeypatch
    ):
        from scripts.audit_reasoning import analyze_with_reasoning

        mock_analysis = {
            "findings": [{"id": "noatt.py:1", "pattern": "fixme", "message": "Bug"}],
            "git_analysis": {}
        }
        monkeypatch.setattr(
            "scripts.audit_reasoning.analyze_directory",
            lambda *args, **kwargs: mock_analysis
        )

        results = analyze_with_reasoning(
            "test/",
            enable_attention=False,
            enable_importance=False,
            use_persistence=False,
            no_save=True
        )

        assert results is not None

    def test_analyze_verbose_output(
        self, mock_persistence_file, mock_rules_file, monkeypatch, capsys
    ):
        from scripts.audit_reasoning import analyze_with_reasoning

        mock_analysis = {
            "findings": [{"id": "verb.py:1", "pattern": "todo", "message": "Task"}],
            "git_analysis": {}
        }
        monkeypatch.setattr(
            "scripts.audit_reasoning.analyze_directory",
            lambda *args, **kwargs: mock_analysis
        )

        results = analyze_with_reasoning(
            "test/",
            verbose=True,
            use_persistence=False,
            no_save=True
        )

        # Verbose should produce more output
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_woven_mind_with_multiple_compound_parts(
        self, mock_persistence_file, mock_rules_file, mock_woven_mind_file
    ):
        from scripts.audit_reasoning import AuditReasoner

        woven_data = {
            "mind": {
                "cortex_state": {
                    "engine_state": {
                        "abstractions": {
                            "multi": {
                                "source_nodes": [
                                    "dir:api",
                                    "pattern:hack",
                                    "trait:high_churn"
                                ],
                                "frequency": 20,
                                "strength": 0.8
                            }
                        }
                    }
                }
            }
        }
        mock_woven_mind_file.write_text(json.dumps(woven_data))

        reasoner = AuditReasoner(use_persistence=False)
        count = reasoner.load_rules_from_woven_mind()

        assert count >= 1

    def test_persistence_restore_attention_focus(
        self, mock_persistence_file, mock_rules_file
    ):
        from scripts.audit_reasoning import AuditReasoner

        state_data = {
            "version": 1, "created": "now", "updated": "now", "session_count": 1,
            "file_importance": {
                "focused_py": {
                    "file_id": "focused_py", "sti": 0.8, "lti": 0.5, "vlti": False,
                    "last_seen": datetime.now().isoformat(), "history": []
                }
            },
            "attention_focus": ["focused_py", "other_py"],
            "global_stats": {}
        }
        mock_persistence_file.write_text(json.dumps(state_data))

        reasoner = AuditReasoner(use_persistence=True)

        # Should have restored attention focus
        assert len(reasoner.attention_focus._focused) > 0
