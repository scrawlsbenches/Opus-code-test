#!/usr/bin/env python3
"""
Unit tests for llm_generate_response.py

Tests prompt generation, template selection, context summarization,
and API detection without making actual API calls.
"""

import json
import os
import sys
import unittest
from io import StringIO
from unittest.mock import patch, MagicMock, mock_open

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))

import llm_generate_response


class TestPromptGeneration(unittest.TestCase):
    """Test prompt generation from various inputs."""

    def setUp(self):
        """Set up test data."""
        self.sample_data = {
            "concepts": [
                {"name": "neural networks", "importance": 0.95},
                {"name": "machine learning", "importance": 0.87}
            ],
            "bridges": [
                {"source": "neural networks", "target": "deep learning", "strength": 0.92}
            ],
            "gaps": [
                {"description": "Limited understanding of transformers"}
            ]
        }

    def test_synthesis_template(self):
        """Test synthesis prompt template generation."""
        prompt = llm_generate_response.generate_prompt(
            self.sample_data,
            template="synthesis"
        )

        self.assertIn("system", prompt)
        self.assertIn("user", prompt)
        self.assertIn("cognitive science", prompt["system"].lower())
        self.assertIn("synthesize", prompt["user"].lower())
        self.assertIn("neural networks", prompt["user"])

    def test_explanation_template(self):
        """Test explanation prompt template generation."""
        prompt = llm_generate_response.generate_prompt(
            self.sample_data,
            template="explanation"
        )

        self.assertIn("system", prompt)
        self.assertIn("user", prompt)
        self.assertIn("educator", prompt["system"].lower())
        self.assertIn("explain", prompt["user"].lower())

    def test_gaps_template(self):
        """Test gaps prompt template generation."""
        prompt = llm_generate_response.generate_prompt(
            self.sample_data,
            template="gaps"
        )

        self.assertIn("system", prompt)
        self.assertIn("user", prompt)
        self.assertIn("gaps", prompt["user"].lower())
        self.assertIn("bridge", prompt["user"].lower())

    def test_custom_template(self):
        """Test custom prompt template."""
        custom = "Custom prompt with {context} placeholder"
        prompt = llm_generate_response.generate_prompt(
            self.sample_data,
            template="custom",
            custom_prompt=custom
        )

        self.assertIn("system", prompt)
        self.assertIn("user", prompt)
        self.assertIn("neural networks", prompt["user"])

    def test_default_template(self):
        """Test default template falls back to synthesis."""
        prompt = llm_generate_response.generate_prompt(
            self.sample_data,
            template="unknown_template"
        )

        self.assertIn("system", prompt)
        self.assertIn("user", prompt)
        self.assertIn("synthesize", prompt["user"].lower())


class TestContextSummarization(unittest.TestCase):
    """Test context extraction and summarization."""

    def test_extract_context_summary(self):
        """Test extracting summary statistics from data."""
        data = {
            "concepts": [{"name": "a"}, {"name": "b"}, {"name": "c"}],
            "bridges": [{"source": "a", "target": "b"}],
            "gaps": [{"description": "gap1"}, {"description": "gap2"}]
        }

        summary = llm_generate_response.extract_context_summary(data)

        self.assertEqual(summary["concepts_analyzed"], 3)
        self.assertEqual(summary["bridges_found"], 1)
        self.assertEqual(summary["gaps_identified"], 2)

    def test_extract_from_cognitive_model(self):
        """Test extracting from cognitive model structure."""
        data = {
            "cognitive_model": {
                "concepts": ["a", "b", "c"],
                "connections": [{"source": "a", "target": "b"}]
            },
            "knowledge_gaps": ["gap1"]
        }

        summary = llm_generate_response.extract_context_summary(data)

        self.assertEqual(summary["concepts_analyzed"], 3)
        self.assertEqual(summary["gaps_identified"], 1)

    def test_extract_with_connections(self):
        """Test extracting bridges from connections field."""
        data = {
            "concepts": ["a", "b"],
            "connections": [{"source": "a", "target": "b"}, {"source": "b", "target": "c"}]
        }

        summary = llm_generate_response.extract_context_summary(data)

        self.assertEqual(summary["bridges_found"], 2)

    def test_format_context(self):
        """Test formatting data into readable context."""
        data = {
            "concepts": [
                {"name": "neural networks", "importance": 0.95}
            ],
            "bridges": [
                {"source": "A", "target": "B", "strength": 0.8}
            ],
            "gaps": [
                {"description": "Understanding transformers"}
            ]
        }

        context = llm_generate_response.format_context(data)

        self.assertIn("CONCEPTS ANALYZED", context)
        self.assertIn("neural networks", context)
        self.assertIn("CONCEPTUAL BRIDGES", context)
        self.assertIn("KNOWLEDGE GAPS", context)
        self.assertIn("Understanding transformers", context)

    def test_format_context_with_cognitive_model(self):
        """Test formatting with cognitive model structure."""
        data = {
            "cognitive_model": {
                "concepts": ["a", "b", "c"],
                "connections": [{"source": "a", "target": "b"}]
            }
        }

        context = llm_generate_response.format_context(data)

        self.assertIn("COGNITIVE MODEL", context)
        self.assertIn("Concepts: 3", context)
        self.assertIn("Connections: 1", context)

    def test_format_context_with_world_model(self):
        """Test formatting with world model structure."""
        data = {
            "world_model": {
                "entities": ["entity1", "entity2"],
                "relations": [{"from": "entity1", "to": "entity2"}]
            }
        }

        context = llm_generate_response.format_context(data)

        self.assertIn("WORLD MODEL", context)
        self.assertIn("Entities: 2", context)
        self.assertIn("Relations: 1", context)

    def test_format_context_truncates_long_lists(self):
        """Test that long lists are truncated."""
        data = {
            "concepts": [{"name": f"concept{i}", "importance": 0.5} for i in range(20)],
            "bridges": [{"source": f"s{i}", "target": f"t{i}", "strength": 0.5} for i in range(10)]
        }

        context = llm_generate_response.format_context(data)

        self.assertIn("and 10 more", context)  # Concepts truncated
        self.assertIn("and 5 more", context)   # Bridges truncated


class TestAPIDetection(unittest.TestCase):
    """Test API availability detection."""

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=True)
    def test_detect_anthropic(self):
        """Test detecting Anthropic API key."""
        available, api_type = llm_generate_response.detect_api_availability()

        self.assertTrue(available)
        self.assertEqual(api_type, "anthropic")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True)
    def test_detect_openai(self):
        """Test detecting OpenAI API key."""
        available, api_type = llm_generate_response.detect_api_availability()

        self.assertTrue(available)
        self.assertEqual(api_type, "openai")

    @patch.dict(os.environ, {}, clear=True)
    def test_no_api_key(self):
        """Test when no API key is available."""
        available, api_type = llm_generate_response.detect_api_availability()

        self.assertFalse(available)
        self.assertIsNone(api_type)

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test1", "OPENAI_API_KEY": "test2"}, clear=True)
    def test_anthropic_priority(self):
        """Test that Anthropic is preferred when both keys present."""
        available, api_type = llm_generate_response.detect_api_availability()

        self.assertTrue(available)
        self.assertEqual(api_type, "anthropic")


class TestGracefulDegradation(unittest.TestCase):
    """Test graceful degradation without API keys."""

    @patch.dict(os.environ, {}, clear=True)
    def test_auto_mode_without_key(self):
        """Test auto mode falls back to prompt-only without API key."""
        data = {"concepts": [{"name": "test"}]}
        prompt = llm_generate_response.generate_prompt(data, "synthesis")
        context_summary = llm_generate_response.extract_context_summary(data)

        api_available, api_type = llm_generate_response.detect_api_availability()
        mode = "api_call" if api_available else "prompt_only"

        self.assertEqual(mode, "prompt_only")

    @patch.dict(os.environ, {}, clear=True)
    def test_api_mode_without_key_logic(self):
        """Test logic when API mode requested without key."""
        api_available, api_type = llm_generate_response.detect_api_availability()

        self.assertFalse(api_available)
        self.assertIsNone(api_type)


class TestInputReading(unittest.TestCase):
    """Test reading input from various sources."""

    def test_read_from_file(self):
        """Test reading JSON from file."""
        test_data = {"concepts": [{"name": "test"}]}
        test_json = json.dumps(test_data)

        with patch("builtins.open", mock_open(read_data=test_json)):
            result = llm_generate_response.read_input("test.json")

        self.assertEqual(result, test_data)

    @patch('sys.stdin', StringIO('{"concepts": [{"name": "stdin_test"}]}'))
    def test_read_from_stdin(self):
        """Test reading JSON from stdin."""
        result = llm_generate_response.read_input(None)

        self.assertIn("concepts", result)
        self.assertEqual(result["concepts"][0]["name"], "stdin_test")


class TestAPICallMocking(unittest.TestCase):
    """Test API calls with mocking (no real API calls)."""

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    def test_anthropic_api_call_mock(self):
        """Test Anthropic API call with mocking."""
        # Create mock anthropic module
        mock_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_message = MagicMock()
        mock_content = MagicMock()
        mock_content.text = "This is a test response from Claude"
        mock_message.content = [mock_content]
        mock_client.messages.create.return_value = mock_message
        mock_anthropic.Anthropic.return_value = mock_client

        # Inject mock into sys.modules
        with patch.dict('sys.modules', {'anthropic': mock_anthropic}):
            prompt = {
                "system": "You are helpful",
                "user": "Test prompt"
            }

            response = llm_generate_response.call_anthropic_api(prompt, 100, "claude-3-haiku-20240307")

            self.assertEqual(response, "This is a test response from Claude")
            mock_client.messages.create.assert_called_once()

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_openai_api_call_mock(self):
        """Test OpenAI API call with mocking."""
        # Create mock openai module
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "This is a test response from GPT"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client

        # Inject mock into sys.modules
        with patch.dict('sys.modules', {'openai': mock_openai}):
            prompt = {
                "system": "You are helpful",
                "user": "Test prompt"
            }

            response = llm_generate_response.call_openai_api(prompt, 100, "gpt-3.5-turbo")

            self.assertEqual(response, "This is a test response from GPT")
            mock_client.chat.completions.create.assert_called_once()

    @patch('llm_generate_response.call_anthropic_api')
    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    def test_call_llm_api_success(self, mock_call_anthropic):
        """Test successful LLM API call."""
        mock_call_anthropic.return_value = "Test response"

        prompt = {"system": "sys", "user": "usr"}
        response, error = llm_generate_response.call_llm_api(
            prompt, "anthropic", 100, "claude-3-haiku-20240307"
        )

        self.assertEqual(response, "Test response")
        self.assertIsNone(error)

    @patch('llm_generate_response.call_anthropic_api')
    def test_call_llm_api_error(self, mock_call_anthropic):
        """Test LLM API call with error."""
        mock_call_anthropic.side_effect = Exception("API error")

        prompt = {"system": "sys", "user": "usr"}
        response, error = llm_generate_response.call_llm_api(
            prompt, "anthropic", 100, "claude-3-haiku-20240307"
        )

        self.assertIsNone(response)
        self.assertIn("API call failed", error)

    def test_call_llm_api_unknown_type(self):
        """Test LLM API call with unknown API type."""
        prompt = {"system": "sys", "user": "usr"}
        response, error = llm_generate_response.call_llm_api(
            prompt, "unknown", 100, "model"
        )

        self.assertIsNone(response)
        self.assertIn("Unknown API type", error)

    def test_anthropic_api_import_error(self):
        """Test handling of missing anthropic package."""
        # Ensure anthropic is not available
        with patch.dict('sys.modules', {'anthropic': None}):
            prompt = {"system": "sys", "user": "usr"}

            with self.assertRaises(ImportError) as context:
                llm_generate_response.call_anthropic_api(prompt, 100, "claude-3-haiku-20240307")

            self.assertIn("anthropic package not installed", str(context.exception))

    def test_openai_api_import_error(self):
        """Test handling of missing openai package."""
        # Ensure openai is not available
        with patch.dict('sys.modules', {'openai': None}):
            prompt = {"system": "sys", "user": "usr"}

            with self.assertRaises(ImportError) as context:
                llm_generate_response.call_openai_api(prompt, 100, "gpt-3.5-turbo")

            self.assertIn("openai package not installed", str(context.exception))

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}, clear=True)
    def test_anthropic_api_no_key(self):
        """Test handling of missing API key."""
        # Mock anthropic module to be available
        mock_anthropic = MagicMock()

        with patch.dict('sys.modules', {'anthropic': mock_anthropic}):
            prompt = {"system": "sys", "user": "usr"}

            with self.assertRaises(ValueError) as context:
                llm_generate_response.call_anthropic_api(prompt, 100, "claude-3-haiku-20240307")

            self.assertIn("ANTHROPIC_API_KEY", str(context.exception))


class TestArgumentParsing(unittest.TestCase):
    """Test command line argument parsing."""

    @patch('sys.argv', ['script.py', '--input', 'test.json', '--mode', 'prompt'])
    def test_parse_basic_args(self):
        """Test parsing basic arguments."""
        args = llm_generate_response.parse_arguments()

        self.assertEqual(args.input, 'test.json')
        self.assertEqual(args.mode, 'prompt')

    @patch('sys.argv', ['script.py', '--template', 'gaps', '--max-tokens', '2000'])
    def test_parse_template_args(self):
        """Test parsing template and token arguments."""
        args = llm_generate_response.parse_arguments()

        self.assertEqual(args.template, 'gaps')
        self.assertEqual(args.max_tokens, 2000)

    @patch('sys.argv', ['script.py', '--custom-prompt', 'Custom prompt here'])
    def test_parse_custom_prompt(self):
        """Test parsing custom prompt argument."""
        args = llm_generate_response.parse_arguments()

        self.assertEqual(args.custom_prompt, 'Custom prompt here')

    @patch('sys.argv', ['script.py', '--model', 'gpt-4'])
    def test_parse_model_arg(self):
        """Test parsing model argument."""
        args = llm_generate_response.parse_arguments()

        self.assertEqual(args.model, 'gpt-4')

    @patch('sys.argv', ['script.py'])
    def test_default_args(self):
        """Test default argument values."""
        args = llm_generate_response.parse_arguments()

        self.assertIsNone(args.input)
        self.assertEqual(args.mode, 'auto')
        self.assertEqual(args.template, 'synthesis')
        self.assertEqual(args.max_tokens, 1000)
        self.assertEqual(args.model, 'claude-3-haiku-20240307')


class TestEndToEnd(unittest.TestCase):
    """Test end-to-end workflow."""

    @patch('sys.stdout', new_callable=StringIO)
    @patch('sys.argv', ['script.py'])
    @patch('sys.stdin', StringIO('{"concepts": [{"name": "test", "importance": 0.9}], "gaps": []}'))
    @patch.dict(os.environ, {}, clear=True)
    def test_prompt_only_workflow(self, mock_stdout):
        """Test complete prompt-only workflow."""
        llm_generate_response.main()

        output = mock_stdout.getvalue()
        result = json.loads(output.strip())

        self.assertIn("prompt", result)
        self.assertIn("context_summary", result)
        self.assertEqual(result["mode"], "prompt_only")
        self.assertIsNone(result["response"])
        self.assertIsNone(result["api_used"])
        self.assertEqual(result["context_summary"]["concepts_analyzed"], 1)

    @patch('sys.stdout', new_callable=StringIO)
    @patch('sys.stderr', new_callable=StringIO)
    @patch('sys.argv', ['script.py', '--mode', 'api'])
    @patch('sys.stdin', StringIO('{"concepts": [{"name": "test"}]}'))
    @patch.dict(os.environ, {}, clear=True)
    def test_api_mode_without_key(self, mock_stderr, mock_stdout):
        """Test API mode falls back gracefully without key."""
        llm_generate_response.main()

        output = mock_stdout.getvalue()
        result = json.loads(output.strip())

        self.assertEqual(result["mode"], "prompt_only")
        self.assertIn("Warning", mock_stderr.getvalue())

    @patch('sys.stdout', new_callable=StringIO)
    @patch('sys.argv', ['script.py', '--template', 'explanation'])
    @patch('sys.stdin', StringIO('{"concepts": [{"name": "test"}], "bridges": [{"source": "a", "target": "b"}]}'))
    @patch.dict(os.environ, {}, clear=True)
    def test_explanation_template_workflow(self, mock_stdout):
        """Test workflow with explanation template."""
        llm_generate_response.main()

        output = mock_stdout.getvalue()
        result = json.loads(output.strip())

        self.assertIn("explain", result["prompt"]["user"].lower())
        self.assertEqual(result["context_summary"]["bridges_found"], 1)


class TestErrorHandling(unittest.TestCase):
    """Test error handling scenarios."""

    @patch('sys.stderr', new_callable=StringIO)
    @patch('sys.argv', ['script.py', '--input', 'nonexistent.json'])
    def test_invalid_input_file(self, mock_stderr):
        """Test handling of invalid input file."""
        with self.assertRaises(SystemExit):
            llm_generate_response.main()

        self.assertIn("Error reading input", mock_stderr.getvalue())

    @patch('sys.stdin', StringIO('invalid json'))
    @patch('sys.stderr', new_callable=StringIO)
    @patch('sys.argv', ['script.py'])
    def test_invalid_json_input(self, mock_stderr):
        """Test handling of invalid JSON input."""
        with self.assertRaises(SystemExit):
            llm_generate_response.main()

        self.assertIn("Error reading input", mock_stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
