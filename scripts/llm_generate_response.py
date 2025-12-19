#!/usr/bin/env python3
"""
LLM Response Generation Pipeline Stage

Generates structured prompts from analysis results and optionally calls
LLM APIs (Claude/OpenAI) if credentials are available.

Usage:
    python scripts/llm_generate_response.py < analysis.json
    python scripts/llm_generate_response.py --input analysis.json --mode api
    python scripts/llm_generate_response.py --template synthesis --max-tokens 2000
"""

import argparse
import json
import os
import sys
from typing import Dict, Any, Optional, Tuple

# Prompt templates for different use cases
PROMPT_TEMPLATES = {
    "synthesis": {
        "system": "You are a cognitive science expert specializing in knowledge synthesis and conceptual integration. Your role is to take disparate concepts and their relationships and weave them into a coherent, insightful understanding.",
        "user": """Based on the following cognitive analysis results, please synthesize a coherent understanding:

{context}

Please provide:
1. A unified conceptual framework that integrates these elements
2. Key insights about the relationships and patterns
3. Implications for understanding and application
4. Any emergent properties or higher-level patterns

Focus on clarity, depth, and actionable insights."""
    },
    "explanation": {
        "system": "You are a cognitive science educator skilled at explaining complex conceptual relationships in clear, accessible language.",
        "user": """Based on the following analysis of concepts and their relationships:

{context}

Please explain:
1. How these concepts relate to each other
2. The nature and strength of the connections
3. Any hierarchical or causal relationships
4. Practical implications of these relationships

Make your explanation clear and well-structured."""
    },
    "gaps": {
        "system": "You are a knowledge engineering expert who identifies gaps in understanding and suggests ways to bridge them.",
        "user": """Based on the following analysis that has identified knowledge gaps:

{context}

Please provide:
1. A clear description of each identified gap
2. Why these gaps are significant
3. Concrete suggestions for how to bridge them
4. Resources or approaches that could help fill the gaps
5. Priority order for addressing the gaps

Be specific and actionable in your recommendations."""
    }
}


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate LLM prompts from cognitive analysis results"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Input JSON file (default: stdin)"
    )
    parser.add_argument(
        "--mode",
        choices=["prompt", "api", "auto"],
        default="auto",
        help="Output mode: prompt-only, api-call, or auto-detect (default: auto)"
    )
    parser.add_argument(
        "--template",
        choices=["synthesis", "explanation", "gaps", "custom"],
        default="synthesis",
        help="Prompt template to use (default: synthesis)"
    )
    parser.add_argument(
        "--custom-prompt",
        type=str,
        help="Custom prompt template (used when --template=custom)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1000,
        help="Maximum tokens for API response (default: 1000)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-3-haiku-20240307",
        help="Model to use (default: claude-3-haiku-20240307)"
    )
    return parser.parse_args()


def read_input(input_file: Optional[str]) -> Dict[str, Any]:
    """Read JSON input from file or stdin."""
    if input_file:
        with open(input_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        return json.load(sys.stdin)


def extract_context_summary(data: Dict[str, Any]) -> Dict[str, int]:
    """Extract summary statistics from analysis data."""
    summary = {
        "concepts_analyzed": 0,
        "bridges_found": 0,
        "gaps_identified": 0
    }

    # Count concepts
    if "concepts" in data:
        summary["concepts_analyzed"] = len(data["concepts"])
    elif "cognitive_model" in data and "concepts" in data["cognitive_model"]:
        summary["concepts_analyzed"] = len(data["cognitive_model"]["concepts"])

    # Count bridges
    if "bridges" in data:
        summary["bridges_found"] = len(data["bridges"])
    elif "connections" in data:
        summary["bridges_found"] = len(data["connections"])

    # Count gaps
    if "gaps" in data:
        summary["gaps_identified"] = len(data["gaps"])
    elif "knowledge_gaps" in data:
        summary["gaps_identified"] = len(data["knowledge_gaps"])

    return summary


def format_context(data: Dict[str, Any]) -> str:
    """Format analysis data into readable context for the prompt."""
    lines = []

    # Format concepts
    if "concepts" in data:
        lines.append("CONCEPTS ANALYZED:")
        for concept in data["concepts"][:10]:  # Limit to top 10
            if isinstance(concept, dict):
                name = concept.get("name", concept.get("term", "Unknown"))
                importance = concept.get("importance", concept.get("pagerank", 0))
                lines.append(f"  - {name} (importance: {importance:.3f})")
            else:
                lines.append(f"  - {concept}")
        if len(data["concepts"]) > 10:
            lines.append(f"  ... and {len(data['concepts']) - 10} more")
        lines.append("")

    # Format cognitive model
    if "cognitive_model" in data:
        model = data["cognitive_model"]
        lines.append("COGNITIVE MODEL:")
        if "concepts" in model:
            lines.append(f"  Concepts: {len(model['concepts'])}")
        if "connections" in model:
            lines.append(f"  Connections: {len(model['connections'])}")
        lines.append("")

    # Format bridges
    if "bridges" in data:
        lines.append("CONCEPTUAL BRIDGES:")
        for bridge in data["bridges"][:5]:  # Top 5 bridges
            if isinstance(bridge, dict):
                source = bridge.get("source", "?")
                target = bridge.get("target", "?")
                strength = bridge.get("strength", bridge.get("weight", 0))
                lines.append(f"  - {source} <-> {target} (strength: {strength:.3f})")
        if len(data["bridges"]) > 5:
            lines.append(f"  ... and {len(data['bridges']) - 5} more")
        lines.append("")

    # Format connections
    if "connections" in data and "bridges" not in data:
        lines.append("CONNECTIONS:")
        for conn in data["connections"][:5]:
            if isinstance(conn, dict):
                source = conn.get("source", "?")
                target = conn.get("target", "?")
                lines.append(f"  - {source} -> {target}")
        if len(data["connections"]) > 5:
            lines.append(f"  ... and {len(data['connections']) - 5} more")
        lines.append("")

    # Format gaps
    if "gaps" in data:
        lines.append("KNOWLEDGE GAPS IDENTIFIED:")
        for gap in data["gaps"][:5]:
            if isinstance(gap, dict):
                description = gap.get("description", gap.get("gap", "Unknown gap"))
                lines.append(f"  - {description}")
            else:
                lines.append(f"  - {gap}")
        if len(data["gaps"]) > 5:
            lines.append(f"  ... and {len(data['gaps']) - 5} more")
        lines.append("")

    # Format world model if present
    if "world_model" in data:
        lines.append("WORLD MODEL:")
        wm = data["world_model"]
        if "entities" in wm:
            lines.append(f"  Entities: {len(wm['entities'])}")
        if "relations" in wm:
            lines.append(f"  Relations: {len(wm['relations'])}")
        lines.append("")

    return "\n".join(lines)


def generate_prompt(
    data: Dict[str, Any],
    template: str,
    custom_prompt: Optional[str] = None
) -> Dict[str, str]:
    """Generate structured prompt from analysis data."""
    context = format_context(data)

    if template == "custom" and custom_prompt:
        return {
            "system": "You are a helpful AI assistant with expertise in cognitive science and knowledge analysis.",
            "user": custom_prompt.format(context=context)
        }
    elif template in PROMPT_TEMPLATES:
        template_data = PROMPT_TEMPLATES[template]
        return {
            "system": template_data["system"],
            "user": template_data["user"].format(context=context)
        }
    else:
        # Default to synthesis
        template_data = PROMPT_TEMPLATES["synthesis"]
        return {
            "system": template_data["system"],
            "user": template_data["user"].format(context=context)
        }


def detect_api_availability() -> Tuple[bool, Optional[str]]:
    """Detect which LLM API is available based on environment variables."""
    if os.getenv("ANTHROPIC_API_KEY"):
        return True, "anthropic"
    elif os.getenv("OPENAI_API_KEY"):
        return True, "openai"
    else:
        return False, None


def call_anthropic_api(
    prompt: Dict[str, str],
    max_tokens: int,
    model: str
) -> str:
    """Call Anthropic Claude API."""
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic package not installed. Install with: pip install anthropic")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    client = anthropic.Anthropic(api_key=api_key)

    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=prompt["system"],
        messages=[
            {"role": "user", "content": prompt["user"]}
        ]
    )

    return message.content[0].text


def call_openai_api(
    prompt: Dict[str, str],
    max_tokens: int,
    model: str
) -> str:
    """Call OpenAI API."""
    try:
        import openai
    except ImportError:
        raise ImportError("openai package not installed. Install with: pip install openai")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    client = openai.OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=model if not model.startswith("claude") else "gpt-3.5-turbo",
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]}
        ]
    )

    return response.choices[0].message.content


def call_llm_api(
    prompt: Dict[str, str],
    api_type: str,
    max_tokens: int,
    model: str
) -> Tuple[Optional[str], Optional[str]]:
    """
    Call LLM API and return response.

    Returns:
        Tuple of (response_text, error_message)
    """
    try:
        if api_type == "anthropic":
            response = call_anthropic_api(prompt, max_tokens, model)
            return response, None
        elif api_type == "openai":
            response = call_openai_api(prompt, max_tokens, model)
            return response, None
        else:
            return None, f"Unknown API type: {api_type}"
    except ImportError as e:
        return None, f"API library not available: {str(e)}"
    except Exception as e:
        return None, f"API call failed: {str(e)}"


def main() -> None:
    """Main entry point."""
    args = parse_arguments()

    # Read input
    try:
        data = read_input(args.input)
    except Exception as e:
        print(f"Error reading input: {e}", file=sys.stderr)
        sys.exit(1)

    # Generate prompt
    prompt = generate_prompt(data, args.template, args.custom_prompt)

    # Extract context summary
    context_summary = extract_context_summary(data)

    # Determine mode
    api_available, api_type = detect_api_availability()

    if args.mode == "auto":
        mode = "api_call" if api_available else "prompt_only"
    else:
        mode = "api_call" if args.mode == "api" else "prompt_only"

    # Build output
    output = {
        "prompt": prompt,
        "context_summary": context_summary,
        "response": None,
        "api_used": None,
        "mode": mode
    }

    # Call API if requested and available
    if mode == "api_call":
        if not api_available:
            print("Warning: API mode requested but no API key found. Falling back to prompt-only mode.", file=sys.stderr)
            output["mode"] = "prompt_only"
        else:
            response, error = call_llm_api(prompt, api_type, args.max_tokens, args.model)
            if response:
                output["response"] = response
                output["api_used"] = api_type
            else:
                print(f"Warning: API call failed: {error}. Falling back to prompt-only mode.", file=sys.stderr)
                output["mode"] = "prompt_only"

    # Output result
    json.dump(output, sys.stdout, indent=2)
    print()  # Newline for readability


if __name__ == "__main__":
    main()
