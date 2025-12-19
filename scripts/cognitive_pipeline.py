#!/usr/bin/env python3
"""
Cognitive Pipeline Orchestrator
================================

Orchestrates the cognitive analysis pipeline with support for:
- Loop-based reanalysis with convergence detection
- Middleware insertion between stages
- Dynamic parameter adjustment
- Interactive mode for exploration

Usage:
    # Run full pipeline
    python scripts/cognitive_pipeline.py --query "world models and prediction"

    # Run with iteration loop
    python scripts/cognitive_pipeline.py --query "learning" --max-iterations 3

    # Insert custom middleware
    python scripts/cognitive_pipeline.py --query "cognition" \\
        --after question_connection scripts/my_filter.py

    # Interactive exploration mode
    python scripts/cognitive_pipeline.py --interactive

    # Use specific stages only
    python scripts/cognitive_pipeline.py --stages "world_model_analysis,knowledge_bridge"

Pipeline Stages:
    1. world_model_analysis - Analyze corpus for concepts and connections
    2. question_connection  - Expand queries and find paths
    3. knowledge_analysis   - Detect patterns and clusters
    4. knowledge_bridge     - Find gaps and suggest bridges
    5. llm_generate_response - Generate synthesis prompts
"""

import argparse
import json
import subprocess
import sys
import os
from typing import Dict, List, Any, Optional, Tuple, Callable
from pathlib import Path
from dataclasses import dataclass, field

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.thought_chain import (
    ThoughtChain, ChainContext, ChainParameters,
    is_chain_format, wrap_existing_output
)


@dataclass
class StageConfig:
    """Configuration for a pipeline stage."""
    name: str
    script: str
    args: List[str] = field(default_factory=list)
    enabled: bool = True
    timeout: int = 60
    retry_count: int = 1


@dataclass
class PipelineConfig:
    """Configuration for the entire pipeline."""
    stages: List[StageConfig] = field(default_factory=list)
    middleware: Dict[str, List[str]] = field(default_factory=dict)  # after_stage: [scripts]
    max_iterations: int = 1
    convergence_threshold: float = 0.1
    verbose: bool = False
    dry_run: bool = False
    samples_dir: str = "samples"


class CognitivePipeline:
    """
    Orchestrates cognitive analysis pipeline execution.

    Supports loop-based reanalysis, middleware insertion, and
    dynamic parameter adjustment based on intermediate results.
    """

    # Default stage definitions
    DEFAULT_STAGES = [
        StageConfig(
            name="world_model_analysis",
            script="scripts/world_model_analysis.py",
            args=["--json"]
        ),
        StageConfig(
            name="question_connection",
            script="scripts/question_connection.py",
            args=["--explore"]
        ),
        StageConfig(
            name="knowledge_analysis",
            script="scripts/knowledge_analysis.py",
            args=[]
        ),
        StageConfig(
            name="knowledge_bridge",
            script="scripts/knowledge_bridge.py",
            args=["--pretty"]
        ),
        StageConfig(
            name="llm_generate_response",
            script="scripts/llm_generate_response.py",
            args=["--mode", "prompt"]
        )
    ]

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        if not self.config.stages:
            self.config.stages = self.DEFAULT_STAGES.copy()

        self.chain: Optional[ThoughtChain] = None
        self._callbacks: Dict[str, List[Callable]] = {
            "pre_stage": [],
            "post_stage": [],
            "pre_iteration": [],
            "post_iteration": []
        }

    def create_chain(
        self,
        query: str = "",
        depth: int = 3,
        exploration_mode: str = "hybrid",
        **kwargs
    ) -> ThoughtChain:
        """Create a new thought chain for pipeline execution."""
        self.chain = ThoughtChain.create(
            query=query,
            depth=depth,
            exploration_mode=exploration_mode,
            max_iterations=self.config.max_iterations,
            convergence_threshold=self.config.convergence_threshold,
            **kwargs
        )
        return self.chain

    def load_chain(self, data: Dict[str, Any]) -> ThoughtChain:
        """Load existing chain for continuation."""
        self.chain = ThoughtChain.from_dict(data)
        return self.chain

    def add_middleware(self, after_stage: str, script_path: str) -> None:
        """Add middleware to run after a specific stage."""
        if after_stage not in self.config.middleware:
            self.config.middleware[after_stage] = []
        self.config.middleware[after_stage].append(script_path)

    def set_stages(self, stage_names: List[str]) -> None:
        """Enable only specific stages."""
        for stage in self.config.stages:
            stage.enabled = stage.name in stage_names

    def update_stage_args(self, stage_name: str, args: List[str]) -> None:
        """Update arguments for a specific stage."""
        for stage in self.config.stages:
            if stage.name == stage_name:
                stage.args = args
                break

    def register_callback(self, event: str, callback: Callable) -> None:
        """Register callback for pipeline events."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def _run_script(
        self,
        script: str,
        args: List[str],
        input_data: Optional[str] = None,
        timeout: int = 60
    ) -> Tuple[bool, str, str]:
        """
        Run a script with input data piped to stdin.

        Returns:
            Tuple of (success, stdout, stderr)
        """
        cmd = [sys.executable, script] + args

        if self.config.verbose:
            print(f"  Running: {' '.join(cmd)}", file=sys.stderr)

        try:
            result = subprocess.run(
                cmd,
                input=input_data,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )

            return (
                result.returncode == 0,
                result.stdout,
                result.stderr
            )
        except subprocess.TimeoutExpired:
            return False, "", f"Script timed out after {timeout}s"
        except Exception as e:
            return False, "", str(e)

    def _run_stage(
        self,
        stage: StageConfig,
        input_data: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """Run a single pipeline stage."""
        # Build arguments based on chain parameters
        args = list(stage.args)

        # Add query if available
        if self.chain and self.chain.context.query:
            if stage.name == "question_connection":
                args.extend(["--query", self.chain.context.query])

        # Add parameter overrides
        if self.chain:
            params = self.chain.parameters

            if stage.name == "question_connection":
                args.extend(["--top_k", str(params.top_k)])
                if params.expand_terms:
                    args.append("--expand")

            elif stage.name == "knowledge_bridge":
                args.extend(["--min-gap-distance", str(params.min_gap_distance)])
                args.extend(["--weak-link-threshold", str(params.weak_link_threshold)])
                args.extend(["--max-bridges", str(params.max_bridges)])

            elif stage.name == "llm_generate_response":
                args.extend(["--template", params.template])
                args.extend(["--max-tokens", str(params.max_tokens)])

        # Add samples directory for world_model_analysis
        if stage.name == "world_model_analysis":
            args.extend(["--samples", self.config.samples_dir])

        if self.config.dry_run:
            print(f"  [DRY RUN] Would run: {stage.script} {' '.join(args)}", file=sys.stderr)
            return True, {}

        success, stdout, stderr = self._run_script(
            stage.script,
            args,
            input_data,
            stage.timeout
        )

        if stderr and self.config.verbose:
            print(f"  stderr: {stderr}", file=sys.stderr)

        if not success:
            return False, {"error": stderr or "Unknown error"}

        try:
            result = json.loads(stdout) if stdout.strip() else {}
            return True, result
        except json.JSONDecodeError as e:
            return False, {"error": f"Invalid JSON output: {e}", "raw": stdout[:500]}

    def _run_middleware(
        self,
        after_stage: str,
        current_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run middleware scripts after a stage."""
        middleware_scripts = self.config.middleware.get(after_stage, [])

        for script in middleware_scripts:
            if self.config.verbose:
                print(f"  Running middleware: {script}", file=sys.stderr)

            if self.config.dry_run:
                print(f"  [DRY RUN] Would run middleware: {script}", file=sys.stderr)
                continue

            success, stdout, stderr = self._run_script(
                script,
                [],
                json.dumps(current_data),
                timeout=30
            )

            if success and stdout.strip():
                try:
                    current_data = json.loads(stdout)
                except json.JSONDecodeError:
                    pass  # Keep original data if middleware output is invalid

        return current_data

    def _fire_callbacks(self, event: str, **kwargs) -> None:
        """Fire registered callbacks for an event."""
        for callback in self._callbacks.get(event, []):
            try:
                callback(**kwargs)
            except Exception as e:
                if self.config.verbose:
                    print(f"  Callback error: {e}", file=sys.stderr)

    def run(
        self,
        initial_data: Optional[Dict[str, Any]] = None
    ) -> ThoughtChain:
        """
        Run the complete pipeline with optional iteration.

        Args:
            initial_data: Optional initial data to feed into pipeline

        Returns:
            Completed ThoughtChain with all results
        """
        if not self.chain:
            self.chain = self.create_chain()

        # Store initial data if provided
        if initial_data:
            self.chain.add_result("input", initial_data)

        iteration = 0
        while True:
            if self.config.verbose:
                print(f"\n=== Iteration {iteration} ===", file=sys.stderr)

            self._fire_callbacks("pre_iteration", iteration=iteration, chain=self.chain)

            # Run through enabled stages
            current_data = initial_data or {}

            for stage in self.config.stages:
                if not stage.enabled:
                    continue

                if self.config.verbose:
                    print(f"\nStage: {stage.name}", file=sys.stderr)

                self._fire_callbacks(
                    "pre_stage",
                    stage=stage.name,
                    iteration=iteration,
                    data=current_data
                )

                # Use previous stage output or chain data
                input_json = json.dumps(current_data) if current_data else ""

                success, result = self._run_stage(stage, input_json)

                if not success:
                    if self.config.verbose:
                        print(f"  Stage failed: {result.get('error', 'Unknown')}", file=sys.stderr)
                    self.chain.add_insight(
                        f"Stage {stage.name} failed: {result.get('error', 'Unknown')}",
                        stage=stage.name,
                        confidence=0.0
                    )
                    continue

                # Store result
                self.chain.add_result(stage.name, result)
                current_data = result

                self._fire_callbacks(
                    "post_stage",
                    stage=stage.name,
                    iteration=iteration,
                    result=result
                )

                # Run middleware
                current_data = self._run_middleware(stage.name, current_data)

            self._fire_callbacks("post_iteration", iteration=iteration, chain=self.chain)

            # Check if we should continue iterating
            iteration += 1
            self.chain.next_iteration()

            if not self.chain.should_continue():
                if self.config.verbose:
                    print(f"\nConverged after {iteration} iterations", file=sys.stderr)
                break

            if iteration >= self.config.max_iterations:
                if self.config.verbose:
                    print(f"\nMax iterations ({self.config.max_iterations}) reached", file=sys.stderr)
                break

            # For next iteration, use accumulated chain data
            initial_data = self.chain.to_dict()

        return self.chain

    def run_interactive(self) -> None:
        """Run pipeline in interactive mode."""
        print("Cognitive Pipeline Interactive Mode")
        print("=" * 40)
        print("Commands:")
        print("  query <text>     - Set exploration query")
        print("  run              - Run pipeline")
        print("  run <stage>      - Run single stage")
        print("  iterate <n>      - Run n iterations")
        print("  params           - Show current parameters")
        print("  set <key> <val>  - Set parameter")
        print("  stages           - List stages")
        print("  enable <stage>   - Enable stage")
        print("  disable <stage>  - Disable stage")
        print("  results          - Show results summary")
        print("  insights         - Show insights")
        print("  export <file>    - Export chain to file")
        print("  quit             - Exit")
        print()

        self.chain = self.create_chain()

        while True:
            try:
                line = input("pipeline> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting...")
                break

            if not line:
                continue

            parts = line.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd == "quit" or cmd == "exit":
                break

            elif cmd == "query":
                self.chain.context.query = arg
                print(f"Query set to: {arg}")

            elif cmd == "run":
                if arg:
                    # Run single stage
                    for stage in self.config.stages:
                        if stage.name == arg:
                            self.config.verbose = True
                            input_data = json.dumps(self.chain.to_dict())
                            success, result = self._run_stage(stage, input_data)
                            if success:
                                self.chain.add_result(stage.name, result)
                                print(json.dumps(result, indent=2)[:1000])
                            else:
                                print(f"Error: {result}")
                            break
                    else:
                        print(f"Unknown stage: {arg}")
                else:
                    # Run full pipeline
                    self.config.verbose = True
                    self.run()
                    print("\nPipeline complete. Use 'results' to see summary.")

            elif cmd == "iterate":
                try:
                    n = int(arg) if arg else 3
                    self.config.max_iterations = n
                    self.run()
                except ValueError:
                    print("Invalid number of iterations")

            elif cmd == "params":
                print(json.dumps(self.chain.parameters.to_dict(), indent=2))

            elif cmd == "set":
                key_val = arg.split(maxsplit=1)
                if len(key_val) == 2:
                    key, val = key_val
                    try:
                        # Try to parse as number
                        if "." in val:
                            val = float(val)
                        else:
                            try:
                                val = int(val)
                            except ValueError:
                                pass  # Keep as string

                        if hasattr(self.chain.parameters, key):
                            setattr(self.chain.parameters, key, val)
                            print(f"Set {key} = {val}")
                        elif hasattr(self.chain.context, key):
                            setattr(self.chain.context, key, val)
                            print(f"Set {key} = {val}")
                        else:
                            print(f"Unknown parameter: {key}")
                    except Exception as e:
                        print(f"Error: {e}")
                else:
                    print("Usage: set <key> <value>")

            elif cmd == "stages":
                for stage in self.config.stages:
                    status = "[enabled]" if stage.enabled else "[disabled]"
                    print(f"  {stage.name} {status}")

            elif cmd == "enable":
                for stage in self.config.stages:
                    if stage.name == arg:
                        stage.enabled = True
                        print(f"Enabled {arg}")
                        break
                else:
                    print(f"Unknown stage: {arg}")

            elif cmd == "disable":
                for stage in self.config.stages:
                    if stage.name == arg:
                        stage.enabled = False
                        print(f"Disabled {arg}")
                        break
                else:
                    print(f"Unknown stage: {arg}")

            elif cmd == "results":
                summary = self.chain.get_iteration_summary()
                print(json.dumps(summary, indent=2))

            elif cmd == "insights":
                for insight in self.chain.insights:
                    print(f"[iter {insight.iteration}] {insight.stage}: {insight.insight}")

            elif cmd == "export":
                if arg:
                    with open(arg, "w") as f:
                        f.write(self.chain.to_json())
                    print(f"Exported to {arg}")
                else:
                    print("Usage: export <filename>")

            else:
                print(f"Unknown command: {cmd}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Cognitive Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline with query
  python scripts/cognitive_pipeline.py --query "world models and prediction"

  # Run with multiple iterations
  python scripts/cognitive_pipeline.py --query "learning" --max-iterations 3

  # Use specific stages only
  python scripts/cognitive_pipeline.py --stages world_model_analysis,knowledge_bridge

  # Insert middleware after a stage
  python scripts/cognitive_pipeline.py --query "cognition" \\
      --after question_connection scripts/my_filter.py

  # Interactive exploration mode
  python scripts/cognitive_pipeline.py --interactive

  # Dry run to see what would execute
  python scripts/cognitive_pipeline.py --query "test" --dry-run --verbose
        """
    )

    parser.add_argument(
        "--query", "-q",
        help="Exploration query"
    )
    parser.add_argument(
        "--depth", "-d",
        type=int,
        default=3,
        help="Exploration depth (default: 3)"
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["breadth", "depth", "hybrid", "focused"],
        default="hybrid",
        help="Exploration mode (default: hybrid)"
    )
    parser.add_argument(
        "--max-iterations", "-i",
        type=int,
        default=1,
        help="Maximum iterations for loop-based reanalysis (default: 1)"
    )
    parser.add_argument(
        "--convergence",
        type=float,
        default=0.1,
        help="Convergence threshold for early stopping (default: 0.1)"
    )
    parser.add_argument(
        "--stages", "-s",
        help="Comma-separated list of stages to run"
    )
    parser.add_argument(
        "--after",
        nargs=2,
        action="append",
        metavar=("STAGE", "SCRIPT"),
        help="Add middleware script after stage"
    )
    parser.add_argument(
        "--samples",
        default="samples",
        help="Samples directory for corpus (default: samples)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would run without executing"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file for results (default: stdout)"
    )

    # Parameter overrides
    parser.add_argument("--top-k", type=int, help="Top K results")
    parser.add_argument("--min-gap-distance", type=int, help="Min gap distance for bridges")
    parser.add_argument("--template", help="LLM template (synthesis/explanation/gaps)")

    args = parser.parse_args()

    # Build configuration
    config = PipelineConfig(
        max_iterations=args.max_iterations,
        convergence_threshold=args.convergence,
        verbose=args.verbose,
        dry_run=args.dry_run,
        samples_dir=args.samples
    )

    # Create pipeline
    pipeline = CognitivePipeline(config)

    # Set stages if specified
    if args.stages:
        pipeline.set_stages(args.stages.split(","))

    # Add middleware
    if args.after:
        for stage, script in args.after:
            pipeline.add_middleware(stage, script)

    # Interactive mode
    if args.interactive:
        pipeline.run_interactive()
        return

    # Create chain with query
    chain = pipeline.create_chain(
        query=args.query or "",
        depth=args.depth,
        exploration_mode=args.mode
    )

    # Apply parameter overrides
    if args.top_k:
        chain.parameters.top_k = args.top_k
    if args.min_gap_distance:
        chain.parameters.min_gap_distance = args.min_gap_distance
    if args.template:
        chain.parameters.template = args.template

    # Run pipeline
    result_chain = pipeline.run()

    # Output results
    output = result_chain.to_json()

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        if args.verbose:
            print(f"Results written to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
