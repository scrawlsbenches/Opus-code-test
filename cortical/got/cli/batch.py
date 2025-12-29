"""
Batch CLI commands for GoT system.

Provides batch operations using a heredoc-friendly DSL:

    python scripts/got_utils.py batch <<'EOF'
    sprint create "Sprint 28" --number 28 as sprint1
    task create "Feature X" --sprint $sprint1 --priority high as t1
    task create "Tests" --sprint $sprint1 as t2
    edge add $t2 $t1 DEPENDS_ON
    EOF

Features:
- Alias assignment with `as NAME`
- Variable resolution with `$NAME`
- Atomic execution with transaction support
- Dry-run mode for previewing changes
- JSON output for scripting
"""

import json
import re
import shlex
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.got_utils import TransactionalGoTAdapter


class BatchError(Exception):
    """Error during batch parsing or execution."""
    pass


@dataclass
class BatchOperation:
    """A single operation parsed from batch DSL."""
    command: str           # e.g., "task", "sprint", "edge"
    action: str            # e.g., "create", "add"
    args: Dict[str, Any]   # Parsed arguments
    alias: Optional[str] = None  # Optional alias for referencing
    line_number: int = 0   # Original line number for error messages
    raw_line: str = ""     # Original line text


@dataclass
class BatchResult:
    """Result of batch execution."""
    success: bool
    created: List[str] = field(default_factory=list)  # Created entity IDs
    aliases: Dict[str, str] = field(default_factory=dict)  # alias -> ID mapping
    planned: List[BatchOperation] = field(default_factory=list)  # For dry-run
    error: Optional[str] = None
    dry_run: bool = False

    def to_json(self) -> Dict[str, Any]:
        """Convert result to JSON-serializable dict."""
        return {
            "success": self.success,
            "created": self.created,
            "aliases": self.aliases,
            "error": self.error,
            "dry_run": self.dry_run,
        }


def parse_batch_line(line: str, line_number: int = 0) -> Optional[BatchOperation]:
    """
    Parse a single batch DSL line.

    Syntax examples:
        task create "Title" --priority high as t1
        sprint create "Name" --number 28 as s1
        edge add $source $target DEPENDS_ON

    Args:
        line: Raw line text
        line_number: Line number for error messages

    Returns:
        BatchOperation or None for empty/comment lines
    """
    line = line.strip()

    # Skip empty lines and comments
    if not line or line.startswith("#"):
        return None

    # Extract alias if present (must be at end: "... as NAME")
    alias = None
    alias_match = re.search(r'\s+as\s+(\w+)\s*$', line)
    if alias_match:
        alias = alias_match.group(1)
        line = line[:alias_match.start()]

    # Parse using shlex to handle quoted strings
    try:
        tokens = shlex.split(line)
    except ValueError as e:
        raise BatchError(f"Line {line_number}: Parse error: {e}")

    if len(tokens) < 2:
        raise BatchError(f"Line {line_number}: Expected 'command action [args]'")

    command = tokens[0].lower()
    action = tokens[1].lower()
    remaining = tokens[2:]

    # Parse command-specific arguments
    args = _parse_command_args(command, action, remaining, line_number)

    return BatchOperation(
        command=command,
        action=action,
        args=args,
        alias=alias,
        line_number=line_number,
        raw_line=line,
    )


def _parse_command_args(
    command: str,
    action: str,
    tokens: List[str],
    line_number: int
) -> Dict[str, Any]:
    """
    Parse command-specific arguments from tokens.

    Args:
        command: Command name (task, sprint, edge, etc.)
        action: Action name (create, add, etc.)
        tokens: Remaining tokens after command/action
        line_number: For error messages

    Returns:
        Dict of parsed arguments
    """
    args: Dict[str, Any] = {}

    if command == "task" and action == "create":
        args = _parse_task_create_args(tokens, line_number)
    elif command == "sprint" and action == "create":
        args = _parse_sprint_create_args(tokens, line_number)
    elif command == "epic" and action == "create":
        args = _parse_epic_create_args(tokens, line_number)
    elif command == "edge" and action == "add":
        args = _parse_edge_add_args(tokens, line_number)
    elif command == "decision" and action == "log":
        args = _parse_decision_log_args(tokens, line_number)
    else:
        raise BatchError(
            f"Line {line_number}: Unknown command '{command} {action}'"
        )

    return args


def _parse_task_create_args(tokens: List[str], line_number: int) -> Dict[str, Any]:
    """Parse 'task create' arguments."""
    args: Dict[str, Any] = {}

    # First non-flag token is the title
    positional_idx = 0
    i = 0
    while i < len(tokens):
        token = tokens[i]

        if token.startswith("--"):
            flag = token[2:].replace("-", "_")
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                value = tokens[i + 1]
                # Type conversion for known flags
                if flag == "priority":
                    args["priority"] = value
                elif flag == "category":
                    args["category"] = value
                elif flag == "sprint":
                    args["sprint"] = value
                elif flag == "depends_on" or flag == "depends":
                    args["depends_on"] = value
                elif flag == "blocks":
                    args["blocks"] = value
                elif flag == "description":
                    args["description"] = value
                else:
                    args[flag] = value
                i += 2
            else:
                # Boolean flag
                args[flag] = True
                i += 1
        else:
            # Positional argument - title
            if positional_idx == 0:
                args["title"] = token
                positional_idx += 1
            i += 1

    if "title" not in args:
        raise BatchError(f"Line {line_number}: task create requires a title")

    return args


def _parse_sprint_create_args(tokens: List[str], line_number: int) -> Dict[str, Any]:
    """Parse 'sprint create' arguments."""
    args: Dict[str, Any] = {}

    positional_idx = 0
    i = 0
    while i < len(tokens):
        token = tokens[i]

        if token.startswith("--"):
            flag = token[2:].replace("-", "_")
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                value = tokens[i + 1]
                if flag == "number":
                    args["number"] = int(value)
                elif flag == "epic":
                    args["epic"] = value
                else:
                    args[flag] = value
                i += 2
            else:
                args[flag] = True
                i += 1
        else:
            if positional_idx == 0:
                args["name"] = token
                positional_idx += 1
            i += 1

    if "name" not in args:
        raise BatchError(f"Line {line_number}: sprint create requires a name")

    return args


def _parse_epic_create_args(tokens: List[str], line_number: int) -> Dict[str, Any]:
    """Parse 'epic create' arguments."""
    args: Dict[str, Any] = {}

    positional_idx = 0
    i = 0
    while i < len(tokens):
        token = tokens[i]

        if token.startswith("--"):
            flag = token[2:].replace("-", "_")
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                args[flag] = tokens[i + 1]
                i += 2
            else:
                args[flag] = True
                i += 1
        else:
            if positional_idx == 0:
                args["name"] = token
                positional_idx += 1
            i += 1

    if "name" not in args:
        raise BatchError(f"Line {line_number}: epic create requires a name")

    return args


def _parse_edge_add_args(tokens: List[str], line_number: int) -> Dict[str, Any]:
    """Parse 'edge add' arguments."""
    if len(tokens) < 3:
        raise BatchError(
            f"Line {line_number}: edge add requires SOURCE TARGET EDGE_TYPE"
        )

    args = {
        "source_id": tokens[0],
        "target_id": tokens[1],
        "edge_type": tokens[2].upper(),
    }

    # Parse optional --weight
    i = 3
    while i < len(tokens):
        if tokens[i] == "--weight" and i + 1 < len(tokens):
            args["weight"] = float(tokens[i + 1])
            i += 2
        else:
            i += 1

    return args


def _parse_decision_log_args(tokens: List[str], line_number: int) -> Dict[str, Any]:
    """Parse 'decision log' arguments."""
    args: Dict[str, Any] = {}

    positional_idx = 0
    i = 0
    while i < len(tokens):
        token = tokens[i]

        if token.startswith("--"):
            flag = token[2:].replace("-", "_")
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                args[flag] = tokens[i + 1]
                i += 2
            else:
                args[flag] = True
                i += 1
        else:
            if positional_idx == 0:
                args["decision"] = token
                positional_idx += 1
            i += 1

    if "decision" not in args:
        raise BatchError(f"Line {line_number}: decision log requires a decision")

    return args


def resolve_variables(
    value: Any,
    aliases: Dict[str, str]
) -> Any:
    """
    Resolve $NAME variables in a value.

    Args:
        value: String, dict, or list to resolve
        aliases: Mapping of alias names to actual IDs

    Returns:
        Resolved value

    Raises:
        BatchError: If variable is not found in aliases
    """
    if isinstance(value, str):
        if value.startswith("$"):
            var_name = value[1:]
            if var_name not in aliases:
                raise BatchError(f"Unknown variable '${var_name}'")
            return aliases[var_name]
        return value

    elif isinstance(value, dict):
        return {k: resolve_variables(v, aliases) for k, v in value.items()}

    elif isinstance(value, list):
        return [resolve_variables(v, aliases) for v in value]

    return value


class BatchParser:
    """Parser for batch DSL scripts."""

    def parse(self, script: str) -> List[BatchOperation]:
        """
        Parse a multi-line batch script.

        Args:
            script: Full batch script text

        Returns:
            List of BatchOperation objects
        """
        operations = []
        lines = script.strip().split("\n")

        for i, line in enumerate(lines, start=1):
            op = parse_batch_line(line, line_number=i)
            if op is not None:
                operations.append(op)

        return operations


class BatchExecutor:
    """Executor for batch operations."""

    def __init__(self, manager: "TransactionalGoTAdapter"):
        """
        Initialize executor with a GoT manager.

        Args:
            manager: TransactionalGoTAdapter for executing operations
        """
        self.manager = manager

    def execute(
        self,
        script: str,
        dry_run: bool = False,
        atomic: bool = True,
    ) -> BatchResult:
        """
        Execute a batch script.

        Args:
            script: Batch DSL script
            dry_run: If True, only parse and validate without executing
            atomic: If True, rollback all on any failure

        Returns:
            BatchResult with execution details
        """
        parser = BatchParser()

        try:
            operations = parser.parse(script)
        except BatchError as e:
            return BatchResult(success=False, error=str(e))

        if dry_run:
            return BatchResult(
                success=True,
                planned=operations,
                dry_run=True,
            )

        # Execute operations
        aliases: Dict[str, str] = {}
        created: List[str] = []

        try:
            for op in operations:
                # Resolve variables in args
                resolved_args = resolve_variables(op.args, aliases)

                # Execute the operation
                result_id = self._execute_operation(op.command, op.action, resolved_args)

                if result_id:
                    created.append(result_id)
                    if op.alias:
                        aliases[op.alias] = result_id

            # Save all changes
            self.manager.save()

            return BatchResult(
                success=True,
                created=created,
                aliases=aliases,
            )

        except Exception as e:
            # On failure, don't save (changes are not persisted)
            return BatchResult(
                success=False,
                created=created,
                aliases=aliases,
                error=str(e),
            )

    def _execute_operation(
        self,
        command: str,
        action: str,
        args: Dict[str, Any]
    ) -> Optional[str]:
        """
        Execute a single operation.

        Args:
            command: Command name
            action: Action name
            args: Resolved arguments

        Returns:
            Created entity ID, or None
        """
        if command == "task" and action == "create":
            return self.manager.create_task(
                title=args["title"],
                priority=args.get("priority", "medium"),
                category=args.get("category", "feature"),
                description=args.get("description", ""),
                sprint_id=args.get("sprint"),
                depends_on=args.get("depends_on"),
                blocks=args.get("blocks"),
            )

        elif command == "sprint" and action == "create":
            return self.manager.create_sprint(
                name=args["name"],
                number=args.get("number"),
                epic_id=args.get("epic"),
            )

        elif command == "epic" and action == "create":
            return self.manager.create_epic(
                name=args["name"],
                description=args.get("description", ""),
            )

        elif command == "edge" and action == "add":
            edge = self.manager.add_edge(
                source_id=args["source_id"],
                target_id=args["target_id"],
                edge_type=args["edge_type"],
                weight=args.get("weight", 1.0),
            )
            # Edges don't have IDs in our system, return None
            return None

        elif command == "decision" and action == "log":
            return self.manager.log_decision(
                decision=args["decision"],
                rationale=args.get("rationale", ""),
                context=args.get("context"),
            )

        else:
            raise BatchError(f"Unknown operation: {command} {action}")


# =============================================================================
# CLI INTEGRATION
# =============================================================================

def setup_batch_parser(subparsers) -> None:
    """Set up the batch subcommand parser."""
    batch_parser = subparsers.add_parser(
        "batch",
        help="Execute batch operations from stdin or file",
        description="""
Execute multiple GoT operations in a single transaction.

Reads from stdin (heredoc) or a file and executes operations atomically.

DSL Syntax:
  command action "arg" --flag value as alias

Examples:
  sprint create "Sprint 28" --number 28 as s1
  task create "Feature X" --sprint $s1 --priority high as t1
  task create "Tests" --sprint $s1 as t2
  edge add $t2 $t1 DEPENDS_ON

Usage:
  python scripts/got_utils.py batch <<'EOF'
  sprint create "Sprint 1" as s1
  task create "Task 1" --sprint $s1 as t1
  EOF

  python scripts/got_utils.py batch --file setup.got
""",
    )

    batch_parser.add_argument(
        "--file", "-f",
        help="Read batch script from file instead of stdin"
    )
    batch_parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Parse and validate without executing"
    )
    batch_parser.add_argument(
        "--output-json", "-j",
        action="store_true",
        help="Output result as JSON"
    )
    batch_parser.add_argument(
        "--no-atomic",
        action="store_true",
        help="Don't use atomic transaction (allow partial success)"
    )


def handle_batch_command(args, manager: "TransactionalGoTAdapter") -> int:
    """Handle the batch command."""
    import sys

    # Read script from file or stdin
    if args.file:
        try:
            with open(args.file, "r") as f:
                script = f.read()
        except FileNotFoundError:
            print(f"Error: File not found: {args.file}")
            return 1
        except IOError as e:
            print(f"Error reading file: {e}")
            return 1
    else:
        # Read from stdin
        if sys.stdin.isatty():
            print("Reading batch script from stdin (Ctrl+D to end):")
        script = sys.stdin.read()

    if not script.strip():
        print("Error: Empty batch script")
        return 1

    # Execute
    executor = BatchExecutor(manager)
    result = executor.execute(
        script,
        dry_run=getattr(args, 'dry_run', False),
        atomic=not getattr(args, 'no_atomic', False),
    )

    # Output
    if getattr(args, 'output_json', False):
        print(json.dumps(result.to_json(), indent=2))
    else:
        if result.dry_run:
            print("=== DRY RUN (no changes made) ===")
            print(f"Operations to execute: {len(result.planned)}")
            for op in result.planned:
                alias_str = f" as {op.alias}" if op.alias else ""
                print(f"  {op.command} {op.action}{alias_str}")
        elif result.success:
            print(f"✓ Batch completed: {len(result.created)} entities created")
            if result.aliases:
                print("\nAliases:")
                for alias, entity_id in result.aliases.items():
                    print(f"  {alias} = {entity_id}")
        else:
            print(f"✗ Batch failed: {result.error}")
            if result.created:
                print(f"\nPartially created ({len(result.created)}):")
                for entity_id in result.created:
                    print(f"  {entity_id}")

    return 0 if result.success else 1
