"""
Audit Tool Command Registry

Commands are auto-discovered from this package. Each command module must define:
- NAME: str - Command name (e.g., 'generate')
- HELP: str - Short help text
- setup_args(parser) - Function to add arguments to subparser
- run(args) - Function to execute the command

Example command module:
    # scan.py
    NAME = 'scan'
    HELP = 'Scan for suspicious comments'

    def setup_args(parser):
        parser.add_argument('directory', help='Directory to scan')

    def run(args):
        print(f"Scanning {args.directory}...")
"""

import importlib
import pkgutil
from pathlib import Path
from typing import Dict, Any, Callable

# Registry of all commands
_COMMANDS: Dict[str, Dict[str, Any]] = {}


def register(name: str, help_text: str, setup_args: Callable, run: Callable):
    """Register a command."""
    _COMMANDS[name] = {
        'help': help_text,
        'setup_args': setup_args,
        'run': run,
    }


def get_commands() -> Dict[str, Dict[str, Any]]:
    """Get all registered commands."""
    return _COMMANDS.copy()


def setup_all_parsers(subparsers):
    """Set up argument parsers for all commands."""
    for name, cmd in _COMMANDS.items():
        parser = subparsers.add_parser(name, help=cmd['help'])
        cmd['setup_args'](parser)


def run_command(name: str, args) -> int:
    """Run a command by name."""
    if name not in _COMMANDS:
        return 1
    _COMMANDS[name]['run'](args)
    return 0


def _discover_commands():
    """Auto-discover and register commands from this package."""
    package_dir = Path(__file__).parent

    for module_info in pkgutil.iter_modules([str(package_dir)]):
        if module_info.name.startswith('_'):
            continue  # Skip private modules

        module = importlib.import_module(f'.{module_info.name}', __package__)

        # Check for required attributes
        if all(hasattr(module, attr) for attr in ['NAME', 'HELP', 'setup_args', 'run']):
            register(
                name=module.NAME,
                help_text=module.HELP,
                setup_args=module.setup_args,
                run=module.run,
            )


# Auto-discover on import
_discover_commands()
