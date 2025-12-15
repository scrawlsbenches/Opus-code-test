#!/usr/bin/env python3
"""
🔥 ANIMATED ASCII CODEBASE VISUALIZER 🔥

A stunning terminal experience featuring:
- Matrix-style falling commit messages
- Animated fire effect based on code activity
- Starfield hyperspace with flying commits
- Pulsing neural network visualization
- Live updating dashboard with animated bars

Usage:
    python scripts/ascii_visualizer_animated.py              # Interactive menu
    python scripts/ascii_visualizer_animated.py --matrix     # Matrix rain
    python scripts/ascii_visualizer_animated.py --fire       # Fire effect
    python scripts/ascii_visualizer_animated.py --stars      # Starfield
    python scripts/ascii_visualizer_animated.py --neural     # Neural pulse
    python scripts/ascii_visualizer_animated.py --dashboard  # Live dashboard
    python scripts/ascii_visualizer_animated.py --demo       # Demo all effects

Controls:
    q, ESC, Ctrl+C - Quit
    SPACE - Pause/Resume
    1-5 - Switch visualization
"""

import argparse
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ============================================================================
# TERMINAL SETUP
# ============================================================================

class Terminal:
    """Terminal control and state management."""

    # ANSI escape codes
    CLEAR = '\033[2J'
    HOME = '\033[H'
    HIDE_CURSOR = '\033[?25l'
    SHOW_CURSOR = '\033[?25h'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

    # Colors (256-color mode)
    @staticmethod
    def fg(n: int) -> str:
        return f'\033[38;5;{n}m'

    @staticmethod
    def bg(n: int) -> str:
        return f'\033[48;5;{n}m'

    @staticmethod
    def rgb(r: int, g: int, b: int) -> str:
        return f'\033[38;2;{r};{g};{b}m'

    @staticmethod
    def bg_rgb(r: int, g: int, b: int) -> str:
        return f'\033[48;2;{r};{g};{b}m'

    @staticmethod
    def move(row: int, col: int) -> str:
        return f'\033[{row};{col}H'

    @staticmethod
    def get_size() -> Tuple[int, int]:
        size = shutil.get_terminal_size((80, 24))
        return size.lines, size.columns

    @staticmethod
    def setup():
        """Setup terminal for animation."""
        print(Terminal.HIDE_CURSOR, end='', flush=True)
        print(Terminal.CLEAR, end='', flush=True)

    @staticmethod
    def cleanup():
        """Restore terminal state."""
        print(Terminal.SHOW_CURSOR, end='', flush=True)
        print(Terminal.RESET, end='', flush=True)
        print(Terminal.CLEAR, end='', flush=True)


# ============================================================================
# DATA SOURCES
# ============================================================================

def get_project_root() -> Path:
    return Path(__file__).parent.parent


def run_git(args: List[str]) -> str:
    try:
        result = subprocess.run(
            ['git'] + args,
            capture_output=True,
            text=True,
            cwd=get_project_root(),
            timeout=5
        )
        return result.stdout.strip()
    except Exception:
        return ""


def get_commits(limit: int = 100) -> List[Tuple[str, str, str]]:
    """Get recent commits as (hash, date, message)."""
    log = run_git(['log', '--format=%h|%ad|%s', '--date=short', '-n', str(limit)])
    commits = []
    for line in log.splitlines():
        parts = line.split('|', 2)
        if len(parts) == 3:
            commits.append(tuple(parts))
    return commits


def get_file_stats() -> Dict[str, int]:
    """Get line counts for Python files."""
    stats = {}
    cortical_dir = get_project_root() / 'cortical'
    for py_file in cortical_dir.rglob('*.py'):
        if '__pycache__' not in str(py_file):
            try:
                lines = len(py_file.read_text().splitlines())
                stats[py_file.name] = lines
            except Exception:
                pass
    return stats


def get_hot_files() -> Dict[str, int]:
    """Get modification counts per file."""
    log = run_git(['log', '--format=', '--name-only', '-n', '200'])
    counts = defaultdict(int)
    for line in log.splitlines():
        if line.endswith('.py'):
            counts[Path(line).name] += 1
    return dict(counts)


# ============================================================================
# MATRIX RAIN EFFECT
# ============================================================================

@dataclass
class MatrixDrop:
    col: int
    row: float
    speed: float
    text: str
    char_idx: int
    brightness: float


class MatrixRain:
    """Matrix-style falling code with commit messages."""

    CHARS = "ｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜﾝ0123456789"

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.drops: List[MatrixDrop] = []
        self.commits = get_commits(50)
        self.commit_idx = 0

        # Initialize drops
        for _ in range(width // 3):
            self._spawn_drop()

    def _spawn_drop(self):
        col = random.randint(0, self.width - 1)
        # Use commit message or random chars
        if self.commits and random.random() < 0.3:
            text = self.commits[self.commit_idx % len(self.commits)][2][:30]
            self.commit_idx += 1
        else:
            text = ''.join(random.choice(self.CHARS) for _ in range(random.randint(5, 20)))

        self.drops.append(MatrixDrop(
            col=col,
            row=random.uniform(-20, 0),
            speed=random.uniform(0.3, 1.2),
            text=text,
            char_idx=0,
            brightness=random.uniform(0.5, 1.0)
        ))

    def update(self):
        new_drops = []
        for drop in self.drops:
            drop.row += drop.speed
            drop.char_idx = int(drop.row) % len(drop.text)

            # Keep if still visible
            if drop.row < self.height + len(drop.text):
                new_drops.append(drop)
            else:
                # Respawn
                self._spawn_drop()

        self.drops = new_drops

        # Maintain drop count
        while len(self.drops) < self.width // 3:
            self._spawn_drop()

    def render(self) -> str:
        # Create buffer
        buffer = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        colors = [[0 for _ in range(self.width)] for _ in range(self.height)]

        for drop in self.drops:
            for i, char in enumerate(drop.text):
                row = int(drop.row) - i
                if 0 <= row < self.height and 0 <= drop.col < self.width:
                    buffer[row][drop.col] = char
                    # Head is brightest
                    if i == 0:
                        colors[row][drop.col] = 3  # Bright white/green
                    elif i < 3:
                        colors[row][drop.col] = 2  # Bright green
                    else:
                        colors[row][drop.col] = 1  # Dim green

        # Render to string
        output = [Terminal.HOME]
        for row in range(self.height):
            line = ""
            for col in range(self.width):
                char = buffer[row][col]
                color = colors[row][col]
                if color == 3:
                    line += f"{Terminal.rgb(255, 255, 255)}{char}"
                elif color == 2:
                    line += f"{Terminal.rgb(0, 255, 70)}{char}"
                elif color == 1:
                    line += f"{Terminal.rgb(0, 150, 40)}{char}"
                else:
                    line += f"{Terminal.rgb(0, 50, 20)}{char}"
            output.append(line + Terminal.RESET)

        # Add title
        title = "[ MATRIX: COMMIT STREAM ]"
        title_pos = (self.width - len(title)) // 2
        output[1] = output[1][:title_pos] + f"{Terminal.BOLD}{Terminal.rgb(0,255,0)}{title}{Terminal.RESET}" + output[1][title_pos+len(title):]

        return '\n'.join(output)


# ============================================================================
# FIRE EFFECT
# ============================================================================

class FireEffect:
    """Animated fire where height represents code activity."""

    FIRE_CHARS = " .:-=+*#%@"
    FIRE_COLORS = [
        (0, 0, 0),       # Black
        (32, 0, 0),      # Very dark red
        (64, 0, 0),      # Dark red
        (128, 0, 0),     # Red
        (192, 32, 0),    # Orange-red
        (255, 64, 0),    # Orange
        (255, 128, 0),   # Light orange
        (255, 192, 0),   # Yellow-orange
        (255, 255, 0),   # Yellow
        (255, 255, 128), # Light yellow
        (255, 255, 255), # White
    ]

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.buffer = [[0.0 for _ in range(width)] for _ in range(height)]
        self.hot_files = get_hot_files()
        self.file_names = list(self.hot_files.keys())[:min(width // 5, 15)]

        # Calculate base heat per column based on file activity
        self.base_heat = []
        if self.file_names:
            max_count = max(self.hot_files.values()) if self.hot_files else 1
            col_width = width // len(self.file_names)
            for filename in self.file_names:
                heat = self.hot_files.get(filename, 0) / max_count
                for _ in range(col_width):
                    self.base_heat.append(heat)
        # Pad to width
        while len(self.base_heat) < width:
            self.base_heat.append(0.3)

    def update(self):
        # Set bottom row based on file heat + randomness
        for x in range(self.width):
            base = self.base_heat[x] if x < len(self.base_heat) else 0.3
            self.buffer[self.height - 1][x] = base + random.uniform(0, 0.5)
            self.buffer[self.height - 2][x] = base + random.uniform(0, 0.4)

        # Propagate fire upward with cooling
        for y in range(self.height - 3, -1, -1):
            for x in range(self.width):
                # Average of cells below + random decay
                below = self.buffer[y + 1][x]
                left = self.buffer[y + 1][max(0, x - 1)]
                right = self.buffer[y + 1][min(self.width - 1, x + 1)]

                # Weighted average with some randomness
                new_val = (below * 0.5 + left * 0.25 + right * 0.25)
                new_val *= random.uniform(0.85, 0.98)  # Cooling
                new_val += random.uniform(-0.05, 0.05)  # Turbulence

                self.buffer[y][x] = max(0, min(1, new_val))

    def render(self) -> str:
        output = [Terminal.HOME]

        for y in range(self.height - 3):  # Leave room for labels
            line = ""
            for x in range(self.width):
                heat = self.buffer[y][x]
                color_idx = min(len(self.FIRE_COLORS) - 1, int(heat * len(self.FIRE_COLORS)))
                char_idx = min(len(self.FIRE_CHARS) - 1, int(heat * len(self.FIRE_CHARS)))

                r, g, b = self.FIRE_COLORS[color_idx]
                char = self.FIRE_CHARS[char_idx]
                line += f"{Terminal.rgb(r, g, b)}{char}"

            output.append(line + Terminal.RESET)

        # Add file labels at bottom
        if self.file_names:
            col_width = self.width // len(self.file_names)
            label_line = ""
            for filename in self.file_names:
                short = filename[:col_width-1].center(col_width)
                heat = self.hot_files.get(filename, 0)
                if heat > 10:
                    label_line += f"{Terminal.rgb(255, 100, 0)}{short}"
                else:
                    label_line += f"{Terminal.rgb(150, 150, 150)}{short}"
            output.append(label_line + Terminal.RESET)

        # Title
        title = "[ 🔥 CODE FIRE: Hot Files Burn Brighter 🔥 ]"
        output.insert(1, f"{Terminal.BOLD}{Terminal.rgb(255,200,0)}{title.center(self.width)}{Terminal.RESET}")

        return '\n'.join(output)


# ============================================================================
# STARFIELD / HYPERSPACE
# ============================================================================

@dataclass
class Star:
    x: float
    y: float
    z: float
    text: str


class Starfield:
    """Hyperspace starfield with commits as flying stars."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.cx = width // 2
        self.cy = height // 2
        self.stars: List[Star] = []
        self.commits = get_commits(100)
        self.commit_idx = 0

        # Initialize stars
        for _ in range(100):
            self._spawn_star()

    def _spawn_star(self, far: bool = True):
        # Random position
        angle = random.uniform(0, 2 * math.pi)
        distance = random.uniform(0.1, 1.0)
        x = math.cos(angle) * distance * self.width
        y = math.sin(angle) * distance * self.height
        z = random.uniform(0.8, 1.0) if far else random.uniform(0.1, 0.3)

        # Get commit message for some stars
        if self.commits and random.random() < 0.2:
            text = self.commits[self.commit_idx % len(self.commits)][2][:20]
            self.commit_idx += 1
        else:
            text = random.choice(['*', '+', '.', '·', '✦', '★', '✧', '○'])

        self.stars.append(Star(x=x, y=y, z=z, text=text))

    def update(self):
        new_stars = []
        for star in self.stars:
            star.z -= 0.02  # Move toward viewer

            if star.z <= 0.01:
                self._spawn_star(far=True)
            else:
                new_stars.append(star)

        self.stars = new_stars

        # Maintain star count
        while len(self.stars) < 100:
            self._spawn_star(far=True)

    def render(self) -> str:
        buffer = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        depths = [[999.0 for _ in range(self.width)] for _ in range(self.height)]
        star_data = [[None for _ in range(self.width)] for _ in range(self.height)]

        for star in self.stars:
            # Project 3D to 2D
            px = int(self.cx + star.x / star.z)
            py = int(self.cy + star.y / star.z)

            if 0 <= px < self.width and 0 <= py < self.height:
                if star.z < depths[py][px]:
                    depths[py][px] = star.z
                    star_data[py][px] = star

        # Render
        output = [Terminal.HOME]
        for y in range(self.height):
            line = ""
            for x in range(self.width):
                star = star_data[y][x]
                if star:
                    # Brightness based on depth
                    brightness = int(255 * (1 - star.z))
                    brightness = max(50, min(255, brightness))

                    # Color: closer = bluer (doppler shift effect)
                    if star.z < 0.3:
                        r, g, b = brightness, brightness, 255
                    elif star.z < 0.6:
                        r, g, b = brightness, brightness, brightness
                    else:
                        r, g, b = 255, brightness, brightness // 2

                    char = star.text[0] if star.text else '*'
                    line += f"{Terminal.rgb(r, g, b)}{char}"
                else:
                    line += f"{Terminal.rgb(10, 10, 30)} "
            output.append(line + Terminal.RESET)

        # Title
        title = "[ ✨ HYPERSPACE: Commits Flying By ✨ ]"
        title_colored = f"{Terminal.BOLD}{Terminal.rgb(100,150,255)}{title.center(self.width)}{Terminal.RESET}"
        output[1] = title_colored

        # Speed indicator
        speed = "WARP SPEED ████████████████▶"
        output[self.height - 2] = f"{Terminal.rgb(100,200,255)}{speed.center(self.width)}{Terminal.RESET}"

        return '\n'.join(output)


# ============================================================================
# NEURAL PULSE
# ============================================================================

class NeuralPulse:
    """Pulsing neural network visualization of the cortical layers."""

    LAYERS = [
        ("DOCUMENTS", "Layer 3 - IT Cortex", 4),
        ("CONCEPTS", "Layer 2 - V4 Cortex", 8),
        ("BIGRAMS", "Layer 1 - V2 Cortex", 12),
        ("TOKENS", "Layer 0 - V1 Cortex", 16),
    ]

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.time = 0
        self.pulses = []  # (layer_idx, node_idx, intensity)
        self.file_stats = get_file_stats()

    def update(self):
        self.time += 0.15

        # Spawn new pulses from bottom
        if random.random() < 0.3:
            layer_idx = 3  # Tokens layer
            node_idx = random.randint(0, self.LAYERS[layer_idx][2] - 1)
            self.pulses.append([layer_idx, node_idx, 1.0, 0])  # layer, node, intensity, age

        # Update existing pulses
        new_pulses = []
        for pulse in self.pulses:
            pulse[2] *= 0.95  # Decay
            pulse[3] += 1     # Age

            # Propagate upward occasionally
            if pulse[3] % 8 == 0 and pulse[0] > 0 and pulse[2] > 0.3:
                new_layer = pulse[0] - 1
                # Connect to random node in upper layer
                new_node = random.randint(0, self.LAYERS[new_layer][2] - 1)
                new_pulses.append([new_layer, new_node, pulse[2] * 0.8, 0])

            if pulse[2] > 0.1:
                new_pulses.append(pulse)

        self.pulses = new_pulses

    def render(self) -> str:
        output = [Terminal.HOME]

        # Calculate positions
        layer_height = (self.height - 6) // len(self.LAYERS)

        # Create intensity map per layer/node
        intensities = {}
        for pulse in self.pulses:
            key = (pulse[0], pulse[1])
            intensities[key] = max(intensities.get(key, 0), pulse[2])

        output.append(f"{Terminal.BOLD}{Terminal.rgb(150,200,255)}{'[ 🧠 NEURAL PULSE: Cortical Layer Activity 🧠 ]'.center(self.width)}{Terminal.RESET}")
        output.append("")

        for layer_idx, (name, desc, node_count) in enumerate(self.LAYERS):
            y = 3 + layer_idx * layer_height

            # Layer label
            label = f"━━━ {name} ({desc}) ━━━"
            output.append(f"{Terminal.rgb(100, 150, 200)}{label.center(self.width)}{Terminal.RESET}")

            # Nodes
            node_spacing = (self.width - 10) // node_count
            node_line = " " * 5

            for node_idx in range(node_count):
                intensity = intensities.get((layer_idx, node_idx), 0)

                # Pulsing glow effect
                glow = math.sin(self.time + node_idx * 0.5) * 0.2 + 0.3
                combined = min(1.0, intensity + glow * 0.3)

                if intensity > 0.5:
                    # Active - bright color
                    r = int(100 + 155 * intensity)
                    g = int(200 * intensity)
                    b = int(255 * intensity)
                    char = "◉"
                elif intensity > 0.2:
                    # Fading
                    r = int(50 + 100 * intensity)
                    g = int(100 * intensity)
                    b = int(150 * intensity)
                    char = "○"
                else:
                    # Idle with subtle pulse
                    brightness = int(50 + 30 * glow)
                    r, g, b = brightness, brightness, brightness + 20
                    char = "·"

                node_line += f"{Terminal.rgb(r, g, b)}{char}{Terminal.RESET}" + " " * (node_spacing - 1)

            output.append(node_line)

            # Connection lines (if not bottom layer)
            if layer_idx < len(self.LAYERS) - 1:
                conn_line = " " * 5
                for i in range(node_count):
                    # Draw connection hint
                    conn_line += f"{Terminal.rgb(60, 80, 100)}│{Terminal.RESET}" + " " * (node_spacing - 1)
                output.append(conn_line)

            output.append("")

        # Stats at bottom
        total_files = len(self.file_stats)
        total_lines = sum(self.file_stats.values())
        stats = f"Files: {total_files} | Lines: {total_lines:,} | Active Pulses: {len(self.pulses)}"
        output.append(f"{Terminal.rgb(100, 150, 100)}{stats.center(self.width)}{Terminal.RESET}")

        # Pad to height
        while len(output) < self.height:
            output.append("")

        return '\n'.join(output[:self.height])


# ============================================================================
# LIVE DASHBOARD
# ============================================================================

class LiveDashboard:
    """Real-time animated dashboard with multiple panels."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.time = 0
        self.file_stats = get_file_stats()
        self.hot_files = get_hot_files()
        self.commits = get_commits(20)

        # Animated bar states
        self.bar_targets = {}
        self.bar_current = {}

        for name, lines in self.file_stats.items():
            self.bar_targets[name] = lines
            self.bar_current[name] = 0.0

    def update(self):
        self.time += 0.1

        # Animate bars toward targets
        for name in self.bar_current:
            target = self.bar_targets.get(name, 0)
            current = self.bar_current[name]
            self.bar_current[name] = current + (target - current) * 0.1

    def _draw_box(self, title: str, x: int, y: int, w: int, h: int) -> List[Tuple[int, int, str]]:
        """Draw a box and return positioned strings."""
        chars = []
        # Top border
        border_color = Terminal.rgb(80, 120, 180)
        title_color = Terminal.rgb(200, 220, 255)

        top = f"╔{'═' * (w - 2)}╗"
        chars.append((y, x, f"{border_color}{top}{Terminal.RESET}"))

        # Title
        title_str = f"║ {title}{' ' * (w - 4 - len(title))} ║"
        chars.append((y + 1, x, f"{border_color}║{Terminal.RESET}{title_color} {title}{Terminal.RESET}{' ' * (w - 4 - len(title))}{border_color}║{Terminal.RESET}"))

        # Separator
        sep = f"╠{'─' * (w - 2)}╣"
        chars.append((y + 2, x, f"{border_color}{sep}{Terminal.RESET}"))

        # Middle rows
        for i in range(3, h - 1):
            chars.append((y + i, x, f"{border_color}║{Terminal.RESET}{' ' * (w - 2)}{border_color}║{Terminal.RESET}"))

        # Bottom border
        bottom = f"╚{'═' * (w - 2)}╝"
        chars.append((y + h - 1, x, f"{border_color}{bottom}{Terminal.RESET}"))

        return chars

    def _animated_bar(self, value: float, max_val: float, width: int) -> str:
        """Create an animated gradient bar."""
        if max_val == 0:
            return " " * width

        ratio = min(1.0, value / max_val)
        filled = int(ratio * width)

        bar = ""
        for i in range(width):
            if i < filled:
                # Gradient from green to yellow to red
                progress = i / width
                if progress < 0.5:
                    r = int(100 + 155 * (progress * 2))
                    g = 200
                    b = 50
                else:
                    r = 255
                    g = int(200 - 150 * ((progress - 0.5) * 2))
                    b = 50

                # Pulsing effect
                pulse = math.sin(self.time * 2 + i * 0.3) * 20
                r = min(255, max(0, r + int(pulse)))

                bar += f"{Terminal.rgb(r, g, b)}█{Terminal.RESET}"
            else:
                bar += f"{Terminal.rgb(40, 40, 50)}░{Terminal.RESET}"

        return bar

    def render(self) -> str:
        buffer = [[' ' for _ in range(self.width)] for _ in range(self.height)]

        # Title
        title = "⚡ CORTICAL TEXT PROCESSOR - LIVE DASHBOARD ⚡"
        pulse = int(127 + 127 * math.sin(self.time * 2))
        title_line = f"{Terminal.BOLD}{Terminal.rgb(pulse, 200, 255)}{title.center(self.width)}{Terminal.RESET}"

        output = [Terminal.HOME, title_line, ""]

        # Calculate panel dimensions
        panel_width = self.width // 2 - 2
        panel_height = (self.height - 5) // 2

        # Panel 1: File sizes (top left)
        output.append(f"{Terminal.rgb(80, 150, 200)}╔{'═' * (panel_width - 2)}╗{Terminal.RESET}" +
                     "  " +
                     f"{Terminal.rgb(80, 150, 200)}╔{'═' * (panel_width - 2)}╗{Terminal.RESET}")

        output.append(f"{Terminal.rgb(80, 150, 200)}║{Terminal.RESET}" +
                     f"{Terminal.rgb(200, 220, 255)} 📊 FILE SIZES{Terminal.RESET}".ljust(panel_width + 10) +
                     f"{Terminal.rgb(80, 150, 200)}║{Terminal.RESET}" +
                     "  " +
                     f"{Terminal.rgb(80, 150, 200)}║{Terminal.RESET}" +
                     f"{Terminal.rgb(200, 220, 255)} 🔥 HOT FILES{Terminal.RESET}".ljust(panel_width + 10) +
                     f"{Terminal.rgb(80, 150, 200)}║{Terminal.RESET}")

        # Sort files
        sorted_files = sorted(self.file_stats.items(), key=lambda x: -x[1])[:panel_height - 4]
        sorted_hot = sorted(self.hot_files.items(), key=lambda x: -x[1])[:panel_height - 4]
        max_lines = max(self.file_stats.values()) if self.file_stats else 1
        max_hot = max(self.hot_files.values()) if self.hot_files else 1

        for i in range(panel_height - 4):
            # Left panel: file sizes
            if i < len(sorted_files):
                name, lines = sorted_files[i]
                short_name = name[:12].ljust(12)
                bar = self._animated_bar(self.bar_current.get(name, 0), max_lines, 15)
                left_content = f" {short_name} {bar} {lines:4d}"
            else:
                left_content = " " * (panel_width - 2)

            # Right panel: hot files
            if i < len(sorted_hot):
                name, count = sorted_hot[i]
                short_name = name[:12].ljust(12)
                heat = count / max_hot
                heat_chars = "🔥" * min(5, int(heat * 5) + 1)
                right_content = f" {short_name} {heat_chars.ljust(10)} {count:3d}"
            else:
                right_content = " " * (panel_width - 2)

            output.append(
                f"{Terminal.rgb(80, 150, 200)}║{Terminal.RESET}" +
                left_content[:panel_width - 2].ljust(panel_width - 2) +
                f"{Terminal.rgb(80, 150, 200)}║{Terminal.RESET}" +
                "  " +
                f"{Terminal.rgb(80, 150, 200)}║{Terminal.RESET}" +
                right_content[:panel_width - 2].ljust(panel_width - 2) +
                f"{Terminal.rgb(80, 150, 200)}║{Terminal.RESET}"
            )

        output.append(f"{Terminal.rgb(80, 150, 200)}╚{'═' * (panel_width - 2)}╝{Terminal.RESET}" +
                     "  " +
                     f"{Terminal.rgb(80, 150, 200)}╚{'═' * (panel_width - 2)}╝{Terminal.RESET}")

        output.append("")

        # Bottom panel: Recent commits (full width)
        output.append(f"{Terminal.rgb(80, 150, 200)}╔{'═' * (self.width - 4)}╗{Terminal.RESET}")
        output.append(f"{Terminal.rgb(80, 150, 200)}║{Terminal.RESET}" +
                     f"{Terminal.rgb(200, 220, 255)} 📜 RECENT COMMITS{Terminal.RESET}".ljust(self.width + 7) +
                     f"{Terminal.rgb(80, 150, 200)}║{Terminal.RESET}")

        for i, (hash_short, date, msg) in enumerate(self.commits[:5]):
            # Animated highlight for recent
            if i == 0:
                glow = int(100 + 50 * math.sin(self.time * 3))
                color = Terminal.rgb(glow, 255, glow)
            else:
                color = Terminal.rgb(150, 150, 150)

            commit_line = f" {color}{hash_short}{Terminal.RESET} {date} {msg[:self.width - 25]}"
            output.append(f"{Terminal.rgb(80, 150, 200)}║{Terminal.RESET}" +
                         commit_line.ljust(self.width + 30)[:self.width - 4] +
                         f"{Terminal.rgb(80, 150, 200)}║{Terminal.RESET}")

        output.append(f"{Terminal.rgb(80, 150, 200)}╚{'═' * (self.width - 4)}╝{Terminal.RESET}")

        # Footer
        footer = f"Press 'q' to quit | Time: {time.strftime('%H:%M:%S')}"
        output.append(f"{Terminal.rgb(100, 100, 120)}{footer.center(self.width)}{Terminal.RESET}")

        return '\n'.join(output)


# ============================================================================
# MAIN ANIMATION LOOP
# ============================================================================

class AnimationController:
    """Controls animation lifecycle and input handling."""

    def __init__(self):
        self.running = True
        self.paused = False
        self.current_vis = None

        # Setup signal handler
        signal.signal(signal.SIGINT, self._handle_interrupt)

    def _handle_interrupt(self, signum, frame):
        self.running = False

    def run(self, visualization, fps: int = 30):
        """Run animation loop."""
        Terminal.setup()

        try:
            frame_time = 1.0 / fps

            while self.running:
                start = time.time()

                if not self.paused:
                    visualization.update()

                print(visualization.render(), end='', flush=True)

                # Frame timing
                elapsed = time.time() - start
                if elapsed < frame_time:
                    time.sleep(frame_time - elapsed)

        finally:
            Terminal.cleanup()


def show_menu():
    """Show interactive menu."""
    print(f"""
{Terminal.BOLD}{Terminal.rgb(100, 200, 255)}
    ╔═══════════════════════════════════════════════════════════════╗
    ║     🎨 ANIMATED ASCII CODEBASE VISUALIZER 🎨                  ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║                                                               ║
    ║   Choose a visualization:                                     ║
    ║                                                               ║
    ║   [1] 💻 Matrix Rain    - Falling commit messages             ║
    ║   [2] 🔥 Fire Effect    - Hot files burn brighter             ║
    ║   [3] ✨ Starfield      - Commits flying through space        ║
    ║   [4] 🧠 Neural Pulse   - Pulsing cortical layers             ║
    ║   [5] 📊 Live Dashboard - Real-time animated stats            ║
    ║                                                               ║
    ║   [q] Quit                                                    ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
{Terminal.RESET}
    """)

    choice = input(f"{Terminal.rgb(150, 200, 255)}    Enter choice (1-5): {Terminal.RESET}")
    return choice.strip()


def main():
    parser = argparse.ArgumentParser(description="Animated ASCII Codebase Visualizer")
    parser.add_argument('--matrix', action='store_true', help='Matrix rain effect')
    parser.add_argument('--fire', action='store_true', help='Fire effect')
    parser.add_argument('--stars', action='store_true', help='Starfield effect')
    parser.add_argument('--neural', action='store_true', help='Neural pulse effect')
    parser.add_argument('--dashboard', action='store_true', help='Live dashboard')
    parser.add_argument('--demo', action='store_true', help='Demo all effects')
    parser.add_argument('--fps', type=int, default=20, help='Frames per second')

    args = parser.parse_args()

    height, width = Terminal.get_size()
    height = min(height - 2, 40)
    width = min(width - 2, 120)

    controller = AnimationController()

    # Determine which visualization to run
    if args.matrix:
        vis = MatrixRain(width, height)
    elif args.fire:
        vis = FireEffect(width, height)
    elif args.stars:
        vis = Starfield(width, height)
    elif args.neural:
        vis = NeuralPulse(width, height)
    elif args.dashboard:
        vis = LiveDashboard(width, height)
    elif args.demo:
        # Run each for a few seconds
        visualizations = [
            ("Matrix Rain", MatrixRain(width, height)),
            ("Fire Effect", FireEffect(width, height)),
            ("Starfield", Starfield(width, height)),
            ("Neural Pulse", NeuralPulse(width, height)),
            ("Dashboard", LiveDashboard(width, height)),
        ]

        for name, vis in visualizations:
            Terminal.setup()
            print(f"\n{Terminal.BOLD}Starting: {name}{Terminal.RESET}\n")
            time.sleep(1)

            start = time.time()
            while time.time() - start < 5 and controller.running:
                vis.update()
                print(vis.render(), end='', flush=True)
                time.sleep(0.05)

            Terminal.cleanup()

        return
    else:
        # Interactive menu
        choice = show_menu()
        if choice == '1':
            vis = MatrixRain(width, height)
        elif choice == '2':
            vis = FireEffect(width, height)
        elif choice == '3':
            vis = Starfield(width, height)
        elif choice == '4':
            vis = NeuralPulse(width, height)
        elif choice == '5':
            vis = LiveDashboard(width, height)
        else:
            print("Goodbye!")
            return

    controller.run(vis, fps=args.fps)


if __name__ == '__main__':
    main()
