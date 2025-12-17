#!/usr/bin/env python3
"""
Generate animated GIFs from ASCII visualizations.

Creates beautiful GIF previews of:
- Matrix rain effect
- Fire effect
- Starfield
- Neural pulse
- Code skyline (static)

Requires: PIL/Pillow
    pip install Pillow

Usage:
    python scripts/generate_ascii_gifs.py              # Generate all GIFs
    python scripts/generate_ascii_gifs.py --matrix     # Just matrix
    python scripts/generate_ascii_gifs.py --output assets/  # Custom output dir
"""

import argparse
import math
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import subprocess

from ascii_effects import MatrixDrop, Star

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: Pillow not installed. Run: pip install Pillow")


# ============================================================================
# CONFIGURATION
# ============================================================================

# GIF dimensions and styling
WIDTH = 800
HEIGHT = 500
FONT_SIZE = 14
LINE_HEIGHT = 16
CHAR_WIDTH = 8
BG_COLOR = (13, 17, 23)  # Dark background like GitHub dark mode
FRAMES = 60
FRAME_DURATION = 50  # milliseconds


def get_font():
    """Get a monospace font for rendering."""
    # Try common monospace fonts
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
        "/System/Library/Fonts/Monaco.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "C:\\Windows\\Fonts\\consola.ttf",
        "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf",
    ]

    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, FONT_SIZE)
            except Exception:
                pass

    # Fallback to default
    try:
        return ImageFont.truetype("DejaVuSansMono.ttf", FONT_SIZE)
    except Exception:
        return ImageFont.load_default()


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


def get_commits(limit: int = 50) -> List[Tuple[str, str, str]]:
    """Get recent commits."""
    log = run_git(['log', '--format=%h|%ad|%s', '--date=short', '-n', str(limit)])
    commits = []
    for line in log.splitlines():
        parts = line.split('|', 2)
        if len(parts) == 3:
            commits.append(tuple(parts))
    return commits or [("abc1234", "2025-12-15", "Example commit message")]


def get_file_stats() -> Dict[str, int]:
    """Get file line counts."""
    stats = {}
    cortical_dir = get_project_root() / 'cortical'
    if cortical_dir.exists():
        for py_file in cortical_dir.rglob('*.py'):
            if '__pycache__' not in str(py_file):
                try:
                    lines = len(py_file.read_text().splitlines())
                    stats[py_file.name] = lines
                except Exception:
                    pass

    # Fallback data if no files found
    if not stats:
        stats = {
            "analysis.py": 1123, "processor.py": 892, "query.py": 756,
            "semantics.py": 915, "persistence.py": 606, "tokenizer.py": 398,
        }
    return stats


def get_hot_files() -> Dict[str, int]:
    """Get file modification counts."""
    log = run_git(['log', '--format=', '--name-only', '-n', '200'])
    counts = defaultdict(int)
    for line in log.splitlines():
        if line.endswith('.py'):
            counts[Path(line).name] += 1

    # Fallback data
    if not counts:
        counts = {"processor.py": 49, "analysis.py": 24, "query.py": 19}
    return dict(counts)


# ============================================================================
# MATRIX RAIN GENERATOR
# ============================================================================

# MatrixDrop imported from ascii_effects


class MatrixRainGenerator:
    """Generate Matrix rain frames."""

    CHARS = "ｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜﾝ0123456789"

    def __init__(self, cols: int, rows: int):
        self.cols = cols
        self.rows = rows
        self.drops: List[MatrixDrop] = []
        self.commits = get_commits(30)
        self.commit_idx = 0

        for _ in range(cols // 2):
            self._spawn_drop()

    def _spawn_drop(self):
        col = random.randint(0, self.cols - 1)
        if self.commits and random.random() < 0.3:
            text = self.commits[self.commit_idx % len(self.commits)][2][:25]
            self.commit_idx += 1
        else:
            text = ''.join(random.choice(self.CHARS) for _ in range(random.randint(5, 15)))

        self.drops.append(MatrixDrop(
            col=col, row=random.uniform(-15, 0),
            speed=random.uniform(0.3, 0.8), text=text, char_idx=0
        ))

    def update(self):
        new_drops = []
        for drop in self.drops:
            drop.row += drop.speed
            if drop.row < self.rows + len(drop.text):
                new_drops.append(drop)
            else:
                self._spawn_drop()
        self.drops = new_drops
        while len(self.drops) < self.cols // 2:
            self._spawn_drop()

    def render_frame(self, font) -> Image.Image:
        img = Image.new('RGB', (WIDTH, HEIGHT), BG_COLOR)
        draw = ImageDraw.Draw(img)

        # Title
        title = "[ MATRIX: COMMIT STREAM ]"
        draw.text((WIDTH // 2 - len(title) * 4, 10), title, fill=(0, 255, 0), font=font)

        for drop in self.drops:
            for i, char in enumerate(drop.text):
                row = int(drop.row) - i
                if 0 <= row < self.rows:
                    x = drop.col * CHAR_WIDTH + 10
                    y = row * LINE_HEIGHT + 30

                    if i == 0:
                        color = (255, 255, 255)
                    elif i < 3:
                        color = (0, 255, 70)
                    else:
                        color = (0, 150, 40)

                    draw.text((x, y), char, fill=color, font=font)

        return img


# ============================================================================
# FIRE EFFECT GENERATOR
# ============================================================================

class FireGenerator:
    """Generate fire effect frames."""

    FIRE_CHARS = " .:-=+*#%@"

    def __init__(self, cols: int, rows: int):
        self.cols = cols
        self.rows = rows
        self.buffer = [[0.0 for _ in range(cols)] for _ in range(rows)]
        self.hot_files = get_hot_files()
        self.file_names = list(self.hot_files.keys())[:min(cols // 6, 12)]

        self.base_heat = []
        if self.file_names:
            max_count = max(self.hot_files.values()) if self.hot_files else 1
            col_width = cols // len(self.file_names)
            for filename in self.file_names:
                heat = self.hot_files.get(filename, 0) / max_count
                for _ in range(col_width):
                    self.base_heat.append(heat)
        while len(self.base_heat) < cols:
            self.base_heat.append(0.3)

    def update(self):
        for x in range(self.cols):
            base = self.base_heat[x] if x < len(self.base_heat) else 0.3
            self.buffer[self.rows - 1][x] = base + random.uniform(0, 0.5)
            self.buffer[self.rows - 2][x] = base + random.uniform(0, 0.4)

        for y in range(self.rows - 3, -1, -1):
            for x in range(self.cols):
                below = self.buffer[y + 1][x]
                left = self.buffer[y + 1][max(0, x - 1)]
                right = self.buffer[y + 1][min(self.cols - 1, x + 1)]
                new_val = (below * 0.5 + left * 0.25 + right * 0.25)
                new_val *= random.uniform(0.85, 0.98)
                self.buffer[y][x] = max(0, min(1, new_val))

    def render_frame(self, font) -> Image.Image:
        img = Image.new('RGB', (WIDTH, HEIGHT), BG_COLOR)
        draw = ImageDraw.Draw(img)

        title = "[ 🔥 CODE FIRE: Hot Files Burn Brighter 🔥 ]"
        draw.text((WIDTH // 2 - len(title) * 4, 10), title, fill=(255, 200, 0), font=font)

        for y in range(self.rows - 3):
            for x in range(self.cols):
                heat = self.buffer[y][x]
                char_idx = min(len(self.FIRE_CHARS) - 1, int(heat * len(self.FIRE_CHARS)))
                char = self.FIRE_CHARS[char_idx]

                # Fire gradient
                if heat < 0.2:
                    color = (int(32 * heat * 5), 0, 0)
                elif heat < 0.4:
                    color = (int(64 + 64 * (heat - 0.2) * 5), 0, 0)
                elif heat < 0.6:
                    color = (int(128 + 64 * (heat - 0.4) * 5), int(32 * (heat - 0.4) * 5), 0)
                elif heat < 0.8:
                    color = (255, int(64 + 128 * (heat - 0.6) * 5), 0)
                else:
                    color = (255, int(192 + 63 * (heat - 0.8) * 5), int(128 * (heat - 0.8) * 5))

                px = x * CHAR_WIDTH + 10
                py = y * LINE_HEIGHT + 30
                draw.text((px, py), char, fill=color, font=font)

        # File labels
        if self.file_names:
            col_width = self.cols // len(self.file_names)
            for i, name in enumerate(self.file_names):
                x = i * col_width * CHAR_WIDTH + 10
                y = HEIGHT - 25
                color = (255, 100, 0) if self.hot_files.get(name, 0) > 10 else (150, 150, 150)
                draw.text((x, y), name[:8], fill=color, font=font)

        return img


# ============================================================================
# STARFIELD GENERATOR
# ============================================================================

# Star imported from ascii_effects


class StarfieldGenerator:
    """Generate starfield frames."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.cx = width // 2
        self.cy = height // 2
        self.stars: List[Star] = []
        self.commits = get_commits(50)
        self.commit_idx = 0

        for _ in range(80):
            self._spawn_star(random.uniform(0.1, 1.0))

    def _spawn_star(self, z: float):
        angle = random.uniform(0, 2 * math.pi)
        distance = random.uniform(0.1, 0.8)
        x = math.cos(angle) * distance * self.width
        y = math.sin(angle) * distance * self.height

        if self.commits and random.random() < 0.15:
            text = self.commits[self.commit_idx % len(self.commits)][2][:15]
            self.commit_idx += 1
        else:
            text = random.choice(['*', '+', '.', '·', '✦', '★', '○'])

        self.stars.append(Star(x=x, y=y, z=z, text=text))

    def update(self):
        new_stars = []
        for star in self.stars:
            star.z -= 0.015
            if star.z > 0.01:
                new_stars.append(star)
            else:
                self._spawn_star(random.uniform(0.8, 1.0))
        self.stars = new_stars
        while len(self.stars) < 80:
            self._spawn_star(random.uniform(0.8, 1.0))

    def render_frame(self, font) -> Image.Image:
        img = Image.new('RGB', (WIDTH, HEIGHT), (10, 10, 30))
        draw = ImageDraw.Draw(img)

        title = "[ ✨ HYPERSPACE: Commits Flying By ✨ ]"
        draw.text((WIDTH // 2 - len(title) * 4, 10), title, fill=(100, 150, 255), font=font)

        for star in self.stars:
            if star.z > 0:
                px = int(self.cx + star.x / star.z)
                py = int(self.cy + star.y / star.z)

                if 0 <= px < self.width and 0 <= py < self.height:
                    brightness = int(255 * (1 - star.z))
                    brightness = max(50, min(255, brightness))

                    if star.z < 0.3:
                        color = (brightness, brightness, 255)
                    elif star.z < 0.6:
                        color = (brightness, brightness, brightness)
                    else:
                        color = (255, brightness, brightness // 2)

                    draw.text((px, py), star.text[0], fill=color, font=font)

        # Speed indicator
        speed = "WARP SPEED ████████████████▶"
        draw.text((WIDTH // 2 - len(speed) * 4, HEIGHT - 30), speed, fill=(100, 200, 255), font=font)

        return img


# ============================================================================
# NEURAL PULSE GENERATOR
# ============================================================================

class NeuralPulseGenerator:
    """Generate neural pulse frames."""

    LAYERS = [
        ("DOCUMENTS", 4),
        ("CONCEPTS", 8),
        ("BIGRAMS", 12),
        ("TOKENS", 16),
    ]

    def __init__(self):
        self.time = 0
        self.pulses = []  # (layer_idx, node_idx, intensity, age)

    def update(self):
        self.time += 0.15

        if random.random() < 0.3:
            self.pulses.append([3, random.randint(0, 15), 1.0, 0])

        new_pulses = []
        for pulse in self.pulses:
            pulse[2] *= 0.95
            pulse[3] += 1

            if pulse[3] % 8 == 0 and pulse[0] > 0 and pulse[2] > 0.3:
                new_layer = pulse[0] - 1
                new_node = random.randint(0, self.LAYERS[new_layer][1] - 1)
                new_pulses.append([new_layer, new_node, pulse[2] * 0.8, 0])

            if pulse[2] > 0.1:
                new_pulses.append(pulse)

        self.pulses = new_pulses

    def render_frame(self, font) -> Image.Image:
        img = Image.new('RGB', (WIDTH, HEIGHT), BG_COLOR)
        draw = ImageDraw.Draw(img)

        title = "[ 🧠 NEURAL PULSE: Cortical Layer Activity 🧠 ]"
        draw.text((WIDTH // 2 - len(title) * 4, 10), title, fill=(150, 200, 255), font=font)

        intensities = {}
        for pulse in self.pulses:
            key = (pulse[0], pulse[1])
            intensities[key] = max(intensities.get(key, 0), pulse[2])

        layer_height = (HEIGHT - 80) // len(self.LAYERS)

        for layer_idx, (name, node_count) in enumerate(self.LAYERS):
            y_base = 50 + layer_idx * layer_height

            # Layer label
            label = f"━━━ {name} ━━━"
            draw.text((WIDTH // 2 - len(label) * 4, y_base), label, fill=(100, 150, 200), font=font)

            # Nodes
            node_spacing = (WIDTH - 100) // node_count
            for node_idx in range(node_count):
                intensity = intensities.get((layer_idx, node_idx), 0)
                glow = math.sin(self.time + node_idx * 0.5) * 0.2 + 0.3

                x = 50 + node_idx * node_spacing
                y = y_base + 25

                if intensity > 0.5:
                    color = (int(100 + 155 * intensity), int(200 * intensity), int(255 * intensity))
                    char = "◉"
                elif intensity > 0.2:
                    color = (int(50 + 100 * intensity), int(100 * intensity), int(150 * intensity))
                    char = "○"
                else:
                    brightness = int(50 + 30 * glow)
                    color = (brightness, brightness, brightness + 20)
                    char = "·"

                draw.text((x, y), char, fill=color, font=font)

            # Connection lines
            if layer_idx < len(self.LAYERS) - 1:
                for node_idx in range(node_count):
                    x = 50 + node_idx * node_spacing + 4
                    y1 = y_base + 40
                    y2 = y_base + layer_height - 10
                    draw.line([(x, y1), (x, y2)], fill=(60, 80, 100), width=1)

        return img


# ============================================================================
# SKYLINE GENERATOR (Static with slight animation)
# ============================================================================

class SkylineGenerator:
    """Generate code skyline frames."""

    def __init__(self):
        self.file_stats = get_file_stats()
        self.sorted_files = sorted(self.file_stats.items(), key=lambda x: -x[1])[:15]
        self.max_lines = max(self.file_stats.values()) if self.file_stats else 1
        self.time = 0

    def update(self):
        self.time += 0.1

    def render_frame(self, font) -> Image.Image:
        img = Image.new('RGB', (WIDTH, HEIGHT), BG_COLOR)
        draw = ImageDraw.Draw(img)

        title = "[ CODE SKYLINE - File Sizes as Buildings ]"
        draw.text((WIDTH // 2 - len(title) * 4, 10), title, fill=(100, 200, 255), font=font)

        building_width = 40
        spacing = 10
        max_height = HEIGHT - 120
        base_y = HEIGHT - 60

        colors = [
            (66, 135, 245), (92, 184, 92), (240, 173, 78),
            (217, 83, 79), (153, 102, 255), (23, 162, 184)
        ]

        for i, (filename, lines) in enumerate(self.sorted_files):
            height = int((lines / self.max_lines) * max_height)
            x = 30 + i * (building_width + spacing)
            y = base_y - height

            color = colors[i % len(colors)]

            # Building with slight glow animation
            glow = int(20 * math.sin(self.time + i * 0.5))
            adjusted_color = tuple(min(255, c + glow) for c in color)

            # Building body
            draw.rectangle([x, y, x + building_width, base_y], fill=adjusted_color)

            # Windows
            window_color = tuple(min(255, c + 60) for c in color)
            for wy in range(y + 10, base_y - 10, 20):
                for wx in range(x + 8, x + building_width - 8, 15):
                    draw.rectangle([wx, wy, wx + 8, wy + 12], fill=window_color)

            # Roof
            draw.polygon([(x, y), (x + building_width // 2, y - 15), (x + building_width, y)],
                        fill=adjusted_color)

            # Label
            short_name = filename[:5]
            draw.text((x + 5, base_y + 5), short_name, fill=(180, 180, 180), font=font)

        # Ground
        draw.rectangle([0, base_y, WIDTH, base_y + 3], fill=(100, 100, 100))

        # Stats
        stats = f"Top file: {self.sorted_files[0][0]} ({self.sorted_files[0][1]} lines)"
        draw.text((20, HEIGHT - 25), stats, fill=(150, 150, 150), font=font)

        return img


# ============================================================================
# MAIN GENERATION
# ============================================================================

def generate_gif(generator, name: str, output_dir: Path, frames: int = FRAMES):
    """Generate a GIF from a frame generator."""
    font = get_font()
    images = []

    print(f"  Generating {name}...")
    for i in range(frames):
        generator.update()
        img = generator.render_frame(font)
        images.append(img)
        if (i + 1) % 10 == 0:
            print(f"    Frame {i + 1}/{frames}")

    output_path = output_dir / f"{name}.gif"
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=FRAME_DURATION,
        loop=0,
        optimize=True
    )
    print(f"  ✓ Saved: {output_path}")
    return output_path


def main():
    if not PIL_AVAILABLE:
        print("Error: Pillow is required. Install with: pip install Pillow")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Generate ASCII art GIFs")
    parser.add_argument('--output', '-o', default='assets', help='Output directory')
    parser.add_argument('--matrix', action='store_true', help='Generate matrix only')
    parser.add_argument('--fire', action='store_true', help='Generate fire only')
    parser.add_argument('--stars', action='store_true', help='Generate starfield only')
    parser.add_argument('--neural', action='store_true', help='Generate neural only')
    parser.add_argument('--skyline', action='store_true', help='Generate skyline only')
    parser.add_argument('--frames', '-f', type=int, default=FRAMES, help='Number of frames')
    args = parser.parse_args()

    output_dir = get_project_root() / args.output
    output_dir.mkdir(exist_ok=True)

    # Determine what to generate
    generate_all = not any([args.matrix, args.fire, args.stars, args.neural, args.skyline])

    cols = WIDTH // CHAR_WIDTH
    rows = HEIGHT // LINE_HEIGHT

    print(f"\n🎨 Generating ASCII Art GIFs...")
    print(f"   Output: {output_dir}/")
    print(f"   Frames: {args.frames}")
    print()

    generated = []

    if generate_all or args.matrix:
        gen = MatrixRainGenerator(cols, rows)
        generated.append(generate_gif(gen, "matrix_rain", output_dir, args.frames))

    if generate_all or args.fire:
        gen = FireGenerator(cols, rows)
        generated.append(generate_gif(gen, "fire_effect", output_dir, args.frames))

    if generate_all or args.stars:
        gen = StarfieldGenerator(WIDTH, HEIGHT)
        generated.append(generate_gif(gen, "starfield", output_dir, args.frames))

    if generate_all or args.neural:
        gen = NeuralPulseGenerator()
        generated.append(generate_gif(gen, "neural_pulse", output_dir, args.frames))

    if generate_all or args.skyline:
        gen = SkylineGenerator()
        generated.append(generate_gif(gen, "code_skyline", output_dir, args.frames // 2))

    print(f"\n✨ Done! Generated {len(generated)} GIF(s)")
    print("\nTo add to README.md:")
    for path in generated:
        rel_path = path.relative_to(get_project_root())
        print(f"  ![{path.stem}]({rel_path})")


if __name__ == '__main__':
    main()
