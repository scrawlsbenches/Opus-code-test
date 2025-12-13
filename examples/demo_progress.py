#!/usr/bin/env python
"""
Demonstration of progress reporting capabilities.

This script shows the different ways to use progress reporting with
the Cortical Text Processor's compute_all() method.
"""

from cortical import CorticalTextProcessor, CallbackProgressReporter


def demo_silent():
    """Default behavior - no progress output."""
    print("=" * 60)
    print("Demo 1: Silent Mode (default)")
    print("=" * 60)

    processor = CorticalTextProcessor()
    processor.process_document("doc1", "Neural networks process information efficiently.")
    processor.process_document("doc2", "Machine learning algorithms analyze large datasets.")
    processor.process_document("doc3", "Deep learning models require substantial training data.")

    print("Running compute_all() with default settings (silent)...")
    processor.compute_all(verbose=False)
    print("Complete! (no progress output)\n")


def demo_console_progress():
    """Console progress bar with nice formatting."""
    print("=" * 60)
    print("Demo 2: Console Progress Bar")
    print("=" * 60)

    processor = CorticalTextProcessor()
    processor.process_document("doc1", "Neural networks process information efficiently.")
    processor.process_document("doc2", "Machine learning algorithms analyze large datasets.")
    processor.process_document("doc3", "Deep learning models require substantial training data.")
    processor.process_document("doc4", "Artificial intelligence systems learn from experience.")
    processor.process_document("doc5", "Data science combines statistics and programming.")

    print("Running compute_all() with show_progress=True:")
    processor.compute_all(show_progress=True, verbose=False)
    print("\nComplete!\n")


def demo_callback():
    """Custom callback for integration with other systems."""
    print("=" * 60)
    print("Demo 3: Custom Callback")
    print("=" * 60)

    processor = CorticalTextProcessor()
    processor.process_document("doc1", "Neural networks process information efficiently.")
    processor.process_document("doc2", "Machine learning algorithms analyze large datasets.")
    processor.process_document("doc3", "Deep learning models require substantial training data.")

    # Track progress with custom callback
    progress_log = []

    def custom_callback(phase, percent, message):
        """Custom callback that logs progress."""
        progress_log.append({
            'phase': phase,
            'percent': percent,
            'message': message
        })
        # Print in custom format
        msg_str = f" - {message}" if message else ""
        print(f"  [{phase}] {percent:5.1f}%{msg_str}")

    reporter = CallbackProgressReporter(custom_callback)

    print("Running compute_all() with custom callback:")
    processor.compute_all(progress_callback=reporter, verbose=False)

    print(f"\nLogged {len(progress_log)} progress updates")
    print(f"Phases completed: {len([p for p in progress_log if p['percent'] == 100.0])}\n")


def demo_verbose_with_progress():
    """Combining verbose logging with progress bar."""
    print("=" * 60)
    print("Demo 4: Verbose Logging + Progress Bar")
    print("=" * 60)

    processor = CorticalTextProcessor()
    processor.process_document("doc1", "Neural networks process information efficiently.")
    processor.process_document("doc2", "Machine learning algorithms analyze large datasets.")

    print("Running compute_all() with both verbose=True and show_progress=True:")
    print("(Shows both logger messages and progress bars)\n")
    processor.compute_all(show_progress=True, verbose=True)
    print("\nComplete!\n")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("CORTICAL TEXT PROCESSOR - PROGRESS REPORTING DEMO")
    print("=" * 60 + "\n")

    # Demo 1: Silent (default)
    demo_silent()

    # Demo 2: Console progress bar
    demo_console_progress()

    # Demo 3: Custom callback
    demo_callback()

    # Demo 4: Verbose + progress
    demo_verbose_with_progress()

    print("=" * 60)
    print("All demos complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
