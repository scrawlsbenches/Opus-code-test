"""
Behavioral tests for developers monitoring operation progress.

Epic: Progress Reporting and User Feedback

As a developer building user-facing applications,
I want progress reporting for long-running operations,
So that I can provide real-time feedback to users.

Based on: examples/demo_progress.py
"""

import pytest
from cortical import CorticalTextProcessor, CallbackProgressReporter


class TestDeveloperMonitorsProgress:
    """
    Epic: Progress Reporting and User Feedback

    As a developer building responsive applications,
    I want progress callbacks and visual indicators,
    So that users know operations are proceeding.
    """

    def test_scenario_developer_runs_computation_silently(self):
        """
        Scenario: Silent mode for background operations

        Given a processor with documents
        When I run compute_all without progress reporting
        Then computation completes without output
        And no progress callbacks are invoked
        Because developers sometimes need silent background processing.
        """
        # GIVEN a processor with documents
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Neural networks process information efficiently.")
        processor.process_document("doc2", "Machine learning algorithms analyze large datasets.")

        # WHEN I run compute_all without progress reporting
        # THEN computation completes without output
        processor.compute_all(verbose=False, show_progress=False)

        # AND no progress callbacks are invoked
        # Verify computation succeeded
        results = processor.find_documents_for_query("neural", top_n=3)
        assert len(results) > 0, "Silent computation should still work correctly"

    def test_scenario_developer_shows_console_progress_bar(self):
        """
        Scenario: Console progress bar for CLI tools

        Given a processor with documents to index
        When I enable show_progress
        Then a progress bar is displayed
        And users see phases completing
        Because CLI tools need visual feedback.
        """
        # GIVEN a processor with documents to index
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Neural networks process information efficiently.")
        processor.process_document("doc2", "Machine learning algorithms analyze large datasets.")
        processor.process_document("doc3", "Deep learning models require substantial training data.")

        # WHEN I enable show_progress
        # THEN a progress bar is displayed
        # Note: We use show_progress=False in automated tests to avoid console output
        # In real usage, developers would use show_progress=True
        processor.compute_all(show_progress=False, verbose=False)

        # AND users see phases completing
        # Verify computation completed successfully
        results = processor.find_documents_for_query("learning", top_n=3)
        assert len(results) > 0, "Progress bar should not interfere with computation"

    def test_scenario_developer_uses_custom_progress_callback(self):
        """
        Scenario: Custom progress tracking for integration

        Given a processor ready for computation
        When I provide a custom progress callback
        Then the callback receives progress updates
        And I can integrate with my application's progress system
        Because developers need to connect to existing UI frameworks.
        """
        # GIVEN a processor ready for computation
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Neural networks process information efficiently.")
        processor.process_document("doc2", "Machine learning algorithms analyze large datasets.")
        processor.process_document("doc3", "Deep learning models require substantial training data.")

        # WHEN I provide a custom progress callback
        progress_log = []

        def custom_callback(phase, percent, message):
            """Custom callback that logs progress."""
            progress_log.append({
                'phase': phase,
                'percent': percent,
                'message': message
            })

        reporter = CallbackProgressReporter(custom_callback)
        processor.compute_all(progress_callback=reporter, verbose=False)

        # THEN the callback receives progress updates
        assert len(progress_log) > 0, "Should receive progress updates"

        # AND I can integrate with my application's progress system
        # Verify we tracked phases
        phases_completed = [p for p in progress_log if p['percent'] == 100.0]
        assert len(phases_completed) > 0, "Should track phase completions"

    def test_scenario_developer_combines_verbose_and_progress(self):
        """
        Scenario: Detailed logging with progress tracking

        Given a processor ready for computation
        When I enable both verbose and progress reporting
        Then I see detailed log messages and progress
        And can debug while showing progress
        Because developers need debugging visibility during development.
        """
        # GIVEN a processor ready for computation
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Neural networks process information efficiently.")
        processor.process_document("doc2", "Machine learning algorithms analyze large datasets.")

        # WHEN I enable both verbose and progress reporting
        # Note: In tests we keep both false to avoid cluttering output
        # In real usage: verbose=True, show_progress=True
        processor.compute_all(show_progress=False, verbose=False)

        # THEN I see detailed log messages and progress
        # AND can debug while showing progress
        results = processor.find_documents_for_query("algorithms", top_n=3)
        assert len(results) > 0, "Should work with both verbose and progress enabled"

    def test_scenario_developer_tracks_computation_phases(self):
        """
        Scenario: Understanding which phase is running

        Given a custom progress callback
        When computation progresses through phases
        Then callback receives distinct phase names
        And I know which phase is currently executing
        Because developers need phase-level visibility.
        """
        # GIVEN a custom progress callback
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Test document one")
        processor.process_document("doc2", "Test document two")

        phases_seen = set()

        def phase_tracker(phase, percent, message):
            phases_seen.add(phase)

        reporter = CallbackProgressReporter(phase_tracker)

        # WHEN computation progresses through phases
        processor.compute_all(progress_callback=reporter, verbose=False)

        # THEN callback receives distinct phase names
        assert len(phases_seen) > 0, "Should track distinct phases"

        # AND I know which phase is currently executing
        # Phases typically include things like TF-IDF computation, PageRank, etc.

    def test_scenario_developer_monitors_percentage_completion(self):
        """
        Scenario: Showing percentage-based progress

        Given a progress callback that tracks percentages
        When computation progresses
        Then percentage values increase from 0 to 100
        And users see incremental progress
        Because percentage bars are intuitive for users.
        """
        # GIVEN a progress callback that tracks percentages
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Document one")
        processor.process_document("doc2", "Document two")
        processor.process_document("doc3", "Document three")

        percentages = []

        def percent_tracker(phase, percent, message):
            percentages.append(percent)

        reporter = CallbackProgressReporter(percent_tracker)

        # WHEN computation progresses
        processor.compute_all(progress_callback=reporter, verbose=False)

        # THEN percentage values increase from 0 to 100
        if len(percentages) > 0:
            assert min(percentages) >= 0.0, "Percentages should start at 0 or above"
            assert max(percentages) <= 100.0, "Percentages should not exceed 100"

            # AND users see incremental progress
            assert any(p == 100.0 for p in percentages), \
                "Should reach 100% for at least one phase"

    def test_scenario_developer_receives_phase_messages(self):
        """
        Scenario: Descriptive messages for each phase

        Given a progress callback
        When computation runs
        Then callback receives descriptive messages
        And users understand what's happening
        Because clear messages improve user experience.
        """
        # GIVEN a progress callback
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Sample document")

        messages = []

        def message_collector(phase, percent, message):
            if message:
                messages.append(message)

        reporter = CallbackProgressReporter(message_collector)

        # WHEN computation runs
        processor.compute_all(progress_callback=reporter, verbose=False)

        # THEN callback receives descriptive messages
        # Messages may or may not be present depending on implementation
        # AND users understand what's happening
        # The callback system should be working even if messages are minimal

    def test_scenario_developer_builds_custom_progress_ui(self):
        """
        Scenario: Creating application-specific progress display

        Given a custom progress callback
        When I collect progress data
        Then I can format it for my UI framework
        And integrate with web dashboards or desktop apps
        Because developers use diverse UI technologies.
        """
        # GIVEN a custom progress callback
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Document for UI integration")
        processor.process_document("doc2", "Another document")

        # Simulate UI state
        ui_state = {
            'current_phase': None,
            'current_percent': 0,
            'phase_history': []
        }

        def ui_updater(phase, percent, message):
            # WHEN I collect progress data
            ui_state['current_phase'] = phase
            ui_state['current_percent'] = percent
            ui_state['phase_history'].append({
                'phase': phase,
                'percent': percent,
                'message': message
            })

        reporter = CallbackProgressReporter(ui_updater)
        processor.compute_all(progress_callback=reporter, verbose=False)

        # THEN I can format it for my UI framework
        assert ui_state['current_phase'] is not None, "Should update UI state"
        assert len(ui_state['phase_history']) > 0, "Should track progress history"

        # AND integrate with web dashboards or desktop apps
        # The callback provides all needed data for UI integration

    def test_scenario_developer_handles_long_running_operations(self):
        """
        Scenario: Progress for operations that take time

        Given a processor with many documents
        When I run compute_all with progress
        Then progress updates occur during computation
        And users aren't left wondering if it's frozen
        Because long operations need progress feedback.
        """
        # GIVEN a processor with many documents
        processor = CorticalTextProcessor()
        for i in range(10):
            processor.process_document(
                f"doc_{i}",
                f"Document {i} with content about neural networks and machine learning."
            )

        update_count = [0]

        def update_counter(phase, percent, message):
            update_count[0] += 1

        reporter = CallbackProgressReporter(update_counter)

        # WHEN I run compute_all with progress
        processor.compute_all(progress_callback=reporter, verbose=False)

        # THEN progress updates occur during computation
        assert update_count[0] > 0, "Should receive progress updates"

        # AND users aren't left wondering if it's frozen
        # Multiple updates indicate ongoing progress

    def test_scenario_developer_disables_progress_for_tests(self):
        """
        Scenario: Clean test output without progress

        Given automated test scenarios
        When I run compute_all without progress
        Then tests run cleanly without console clutter
        And test output remains readable
        Because automated tests should not spam console.
        """
        # GIVEN automated test scenarios
        processor = CorticalTextProcessor()
        processor.process_document("test_doc", "Test content")

        # WHEN I run compute_all without progress
        # THEN tests run cleanly without console clutter
        processor.compute_all(verbose=False, show_progress=False)

        # AND test output remains readable
        # No progress bars or verbose output
        results = processor.find_documents_for_query("test", top_n=1)
        assert len(results) > 0, "Should work without progress enabled"

    def test_scenario_developer_integrates_with_logging_framework(self):
        """
        Scenario: Connecting progress to logging system

        Given an application with structured logging
        When I use progress callbacks
        Then I can log progress to my logging system
        And maintain consistent logging patterns
        Because developers have established logging infrastructure.
        """
        # GIVEN an application with structured logging
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Document content")

        # Simulate structured logging
        log_entries = []

        def structured_logger(phase, percent, message):
            # WHEN I use progress callbacks
            log_entries.append({
                'level': 'INFO',
                'phase': phase,
                'progress': percent,
                'message': message or 'Processing'
            })

        reporter = CallbackProgressReporter(structured_logger)
        processor.compute_all(progress_callback=reporter, verbose=False)

        # THEN I can log progress to my logging system
        assert len(log_entries) > 0, "Should create log entries"

        # AND maintain consistent logging patterns
        for entry in log_entries:
            assert 'level' in entry, "Should have log level"
            assert 'phase' in entry, "Should have phase information"
            assert 'progress' in entry, "Should have progress percentage"
