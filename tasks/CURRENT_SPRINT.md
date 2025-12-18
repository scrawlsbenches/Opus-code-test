# Current Sprint

**Sprint ID:** sprint-005-ux-documentation
**Epic:** Hubris MoE System (efba)
**Started:** 2025-12-18
**Status:** In Progress

## Goals
- [ ] Fix cold-start UX - clearer message when experts untrained (T-dbf8-014)
- [ ] Complete Hubris documentation - book chapter + README (T-2e92-002)
- [ ] Add calibration status to CLI dashboard

## In Progress
(None yet)

## Completed This Sprint
(None yet)

## Blocked
(None currently)

## Notes
- Sprint 4 completed all meta-learning goals
- Calibration tracking now live (Phase 1+2 done)
- Focus on user experience and documentation this sprint
- Cold-start UX: address "Unknown cause" output with fallback to ML model

---

# Previous Sprints

## Sprint 4: Meta-Learning (Complete)
**Dates:** 2025-12-18
**Epic:** Hubris MoE System (efba)

Completed:
- ✅ Wire feedback_collector to live git hooks
- ✅ Add EXPERIMENTAL banner to predictions
- ✅ Create sprint tracking persistence
- ✅ Add git lock detection
- ✅ Exception handling audit (9 fixes)
- ✅ Calibration tracking Phase 1+2 (ResolvedPrediction + CalibrationTracker)

## Sprint 3: MoE Integration (Complete)
**Dates:** 2025-12-15 to 2025-12-17
**Epic:** Hubris MoE System (efba)

Completed:
- ✅ CLI interface (hubris_cli.py)
- ✅ Feedback collector (feedback_collector.py)
- ✅ README documentation
- ✅ Integration tests

## Sprint 2: Credit System (Complete)
**Dates:** 2025-12-13 to 2025-12-14
**Epic:** Hubris MoE System (efba)

Completed:
- ✅ CreditAccount, CreditLedger
- ✅ ValueSignal, ValueAttributor
- ✅ CreditRouter, Staking
- ✅ Credit propagation algorithms

## Sprint 1: Expert Foundation (Complete)
**Dates:** 2025-12-10 to 2025-12-12
**Epic:** Hubris MoE System (efba)

Completed:
- ✅ MicroExpert base class
- ✅ FileExpert, TestExpert
- ✅ ErrorDiagnosisExpert, EpisodeExpert
- ✅ ExpertConsolidator
- ✅ Expert routing system

---

# Epics

## Active: Hubris MoE (efba)
**Started:** 2025-12-10
**Status:** Phase 4 - Meta-Learning

Building a mixture of experts system that learns from usage.

### Phases:
- **Phase 1:** Expert foundation ✅ (Sprint 1)
- **Phase 2:** Credit system ✅ (Sprint 2)
- **Phase 3:** Integration ✅ (Sprint 3)
- **Phase 4:** Meta-learning 🔄 (Sprint 4 - current)
  - Sprint tracking persistence
  - Live feedback loops
  - Git lock detection
  - Experimental feature flagging

### Success Criteria:
- [ ] System learns from real usage data
- [ ] Expert credits update based on outcomes
- [ ] Safe concurrent operation with git
- [ ] Clear experimental feature boundaries

## Backlog: Future Epics

### LEGACY Modernization
- Async API support
- REST wrapper
- Plugin registry system

### Quality Sweep
- Exception handling audit
- Magic number elimination
- Type coverage improvements
- Documentation completeness

---

# How to Use This File

**For Claude at Session Start:**
Read this file to understand:
- What sprint we're in
- What tasks are in progress
- What's been completed recently
- What's blocked

**For Humans:**
Update this file when:
- Starting a new sprint
- Completing sprint goals
- Discovering blockers
- Making strategic decisions

**Update Commands:**
```bash
# View current sprint status
python scripts/task_utils.py sprint status

# Mark a sprint goal complete
python scripts/task_utils.py sprint complete "goal description"

# Add a note to current sprint
python scripts/task_utils.py sprint note "your note here"
```

---

# Sprint History Stats

| Sprint | Duration | Tasks | Commits | Files Changed |
|--------|----------|-------|---------|---------------|
| Sprint 3 | 3 days | 8 | 15 | 12 |
| Sprint 2 | 2 days | 6 | 10 | 8 |
| Sprint 1 | 3 days | 7 | 12 | 15 |
| **Total** | **8 days** | **21** | **37** | **35** |
