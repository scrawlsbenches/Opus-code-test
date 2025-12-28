"""
Evolutionary Algorithm for Strategy Improvement

This module implements the evolutionary layer that enables self-improvement:
- Survey: Instrument and observe executions
- Study: Analyze traces and attribute outcomes to strategies
- Evolve: Select, crossover, mutate, and propagate strategies

The "genetic material" is cognitive strategies, not weights:
- Decomposition patterns
- Delegation strategies
- Context compression methods
- Coordination protocols
- Failure recovery strategies
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Iterator, Literal

from .types import (
    AgentNode,
    AgentTree,
    Event,
    EventBus,
    Result,
)


# =============================================================================
# STRATEGY GENOME
# =============================================================================


@dataclass
class DecompositionPattern:
    """Pattern for breaking down goals into tasks."""

    name: str
    goal_type: str  # e.g., "feature", "bugfix", "refactor"
    phases: list[str]  # e.g., ["research", "design", "implement", "test"]
    parallel_phases: list[list[str]] = field(default_factory=list)


@dataclass
class DelegationStrategy:
    """Strategy for assigning work to workers."""

    name: str
    parallelism: Literal["sequential", "parallel", "adaptive"]
    max_concurrent: int = 3
    batch_size: int = 1


@dataclass
class CompressionMethod:
    """Method for compressing context for sub-agents."""

    name: str
    max_tokens: int = 2000
    include_decisions: bool = True
    include_rationale: bool = False


@dataclass
class CoordinationProtocol:
    """Protocol for coordinating workers."""

    name: str
    check_in_interval: timedelta = field(
        default_factory=lambda: timedelta(minutes=5)
    )
    interrupt_on_blocker: bool = True
    swarm_on_block: bool = True


@dataclass
class FailureStrategy:
    """Strategy for handling failures."""

    name: str
    max_retries: int = 3
    backoff_multiplier: float = 2.0
    escalate_after: int = 2


@dataclass
class SynthesisPattern:
    """Pattern for combining outputs."""

    name: str
    merge_strategy: Literal["sequential", "semantic", "priority"]
    conflict_resolution: Literal["first", "last", "highest_confidence"]


@dataclass
class StrategyGenome:
    """
    The heritable traits that define how agents operate.

    This is the "genetic material" that evolves over time.
    """

    genome_id: str

    # Decomposition genes
    decomposition_patterns: list[DecompositionPattern] = field(
        default_factory=list
    )

    # Delegation genes
    delegation_strategies: list[DelegationStrategy] = field(
        default_factory=list
    )

    # Context genes
    context_compression_methods: list[CompressionMethod] = field(
        default_factory=list
    )

    # Coordination genes
    coordination_protocols: list[CoordinationProtocol] = field(
        default_factory=list
    )

    # Recovery genes
    failure_strategies: list[FailureStrategy] = field(default_factory=list)

    # Synthesis genes
    synthesis_patterns: list[SynthesisPattern] = field(default_factory=list)

    # Meta genes
    exploration_rate: float = 0.1
    confidence_threshold: float = 0.7
    parallelism_preference: float = 0.5

    # Tracking
    fitness_history: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def genes(self) -> Iterator[tuple[str, Any]]:
        """Iterate over all genes."""
        yield "decomposition_patterns", self.decomposition_patterns
        yield "delegation_strategies", self.delegation_strategies
        yield "context_compression_methods", self.context_compression_methods
        yield "coordination_protocols", self.coordination_protocols
        yield "failure_strategies", self.failure_strategies
        yield "synthesis_patterns", self.synthesis_patterns
        yield "exploration_rate", self.exploration_rate
        yield "confidence_threshold", self.confidence_threshold
        yield "parallelism_preference", self.parallelism_preference

    def copy(self) -> StrategyGenome:
        """Create a copy of this genome."""
        return StrategyGenome(
            genome_id=f"{self.genome_id}-copy",
            decomposition_patterns=list(self.decomposition_patterns),
            delegation_strategies=list(self.delegation_strategies),
            context_compression_methods=list(self.context_compression_methods),
            coordination_protocols=list(self.coordination_protocols),
            failure_strategies=list(self.failure_strategies),
            synthesis_patterns=list(self.synthesis_patterns),
            exploration_rate=self.exploration_rate,
            confidence_threshold=self.confidence_threshold,
            parallelism_preference=self.parallelism_preference,
        )


# =============================================================================
# STRATEGY POOL
# =============================================================================


class StrategyPool:
    """Pool of strategy genomes for selection."""

    def __init__(self):
        self._genomes: dict[str, StrategyGenome] = {}
        self._fitness: dict[str, float] = {}
        self._generation = 0

    def add(self, genome: StrategyGenome) -> None:
        """Add a genome to the pool."""
        self._genomes[genome.genome_id] = genome

    def get(self, genome_id: str) -> StrategyGenome | None:
        """Get a genome by ID."""
        return self._genomes.get(genome_id)

    def get_random(self) -> StrategyGenome | None:
        """Get a random genome."""
        if not self._genomes:
            return None
        return random.choice(list(self._genomes.values()))

    def get_best_for(self, goal_type: str) -> StrategyGenome | None:
        """Get the best genome for a goal type."""
        if not self._genomes:
            return None

        # Return highest fitness
        best_id = max(self._fitness.keys(), key=lambda k: self._fitness[k])
        return self._genomes.get(best_id)

    def get_current_generation(self) -> list[StrategyGenome]:
        """Get all genomes in current generation."""
        return list(self._genomes.values())

    def update(
        self,
        survivors: list[StrategyGenome],
        offspring: list[StrategyGenome],
        fitness_scores: dict[str, float],
    ) -> list[StrategyGenome]:
        """Update pool with new generation."""
        self._generation += 1

        # Clear and repopulate
        self._genomes.clear()
        for genome in survivors + offspring:
            self._genomes[genome.genome_id] = genome

        self._fitness = fitness_scores
        return list(self._genomes.values())

    def update_fitness(self, genome_id: str, fitness: float) -> None:
        """Update fitness for a genome."""
        self._fitness[genome_id] = fitness
        if genome_id in self._genomes:
            self._genomes[genome_id].fitness_history.append(fitness)


# =============================================================================
# EXECUTION METRICS AND TRACES
# =============================================================================


@dataclass
class ExecutionMetrics:
    """Metrics collected during execution."""

    # Efficiency
    total_duration_ms: float = 0.0
    agent_count: int = 0
    tool_calls: int = 0
    context_tokens_used: int = 0

    # Quality
    goal_achieved: bool = False
    completeness_score: float = 0.0
    correctness_score: float = 0.0

    # Coordination
    escalation_count: int = 0
    blocker_count: int = 0
    recovery_success_rate: float = 0.0

    # Resource efficiency
    parallel_utilization: float = 0.0
    redundant_work_ratio: float = 0.0
    context_efficiency: float = 0.0

    # Stability
    retry_count: int = 0
    error_count: int = 0
    checkpoint_count: int = 0


@dataclass
class ExecutionTrace:
    """Complete record of an execution for evolutionary analysis."""

    trace_id: str
    goal: str
    strategy_genome_id: str

    # Structure
    agent_tree: AgentTree = field(default_factory=AgentTree)
    event_log: list[Event] = field(default_factory=list)

    # Measurements
    metrics: ExecutionMetrics = field(default_factory=ExecutionMetrics)

    # Outcome
    result: Result | None = None
    user_feedback: dict[str, Any] | None = None


# =============================================================================
# FITNESS SCORING
# =============================================================================


@dataclass
class FitnessScore:
    """Multi-objective fitness score."""

    # Core objectives
    success: float = 0.0
    efficiency: float = 0.0
    quality: float = 0.0

    # Secondary objectives
    stability: float = 0.0
    elegance: float = 0.0

    # User-weighted
    user_satisfaction: float = 0.0

    def aggregate(self, weights: dict[str, float] | None = None) -> float:
        """Aggregate into single score."""
        if weights is None:
            weights = {
                "success": 0.3,
                "efficiency": 0.2,
                "quality": 0.2,
                "stability": 0.1,
                "elegance": 0.1,
                "user_satisfaction": 0.1,
            }

        return sum(
            getattr(self, key) * weight
            for key, weight in weights.items()
        )


@dataclass
class Attribution:
    """Attribution of outcome to a specific gene."""

    gene: str
    contribution: float  # -1 to 1, negative = harmful
    evidence: list[str] = field(default_factory=list)


@dataclass
class StrategyAttribution:
    """Full attribution of an execution to strategy genes."""

    trace_id: str
    genome_id: str
    overall_fitness: FitnessScore
    gene_attributions: list[Attribution] = field(default_factory=list)


# =============================================================================
# EXECUTION SURVEYOR
# =============================================================================


class ExecutionSurveyor:
    """Instruments and observes executions."""

    def __init__(self, event_bus: EventBus | None = None):
        self.event_bus = event_bus or EventBus()
        self.traces: dict[str, ExecutionTrace] = {}

        # Subscribe to all events
        if self.event_bus:
            self.event_bus.subscribe("*", self.record_event)

    def start_trace(self, goal: str, genome_id: str) -> str:
        """Begin tracing an execution."""
        trace_id = f"trace-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        self.traces[trace_id] = ExecutionTrace(
            trace_id=trace_id,
            goal=goal,
            strategy_genome_id=genome_id,
        )

        return trace_id

    def record_event(self, event: Event) -> None:
        """Record an event to its trace."""
        trace_id = event.trace_id
        if trace_id and trace_id in self.traces:
            self.traces[trace_id].event_log.append(event)
            self._update_metrics(self.traces[trace_id], event)

    def _update_metrics(self, trace: ExecutionTrace, event: Event) -> None:
        """Update metrics based on event."""
        if "agent.spawned" in event.type:
            trace.metrics.agent_count += 1
        elif "error" in event.type:
            trace.metrics.error_count += 1
        elif "blocker" in event.type:
            trace.metrics.blocker_count += 1
        elif "escalation" in event.type:
            trace.metrics.escalation_count += 1

    def finalize_trace(
        self,
        trace_id: str,
        result: Result,
        feedback: dict[str, Any] | None = None,
    ) -> ExecutionTrace:
        """Complete a trace with outcome data."""
        if trace_id not in self.traces:
            raise ValueError(f"Unknown trace: {trace_id}")

        trace = self.traces[trace_id]
        trace.result = result
        trace.user_feedback = feedback
        trace.metrics.goal_achieved = result.success

        return trace


# =============================================================================
# STRATEGY ANALYZER
# =============================================================================


class StrategyAnalyzer:
    """Analyzes execution traces to understand strategy effectiveness."""

    def compute_fitness(self, trace: ExecutionTrace) -> FitnessScore:
        """Compute multi-objective fitness from a trace."""
        metrics = trace.metrics

        return FitnessScore(
            success=1.0 if metrics.goal_achieved else 0.0,
            efficiency=self._compute_efficiency(metrics),
            quality=metrics.completeness_score,
            stability=1.0 - (metrics.error_count / max(metrics.agent_count, 1)),
            elegance=1.0 - (metrics.redundant_work_ratio),
            user_satisfaction=(
                trace.user_feedback.get("satisfaction", 0.5)
                if trace.user_feedback else 0.5
            ),
        )

    def _compute_efficiency(self, metrics: ExecutionMetrics) -> float:
        """Compute efficiency score."""
        # Lower is better for duration, normalize to 0-1
        duration_score = max(0, 1 - (metrics.total_duration_ms / 300000))

        # Fewer agents is better (for same output)
        agent_score = max(0, 1 - (metrics.agent_count / 20))

        return (duration_score + agent_score) / 2

    def attribute_outcomes(
        self,
        trace: ExecutionTrace,
    ) -> StrategyAttribution:
        """Attribute success/failure to specific strategy choices."""
        attributions = []
        fitness = self.compute_fitness(trace)

        # Analyze each gene type
        # (Simplified - real implementation would analyze event patterns)

        # Decomposition effectiveness
        decomp_score = self._analyze_decomposition(trace)
        attributions.append(Attribution(
            gene="decomposition",
            contribution=decomp_score,
            evidence=["Based on phase completion rate"],
        ))

        # Delegation effectiveness
        deleg_score = self._analyze_delegation(trace)
        attributions.append(Attribution(
            gene="delegation",
            contribution=deleg_score,
            evidence=["Based on parallelism utilization"],
        ))

        # Coordination effectiveness
        coord_score = self._analyze_coordination(trace)
        attributions.append(Attribution(
            gene="coordination",
            contribution=coord_score,
            evidence=["Based on blocker resolution time"],
        ))

        return StrategyAttribution(
            trace_id=trace.trace_id,
            genome_id=trace.strategy_genome_id,
            overall_fitness=fitness,
            gene_attributions=attributions,
        )

    def _analyze_decomposition(self, trace: ExecutionTrace) -> float:
        """Analyze decomposition effectiveness."""
        # Placeholder - would analyze event patterns
        if trace.result and trace.result.success:
            return 0.8
        return 0.2

    def _analyze_delegation(self, trace: ExecutionTrace) -> float:
        """Analyze delegation effectiveness."""
        return trace.metrics.parallel_utilization

    def _analyze_coordination(self, trace: ExecutionTrace) -> float:
        """Analyze coordination effectiveness."""
        if trace.metrics.blocker_count == 0:
            return 1.0
        return trace.metrics.recovery_success_rate

    def compare_strategies(
        self,
        traces: list[ExecutionTrace],
        control_variable: str,
    ) -> dict[str, Any]:
        """Compare strategies on similar goals."""
        # Group by strategy
        by_strategy: dict[str, list[ExecutionTrace]] = {}
        for trace in traces:
            sid = trace.strategy_genome_id
            if sid not in by_strategy:
                by_strategy[sid] = []
            by_strategy[sid].append(trace)

        # Compare fitness distributions
        comparisons = {}
        for sid, strategy_traces in by_strategy.items():
            fitness_scores = [
                self.compute_fitness(t).aggregate()
                for t in strategy_traces
            ]
            comparisons[sid] = {
                "count": len(strategy_traces),
                "avg_fitness": sum(fitness_scores) / len(fitness_scores),
                "min_fitness": min(fitness_scores),
                "max_fitness": max(fitness_scores),
            }

        return comparisons

    def identify_patterns(
        self,
        traces: list[ExecutionTrace],
    ) -> list[dict[str, Any]]:
        """Discover emergent patterns across executions."""
        patterns = []

        # Success patterns
        successful = [t for t in traces if t.result and t.result.success]
        if successful:
            patterns.append({
                "type": "success_pattern",
                "count": len(successful),
                "common_traits": self._find_common_traits(successful),
            })

        # Failure patterns
        failed = [t for t in traces if t.result and not t.result.success]
        if failed:
            patterns.append({
                "type": "failure_pattern",
                "count": len(failed),
                "common_traits": self._find_common_traits(failed),
            })

        return patterns

    def _find_common_traits(
        self,
        traces: list[ExecutionTrace],
    ) -> list[str]:
        """Find common traits across traces."""
        # Placeholder
        return []


# =============================================================================
# STRATEGY EVOLVER
# =============================================================================


@dataclass
class EvolutionResult:
    """Result of an evolution generation."""

    generation: int
    population_size: int
    best_fitness: FitnessScore
    avg_fitness: float
    novel_strategies_added: int
    strategies_retired: int


class EvolutionHistory:
    """History of evolution generations."""

    def __init__(self):
        self.generations: list[EvolutionResult] = []

    def record(self, result: EvolutionResult) -> None:
        """Record a generation result."""
        self.generations.append(result)

    def last_n_generations(self, n: int) -> list[EvolutionResult]:
        """Get last N generations."""
        return self.generations[-n:]


class StrategyEvolver:
    """Evolves the strategy pool over time."""

    def __init__(self, strategy_pool: StrategyPool):
        self.pool = strategy_pool
        self.analyzer = StrategyAnalyzer()
        self.generation = 0
        self.history = EvolutionHistory()

    # =========================================================================
    # SELECTION
    # =========================================================================

    def select_parents(
        self,
        population: list[StrategyGenome],
        fitness_scores: dict[str, FitnessScore],
        selection_pressure: float = 0.3,
    ) -> list[StrategyGenome]:
        """Select high-fitness strategies for reproduction."""
        selected = []

        # Tournament selection
        tournament_size = max(2, int(len(population) * selection_pressure))

        for _ in range(len(population) // 2):
            tournament = random.sample(population, min(tournament_size, len(population)))
            winner = max(
                tournament,
                key=lambda g: fitness_scores.get(g.genome_id, FitnessScore()).aggregate()
            )
            selected.append(winner)

        # Preserve diversity
        diversity_slots = max(1, int(len(population) * 0.1))
        remaining = [g for g in population if g not in selected]
        if remaining:
            selected.extend(random.sample(
                remaining,
                min(diversity_slots, len(remaining))
            ))

        return selected

    # =========================================================================
    # CROSSOVER
    # =========================================================================

    def crossover(
        self,
        parent_a: StrategyGenome,
        parent_b: StrategyGenome,
    ) -> StrategyGenome:
        """Combine two successful strategies."""
        child_id = f"genome-{datetime.now().strftime('%Y%m%d%H%M%S')}-{random.randint(0, 9999)}"

        child = StrategyGenome(
            genome_id=child_id,
            # For list genes, randomly choose from parents
            decomposition_patterns=random.choice([
                parent_a.decomposition_patterns,
                parent_b.decomposition_patterns,
            ]),
            delegation_strategies=random.choice([
                parent_a.delegation_strategies,
                parent_b.delegation_strategies,
            ]),
            context_compression_methods=random.choice([
                parent_a.context_compression_methods,
                parent_b.context_compression_methods,
            ]),
            coordination_protocols=random.choice([
                parent_a.coordination_protocols,
                parent_b.coordination_protocols,
            ]),
            failure_strategies=random.choice([
                parent_a.failure_strategies,
                parent_b.failure_strategies,
            ]),
            synthesis_patterns=random.choice([
                parent_a.synthesis_patterns,
                parent_b.synthesis_patterns,
            ]),
            # For numeric genes, average
            exploration_rate=(
                parent_a.exploration_rate + parent_b.exploration_rate
            ) / 2,
            confidence_threshold=random.choice([
                parent_a.confidence_threshold,
                parent_b.confidence_threshold,
            ]),
            parallelism_preference=(
                parent_a.parallelism_preference + parent_b.parallelism_preference
            ) / 2,
        )

        return child

    # =========================================================================
    # MUTATION
    # =========================================================================

    def mutate(
        self,
        genome: StrategyGenome,
        mutation_rate: float = 0.1,
    ) -> StrategyGenome:
        """Introduce variations to explore new strategies."""
        mutated = genome.copy()
        mutated.genome_id = f"{genome.genome_id}-mutated"

        # Mutate numeric genes with small perturbations
        if random.random() < mutation_rate:
            mutated.exploration_rate = max(0, min(1,
                mutated.exploration_rate + random.gauss(0, 0.05)
            ))

        if random.random() < mutation_rate:
            mutated.confidence_threshold = max(0, min(1,
                mutated.confidence_threshold + random.gauss(0, 0.05)
            ))

        if random.random() < mutation_rate:
            mutated.parallelism_preference = max(0, min(1,
                mutated.parallelism_preference + random.gauss(0, 0.1)
            ))

        return mutated

    # =========================================================================
    # NOVEL GENERATION
    # =========================================================================

    def generate_novel_strategy(
        self,
        inspiration_traces: list[ExecutionTrace],
    ) -> StrategyGenome:
        """Generate a genuinely new strategy."""
        # Analyze successful patterns
        patterns = self.analyzer.identify_patterns(inspiration_traces)

        # Create novel genome
        novel = StrategyGenome(
            genome_id=f"novel-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            # Would use patterns to inform gene choices
            exploration_rate=random.uniform(0.05, 0.2),
            confidence_threshold=random.uniform(0.5, 0.9),
            parallelism_preference=random.uniform(0.3, 0.8),
        )

        novel.metadata["origin"] = "novel_generation"
        novel.metadata["patterns_used"] = len(patterns)

        return novel

    # =========================================================================
    # EVOLUTION CYCLE
    # =========================================================================

    def evolve_generation(
        self,
        traces: list[ExecutionTrace],
    ) -> EvolutionResult:
        """Complete one evolution cycle."""
        self.generation += 1

        # 1. Evaluate fitness
        fitness_scores: dict[str, FitnessScore] = {}
        for trace in traces:
            fitness = self.analyzer.compute_fitness(trace)
            fitness_scores[trace.strategy_genome_id] = fitness

        # 2. Get current population
        population = self.pool.get_current_generation()
        if not population:
            # Bootstrap with novel strategies
            for _ in range(10):
                novel = self.generate_novel_strategy(traces)
                self.pool.add(novel)
            population = self.pool.get_current_generation()

        # 3. Select parents
        parents = self.select_parents(population, fitness_scores)

        # 4. Generate offspring
        offspring = []

        # Crossover
        for i in range(0, len(parents) - 1, 2):
            child = self.crossover(parents[i], parents[i + 1])
            child = self.mutate(child)
            offspring.append(child)

        # Novel strategies
        novel_count = max(1, int(len(population) * 0.05))
        for _ in range(novel_count):
            novel = self.generate_novel_strategy(traces)
            offspring.append(novel)

        # 5. Update pool
        new_generation = self.pool.update(
            survivors=parents,
            offspring=offspring,
            fitness_scores={
                g.genome_id: fitness_scores.get(g.genome_id, FitnessScore()).aggregate()
                for g in parents + offspring
            },
        )

        # 6. Record history
        best_fitness = max(
            fitness_scores.values(),
            key=lambda f: f.aggregate()
        ) if fitness_scores else FitnessScore()

        avg_fitness = (
            sum(f.aggregate() for f in fitness_scores.values()) /
            len(fitness_scores)
        ) if fitness_scores else 0.0

        result = EvolutionResult(
            generation=self.generation,
            population_size=len(new_generation),
            best_fitness=best_fitness,
            avg_fitness=avg_fitness,
            novel_strategies_added=novel_count,
            strategies_retired=len(population) - len(parents),
        )

        self.history.record(result)
        return result


# =============================================================================
# EVOLUTION SAFEGUARDS
# =============================================================================


@dataclass
class GoldenStrategy:
    """A strategy that must always be preserved."""

    genome: StrategyGenome
    reason: str
    defined_at: datetime = field(default_factory=datetime.now)


@dataclass
class RegressionTest:
    """A test that all strategies must pass."""

    name: str
    test_fn: Callable[[StrategyGenome], bool]
    threshold: float = 0.9


@dataclass
class ValidationResult:
    """Result of validating a generation."""

    valid: bool
    issues: list[str] = field(default_factory=list)


class EvolutionSafeguards:
    """Prevent evolutionary regression."""

    def __init__(self):
        self.golden_strategies: list[GoldenStrategy] = []
        self.regression_tests: list[RegressionTest] = []

    def validate_new_generation(
        self,
        new_generation: list[StrategyGenome],
        old_generation: list[StrategyGenome],
    ) -> ValidationResult:
        """Ensure new generation isn't worse."""
        issues = []

        # Check: best strategy preserved (elitism)
        if old_generation:
            old_best = max(
                old_generation,
                key=lambda g: (
                    sum(g.fitness_history[-5:]) / max(len(g.fitness_history[-5:]), 1)
                    if g.fitness_history else 0
                )
            )
            if old_best.genome_id not in [g.genome_id for g in new_generation]:
                issues.append("best_strategy_lost")

        # Check: diversity maintained
        old_diversity = self._compute_diversity(old_generation)
        new_diversity = self._compute_diversity(new_generation)
        if new_diversity < old_diversity * 0.5:
            issues.append("diversity_collapse")

        # Check: golden strategies preserved
        for golden in self.golden_strategies:
            if not self._strategy_covered(golden.genome, new_generation):
                issues.append(f"golden_strategy_lost: {golden.genome.genome_id}")

        # Check: regression tests pass
        for test in self.regression_tests:
            if not self._passes_regression(new_generation, test):
                issues.append(f"regression: {test.name}")

        return ValidationResult(
            valid=len(issues) == 0,
            issues=issues,
        )

    def define_golden_strategy(
        self,
        genome: StrategyGenome,
        reason: str,
    ) -> None:
        """Mark a strategy as golden - must always be preserved."""
        self.golden_strategies.append(GoldenStrategy(
            genome=genome,
            reason=reason,
        ))

    def add_regression_test(self, test: RegressionTest) -> None:
        """Add a regression test."""
        self.regression_tests.append(test)

    def _compute_diversity(
        self,
        population: list[StrategyGenome],
    ) -> float:
        """Compute population diversity."""
        if len(population) < 2:
            return 1.0

        # Simple diversity: variance in meta-genes
        exploration_rates = [g.exploration_rate for g in population]
        confidence_thresholds = [g.confidence_threshold for g in population]

        var_exploration = self._variance(exploration_rates)
        var_confidence = self._variance(confidence_thresholds)

        return (var_exploration + var_confidence) / 2

    def _variance(self, values: list[float]) -> float:
        """Compute variance."""
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values) / len(values)

    def _strategy_covered(
        self,
        target: StrategyGenome,
        population: list[StrategyGenome],
    ) -> bool:
        """Check if a strategy's traits are covered in population."""
        # Check if genome or similar exists
        for genome in population:
            if genome.genome_id == target.genome_id:
                return True
            # Could also check for similar trait values
        return False

    def _passes_regression(
        self,
        population: list[StrategyGenome],
        test: RegressionTest,
    ) -> bool:
        """Check if population passes a regression test."""
        passing = sum(1 for g in population if test.test_fn(g))
        pass_rate = passing / len(population) if population else 0
        return pass_rate >= test.threshold
