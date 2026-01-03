#!/usr/bin/env python3
"""
Example: Genetic Algorithm for Strategy Evolution

This example demonstrates the completed genetic algorithm implementation
with crossover, mutation, elitism, and diversity maintenance.
"""

from llm_orchestration.evolution import (
    StrategyEvolver,
    StrategyPool,
    StrategyGenome,
    ExecutionTrace,
    ExecutionMetrics,
    Result,
    FitnessScore,
)


def create_sample_genome(genome_id: str, exploration: float = 0.1) -> StrategyGenome:
    """Create a sample genome with specified parameters."""
    return StrategyGenome(
        genome_id=genome_id,
        exploration_rate=exploration,
        confidence_threshold=0.7,
        parallelism_preference=0.5,
    )


def create_sample_trace(genome_id: str, success: bool = True) -> ExecutionTrace:
    """Create a sample execution trace for testing."""
    trace = ExecutionTrace(
        trace_id=f"trace-{genome_id}",
        goal="Sample goal",
        strategy_genome_id=genome_id,
    )

    # Set metrics
    trace.metrics.goal_achieved = success
    trace.metrics.completeness_score = 0.9 if success else 0.4
    trace.metrics.agent_count = 5
    trace.metrics.total_duration_ms = 10000

    # Set result
    trace.result = Result(
        success=success,
        output="Sample output",
    )

    return trace


def main():
    print("="*70)
    print("GENETIC ALGORITHM EXAMPLE")
    print("="*70)
    print()

    # 1. Initialize pool and evolver
    pool = StrategyPool()
    evolver = StrategyEvolver(pool)

    print("1. Creating initial population...")
    genomes = [
        create_sample_genome("genome-1", exploration=0.1),
        create_sample_genome("genome-2", exploration=0.15),
        create_sample_genome("genome-3", exploration=0.2),
        create_sample_genome("genome-4", exploration=0.25),
    ]

    for genome in genomes:
        pool.add(genome)

    print(f"   ✓ Created {len(genomes)} genomes")
    print()

    # 2. Test crossover operators
    print("2. Testing crossover operators...")
    parent1, parent2 = genomes[0], genomes[1]

    child_sp = evolver.crossover_single_point(parent1, parent2)
    print(f"   ✓ Single-point: exploration={child_sp.exploration_rate:.3f}")

    child_uniform = evolver.crossover_uniform(parent1, parent2)
    print(f"   ✓ Uniform: exploration={child_uniform.exploration_rate:.3f}")

    child_blend = evolver.crossover_blend(parent1, parent2, alpha=0.5)
    print(f"   ✓ Blend: exploration={child_blend.exploration_rate:.3f}")
    print()

    # 3. Test mutation operators
    print("3. Testing mutation operators...")

    mutated_gauss = evolver.mutate_gaussian(genomes[0], mutation_rate=1.0)
    print(f"   ✓ Gaussian: {genomes[0].exploration_rate:.3f} → {mutated_gauss.exploration_rate:.3f}")

    mutated_uniform = evolver.mutate_uniform(genomes[0], mutation_rate=1.0)
    print(f"   ✓ Uniform: {genomes[0].exploration_rate:.3f} → {mutated_uniform.exploration_rate:.3f}")

    mutated_adaptive = evolver.mutate_adaptive(genomes[0], genomes)
    print(f"   ✓ Adaptive: {genomes[0].exploration_rate:.3f} → {mutated_adaptive.exploration_rate:.3f}")
    print()

    # 4. Test diversity computation
    print("4. Measuring population diversity...")
    diversity = evolver._compute_diversity(genomes)
    print(f"   ✓ Diversity score: {diversity:.3f}")
    print()

    # 5. Test elitism
    print("5. Testing elitism...")
    fitness_scores = {
        "genome-1": FitnessScore(success=0.9, efficiency=0.8, quality=0.85),
        "genome-2": FitnessScore(success=0.7, efficiency=0.6, quality=0.65),
        "genome-3": FitnessScore(success=0.95, efficiency=0.9, quality=0.92),
        "genome-4": FitnessScore(success=0.5, efficiency=0.4, quality=0.45),
    }

    elites = evolver.select_elites(genomes, fitness_scores, elite_count=2)
    print(f"   ✓ Selected {len(elites)} elites:")
    for elite in elites:
        score = fitness_scores[elite.genome_id].aggregate()
        print(f"      - {elite.genome_id}: fitness={score:.3f}")
    print()

    # 6. Run full evolution cycle
    print("6. Running full evolution cycle...")

    # Create traces for each genome
    traces = [
        create_sample_trace("genome-1", success=True),
        create_sample_trace("genome-2", success=True),
        create_sample_trace("genome-3", success=True),
        create_sample_trace("genome-4", success=False),
    ]

    # Evolve one generation
    result = evolver.evolve_generation(
        traces=traces,
        elite_count=2,
        min_diversity=0.2,
        crossover_method="blend",
        mutation_method="adaptive",
    )

    print(f"   ✓ Generation {result.generation} complete:")
    print(f"      - Population size: {result.population_size}")
    print(f"      - Best fitness: {result.best_fitness.aggregate():.3f}")
    print(f"      - Average fitness: {result.avg_fitness:.3f}")
    print(f"      - Novel strategies added: {result.novel_strategies_added}")
    print(f"      - Strategies retired: {result.strategies_retired}")
    print()

    # 7. Demonstrate evolution over multiple generations
    print("7. Evolving over 3 generations...")

    for gen in range(3):
        # Create new traces (simulating new executions)
        new_traces = [
            create_sample_trace(f"trace-gen{gen}-{i}", success=(i % 2 == 0))
            for i in range(4)
        ]

        result = evolver.evolve_generation(
            traces=new_traces,
            elite_count=2,
            min_diversity=0.15,
            crossover_method="blend",
            mutation_method="adaptive",
        )

        print(f"   Gen {result.generation}: "
              f"pop={result.population_size}, "
              f"best={result.best_fitness.aggregate():.3f}, "
              f"avg={result.avg_fitness:.3f}")

    print()

    # 8. Show evolution history
    print("8. Evolution history:")
    history = evolver.history.last_n_generations(4)
    print(f"   ✓ Tracked {len(history)} generations")
    for i, gen_result in enumerate(history):
        print(f"      Gen {gen_result.generation}: "
              f"avg_fitness={gen_result.avg_fitness:.3f}")
    print()

    print("="*70)
    print("✅ GENETIC ALGORITHM DEMONSTRATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
