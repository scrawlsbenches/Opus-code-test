# Genetic Algorithm API Reference

## Overview

The `llm_orchestration.evolution` module implements a complete genetic algorithm for evolving cognitive strategies. This implementation includes crossover operators, mutation operators, elitism, diversity maintenance, and fitness evaluation.

## Core Classes

### StrategyGenome

The heritable traits that define how agents operate.

```python
@dataclass
class StrategyGenome:
    genome_id: str

    # Strategy genes
    decomposition_patterns: list[DecompositionPattern]
    delegation_strategies: list[DelegationStrategy]
    context_compression_methods: list[CompressionMethod]
    coordination_protocols: list[CoordinationProtocol]
    failure_strategies: list[FailureStrategy]
    synthesis_patterns: list[SynthesisPattern]

    # Meta genes (continuous values in [0, 1])
    exploration_rate: float = 0.1
    confidence_threshold: float = 0.7
    parallelism_preference: float = 0.5

    # Tracking
    fitness_history: list[float]
    metadata: dict[str, Any]
```

### StrategyEvolver

Evolves the strategy pool over time using genetic algorithm operations.

```python
class StrategyEvolver:
    def __init__(self, strategy_pool: StrategyPool):
        self.pool = strategy_pool
        self.analyzer = StrategyAnalyzer()
        self.generation = 0
        self.history = EvolutionHistory()
```

## Crossover Operators

### 1. Single-Point Crossover

Splits genome at a random point and combines parent genes.

```python
def crossover_single_point(
    self,
    parent_a: StrategyGenome,
    parent_b: StrategyGenome,
) -> StrategyGenome:
    """
    Single-point crossover: split genome at random point.

    Example:
        Parent A: [A1, A2, A3, A4, A5]
        Parent B: [B1, B2, B3, B4, B5]
        Point: 3
        Child:    [A1, A2, A3, B4, B5]
    """
```

**Use case**: Good for preserving gene linkage (genes that work well together).

### 2. Uniform Crossover

Randomly selects each gene from either parent independently.

```python
def crossover_uniform(
    self,
    parent_a: StrategyGenome,
    parent_b: StrategyGenome,
) -> StrategyGenome:
    """
    Uniform crossover: randomly select each gene from either parent.

    Example:
        Parent A: [A1, A2, A3, A4, A5]
        Parent B: [B1, B2, B3, B4, B5]
        Child:    [A1, B2, A3, B4, A5]  (50/50 chance per gene)
    """
```

**Use case**: Maximum gene mixing, good for exploring new combinations.

### 3. Blend Crossover

Interpolates continuous genes using alpha parameter.

```python
def crossover_blend(
    self,
    parent_a: StrategyGenome,
    parent_b: StrategyGenome,
    alpha: float = 0.5,
) -> StrategyGenome:
    """
    Blend crossover: interpolate continuous genes, random for discrete.

    For continuous genes:
        child_value = alpha * parent_a_value + (1 - alpha) * parent_b_value

    For discrete genes (lists):
        Randomly choose from either parent

    Args:
        alpha: Blending parameter in [0, 1]
               0.0 = all from parent_b
               0.5 = average of both (default)
               1.0 = all from parent_a
    """
```

**Use case**: Good for continuous optimization, produces children between parents.

### Unified Interface

```python
def crossover(
    self,
    parent_a: StrategyGenome,
    parent_b: StrategyGenome,
    method: str = "blend",
) -> StrategyGenome:
    """
    Combine two successful strategies.

    Args:
        method: "single_point", "uniform", or "blend"
    """
```

## Mutation Operators

### 1. Gaussian Mutation

Small random perturbations with normal distribution.

```python
def mutate_gaussian(
    self,
    genome: StrategyGenome,
    mutation_rate: float = 0.1,
    stddev: float = 0.05,
) -> StrategyGenome:
    """
    Gaussian mutation: small perturbations to numeric genes.

    For each gene:
        if random() < mutation_rate:
            gene_value += gauss(0, stddev)
            gene_value = clamp(gene_value, 0.0, 1.0)

    Args:
        mutation_rate: Probability of mutation per gene
        stddev: Standard deviation of perturbation
    """
```

**Use case**: Fine-tuning, local search around current solution.

**Bounds**: All mutations are clamped to valid ranges:
- `exploration_rate`: [0.0, 1.0]
- `confidence_threshold`: [0.0, 1.0]
- `parallelism_preference`: [0.0, 1.0]

### 2. Uniform Mutation

Random reset within valid bounds.

```python
def mutate_uniform(
    self,
    genome: StrategyGenome,
    mutation_rate: float = 0.1,
) -> StrategyGenome:
    """
    Uniform mutation: random reset of genes within valid bounds.

    For each gene:
        if random() < mutation_rate:
            gene_value = random_uniform(min_bound, max_bound)

    Bounds:
        exploration_rate: [0.0, 0.3]
        confidence_threshold: [0.5, 0.95]
        parallelism_preference: [0.0, 1.0]
    """
```

**Use case**: Large jumps, escaping local optima, exploration.

### 3. Adaptive Mutation

Mutation rate adapts based on population diversity.

```python
def mutate_adaptive(
    self,
    genome: StrategyGenome,
    population: list[StrategyGenome],
    base_rate: float = 0.1,
) -> StrategyGenome:
    """
    Adaptive mutation: rate based on population diversity.

    Low diversity -> higher mutation rate (explore more)
    High diversity -> lower mutation rate (exploit more)

    adapted_rate = base_rate * (3.0 - 2.5 * diversity)

    Where diversity ∈ [0, 1]:
        diversity = 0.0 -> adapted_rate = 3.0 * base_rate
        diversity = 0.5 -> adapted_rate = 1.75 * base_rate
        diversity = 1.0 -> adapted_rate = 0.5 * base_rate

    Clamped to [0.01, 0.5]
    """
```

**Use case**: Self-adaptive exploration/exploitation balance.

### Unified Interface

```python
def mutate(
    self,
    genome: StrategyGenome,
    mutation_rate: float = 0.1,
    method: str = "gaussian",
    population: list[StrategyGenome] | None = None,
) -> StrategyGenome:
    """
    Introduce variations to explore new strategies.

    Args:
        method: "gaussian", "uniform", or "adaptive"
        population: Required for adaptive mutation
    """
```

## Elitism

Preserve top N strategies across generations.

```python
def select_elites(
    self,
    population: list[StrategyGenome],
    fitness_scores: dict[str, FitnessScore],
    elite_count: int = 2,
) -> list[StrategyGenome]:
    """
    Select top N strategies to preserve across generations.

    Ensures the best solutions are never lost due to
    crossover or mutation introducing inferior offspring.

    Args:
        elite_count: Number of top strategies to preserve (default: 2)

    Returns:
        Sorted list of top N genomes by fitness
    """
```

**Benefits**:
- Prevents loss of best solutions
- Guarantees monotonic improvement in best fitness
- Provides stable baseline for comparison

## Diversity Maintenance

### Diversity Measurement

```python
def _compute_diversity(
    self,
    population: list[StrategyGenome]
) -> float:
    """
    Compute population diversity.

    Measures variance in meta-genes (exploration_rate,
    confidence_threshold, parallelism_preference).

    Returns:
        Diversity score in [0, 1]:
            0.0 = No diversity (all genomes identical)
            1.0 = Maximum diversity (uniform distribution)

    Formula:
        avg_variance = mean([var(exploration), var(confidence), var(parallelism)])
        diversity = min(1.0, avg_variance / 0.083)

        (0.083 is max variance for uniform distribution in [0,1])
    """
```

### Diversity Injection

```python
def maintain_diversity(
    self,
    population: list[StrategyGenome],
    traces: list[ExecutionTrace],
    min_diversity: float = 0.2,
) -> list[StrategyGenome]:
    """
    Maintain population diversity by injecting novel strategies.

    If diversity < min_diversity:
        1. Generate novel strategies (10% of population)
        2. Replace low-fitness individuals (bottom 50%)

    Args:
        min_diversity: Minimum acceptable diversity threshold

    Returns:
        Population with diversity maintained
    """
```

**Strategy**:
- Monitor diversity each generation
- Inject random genomes if diversity drops
- Replace worst performers to maintain population size

## Fitness Evaluation

### FitnessScore

Multi-objective fitness scoring.

```python
@dataclass
class FitnessScore:
    # Core objectives
    success: float = 0.0         # Did it achieve the goal?
    efficiency: float = 0.0      # Speed and resource usage
    quality: float = 0.0         # Completeness and correctness

    # Secondary objectives
    stability: float = 0.0       # Error rate, reliability
    elegance: float = 0.0        # Minimal redundant work

    # User feedback
    user_satisfaction: float = 0.0

    def aggregate(self, weights: dict[str, float] | None = None) -> float:
        """
        Aggregate into single score.

        Default weights:
            success: 0.3
            efficiency: 0.2
            quality: 0.2
            stability: 0.1
            elegance: 0.1
            user_satisfaction: 0.1
        """
```

### StrategyAnalyzer

Analyzes execution traces to compute fitness.

```python
class StrategyAnalyzer:
    def compute_fitness(self, trace: ExecutionTrace) -> FitnessScore:
        """
        Compute multi-objective fitness from a trace.

        Metrics used:
            - success: trace.metrics.goal_achieved
            - efficiency: f(duration, agent_count)
            - quality: trace.metrics.completeness_score
            - stability: 1 - (errors / agents)
            - elegance: 1 - redundant_work_ratio
            - user_satisfaction: trace.user_feedback["satisfaction"]
        """
```

## Evolution Cycle

### Full Generation Evolution

```python
def evolve_generation(
    self,
    traces: list[ExecutionTrace],
    elite_count: int = 2,
    min_diversity: float = 0.2,
    crossover_method: str = "blend",
    mutation_method: str = "adaptive",
) -> EvolutionResult:
    """
    Complete one evolution cycle with elitism and diversity maintenance.

    Process:
        1. Evaluate fitness for all traces
        2. Get current population (or bootstrap if empty)
        3. Select elites (top N by fitness)
        4. Select parents (tournament selection)
        5. Generate offspring:
           - Crossover pairs of parents
           - Mutate offspring
           - Generate novel strategies (5% of population)
        6. Combine: elites + parents + offspring
        7. Maintain diversity (inject if needed)
        8. Update pool with new generation
        9. Record history

    Args:
        traces: Execution traces to evaluate fitness
        elite_count: Number of top strategies to preserve
        min_diversity: Minimum diversity threshold
        crossover_method: "single_point", "uniform", or "blend"
        mutation_method: "gaussian", "uniform", or "adaptive"

    Returns:
        EvolutionResult with generation statistics
    """
```

### EvolutionResult

```python
@dataclass
class EvolutionResult:
    generation: int
    population_size: int
    best_fitness: FitnessScore
    avg_fitness: float
    novel_strategies_added: int
    strategies_retired: int
```

## Selection

### Tournament Selection

```python
def select_parents(
    self,
    population: list[StrategyGenome],
    fitness_scores: dict[str, FitnessScore],
    selection_pressure: float = 0.3,
) -> list[StrategyGenome]:
    """
    Select high-fitness strategies for reproduction.

    Method: Tournament selection
        - Create tournaments of size k (30% of population)
        - Select winner (highest fitness) from each tournament
        - Repeat for population_size // 2 parents

    Diversity preservation:
        - Reserve 10% of slots for random selection
        - Ensures genetic diversity is maintained

    Args:
        selection_pressure: Tournament size as fraction of population
                           (higher = more pressure, less diversity)
    """
```

## Usage Examples

### Basic Evolution

```python
from llm_orchestration.evolution import StrategyEvolver, StrategyPool

# Initialize
pool = StrategyPool()
evolver = StrategyEvolver(pool)

# Create initial population
for i in range(10):
    genome = StrategyGenome(
        genome_id=f"genome-{i}",
        exploration_rate=random.uniform(0.05, 0.2),
    )
    pool.add(genome)

# Run evolution
traces = collect_execution_traces()  # Your trace collection
result = evolver.evolve_generation(traces)

print(f"Generation {result.generation}")
print(f"Best fitness: {result.best_fitness.aggregate():.3f}")
print(f"Population size: {result.population_size}")
```

### Custom Evolution Parameters

```python
# Fine-tune evolution parameters
result = evolver.evolve_generation(
    traces=traces,
    elite_count=3,              # Preserve top 3 strategies
    min_diversity=0.25,         # Higher diversity requirement
    crossover_method="uniform",  # Maximum gene mixing
    mutation_method="adaptive",  # Self-adaptive mutation rate
)
```

### Multi-Generation Evolution

```python
# Evolve over multiple generations
for generation in range(100):
    # Collect new execution traces
    traces = run_executions_with_current_population()

    # Evolve
    result = evolver.evolve_generation(traces)

    # Monitor progress
    if result.best_fitness.aggregate() > 0.95:
        print(f"Target fitness reached at generation {generation}")
        break

    # Adapt if stagnating
    if generation > 10:
        recent_history = evolver.history.last_n_generations(10)
        avg_improvement = (
            recent_history[-1].avg_fitness - recent_history[0].avg_fitness
        )
        if avg_improvement < 0.01:
            print("Stagnation detected, increasing diversity requirement")
            min_diversity = 0.35
```

## Performance Characteristics

### Crossover Operators

| Operator | Complexity | Gene Mixing | Preserves Linkage |
|----------|-----------|-------------|-------------------|
| Single-point | O(n) | Low | High |
| Uniform | O(n) | High | Low |
| Blend | O(n) | Medium | Medium |

### Mutation Operators

| Operator | Exploration | Exploitation | Adaptive |
|----------|------------|--------------|----------|
| Gaussian | Low | High | No |
| Uniform | High | Low | No |
| Adaptive | Dynamic | Dynamic | Yes |

### Population Management

- **Elitism overhead**: O(n log n) for sorting
- **Diversity computation**: O(n) where n = population size
- **Selection**: O(k * n) where k = tournament size
- **Crossover + Mutation**: O(n * g) where g = genes per genome

## Best Practices

### 1. Population Size

```python
# Rule of thumb: 10-50 genomes
# Too small: Insufficient diversity
# Too large: Slow convergence

population_size = max(10, problem_complexity * 5)
```

### 2. Elite Count

```python
# Rule of thumb: 5-10% of population
# Too few: Risk losing good solutions
# Too many: Reduces exploration

elite_count = max(2, population_size // 10)
```

### 3. Mutation Rate

```python
# Rule of thumb: 0.05-0.2
# Too low: Slow exploration
# Too high: Random search

mutation_rate = 0.1  # 10% per gene
```

### 4. Diversity Threshold

```python
# Rule of thumb: 0.15-0.30
# Too low: Premature convergence
# Too high: Prevents exploitation

min_diversity = 0.2
```

### 5. Selection Pressure

```python
# Rule of thumb: 0.2-0.4
# Too low: Weak selection, slow progress
# Too high: Premature convergence

selection_pressure = 0.3
```

## Safeguards

The evolution system includes safeguards to prevent regression:

```python
from llm_orchestration.evolution import EvolutionSafeguards

safeguards = EvolutionSafeguards()

# Define golden strategies (must always be preserved)
safeguards.define_golden_strategy(
    genome=best_known_genome,
    reason="Highest success rate ever achieved"
)

# Add regression tests
safeguards.add_regression_test(RegressionTest(
    name="exploration_rate_bounds",
    test_fn=lambda g: 0.0 <= g.exploration_rate <= 0.3,
    threshold=1.0  # All must pass
))

# Validate new generation
validation = safeguards.validate_new_generation(
    new_generation=new_pop,
    old_generation=old_pop,
)

if not validation.valid:
    print(f"Validation failed: {validation.issues}")
```

## References

- Genetic Algorithms: Holland (1975)
- Blend Crossover: Eshelman & Schaffer (1993)
- Adaptive Mutation: Bäck & Schütz (1996)
- Multi-objective Optimization: Deb et al. (2002)
