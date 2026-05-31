"""Genetic algorithm for FLL Scheduler GA."""

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from fll_scheduler_ga.genetic.island import Island, SchedulePopulation
from fll_scheduler_ga.genetic.stagnation import FitnessHistory, OperatorStats, StagnationHandler

if TYPE_CHECKING:
    from fll_scheduler_ga.adapter.observer import GaObserver
    from fll_scheduler_ga.domain.schedule import Schedule
    from fll_scheduler_ga.domain.schema import GeneticModel
    from fll_scheduler_ga.genetic.context import GaContext

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class GA:
    """Genetic algorithm for the FLL Scheduler GA."""

    context: GaContext
    genetic_model: GeneticModel
    rng: np.random.Generator
    observers: tuple[GaObserver, ...]
    generation: int
    operator_stats: OperatorStats
    fitness_history: FitnessHistory
    generations_array: np.ndarray
    migrate_generations: np.ndarray
    seed_pop: list[Schedule] = field(default_factory=list)
    island_seed_map: dict[int, list[int]] = field(default_factory=dict)
    total_population: list[Schedule] = field(default_factory=list)
    islands: list[Island] = field(default_factory=list)
    start_time: float = 0.0
    _n_islands: int = field(init=False)
    _n_generations: int = field(init=False)

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        self._n_islands = self.genetic_model.parameters.num_islands
        self._n_generations = self.genetic_model.parameters.generations
        n_islands = self._n_islands
        for i in range(n_islands):
            island = Island(
                identity=i,
                generation=self.generation,
                context=self.context,
                genetic_model=self.genetic_model,
                rng=self.rng,
                operator_stats=self.operator_stats,
                fitness_history=self.fitness_history.copy(),
                builder=self.context.builder,
                population=SchedulePopulation(ranks=np.empty((0,), dtype=int)),
            )
            island.stagnation = StagnationHandler(
                rng=self.rng,
                generation=self.generation,
                fitness_history=island.fitness_history,
                model=self.genetic_model.stagnation,
            )
            self.islands.append(island)

    def __len__(self) -> int:
        """Return the number of individuals in the population."""
        return sum(len(i) for i in self.islands)

    def run(self) -> None:
        """Run the genetic algorithm and return the best schedule found."""
        try:
            self.start_time = time.time()
            self.notify_on_start(self._n_generations)
            if self.seed_pop:
                self.seed_population()
            self.initialize_population()
            if not any(i.selected for i in self.islands):
                logger.critical("No valid schedule meeting all hard constraints was found.")
                return
            self.run_epochs()
        except KeyboardInterrupt:
            logger.exception("Genetic algorithm run interrupted by user. Saving...")
        finally:
            self._deduplicate_population()
            self.notify_on_finish(self.total_population, self.pareto_front())

    def pareto_front(self) -> list[Schedule]:
        """Get the Pareto front for each island in the population."""
        return [p for p in self.total_population if p.rank == 0]

    def aggregate_island_fitness(self) -> np.ndarray:
        """Calculate the average fitness of the current generation."""
        island_fitnesses = np.asarray([i.fitness_history.get_last_gen_fitness() for i in self.islands], dtype=float)
        return island_fitnesses.mean(axis=0)

    def seed_population(self) -> None:
        """Seed the population for each island."""
        for ii, s_idx in self.island_seed_map.items():
            island = self.islands[ii]
            for si in s_idx:
                if island.add_to_population(self.seed_pop[si]):
                    island.population.add(self.seed_pop[si].schedule)

    def initialize_population(self) -> None:
        """Initialize the population for each island."""
        logger.debug("Initializing %d islands...", self._n_islands)
        for island in self.islands:
            island.initialize()

    def run_epochs(self) -> None:
        """Perform main evolution loop: generations and migrations."""
        for gen in self.generations_array:
            if self.migrate_generations[gen]:
                self.migrate()
            # Run the generations
            for island in self.islands:
                island.run_epoch()
            self.fitness_history.current = self.aggregate_island_fitness()
            self.fitness_history.update_fitness_history()
            self.generation += 1
            self.notify_on_generation_end(
                generation=gen,
                num_generations=self._n_generations,
                best_fitness=self.fitness_history.get_last_gen_fitness(),
                pop_size=len(self),
            )

    def migrate(self) -> None:
        """Migrate the best individuals between islands using a ring topology."""
        n = len(self.islands)
        for i, dest in enumerate(self.islands):
            src = self.islands[(i + 1) % n]
            migrants = src.give_migrants()
            dest.receive_migrants(migrants)

    def _deduplicate_population(self) -> None:
        """Remove duplicate individuals from the population."""
        unique_pop = [ind for island in self.islands for ind in island.selected]
        pop_array = np.asarray([s.schedule for island in self.islands for s in island.selected])
        schedule_fitness, team_fitnesses = self.context.evaluate(pop_array)
        _, flat, ranks = self.context.select_nsga3(schedule_fitness, len(unique_pop))
        selected = {}
        for rank, idx in zip(ranks, flat, strict=True):
            idx: int
            sch = unique_pop[idx]
            sch.fitness = schedule_fitness[idx]
            sch.team_fitnesses = team_fitnesses[idx]
            sch.rank = rank
            selected[hash(sch)] = sch
        self.total_population = sorted(selected.values(), key=lambda s: (s.rank, -s.fitness.sum()))

    def notify_on_start(self, num_generations: int) -> None:
        """Notify observers when the genetic algorithm run starts."""
        for obs in self.observers:
            obs.on_start(num_generations)

    def notify_on_generation_end(
        self, generation: int, num_generations: int, best_fitness: np.ndarray, pop_size: int
    ) -> None:
        """Notify observers at the end of a generation."""
        for obs in self.observers:
            obs.on_generation_end(generation, num_generations, best_fitness, pop_size)

    def notify_on_finish(self, pop: list[Schedule], pareto_front: list[Schedule]) -> None:
        """Notify observers when the genetic algorithm run is finished."""
        for obs in self.observers:
            obs.on_finish(pop, pareto_front)
