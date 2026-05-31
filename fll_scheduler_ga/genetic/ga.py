"""Genetic algorithm for FLL Scheduler GA."""

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from fll_scheduler_ga.genetic.island import Island
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
    operator_stats: OperatorStats
    fitness_history: FitnessHistory
    generations_array: np.ndarray
    migrate_generations: np.ndarray
    seed_pop: list[Schedule] = field(default_factory=list)
    island_seed_map: dict[int, list[int]] = field(default_factory=dict)
    total_population: list[Schedule] = field(default_factory=list)
    islands: list[Island] = field(default_factory=list)
    start_time: float = 0.0

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        for i in range(self.genetic_model.parameters.num_islands):
            self.islands.append(
                Island(
                    identity=i,
                    context=self.context,
                    ga_param=self.genetic_model.parameters,
                    rng=self.rng,
                    operator_stats=self.operator_stats,
                    fitness_history=self.fitness_history.copy(),
                    stagnation=StagnationHandler(
                        enabled=self.genetic_model.stagnation.enable,
                        threshold=self.genetic_model.stagnation.threshold,
                        proportion=self.genetic_model.stagnation.proportion,
                        cooldown=self.genetic_model.stagnation.cooldown,
                    ),
                )
            )

    def __len__(self) -> int:
        """Return the number of individuals in the population."""
        return sum(len(i) for i in self.islands)

    @property
    def pareto_front(self) -> list[Schedule]:
        """Get the Pareto front for each island in the population."""
        return [p for p in self.total_population if p.rank == 0]

    @property
    def avg_island_fit(self) -> np.ndarray:
        """Calculate the average fitness of the current generation."""
        return np.asarray([i.fitness_history.get_last_gen_fitness() for i in self.islands], dtype=float).mean(axis=0)

    def run(self) -> None:
        """Run the genetic algorithm and return the best schedule found."""
        try:
            self.start_time = time.perf_counter()
            self.notify_on_start()
            logger.debug("Seeding population...")
            self.seed_population()
            logger.debug("Initializing population...")
            self.initialize_population()
            if not any(i.selected for i in self.islands):
                msg = "No valid schedule meeting all hard constraints was found."
                raise ValueError(msg)
            self.run_epochs()
        except KeyboardInterrupt:
            logger.exception("Genetic algorithm run interrupted by user. Saving...")
        finally:
            self._deduplicate_population()
            self.notify_on_finish()

    def seed_population(self) -> None:
        """Seed the population for each island."""
        if self.seed_pop:
            for ii, s_idx in self.island_seed_map.items():
                island = self.islands[ii]
                for si in s_idx:
                    if island.add_to_population(self.seed_pop[si]):
                        island.population.add(self.seed_pop[si].schedule)

    def initialize_population(self) -> None:
        """Initialize the population for each island."""
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
            self.fitness_history.current = self.avg_island_fit
            self.fitness_history.update_fitness_history()
            self.notify_on_generation_end(gen)

    def migrate(self) -> None:
        """Migrate the best individuals between islands using a ring topology."""
        n = self.genetic_model.parameters.num_islands
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
        selected = set()
        for rank, idx in zip(ranks, flat, strict=True):
            idx: int
            sch = unique_pop[idx]
            sch.fitness = schedule_fitness[idx]
            sch.team_fitnesses = team_fitnesses[idx]
            sch.rank = rank
            selected.add(sch)
        self.total_population = sorted(selected, key=lambda s: (s.rank, -s.fitness.sum()))

    def notify_on_start(self) -> None:
        """Notify observers when the genetic algorithm run starts."""
        for obs in self.observers:
            obs.on_start(self.genetic_model.parameters.generations)

    def notify_on_generation_end(self, gen: int) -> None:
        """Notify observers at the end of a generation."""
        for obs in self.observers:
            obs.on_generation_end(
                gen, self.genetic_model.parameters.generations, self.fitness_history.get_last_gen_fitness(), len(self)
            )

    def notify_on_finish(self) -> None:
        """Notify observers when the genetic algorithm run is finished."""
        for obs in self.observers:
            obs.on_finish(self.total_population, self.pareto_front)
