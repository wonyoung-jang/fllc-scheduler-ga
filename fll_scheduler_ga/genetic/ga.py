"""Genetic algorithm for FLL Scheduler GA."""

import logging
import multiprocessing
from dataclasses import dataclass, field
from random import Random

from ..config.config import TournamentConfig
from ..data_model.event import EventFactory
from ..data_model.team import TeamFactory
from ..observers.base_observer import GaObserver
from ..operators.crossover import Crossover
from ..operators.mutation import Mutation
from ..operators.nsga2 import non_dominated_sort
from ..operators.selection import Selection
from .builder import ScheduleBuilder
from .fitness import FitnessEvaluator
from .ga_parameters import GaParameters
from .schedule import Population, Schedule

logger = logging.getLogger(__name__)

RANDOM_SEED = (1, 2**32 - 1)


def create_and_evaluate_schedule(
    args: tuple[TeamFactory, EventFactory, TournamentConfig, int, FitnessEvaluator],
) -> Schedule | None:
    """Create and evaluate a schedule in a separate process."""
    team_factory, event_factory, config, seed, evaluator = args
    schedule = ScheduleBuilder(team_factory, event_factory, config, Random(seed)).build()
    if fitness_scores := evaluator.evaluate(schedule):
        schedule.fitness = fitness_scores
        return schedule
    return None


@dataclass(slots=True)
class GA:
    """Genetic algorithm for the FLL Scheduler GA."""

    ga_parameters: GaParameters
    config: TournamentConfig
    rng: Random
    event_factory: EventFactory
    team_factory: TeamFactory
    selection: Selection
    elitism: Selection
    crossovers: tuple[Crossover]
    mutations: tuple[Mutation]
    logger: logging.Logger
    observers: list[GaObserver]
    fitness: FitnessEvaluator
    fitness_history: list[tuple] = field(default_factory=list, init=False, repr=False)
    population: Population = field(default_factory=list, init=False, repr=False)

    _last_reported_fitness: tuple[float, ...] = field(default=None, init=False)

    def pareto_front(self) -> Population:
        """Get the current Pareto front from the population."""
        if not self.population:
            return []
        return [p for p in self.population if p.rank == 0]

    def _calculate_this_gen_fitness(self, front: Population) -> tuple[float, ...]:
        """Calculate the average fitness of the current generation."""
        num_objectives = len(front[0].fitness)
        avg_fitness_front1 = [0.0] * num_objectives

        for p in front:
            for i in range(num_objectives):
                avg_fitness_front1[i] += p.fitness[i]

        return tuple(s / len(front) for s in avg_fitness_front1)

    def run(self) -> bool:
        """Run the genetic algorithm and return the best schedule found."""
        self._notify_on_start(self.ga_parameters.generations)
        self.initialize_population()
        self.generation()

        if not self.population:
            self.logger.critical("No valid schedule meeting all hard constraints was found.")
            return False

        non_dominated_sort(self.population)
        front = self.pareto_front()
        self._notify_on_finish(self.population, front)
        return True

    def initialize_population(self) -> None:
        """Initialize the population with random organisms using multiprocessing."""
        self.logger.info("GA is in Initializing state.")
        num_to_create = self.ga_parameters.population_size
        seeder = Random(self.rng.randint(*RANDOM_SEED))
        worker_seeds = [seeder.randint(*RANDOM_SEED) for _ in range(num_to_create)]
        worker_args = [
            (
                self.team_factory,
                self.event_factory,
                self.config,
                seed,
                self.fitness,
            )
            for seed in worker_seeds
        ]

        self.logger.info("Initializing population with %d individuals using multiprocessing.", num_to_create)
        attempts, max_attempts = 0, 10

        while len(self.population) < num_to_create and attempts < max_attempts:
            with multiprocessing.Pool() as pool:
                population = pool.map(create_and_evaluate_schedule, worker_args)

            for p in filter(None, population):
                self.population.append(p)

                if len(self.population) >= num_to_create:
                    break

            attempts += 1
            self.logger.info("Attempt %d: Created %d valid schedules so far.", attempts, len(self.population))

        if not self.population:
            self.logger.critical("No valid initial schedules could be built.")
            return

        non_dominated_sort(self.population)
        front = self.pareto_front()
        this_gen_fitness = self._calculate_this_gen_fitness(front)
        self.fitness_history.append(this_gen_fitness)
        self.logger.info("Created %d valid schedules.", len(self.population))

    def generation(self) -> None:
        """Perform a single generation step of the genetic algorithm."""
        self.logger.info("GA is in Evolving state.")
        if not self.population or not self.population[0].fitness:
            self.logger.warning("Initial population has no valid individuals. Stopping evolution.")
            return

        for generation in range(self.ga_parameters.generations):
            combined_population = self.population + self.evolve(self.ga_parameters.population_size)
            non_dominated_sort(combined_population)
            self.population = list(self.elitism.select(combined_population, self.ga_parameters.population_size))

            if not (front := self.pareto_front()):
                self.logger.warning("No valid individuals in the current population.")
                return

            this_gen_fitness = self._calculate_this_gen_fitness(front)
            self.fitness_history.append(this_gen_fitness)
            self._notify_gen_end(generation, this_gen_fitness)

    def evolve(self, num_offspring: int) -> Population:
        """Evolve the population to create a new generation."""
        new_population: Population = []
        attempts, max_attempts = 0, num_offspring * 4

        while len(new_population) < num_offspring and attempts < max_attempts:
            new_population.append(self.produce_offspring())
            attempts += 1

        if len(new_population) < num_offspring:
            logger.warning("Only created %d/%d offspring.", len(new_population), num_offspring)

        return new_population

    def produce_offspring(self) -> Schedule:
        """Evolve the population by one individual and return the best of the parents and child."""
        parents: list[Schedule] = list(self.selection.select(self.population, 2))
        max_parent = max(parents, key=lambda p: sum(p.fitness)).clone()
        child: Schedule | None = None
        if self.rng.random() < self.ga_parameters.crossover_chance:
            c = self.rng.choice(self.crossovers)
            child = c.crossover(parents)
            self._notify_crossover(c.__class__.__name__, successful=bool(child))

        if child is None:
            child = max_parent
        elif (score := self.fitness.evaluate(child)) is None:
            return max_parent
        elif score:
            child.fitness = score

        total_score = sum(child.fitness)
        total_last_avg = sum(self.fitness_history[-1]) if self.fitness_history else 0
        mutation_chance = (
            self.ga_parameters.mutation_chance_low
            if total_score >= total_last_avg
            else self.ga_parameters.mutation_chance_high
        )

        if self.rng.random() < mutation_chance:
            m = self.rng.choice(self.mutations)
            m.mutate(child)
            if (new_fitness := self.fitness.evaluate(child)) is not None:
                child.fitness = new_fitness
                self._notify_mutation(m.__class__.__name__, successful=True)
                return child
            self._notify_mutation(m.__class__.__name__)
            return max_parent

        return child

    def _notify_gen_end(self, generation: int, best_fitness: tuple[float, ...]) -> None:
        """Notify observers at the end of a generation."""
        for obs in self.observers:
            obs.on_generation_end(
                generation + 1,
                self.ga_parameters.generations,
                self.ga_parameters.population_size,
                best_fitness,
                len(self.pareto_front()),
            )

    def _notify_mutation(self, mutation_name: str, *, successful: bool) -> None:
        """Notify observers when a mutation is applied."""
        for obs in self.observers:
            obs.on_mutation(mutation_name, successful=successful)

    def _notify_crossover(self, crossover_name: str, *, successful: bool) -> None:
        """Notify observers when a crossover is applied."""
        for obs in self.observers:
            obs.on_crossover(crossover_name, successful=successful)

    def _notify_on_start(self, num_generations: int) -> None:
        """Notify observers when the genetic algorithm run starts."""
        for obs in self.observers:
            obs.on_start(num_generations)

    def _notify_on_finish(self, pop: Population, front: Population) -> None:
        """Notify observers when the genetic algorithm run is finished."""
        for obs in self.observers:
            obs.on_finish(pop, front)
