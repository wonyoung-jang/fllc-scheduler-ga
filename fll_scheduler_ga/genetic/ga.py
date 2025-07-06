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

    def __post_init__(self) -> None:
        """Initialize the genetic algorithm with base schedule and teams."""
        for obs in self.observers:
            obs.on_start(self.ga_parameters.generations)

    def pareto_front(self) -> Population:
        """Get the current Pareto front from the population."""
        if not self.population:
            return []
        return [p for p in self.population if p.rank == 0]

    def run(self) -> Population | None:
        """Run the genetic algorithm and return the best schedule found."""
        self.logger.info("GA is in Initializing state.")
        self.initialize_population()

        self.logger.info("GA is in Evolving state.")
        self.generation()

        self.logger.info("GA has terminated.")
        self.log_final_summary()

        for obs in self.observers:
            obs.on_finish()

        if self.population:
            non_dominated_sort(self.population)
            front = self.pareto_front()
            self.logger.info("Pareto front size: %d", len(front))
            return front

        self.logger.warning("No valid schedule meeting all hard constraints was found.")
        return None

    def initialize_population(self) -> None:
        """Initialize the population with random organisms using multiprocessing."""
        num_to_create = self.ga_parameters.population_size
        worker_args = [
            (
                self.team_factory,
                self.event_factory,
                self.config,
                self.rng.randint(0, 2**32 - 1),
                self.fitness,
            )
            for _ in range(num_to_create)
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
        num_objectives = len(front[0].fitness)
        avg_fitness_front1 = [0.0] * num_objectives
        for p in front:
            for i in range(num_objectives):
                avg_fitness_front1[i] += p.fitness[i]
        this_gen_fitness = tuple(s / len(front) for s in avg_fitness_front1)
        self.fitness_history.append(this_gen_fitness)

        self.logger.info("Created %d valid schedules.", len(self.population))

    def generation(self) -> None:
        """Perform a single generation step of the genetic algorithm."""
        if not self.population or not self.population[0].fitness:
            self.logger.warning("Initial population has no valid individuals. Stopping evolution.")
            return

        for generation in range(self.ga_parameters.generations):
            self._generation_step(generation)

    def _generation_step(self, generation: int) -> None:
        """Perform a single generation step of the genetic algorithm."""
        combined_population = self.population + self.evolve(self.ga_parameters.population_size)
        non_dominated_sort(combined_population)
        self.population = list(self.elitism.select(combined_population, self.ga_parameters.population_size))
        if not (front := self.pareto_front()):
            self.logger.warning("No valid individuals in the current population.")
            return

        num_objectives = len(front[0].fitness)
        avg_fitness_front1 = [0.0] * num_objectives
        for p in front:
            for i in range(num_objectives):
                avg_fitness_front1[i] += p.fitness[i]
        this_gen_fitness = tuple(s / len(front) for s in avg_fitness_front1)
        self.fitness_history.append(this_gen_fitness)

        self._notify_gen_end(generation, this_gen_fitness)

    def evolve(self, num_offspring: int) -> Population:
        """Evolve the population to create a new generation."""
        new_population: Population = []
        attempts, max_attempts = 0, num_offspring * 4
        while len(new_population) < num_offspring and attempts < max_attempts:
            if child := self._evolve_step():
                new_population.append(child)
            attempts += 1
        if len(new_population) < num_offspring:
            logger.warning("Only created %d/%d offspring.", len(new_population), num_offspring)
        return new_population

    def _evolve_step(self) -> Schedule | None:
        """Evolve the population by one individual and return the best of the parents and child."""
        parents = list(self.selection.select(self.population, 2))
        child: Schedule | None = None

        if self.rng.random() < self.ga_parameters.crossover_chance:
            c = self.rng.choice(self.crossovers)
            child = c.crossover(parents)
            if child is not None:
                self._notify_crossover(c.__class__.__name__)

        if child is None:
            child = self.rng.choice(parents).clone()

        roll = self.rng.random()
        total_last_avg = sum(self.fitness_history[-1]) if self.fitness_history else 0
        _low = self.ga_parameters.mutation_chance_low
        _high = self.ga_parameters.mutation_chance_high

        if not child.fitness:
            if score := self.fitness.evaluate(child):
                total_score = sum(score)
                if (total_score >= total_last_avg and roll < _low) or (total_score < total_last_avg and roll < _high):
                    m = self.rng.choice(self.mutations)
                    m.mutate(child)
                    self._notify_mutation(m.__class__.__name__)
                child.fitness = score
                return child
            return None

        total_score = sum(child.fitness)
        if (total_score >= total_last_avg and roll < _low) or (total_score < total_last_avg and roll < _high):
            m = self.rng.choice(self.mutations)
            m.mutate(child)
            self._notify_mutation(m.__class__.__name__)
        child.fitness = self.fitness.evaluate(child)
        return child

    def log_final_summary(self) -> None:
        """Log the final summary of the genetic algorithm."""
        if not self.population:
            self.logger.warning("No valid schedule was found after all generations.")
            return

        front = self.pareto_front()
        self.logger.info("Final Pareto Front Size: %d", len(front))
        self.logger.info("Objective scores for a sample of the Pareto front solutions:")
        for i, schedule in enumerate(front[:5]):
            obj_names = list(self.fitness.soft_constraints)
            scores_str = ", ".join(
                [f"{name}: {score:.4f}" for name, score in zip(obj_names, schedule.fitness, strict=False)]
            )
            self.logger.info("  - Solution %d: %s", i + 1, scores_str)

    def _notify_gen_end(self, generation: int, best_fitness: tuple[float, ...]) -> None:
        """Notify observers at the end of a generation."""
        fitness_str = ", ".join(f"{f:.8f}" for f in best_fitness)
        for obs in self.observers:
            obs.on_generation_end(
                generation + 1, self.ga_parameters.generations, self.ga_parameters.population_size, fitness_str
            )

    def _notify_mutation(self, mutation_name: str) -> None:
        """Notify observers when a mutation is applied."""
        for obs in self.observers:
            obs.on_mutation(mutation_name)

    def _notify_crossover(self, crossover_name: str) -> None:
        """Notify observers when a crossover is applied."""
        for obs in self.observers:
            obs.on_crossover(crossover_name)
