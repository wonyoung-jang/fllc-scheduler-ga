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

RANDOM_SEED = (1, 2**32 - 1)


def create_and_evaluate_schedule(
    args: tuple[TeamFactory, EventFactory, TournamentConfig, FitnessEvaluator, int],
) -> Schedule | None:
    """Create and evaluate a schedule in a separate process."""
    team_factory, event_factory, config, evaluator, seed = args
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
    selections: tuple[Selection]
    elitism: Selection
    crossovers: tuple[Crossover]
    mutations: tuple[Mutation]
    logger: logging.Logger
    observers: list[GaObserver]
    fitness: FitnessEvaluator
    fitness_history: list[tuple] = field(default_factory=list, init=False, repr=False)
    population: Population = field(default_factory=list, init=False, repr=False)

    _last_reported_fitness: tuple[float, ...] = field(default=None, init=False)
    _crossover_ratio: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _mutation_ratio: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    _best_schedules: dict[str, Schedule] = field(default_factory=dict, init=False)

    def pareto_front(self) -> Population:
        """Get the current Pareto front from the population."""
        if not self.population:
            return []
        return [p for p in self.population if p.rank == 0]

    def _calculate_this_gen_fitness(self) -> tuple[float, ...]:
        """Calculate the average fitness of the current generation."""
        front = self.pareto_front()
        num_objectives = len(front[0].fitness)
        avg_fitness_front1 = [0.0] * num_objectives

        for p in front:
            for i in range(num_objectives):
                avg_fitness_front1[i] += p.fitness[i]

        return tuple(s / len(front) for s in avg_fitness_front1)

    def _update_fitness_history(self) -> tuple[float, ...]:
        """Update the fitness history with the current generation's fitness."""
        this_gen_fitness = self._calculate_this_gen_fitness()
        self.fitness_history.append(this_gen_fitness)
        return this_gen_fitness

    def run(self) -> bool:
        """Run the genetic algorithm and return the best schedule found."""
        self._notify_on_start(self.ga_parameters.generations)

        self.initialize_population()
        self.generation()

        if not self.population:
            self.logger.critical("No valid schedule meeting all hard constraints was found.")
            return False

        non_dominated_sort(self.population)

        self._notify_on_finish(self.population, self.pareto_front())
        crossover_total = self._crossover_ratio.get("success", 0) + self._crossover_ratio.get("failure", 0)
        crossover_success_percentage = (
            self._crossover_ratio.get("success", 0) / crossover_total if crossover_total > 0 else 0.0
        )
        mutation_total = self._mutation_ratio.get("success", 0) + self._mutation_ratio.get("failure", 0)
        mutation_success_percentage = (
            self._mutation_ratio.get("success", 0) / mutation_total if mutation_total > 0 else 0.0
        )
        self.logger.info(
            "Crossovers success ratio: %s/%s = %s",
            self._crossover_ratio.get("success", 0),
            crossover_total,
            f"{crossover_success_percentage:.2%}",
        )
        self.logger.info(
            "Mutations success ratio: %s/%s = %s",
            self._mutation_ratio.get("success", 0),
            mutation_total,
            f"{mutation_success_percentage:.2%}",
        )
        self.logger.info(
            "Unique/Total individuals: %s/%s",
            len({hash(s) for s in self.population}),
            len(self.population),
        )
        return True

    def initialize_population(self) -> None:
        """Initialize the population with random organisms using multiprocessing."""
        num_to_create = self.ga_parameters.population_size
        seeder = Random(self.rng.randint(*RANDOM_SEED))
        worker_seeds = [seeder.randint(*RANDOM_SEED) for _ in range(num_to_create)]
        worker_args = [
            (
                self.team_factory,
                self.event_factory,
                self.config,
                self.fitness,
                seed,
            )
            for seed in worker_seeds
        ]

        self.logger.info("Initializing population with %d individuals.", num_to_create)
        attempts, max_attempts = 0, 1000
        init_pop: Population = []
        add_to_init_pop = init_pop.append

        while len(init_pop) < num_to_create and attempts < max_attempts:
            with multiprocessing.Pool() as pool:
                population = pool.map(create_and_evaluate_schedule, worker_args)

            for p in filter(None, population):
                add_to_init_pop(p)

                if len(init_pop) >= num_to_create:
                    break

            attempts += 1
            self.logger.info("Attempt %d: Created %d valid schedules so far.", attempts, len(init_pop))

        if not init_pop or not init_pop[0].fitness:
            self.logger.critical("No valid initial schedules could be built.")
            return

        self.population = init_pop

    def generation(self) -> None:
        """Perform a single generation step of the genetic algorithm."""
        non_dominated_sort(self.population)
        this_gen_fitness = self._update_fitness_history()
        num_offspring = self.ga_parameters.population_size - self.ga_parameters.elite_size

        for generation in range(self.ga_parameters.generations):
            elites = list(self.elitism.select(self.population, self.ga_parameters.elite_size))
            elite_hashes = {hash(e) for e in elites}
            offspring = self.evolve(num_offspring, elite_hashes)
            self.population = elites + offspring

            if not self.population:
                self.logger.warning("No valid individuals in the current population.")
                return

            non_dominated_sort(self.population)
            this_gen_fitness = self._update_fitness_history()

            self._notify_gen_end(generation, this_gen_fitness)

    def evolve(self, num_offspring: int, existing_hashes: set[int]) -> Population:
        """Evolve the population to create a new generation."""
        new_population: Population = []
        child_hashes = existing_hashes.copy()
        attempts, max_attempts = 0, num_offspring * 9999

        while len(new_population) < num_offspring and attempts < max_attempts:
            if child := self.crossover_population(self.population):
                child_hash = hash(child)
                if child_hash not in child_hashes:
                    child_hashes.add(child_hash)
                    new_population.append(child)
            attempts += 1

        if len(new_population) < num_offspring:
            self.logger.debug(
                "Only %d offspring created after %d attempts, expected %d.",
                len(new_population),
                attempts,
                num_offspring,
            )

        non_dominated_sort(new_population)
        self.mutate_population(new_population)

        return new_population

    def mutate_population(self, population: Population) -> None:
        """Mutate the population by applying mutations to each individual."""
        low = self.ga_parameters.mutation_chance_low
        high = self.ga_parameters.mutation_chance_high

        for individual in population:
            mutation_chance = low if individual.rank == 0 else high
            if mutation_chance > self.rng.random():
                m = self.rng.choice(self.mutations)
                mutation_success = m.mutate(individual)
                if mutation_success:
                    self._notify_mutation(m.__class__.__name__, successful=True)
                    self._mutation_ratio["success"] = self._mutation_ratio.get("success", 0) + 1
                    if (new_fitness := self.fitness.evaluate(individual)) is not None:
                        individual.fitness = new_fitness
                else:
                    self._notify_mutation(m.__class__.__name__, successful=False)
                    self._mutation_ratio["failure"] = self._mutation_ratio.get("failure", 0) + 1

    def crossover_population(self, population: Population) -> Schedule | None:
        """Evolve the population by one individual and return the best of the parents and child."""
        selector = self.rng.choice(self.selections)
        parents: list[Schedule] = list(selector.select(population, 2))
        child: Schedule = None
        if self.ga_parameters.crossover_chance > self.rng.random():
            c = self.rng.choice(self.crossovers)
            child = c.crossover(parents)
            if child is not None:
                self._notify_crossover(f"{c.__class__.__name__} 0", successful=True)
                self._crossover_ratio["success"] = self._crossover_ratio.get("success", 0) + 1
                child.fitness = self.fitness.evaluate(child)
            else:
                self._notify_crossover(f"{c.__class__.__name__} 0", successful=False)
                self._crossover_ratio["failure"] = self._crossover_ratio.get("failure", 0) + 1

        return child

    def _notify_on_start(self, num_generations: int) -> None:
        """Notify observers when the genetic algorithm run starts."""
        for obs in self.observers:
            obs.on_start(num_generations)

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

    def _notify_on_finish(self, pop: Population, front: Population) -> None:
        """Notify observers when the genetic algorithm run is finished."""
        for obs in self.observers:
            obs.on_finish(pop, front)
