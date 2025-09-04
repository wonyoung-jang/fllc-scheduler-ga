"""Island structure for FLL Scheduler GA."""

from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from logging import getLogger
from random import Random
from typing import TYPE_CHECKING

from ..config.constants import ATTEMPTS_RANGE, RANDOM_SEED_RANGE
from ..config.ga_context import GaContext
from ..config.ga_parameters import GaParameters
from ..data_model.schedule import Population, Schedule
from .builder import ScheduleBuilder

if TYPE_CHECKING:
    from ..operators.crossover import Crossover
    from ..operators.mutation import Mutation
    from ..operators.selection import Selection

logger = getLogger(__name__)


@dataclass(slots=True)
class Island:
    """Genetic algorithm island for the FLL Scheduler GA."""

    identity: int
    rng: Random
    builder: ScheduleBuilder
    context: GaContext
    ga_params: GaParameters

    selected: dict[int, Schedule] = field(default_factory=dict, repr=False)
    fitness_history: list[tuple] = field(default_factory=list, repr=False)
    offspring_ratio: Counter = field(default_factory=Counter, repr=False)
    crossover_ratio: dict = field(default=None, repr=False)
    mutation_ratio: dict = field(default=None, repr=False)

    op_map: dict[str, dict[str, int]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        self.crossover_ratio = {tracker: Counter() for tracker in ("success", "total")}
        self.mutation_ratio = {tracker: Counter() for tracker in ("success", "total")}
        self._initialize_operator_maps()
        self.initialize()

    def __len__(self) -> int:
        """Return the number of individuals in the island's population."""
        return len(self.selected)

    def _initialize_operator_maps(self) -> None:
        """Create a mapping of generation to operator indices."""
        gens = self.ga_params.generations
        randrange = self.rng.randrange
        self.op_map["selections"] = {g: randrange(len(self.context.selections)) for g in range(gens)}
        self.op_map["crossovers"] = {g: randrange(len(self.context.crossovers)) for g in range(gens)}
        self.op_map["mutations"] = {g: randrange(len(self.context.mutations)) for g in range(gens)}

    def _get_this_gen_fitness(self) -> tuple[float, ...]:
        """Calculate the average fitness of the current generation."""
        # if not (pop := self.pareto_front()):
        if not (pop := self.selected.values()):
            return ()

        totals = (sum(vals) for vals in zip(*(p.fitness for p in pop), strict=True))
        n = len(pop)
        return tuple(total / n for total in totals)

    def get_last_gen_fitness(self) -> tuple[float, ...]:
        """Get the fitness of the last generation."""
        return self.fitness_history[-1] if self.fitness_history else ()

    def update_fitness_history(self) -> None:
        """Update the fitness history with the current generation's fitness."""
        self.fitness_history.append(self._get_this_gen_fitness())

    def pareto_front(self) -> Population:
        """Get the Pareto front for each island in the population."""
        return [p for p in self.selected.values() if p.rank == 0]

    def add_to_population(self, schedule: Schedule, s_hash: int | None = None) -> bool:
        """Add a schedule to a specific island's population if it's not a duplicate."""
        self.offspring_ratio["total"] += 1
        key = s_hash if s_hash is not None else hash(schedule)
        if key in self.selected:
            return False
        self.selected[key] = schedule
        self.offspring_ratio["success"] += 1
        return True

    def _nsga3_select(self) -> None:
        """NSGA-III selection."""
        population = list(self.selected.values())
        self.selected = self.context.nsga3.select(
            population=population,
            population_size=self.ga_params.population_size,
        )
        self.handle_underpopulation()

    def initialize(self) -> None:
        """Initialize the population for each island."""
        pop_size = self.ga_params.population_size
        to_create = pop_size - len(self.selected)
        if to_create <= 0:
            logger.debug("Initializing island %d with 0 individuals.", self.identity)
            return

        logger.debug("Initializing island %d with %d individuals.", self.identity, to_create)

        seeder = Random(self.rng.randint(*RANDOM_SEED_RANGE))
        attempts, max_attempts = ATTEMPTS_RANGE
        created = 0

        build = self.builder.build
        ctx = self.context
        evaluate = ctx.evaluator.evaluate
        repair = ctx.repairer.repair

        while len(self.selected) < pop_size and attempts < max_attempts:
            schedule = build(rng=Random(seeder.randint(*RANDOM_SEED_RANGE)))
            if repair(schedule) and self.add_to_population(schedule):
                evaluate(schedule)
                created += 1
            elif created == 0:
                attempts += 1

        if created == 0:
            msg = "Island %d: No valid individuals created after %d attempts. Try adjusting parameters."
            raise RuntimeError(msg % (self.identity, attempts))

        if created < to_create:
            logger.warning("Island %d: only created %d/%d valid individuals.", self.identity, created, to_create)

    def mutate_child(self, child: Schedule, mutation: "Mutation", *, m_roll: bool) -> bool:
        """Mutate a child schedule."""
        if not (m_roll and mutation):
            return False

        m_str = str(mutation)
        self.mutation_ratio["total"][m_str] += 1

        if not mutation.mutate(child):
            return False

        self.mutation_ratio["success"][m_str] += 1
        return True

    def evolve(self, generation: int) -> None:
        """Perform main evolution loop: generations and migrations."""
        if not (pop := list(self.selected.values())):
            return

        child_count = 0
        params = self.ga_params
        ctx = self.context
        evaluate = ctx.evaluator.evaluate
        repair = ctx.repairer.repair

        si = self.op_map["selections"][generation]
        ci = self.op_map["crossovers"][generation]
        mi = self.op_map["mutations"][generation]

        selection: Selection = ctx.selections[si] if ctx.selections else None
        crossover: Crossover = ctx.crossovers[ci] if ctx.crossovers else None
        mutation: Mutation = ctx.mutations[mi] if ctx.mutations else None

        while child_count < params.offspring_size:
            parents = selection.select(pop, parents=2)
            children = []
            must_mutate = False
            c_roll = params.crossover_chance > self.rng.random()
            if c_roll and crossover:
                for child in crossover.cross(parents):
                    c_str = str(crossover)
                    self.crossover_ratio["total"][c_str] += 1
                    if repair(child):
                        self.crossover_ratio["success"][c_str] += 1
                        evaluate(child)
                        children.append(child)
            else:
                must_mutate = True  # If no crossover is performed, we must mutate clones of parents
                children.extend(p.clone() for p in parents)

            for child in children:
                m_roll = True if must_mutate else params.mutation_chance > self.rng.random()
                if not self.mutate_child(child, mutation, m_roll=m_roll) and must_mutate:
                    continue

                if not self.add_to_population(child):
                    continue

                evaluate(child)
                child_count += 1

                if child_count >= params.offspring_size:
                    break

        self._nsga3_select()

    def give_migrants(self) -> Iterator[Schedule]:
        """Randomly yield migrants from population."""
        keys = list(self.selected.keys())
        for s_hash in self.rng.sample(keys, k=self.ga_params.migration_size):
            yield self.selected.pop(s_hash)

    def receive_migrants(self, migrants: Iterator[Schedule]) -> None:
        """Receive migrants from another island and add them to the current island's population."""
        for migrant in migrants:
            self.add_to_population(migrant)
        self._nsga3_select()

    def handle_underpopulation(self) -> None:
        """Handle underpopulation by adding new individuals to the island."""
        pop_size = len(self.selected)
        target = self.ga_params.population_size
        if pop_size >= target:
            return

        choice = self.rng.choice
        randint = self.rng.randint
        build = self.builder.build
        ctx = self.context
        mutations = ctx.mutations
        repair = ctx.repairer.repair
        evaluate = ctx.evaluator.evaluate

        while pop_size < target:
            chosen: Schedule
            if choice((True, False)) and self.selected and mutations:
                # clone & mutate existing individual
                curr_pop = list(self.selected.values())
                chosen = choice(curr_pop).clone()
                for m in mutations:
                    _m_str = str(m)
                    self.mutation_ratio["total"][_m_str] += 1
                    if m.mutate(chosen):
                        self.mutation_ratio["success"][_m_str] += 1
            else:
                # brand new individual
                seeder = Random(randint(*RANDOM_SEED_RANGE))
                chosen = build(rng=Random(seeder.randint(*RANDOM_SEED_RANGE)))

            if repair(chosen) and self.add_to_population(chosen):
                evaluate(chosen)
                pop_size += 1

    def finalize_island(self) -> Iterator[Schedule]:
        """Finalize the island's state after evolution."""
        for sched in self.selected.values():
            self.context.evaluator.evaluate(sched)
            yield sched
