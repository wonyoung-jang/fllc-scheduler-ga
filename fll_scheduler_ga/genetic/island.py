"""Island structure for FLL Scheduler GA."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from logging import getLogger
from random import Random
from typing import TYPE_CHECKING

from ..config.constants import RANDOM_SEED_RANGE

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ..config.ga_context import GaContext
    from ..config.ga_parameters import GaParameters
    from ..data_model.schedule import Schedule
    from ..operators.crossover import Crossover
    from ..operators.mutation import Mutation
    from .builder import ScheduleBuilder

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

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        trackers = ("success", "total")
        self.crossover_ratio = {t: Counter() for t in trackers}
        self.mutation_ratio = {t: Counter() for t in trackers}

    def __len__(self) -> int:
        """Return the number of individuals in the island's population."""
        return len(self.selected)

    def _get_this_gen_fitness(self) -> tuple[float, ...]:
        """Calculate the average fitness of the current generation."""
        if not (pop := list(self.pareto_front())):
            return ()

        gen_fit_iter = (p.fitness for p in pop)
        totals = (sum(vals) for vals in zip(*gen_fit_iter, strict=True))
        n = len(pop)
        return tuple(total / n for total in totals)

    def get_last_gen_fitness(self) -> tuple[float, ...]:
        """Get the fitness of the last generation."""
        return self.fitness_history[-1] if self.fitness_history else ()

    def update_fitness_history(self) -> None:
        """Update the fitness history with the current generation's fitness."""
        self.fitness_history.append(self._get_this_gen_fitness())

    def pareto_front(self) -> Iterator[Schedule]:
        """Get the Pareto front for each island in the population."""
        yield from (p for p in self.selected.values() if p.rank == 0)

    def add_to_population(self, schedule: Schedule, s_hash: int | None = None, *, recurse: bool = False) -> bool:
        """Add a schedule to a specific island's population if it's not a duplicate."""
        if not recurse:
            self.offspring_ratio["total"] += 1

        key = s_hash if s_hash is not None else hash(schedule)
        if key not in self.selected:
            self.selected[key] = schedule
            self.offspring_ratio["success"] += 1
            return True

        if not self.context.mutations:
            return False

        self.mutate_schedule(schedule)
        return self.add_to_population(schedule, s_hash=hash(schedule), recurse=True)

    def build_schedule(self) -> Schedule:
        """Build a new offspring schedule."""
        seed_x = self.rng.randint(*RANDOM_SEED_RANGE)
        build_x = Random(seed_x).randint(*RANDOM_SEED_RANGE)
        return self.builder.build(rng=Random(build_x))

    def build_n_schedules(self, needed: int) -> None:
        """Build a number of schedules."""
        if needed == 0:
            return

        evaluate = self.context.evaluator.evaluate
        repair = self.context.repairer.repair
        for _ in range(needed):
            schedule = self.build_schedule()
            repair(schedule)
            evaluate(schedule)
            self.add_to_population(schedule)

    def initialize(self) -> None:
        """Initialize the population for each island."""
        logger.debug(
            "Island %d: Initializing population with %d individuals",
            self.identity,
            self.ga_params.population_size - len(self.selected),
        )
        self.build_n_schedules(self.ga_params.population_size - len(self.selected))

    def handle_underpopulation(self) -> None:
        """Handle underpopulation by creating new individuals."""
        self.build_n_schedules(self.ga_params.population_size - len(self.selected))

    def mutate_schedule(self, schedule: Schedule, *, m_roll: bool = True) -> bool:
        """Mutate a child schedule."""
        ctx = self.context
        mutation: Mutation = self.rng.choice(ctx.mutations) if ctx.mutations else None
        if not (m_roll and mutation):
            return False

        m_str = str(mutation)
        self.mutation_ratio["total"][m_str] += 1
        if mutation.mutate(schedule):
            self.mutation_ratio["success"][m_str] += 1
            schedule.mutations += 1
            ctx.evaluator.evaluate(schedule)
            return True
        return False

    def crossover_schedule(self, parents: tuple[Schedule], *, c_roll: bool = True) -> Iterator[Schedule]:
        """Perform crossover between two parent schedules."""
        ctx = self.context
        crossover: Crossover = self.rng.choice(ctx.crossovers) if ctx.crossovers else None
        if not (c_roll and crossover):
            yield from (p.clone() for p in parents)
            return

        c_str = str(crossover)
        for child in crossover.cross(parents):
            self.crossover_ratio["total"][c_str] += 1
            if ctx.checker.check(child):
                self.crossover_ratio["success"][c_str] += 1
            ctx.repairer.repair(child)
            ctx.evaluator.evaluate(child)
            yield child

    def evolve(self) -> None:
        """Perform main evolution loop: generations and migrations."""
        if not (pop := list(self.selected.values())):
            return

        param = self.ga_params
        noffspring = param.offspring_size
        crossover_chance = param.crossover_chance
        mutation_chance = param.mutation_chance
        ctx = self.context
        random = self.rng.random
        select = ctx.selection.select

        nchild = 0
        while nchild < noffspring:
            for child in self.crossover_schedule(select(pop, k=2), c_roll=crossover_chance > random()):
                if self.mutate_schedule(child, m_roll=mutation_chance > random()):
                    self.add_to_population(child)
                    nchild += 1
                    if nchild >= noffspring:
                        break

        pop = list(self.selected.values())
        self.selected = self.context.nsga3.select(pop)

    def give_migrants(self) -> Iterator[Schedule]:
        """Randomly yield migrants from population."""
        keys = list(self.selected.keys())
        selection = self.context.selection
        for s in selection.select(keys, k=self.ga_params.migration_size):
            yield self.selected.pop(s)

    def receive_migrants(self, migrants: Iterator[Schedule]) -> None:
        """Receive migrants from another island and add them to the current island's population."""
        for migrant in migrants:
            self.add_to_population(migrant)

    def finalize_island(self) -> Iterator[Schedule]:
        """Finalize the island's state after evolution."""
        for sched in self.selected.values():
            self.context.evaluator.evaluate(sched)
            yield sched

    def check_for_stagnation(self) -> bool:
        """Check if the island's best fitness has stagnated."""
        stagnation_len = min(self.ga_params.migration_interval, self.ga_params.generations // 5)
        if len(self.fitness_history) < stagnation_len:
            return False

        last_fits = (sum(f) for f in self.fitness_history[-stagnation_len:])
        first = next(last_fits)
        return not any(f > first for f in last_fits)

    def trigger_cataclysm(self) -> None:
        """Heavily mutates a portion of the population to reintroduce diversity."""
        if len(self.selected) < 2:
            return

        num_to_mutate = max(len(self.selected) // 5, 1)
        logger.debug(
            "Island %d: Stagnation detected. Triggering cataclysm, mutating %d individuals.",
            self.identity,
            num_to_mutate,
        )
        random = self.rng.random
        keys = list(self.selected.keys())
        for key in self.rng.sample(keys, k=min(num_to_mutate, len(keys))):
            if random() < 0.05:
                del self.selected[key]
            else:
                s = self.selected.pop(key) if random() < 0.5 else self.selected[key].clone()
                for _ in self.context.mutations:
                    self.mutate_schedule(s, m_roll=True)
                self.add_to_population(s)
