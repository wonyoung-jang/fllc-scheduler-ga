"""Genetic algorithm for FLL Scheduler GA."""

from __future__ import annotations

import pickle
from collections import Counter
from dataclasses import dataclass, field
from logging import getLogger
from time import time
from typing import TYPE_CHECKING, Any

import numpy as np

from .builder import ScheduleBuilder
from .island import Island

if TYPE_CHECKING:
    from pathlib import Path

    from ..config.ga_context import GaContext
    from ..config.ga_parameters import GaParameters
    from ..data_model.config import TournamentConfig
    from ..data_model.schedule import Population
    from ..io.observers import GaObserver
    from ..operators.crossover import Crossover
    from ..operators.mutation import Mutation

logger = getLogger(__name__)


@dataclass(slots=True)
class GA:
    """Genetic algorithm for the FLL Scheduler GA."""

    context: GaContext
    rng: np.random.Generator
    observers: tuple[GaObserver]
    seed_file: Path | None
    save_front_only: bool

    curr_gen: int = 0
    fitness_history: np.ndarray = None
    total_population: Population = field(default_factory=list, repr=False)
    islands: list[Island] = field(default_factory=list, repr=False)
    ga_params: GaParameters = None

    generations_array: np.ndarray = None
    migrate_generations: np.ndarray = None

    _offspring_ratio: Counter = field(default_factory=Counter, repr=False)
    _crossover_ratio: dict[str, Counter] = None
    _mutation_ratio: dict[str, Counter] = None

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        self.ga_params = self.context.app_config.ga_params
        self.generations_array = np.arange(1, self.ga_params.generations + 1)
        self.migrate_generations: np.ndarray = np.zeros(self.ga_params.generations + 1, dtype=int)
        if self.ga_params.num_islands > 1 and self.ga_params.migration_size > 0:
            self.migrate_generations[:: self.ga_params.migration_interval] = 1

        self.fitness_history = np.zeros(
            (self.ga_params.generations, len(self.context.evaluator.objectives)), dtype=float
        )

        builder = ScheduleBuilder(
            team_factory=self.context.team_factory,
            event_factory=self.context.event_factory,
            config=self.context.app_config.tournament,
            rng=self.context.app_config.rng,
        )

        trackers = ("success", "total")
        self._crossover_ratio = {tr: Counter({str(c): 0 for c in self.context.crossovers}) for tr in trackers}
        self._mutation_ratio = {tr: Counter({str(m): 0 for m in self.context.mutations}) for tr in trackers}

        self.islands.extend(
            Island(
                i,
                self.context.app_config.rng,
                builder,
                self.context,
                self.ga_params.clone(),
                self._offspring_ratio.copy(),
                self._crossover_ratio.copy(),
                self._mutation_ratio.copy(),
            )
            for i in range(self.ga_params.num_islands)
        )

    def __len__(self) -> int:
        """Return the number of individuals in the population."""
        return sum(len(i) for i in self.islands)

    def run(self) -> None:
        """Run the genetic algorithm and return the best schedule found."""
        start_time = time()
        self._notify_on_start(self.ga_params.generations)
        try:
            self.initialize_population()
            if not any(i.selected for i in self.islands):
                logger.critical("No valid schedule meeting all hard constraints was found.")
                return False
            self.run_epochs()
        except Exception:
            logger.exception("An error occurred during the genetic algorithm run.")
            self.update_fitness_history()
        except KeyboardInterrupt:
            logger.debug("Genetic algorithm run interrupted by user. Saving...")
            self.update_fitness_history()
        finally:
            self.finalize(start_time)
            self._notify_on_finish(self.total_population, self.pareto_front())
            self.save()

    def pareto_front(self) -> Population:
        """Get the Pareto front for each island in the population."""
        if not self.total_population:
            return [p for i in self.islands for p in i.pareto_front()]
        return [p for p in self.total_population if p.rank == 0]

    def get_this_gen_fitness(self) -> tuple[float, ...]:
        """Calculate the average fitness of the current generation."""
        island_fitnesses = np.asarray([i.get_last_gen_fitness() for i in self.islands])
        return island_fitnesses.mean(axis=0)

    def get_last_gen_fitness(self) -> tuple[float, ...]:
        """Get the fitness of the last generation."""
        return self.fitness_history[self.curr_gen - 1] if self.curr_gen > 0 else ()

    def update_fitness_history(self) -> None:
        """Update the fitness history with the current generation's fitness."""
        self.fitness_history[self.curr_gen] = self.get_this_gen_fitness()

    def initialize_population(self) -> None:
        """Initialize the population for each island."""
        s_path = self.seed_file
        if s_path and s_path.exists() and (seed_pop := self.load()):
            self.rng.shuffle(seed_pop)
            n = self.ga_params.num_islands
            for idx, schedule in enumerate(seed_pop):
                self.islands[idx % n].add_to_population(schedule)

        logger.debug("Initializing %d islands...", self.ga_params.num_islands)
        for island in self.islands:
            island.initialize()
            island.evaluate_pop()

    def run_epochs(self) -> None:
        """Perform main evolution loop: generations and migrations."""
        for gen in self.generations_array:
            if self.migrate_generations[gen]:
                self.migrate()
            self.run_generation()
            self.update_fitness_history()
            self.curr_gen += 1
            self._notify_on_generation_end(
                generation=gen,
                num_generations=self.ga_params.generations,
                best_fitness=self.get_last_gen_fitness(),
                pop_size=len(self),
            )

    def run_generation(self) -> None:
        """Run a single epoch of the genetic algorithm."""
        for island in self.islands:
            island.handle_underpopulation()
            island.evolve()
            island.update_fitness_history()
            island.curr_gen += 1

    def migrate(self) -> None:
        """Migrate the best individuals between islands using a random 1 -> 2 ring topology."""
        n = len(self.islands)
        o = self.rng.integers(1, n)
        for i, island in enumerate(self.islands):
            j = (i + o) % n
            island.receive_migrants(self.islands[j].give_migrants())

    def _notify_on_start(self, num_generations: int) -> None:
        """Notify observers when the genetic algorithm run starts."""
        for obs in self.observers:
            obs.on_start(num_generations)

    def _notify_on_generation_end(
        self,
        generation: int,
        num_generations: int,
        best_fitness: np.ndarray[float],
        pop_size: int,
    ) -> None:
        """Notify observers at the end of a generation."""
        for obs in self.observers:
            obs.on_generation_end(generation, num_generations, best_fitness, pop_size)

    def _notify_on_finish(self, pop: Population, pareto_front: Population) -> None:
        """Notify observers when the genetic algorithm run is finished."""
        for obs in self.observers:
            obs.on_finish(pop, pareto_front)

    def _deduplicate_population(self) -> None:
        """Remove duplicate individuals from the population."""
        unique_pop = [ind for island in self.islands for ind in island.selected]
        pop_array = np.asarray([s.schedule for island in self.islands for s in island.selected])
        schedule_fitness, team_fitnesses = self.context.evaluator.evaluate_population(pop_array, self.context)
        fronts = self.context.nsga3.non_dominated_sort(schedule_fitness, len(pop_array))
        selected = {}
        for rank, front in enumerate(fronts):
            for idx in front:
                idx: int
                sch = unique_pop[idx]
                sch.fitness = schedule_fitness[idx]
                sch.team_fitnesses = team_fitnesses[idx]
                sch.rank = rank
                selected[hash(sch)] = sch
        self.total_population = sorted(selected.values(), key=lambda s: (s.rank, -s.fitness.sum()))

    def _aggregate_stats_from_islands(self) -> None:
        """Aggregate statistics from all islands."""
        for island in self.islands:
            for tracker, crossovers in island.crossover_ratio.items():
                self._crossover_ratio[tracker].update(crossovers)
            for tracker, mutations in island.mutation_ratio.items():
                self._mutation_ratio[tracker].update(mutations)
            for tracker, count in island.offspring_ratio.items():
                self._offspring_ratio[tracker] += count

    def _log_operators(self, name: str, ratios: dict[str, Counter], ops: tuple[Crossover | Mutation]) -> None:
        """Log statistics for crossover and mutation operators."""
        if not (op_strings := [f"{op!s}" for op in ops]):
            return

        log = f"{name.capitalize()} statistics:"
        max_len = max(len(s) for s in op_strings) + 1
        for op in op_strings:
            success = ratios.get("success", {}).get(op, 0)
            total = ratios.get("total", {}).get(op, 0)
            rate = success / total if total > 0 else 0.0
            log += f"\n  {op:<{max_len}}: {success}/{total} ({rate:.2%})"
        logger.debug(log)

    def _log_aggregate_stats(self) -> None:
        """Log aggregate statistics across all islands."""
        final_log = f"{'=' * 20}\nFinal statistics"
        crs_suc = sum(self._crossover_ratio.get("success", {}).values())
        crs_tot = sum(self._crossover_ratio.get("total", {}).values())
        crs_rte = f"{crs_suc / crs_tot if crs_tot > 0 else 0.0:.2%}"
        mut_suc = sum(self._mutation_ratio.get("success", {}).values())
        mut_tot = sum(self._mutation_ratio.get("total", {}).values())
        mut_rte = f"{mut_suc / mut_tot if mut_tot > 0 else 0.0:.2%}"
        off_suc = self._offspring_ratio.get("success", 0)
        off_tot = self._offspring_ratio.get("total", 0)
        off_rte = f"{off_suc / off_tot if off_tot > 0 else 0.0:.2%}"
        unique_inds = len(self.total_population)
        total_inds = len(self)
        unique_rte = f"{unique_inds / total_inds if total_inds > 0 else 0.0:.2%}"
        final_log += (
            f"\n  Total islands          : {len(self.islands)}"
            f"\n  Unique individuals     : {unique_inds}/{total_inds} ({unique_rte})"
            f"\n  Crossover success rate : {crs_suc}/{crs_tot} ({crs_rte})"
            f"\n  Mutation success rate  : {mut_suc}/{mut_tot} ({mut_rte})"
            f"\n  Offspring success rate : {off_suc}/{off_tot} ({off_rte})"
        )
        logger.debug(final_log)

    def _log_unique_events(self) -> None:
        """Count unique event lists."""
        unique_genes = Counter()
        for schedule in self.total_population:
            for events in schedule.team_events.values():
                unique_genes[tuple(sorted(events))] += 1
            # for team in schedule.teams:
            #     unique_genes[tuple(sorted(team.events))] += 1
        for gene, count in unique_genes.most_common(n=10):
            logger.debug("Event: %s, Count: %d", gene, count)

    def finalize(self, start_time: float) -> None:
        """Aggregate islands and run a final selection to produce the final population."""
        self._deduplicate_population()
        self._log_unique_events()
        self._aggregate_stats_from_islands()
        self._log_operators(name="crossover", ratios=self._crossover_ratio, ops=self.context.crossovers)
        self._log_operators(name="mutation", ratios=self._mutation_ratio, ops=self.context.mutations)
        self._log_aggregate_stats()
        for island in self.islands:
            logger.debug("Island %d Fitness: %.2f", island.identity, sum(island.get_last_gen_fitness()))
        logger.debug("Total time taken: %.2f seconds", time() - start_time)

    def load(self) -> Population | None:
        """Load and integrate a population from a seed file."""
        logger.debug("Loading seed population from: %s", self.seed_file)
        seed_data: dict[str, Any] = {}
        try:
            with self.seed_file.open("rb") as f:
                seed_data = pickle.load(f)
        except (OSError, pickle.PicklingError):
            logger.exception("Could not load or parse seed file. Starting with a fresh population.")
        except EOFError:
            logger.debug("Pickle file is empty")

        seed_config: TournamentConfig | None = seed_data.get("config")
        if not seed_config:
            logger.warning("Seed population is missing config. Using current...")
            return None

        if (
            self.context.app_config.tournament.num_teams != seed_config.num_teams
            or self.context.app_config.tournament.rounds != seed_config.rounds
        ):
            logger.warning("Seed population does not match current config. Using current...")
            return None

        population = seed_data.get("population")
        if not population:
            logger.warning("Seed population is missing. Using current...")
            return None

        return population

    def save(self) -> None:
        """Save the final population to a file to be used as a seed for a future run."""
        pop = self.pareto_front() if self.save_front_only else self.total_population

        if not pop:
            logger.warning("No population to save to seed file.")
            return

        data_to_cache = {
            "population": pop,
            "config": self.context.app_config.tournament,
        }

        path = self.seed_file
        logger.debug("Saving final population of size %d to seed file: %s", len(pop), path)

        try:
            with path.open("wb") as f:
                pickle.dump(data_to_cache, f)
        except (OSError, pickle.PicklingError, EOFError):
            logger.exception("Error saving population to seed file: %s", path)
