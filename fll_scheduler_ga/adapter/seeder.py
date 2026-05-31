"""Seed data I/O for genetic algorithm."""

import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from fll_scheduler_ga.adapter.exporter import CsvScheduleExporter, ScheduleSummaryGenerator
from fll_scheduler_ga.adapter.importer import CsvImporter
from fll_scheduler_ga.constants import DATA_MODEL_VERSION, SeedIslandStrategy, SeedPopSort

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from fll_scheduler_ga.domain.model import TournamentConfig
    from fll_scheduler_ga.domain.schedule import Schedule
    from fll_scheduler_ga.domain.schema import AppConfig, ImportModel
    from fll_scheduler_ga.genetic.context import GaContext

logger = getLogger(__name__)


@dataclass(slots=True)
class GASeedData:
    """GA seed data object."""

    config: TournamentConfig | None = None
    population: list[Schedule] = field(default_factory=list)
    version: int = DATA_MODEL_VERSION


def load_ga(path: Path, config: TournamentConfig) -> GASeedData | None:
    """Load GA seed data from a file."""
    try:
        logger.debug("Loading seed population from: %s", path)
        with path.open("rb") as f:
            data: GASeedData = pickle.load(f)
    except OSError, pickle.PicklingError, AttributeError, ModuleNotFoundError:
        logger.warning("Could not load or parse seed file. Starting with a fresh population.")
        return None
    except EOFError:
        logger.debug("Pickle file is empty")
        return None
    try:
        pop = []
        if data.version != DATA_MODEL_VERSION:
            logger.warning(
                "Seed population data version mismatch: Expected (%d), found (%d). Dismissing old seed file...",
                DATA_MODEL_VERSION,
                data.version,
            )
        elif data.config != config:
            logger.warning("Seed population does not match current config. Using current...")
            data.config = config
        elif not data.population:
            logger.warning("Seed population is missing. Using current...")
        else:
            pop = data.population
    except AttributeError:
        logger.warning("Seed population is malformed. Starting with a fresh population.")
        return None
    data.population = pop
    return data


def save_ga(path: Path, data: GASeedData) -> None:
    """Save the final population to a file to be used as a seed for a future run."""
    try:
        logger.debug("Saving final population of size %d to seed file: %s", len(data.population), path)
        with path.open("wb") as f:
            pickle.dump(data, f)
    except OSError, pickle.PicklingError, EOFError:
        logger.exception("Error saving population to seed file: %s", path)


class DistributedSeedingStrategy:
    """Distributed seeding strategy for GA islands."""

    def __call__(self, seed_indices: Iterator[int], n_islands: int) -> dict[int, list[int]]:
        """Get the seed indices for each island."""
        island_to_seed: dict[int, list[int]] = defaultdict(list)
        for idx in seed_indices:
            island_to_seed[idx % n_islands].append(idx)
        return island_to_seed


class ConcentratedSeedingStrategy:
    """Concentrated seeding strategy for GA islands."""

    def __call__(self, seed_indices: Iterator[int], n_islands: int, n_pop: int) -> dict[int, list[int]]:
        """Get the seed indices for each island."""
        island_to_seed: dict[int, list[int]] = defaultdict(list)
        for i in range(n_islands):
            while len(island_to_seed[i]) < n_pop:
                if (idx := next(seed_indices, None)) is None:
                    break
                island_to_seed[i].append(idx)
        return island_to_seed


def _flush_seed_file(seed_file: Path) -> None:
    """Flush the seed file if specified in runtime settings."""
    seed_file.unlink(missing_ok=True)
    logger.debug("Flushed seed file at: %s", seed_file)
    seed_file.touch(exist_ok=True)


@dataclass(slots=True)
class RuntimeStartup:
    """Handle start of runtime with seed file handling and CSV import."""

    config: AppConfig
    context: GaContext

    def run(self) -> None:
        """Run the import schedule handler."""
        seed_file = Path(self.config.runtime.seed_file).resolve()
        if self.config.runtime.flush and seed_file.exists():
            _flush_seed_file(seed_file)
        if imported_schedule := self._import():
            self._add(seed_file, imported_schedule)

    def _import(self) -> Schedule | None:
        """Handle the import file for the genetic algorithm."""
        if not self.config.runtime.import_file:
            logger.debug("No import file specified, skipping import step.")
            return None
        import_path = Path(self.config.runtime.import_file).resolve()
        csv_importer = CsvImporter(
            import_path, self.config.tournament, self.context.event_factory, self.context.event_properties
        )
        if not csv_importer.validate_inputs():
            return None
        csv_importer.run()
        imported_schedule = csv_importer.schedule
        if not self.context.check(imported_schedule):
            self.context.repair(imported_schedule)
        if fits := self.context.evaluate(np.array([imported_schedule.schedule], dtype=int)):
            sched_fits, team_fits = fits
            imported_schedule.fitness = sched_fits
            imported_schedule.team_fitnesses = team_fits
            parent_dir = import_path.parent
            parent_dir.mkdir(parents=True, exist_ok=True)
            report_path = parent_dir / "report.txt"
            team_ids = self.config.io.exports.team_identities
            summary_gen = ScheduleSummaryGenerator(team_ids)
            csv_schedule_exporter = CsvScheduleExporter(
                time_fmt=self.context.app_config.tournament.time_fmt,
                team_identities=team_ids,
                event_properties=self.context.event_properties,
            )
            summary_gen.export(imported_schedule, report_path)
            csv_schedule_exporter.export(imported_schedule, parent_dir / "schedule.csv")
        return imported_schedule

    def _add(self, seed_file: Path, imported_schedule: Schedule) -> None:
        """Add an imported schedule to the GA population."""
        if not self.config.runtime.add_import_to_population:
            logger.debug("Not adding imported schedule to population.")
            return
        seed_data = load_ga(path=seed_file, config=self.config.tournament)
        if seed_data is None:
            seed_data = GASeedData(config=self.config.tournament, population=[])
        if imported_schedule not in seed_data.population:
            seed_data.population.append(imported_schedule)
        save_ga(seed_file, seed_data)


@dataclass(slots=True)
class GASeeder:
    """Seeding strategies for GA instances."""

    imports: ImportModel
    seed_pop: list[Schedule] | None
    rng: np.random.Generator
    seed_island_strategy: str
    n_islands: int
    pop_size: int

    def get_island_seed_map(self) -> dict[int, list[int]]:
        """Get the mapping of islands to seed indices based on the seeding strategy."""
        if not self.is_valid():
            return {}
        match self.seed_island_strategy:
            case SeedIslandStrategy.CONCENTRATED:
                return ConcentratedSeedingStrategy()(self.iter_seeds(), self.n_islands, self.pop_size)
            case SeedIslandStrategy.DISTRIBUTED:
                return DistributedSeedingStrategy()(self.iter_seeds(), self.n_islands)
            case _:
                return DistributedSeedingStrategy()(self.iter_seeds(), self.n_islands)

    def is_valid(self) -> bool:
        """Check if seeding is valid based on the provided seed population."""
        if not self.seed_pop or self.seed_pop is None:
            logger.debug("No seed population provided. Starting with a fresh population.")
            return False
        logger.debug("Seeding population with %d individuals from seed file.", len(self.seed_pop))
        logger.debug(
            "Seed pop sort: %s | Seed island strategy: %s",
            self.imports.seed_pop_sort,
            self.imports.seed_island_strategy,
        )
        return True

    def iter_seeds(self) -> Iterator[int]:
        """Yield indices for seeding strategies."""
        iter_fn_map: dict[str, Callable] = {SeedPopSort.RANDOM: self.rng.permutation, SeedPopSort.BEST: np.arange}
        iter_fn = iter_fn_map.get(self.imports.seed_pop_sort, self.rng.permutation)
        if isinstance(self.seed_pop, list):
            yield from iter_fn(len(self.seed_pop))
