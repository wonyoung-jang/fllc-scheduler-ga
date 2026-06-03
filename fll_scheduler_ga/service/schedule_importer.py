"""Service layer import schedule."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from fll_scheduler_ga.adapter.exporter import CsvScheduleExporter, ScheduleSummaryGenerator
from fll_scheduler_ga.adapter.importer import CsvImporter
from fll_scheduler_ga.adapter.seeder import load_ga, save_ga
from fll_scheduler_ga.domain.model import GASeedData, Schedule

if TYPE_CHECKING:
    from fll_scheduler_ga.adapter.schema import AppConfig
    from fll_scheduler_ga.genetic.context import GaContext

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ScheduleImporter:
    """Service for importing schedules into the genetic algorithm."""

    cfg: AppConfig
    ctx: GaContext

    def run(self) -> None:
        """Run the import schedule handler."""
        path = Path(self.cfg.runtime.seed_file).resolve()
        if self.cfg.runtime.flush and path.exists():
            path.unlink(missing_ok=True)
            path.touch(exist_ok=True)
            logger.debug("Flushed seed file at: %s", path)
        if not self.cfg.runtime.import_file:
            logger.debug("No import file specified, skipping import step.")
            return
        importsched = self._import()
        if importsched is not None and self.cfg.runtime.add_import_to_population:
            population = load_ga(path=path, config=self.cfg.tournament)
            if importsched not in population:
                population.append(importsched)
            save_ga(path, GASeedData(self.cfg.tournament, population))

    def _import(self) -> Schedule | None:
        """Handle the import file for the genetic algorithm."""
        path = Path(self.cfg.runtime.import_file).resolve()
        csv_importer = CsvImporter(path, self.cfg.tournament, self.ctx.event_repo, self.ctx.event_properties)
        if not csv_importer.validate_inputs():
            return None
        csv_importer.run()
        importsched = csv_importer.sched
        if not self.ctx.check(importsched):
            self.ctx.repair(importsched)
        if fits := self.ctx.evaluate(np.array([importsched.schedule], dtype=int)):
            sched_fits, team_fits = fits
            importsched.fitness = sched_fits
            importsched.team_fitnesses = team_fits
            parent_dir = path.parent
            parent_dir.mkdir(parents=True, exist_ok=True)
            report_path = parent_dir / "report.txt"
            ScheduleSummaryGenerator(self.cfg.team_identities).export(importsched, report_path)
            CsvScheduleExporter(
                time_fmt=self.cfg.tournament.time_fmt,
                team_identities=self.cfg.team_identities,
                event_properties=self.ctx.event_properties,
            ).export(importsched, parent_dir / "schedule.csv")
        return importsched
