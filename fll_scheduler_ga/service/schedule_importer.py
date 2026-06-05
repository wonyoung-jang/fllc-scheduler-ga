"""Service layer import schedule."""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from fll_scheduler_ga.adapter.exporter import ScheduleSummaryGenerator
from fll_scheduler_ga.adapter.importer import CsvImporter

if TYPE_CHECKING:
    from pathlib import Path

    from fll_scheduler_ga.domain.model import Schedule, TournamentConfig
    from fll_scheduler_ga.genetic.context import GaContext

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ScheduleImporter:
    """Service for importing schedules into the genetic algorithm."""

    tournament: TournamentConfig
    path: Path
    team_ids: dict[int, str]
    ctx: GaContext

    def import_schedule(self) -> Schedule | None:
        """Run the import schedule handler."""
        csv_importer = CsvImporter(self.path, self.tournament, self.ctx.evt_repo, self.ctx.evt_prop)
        if (sched := csv_importer.run()) is None:
            return None
        if not self.ctx.check(sched):
            return None
        if fits := self.ctx.evaluate(np.array([sched.schedule], dtype=int)):
            sched_fit, team_fit = fits
            sched.fitness = sched_fit[0]
            sched.team_fitnesses = team_fit[0]
            parent = self.path.parent
            parent.mkdir(parents=True, exist_ok=True)
            ScheduleSummaryGenerator(self.team_ids).export(sched, parent / "report.txt")
        return sched
