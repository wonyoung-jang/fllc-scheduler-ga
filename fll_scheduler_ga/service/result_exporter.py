"""Module for exporting GA results, including schedules and visualizations."""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from fll_scheduler_ga.adapter.exporter import MatplotlibVisualizer, SummaryManager
from fll_scheduler_ga.adapter.monitoring import log_ga
from fll_scheduler_ga.adapter.seeder import save_pkl
from fll_scheduler_ga.domain.model import GASeedData

if TYPE_CHECKING:
    from pathlib import Path

    from fll_scheduler_ga.adapter.schema import AppConfig
    from fll_scheduler_ga.genetic.context import GaContext
    from fll_scheduler_ga.genetic.ga import GA


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ResultExporter:
    """Service for exporting GA results, including schedules and visualizations."""

    cfg: AppConfig
    ctx: GaContext
    seed_file: Path
    ga: GA
    outdir: Path

    def export(self) -> None:
        """Finalize the GA results by exporting schedules and generating summaries."""
        log_ga(self.ga)
        data = GASeedData(
            self.cfg.tournament, self.ga.pareto_front if self.cfg.io.exports.front_only else self.ga.total_population
        )
        save_pkl(self.seed_file, data)
        summarizer = SummaryManager(
            self.outdir,
            plot=(
                MatplotlibVisualizer(
                    total_population=self.ga.total_population,
                    fitness_history=self.ga.fitness_history.history,
                    save_dir=self.outdir,
                    ref_points=self.ctx.nsga3.points,
                    cmap_name=self.cfg.io.exports.cmap_name,
                    is_plot_fitness=self.cfg.io.exports.plot_fitness,
                    is_plot_parallel=self.cfg.io.exports.plot_parallel,
                    is_plot_scatter=self.cfg.io.exports.plot_scatter,
                )
                if not self.cfg.io.exports.no_plotting and self.ga.total_population
                else None
            ),
            export_pareto_summary=self.cfg.io.exports.pareto_summary,
            export_schedules_csv=self.cfg.io.exports.schedules_csv,
            export_schedules_html=self.cfg.io.exports.schedules_html,
            export_summary_reports=self.cfg.io.exports.summary_reports,
            export_schedules_team_csv=self.cfg.io.exports.schedules_team_csv,
            export_front_only=self.cfg.io.exports.front_only,
            pareto_front=self.ga.pareto_front,
            total_population=self.ga.total_population,
            roundreqs=self.cfg.tournament.roundreqs,
            time_fmt=self.cfg.tournament.time_fmt,
            team_ids=self.cfg.team_identities,
            evt_prop=self.ctx.evt_prop,
        )
        summarizer.generate()
