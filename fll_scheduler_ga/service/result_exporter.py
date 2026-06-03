"""Module for exporting GA results, including schedules and visualizations."""

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from fll_scheduler_ga.adapter.exporter import MatplotlibVisualizer, SummaryManager
from fll_scheduler_ga.adapter.monitoring import log_ga
from fll_scheduler_ga.adapter.seeder import save_ga
from fll_scheduler_ga.domain.model import GASeedData

if TYPE_CHECKING:
    from fll_scheduler_ga.adapter.schema import AppConfig
    from fll_scheduler_ga.genetic.context import GaContext
    from fll_scheduler_ga.genetic.ga import GA


logger = logging.getLogger(__name__)


def export_results(cfg: AppConfig, ctx: GaContext, seed_file: Path, ga: GA) -> None:
    """Finalize the GA results by exporting schedules and generating summaries."""
    log_ga(ga)
    data = GASeedData(cfg.tournament, ga.pareto_front if cfg.io.exports.front_only else ga.total_population)
    save_ga(seed_file, data)
    outdir = Path(cfg.io.exports.output_dir).resolve()
    if outdir.exists():
        logger.debug("Output directory %s already exists. Clearing contents.", outdir)
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    plot = (
        MatplotlibVisualizer(
            total_population=ga.total_population,
            fitness_history=ga.fitness_history.history,
            save_dir=outdir,
            ref_points=ctx.nsga3.points,
            cmap_name=cfg.io.exports.cmap_name,
            is_plot_fitness=cfg.io.exports.plot_fitness,
            is_plot_parallel=cfg.io.exports.plot_parallel,
            is_plot_scatter=cfg.io.exports.plot_scatter,
        )
        if not cfg.io.exports.no_plotting and ga.total_population
        else None
    )
    summarizer = SummaryManager(
        outdir,
        plot=plot,
        export_pareto_summary=cfg.io.exports.pareto_summary,
        export_schedules_csv=cfg.io.exports.schedules_csv,
        export_schedules_html=cfg.io.exports.schedules_html,
        export_summary_reports=cfg.io.exports.summary_reports,
        export_schedules_team_csv=cfg.io.exports.schedules_team_csv,
        export_front_only=cfg.io.exports.front_only,
        pareto_front=ga.pareto_front,
        total_population=ga.total_population,
        roundreqs=cfg.tournament.roundreqs,
        time_fmt=cfg.tournament.time_fmt,
        team_ids=cfg.team_identities,
        evt_prop=ctx.event_properties,
    )
    summarizer.generate()
