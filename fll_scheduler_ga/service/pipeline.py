"""Main pipeline."""

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from fll_scheduler_ga.adapter.exporter import ScheduleSummaryGenerator
from fll_scheduler_ga.adapter.importer import CsvImporter
from fll_scheduler_ga.adapter.schema import build_app_config_model
from fll_scheduler_ga.adapter.seeder import load_pkl
from fll_scheduler_ga.service.config_builder import AppConfigBuilder
from fll_scheduler_ga.service.context_builder import GaContextBuilder
from fll_scheduler_ga.service.ga_builder import GaBuilder
from fll_scheduler_ga.service.result_exporter import ResultExporter

if TYPE_CHECKING:
    from rich.progress import Progress, TaskID

    from fll_scheduler_ga.domain.model import GASeedData
    from fll_scheduler_ga.genetic.ga import GA


logger = logging.getLogger(__name__)


def run_pipeline(path: Path, progress: Progress | None = None, task_id: TaskID | None = None) -> GA:
    """Core logic to build and run the GA."""
    cfg_builder = AppConfigBuilder(build_app_config_model(path))
    cfg = cfg_builder.build()

    ctx_builder = GaContextBuilder(cfg)
    ctx, evt_repo, evt_prop = ctx_builder.build()

    seedpth = Path(cfg.runtime.seed_file).resolve()
    if cfg.runtime.flush and seedpth.exists():
        seedpth.unlink(missing_ok=True)
        seedpth.touch(exist_ok=True)
        logger.debug("Flushed seed file at: %s", seedpth)

    seed: GASeedData | None = load_pkl(seedpth)
    if not seed:
        seed_pop = []
    elif seed.config != cfg.tournament:
        logger.warning("Seed population does not match current config. Using current...")
        seed_pop = []
    else:
        seed_pop = seed.population

    if cfg.runtime.import_file:
        csv_importer = CsvImporter(
            path=Path(cfg.runtime.import_file).resolve(),
            config=cfg.tournament,
            evt_repo=evt_repo,
            evt_prop=evt_prop,
        )
        importsched = csv_importer.run()
        if (
            importsched is not None
            and ctx.check(importsched)
            and cfg.runtime.add_import_to_population
            and importsched not in seed_pop
        ):
            fits = ctx.evaluate(np.array([importsched.schedule], dtype=int))
            sched_fit, team_fit = fits
            importsched.fitness = sched_fit[0]
            importsched.team_fitnesses = team_fit[0]
            seed_pop.append(importsched)
            parent = csv_importer.path.parent
            parent.mkdir(parents=True, exist_ok=True)
            ScheduleSummaryGenerator(team_ids=cfg.team_identities).export(importsched, parent / "report.txt")

    ga_builder = GaBuilder(progress, task_id, cfg, ctx, seed_pop)
    ga = ga_builder.build()
    ga.run()

    outdir = Path(cfg.io.exports.output_dir).resolve()
    if outdir.exists():
        logger.debug("Output directory %s already exists. Clearing contents.", outdir)
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    exporter = ResultExporter(cfg, ctx, seedpth, ga, outdir, evt_prop)
    exporter.export()
    return ga
