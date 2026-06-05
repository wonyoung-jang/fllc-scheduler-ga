"""Main pipeline."""

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from fll_scheduler_ga.adapter.schema import build_app_config_model
from fll_scheduler_ga.adapter.seeder import load_pkl
from fll_scheduler_ga.service.config_builder import AppConfigBuilder
from fll_scheduler_ga.service.context_builder import GaContextBuilder
from fll_scheduler_ga.service.ga_builder import GaBuilder
from fll_scheduler_ga.service.result_exporter import ResultExporter
from fll_scheduler_ga.service.schedule_importer import ScheduleImporter

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
    ctx = ctx_builder.build()

    seed_file = Path(cfg.runtime.seed_file).resolve()
    if cfg.runtime.flush and path.exists():
        path.unlink(missing_ok=True)
        path.touch(exist_ok=True)
        logger.debug("Flushed seed file at: %s", path)

    seed: GASeedData | None = load_pkl(seed_file)
    if not seed:
        seed_pop = []
    elif seed.config != cfg.tournament:
        logger.warning("Seed population does not match current config. Using current...")
        seed_pop = []
    else:
        seed_pop = seed.population

    if cfg.runtime.import_file:
        importer = ScheduleImporter(cfg.tournament, Path(cfg.runtime.import_file).resolve(), cfg.team_identities, ctx)
        importsched = importer.import_schedule()

        if importsched is not None and cfg.runtime.add_import_to_population and importsched not in seed_pop:
            seed_pop.append(importsched)

    ga_builder = GaBuilder(progress, task_id, cfg, ctx, seed_pop)
    ga = ga_builder.build()
    ga.run()

    outdir = Path(cfg.io.exports.output_dir).resolve()
    if outdir.exists():
        logger.debug("Output directory %s already exists. Clearing contents.", outdir)
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    exporter = ResultExporter(cfg, ctx, seed_file, ga, outdir)
    exporter.export()
    return ga
