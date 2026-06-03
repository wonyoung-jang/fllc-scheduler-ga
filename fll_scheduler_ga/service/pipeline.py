"""Main pipeline."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from fll_scheduler_ga.adapter.schema import build_app_config_model
from fll_scheduler_ga.service.config_builder import build_app_config
from fll_scheduler_ga.service.context_builder import build_ga_context
from fll_scheduler_ga.service.ga_builder import build_ga
from fll_scheduler_ga.service.result_exporter import export_results
from fll_scheduler_ga.service.schedule_importer import ScheduleImporter

if TYPE_CHECKING:
    from rich.progress import Progress, TaskID

    from fll_scheduler_ga.genetic.ga import GA


logger = logging.getLogger(__name__)


def run_pipeline(path: Path, progress: Progress | None = None, task_id: TaskID | None = None) -> GA:
    """Core logic to build and run the GA."""
    cfg = build_app_config(build_app_config_model(path))
    ctx = build_ga_context(cfg)
    importer = ScheduleImporter(cfg, ctx)
    importer.run()
    seed_file = Path(cfg.runtime.seed_file).resolve()
    ga = build_ga(progress, task_id, cfg, ctx, seed_file)
    ga.run()
    export_results(cfg, ctx, seed_file, ga)
    return ga
