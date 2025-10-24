"""Manager for handling background genetic algorithm runs."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from multiprocessing import Process
from typing import TYPE_CHECKING, Any

from ..config.app_config import AppConfig
from ..engine import init_logging, run_ga_instance
from .models import RunStatusEnum

if TYPE_CHECKING:
    from pathlib import Path

    from ..config.schemas import AppConfigModel

logger = logging.getLogger(__name__)


def _ga_process_wrapper(config_model_dict: dict) -> None:
    """Target function to run the GA in a separate process.

    Args:
        config_model_dict (dict): A dictionary representation of the AppConfigModel.

    """
    try:
        from fll_scheduler_ga.config.schemas import AppConfigModel  # noqa: PLC0415

        config_model = AppConfigModel.model_validate(config_model_dict)
        app_config = AppConfig.build_from_model(config_model)
        init_logging(app_config)

        logger.info("Starting GA process for output: %s", app_config.arguments.output_dir)
        run_ga_instance(app_config)
        logger.info("GA process finished for output: %s", app_config.arguments.output_dir)
    except Exception:
        logger.exception("GA process failed unexpectedly.")
        raise


class RunManager:
    """Manages the lifecycle of multiple GA scheduling runs."""

    def __init__(self, base_output_dir: Path) -> None:
        """Initialize the RunManager."""
        self.base_output_dir = base_output_dir
        self.base_output_dir.mkdir(exist_ok=True)
        self.runs: dict[str, dict[str, Any]] = {}

    def start_run(self, config_model: AppConfigModel) -> str:
        """Start a new GA run in a background process."""
        run_id = datetime.now(tz=UTC).strftime("%Y_%m_%d-%H_%M_%S-%f")
        run_output_dir = self.base_output_dir / run_id
        run_output_dir.mkdir()

        config_model.arguments.output_dir = str(run_output_dir)
        config_dict = config_model.model_dump()
        process = Process(target=_ga_process_wrapper, args=(config_dict,))
        process.start()

        self.runs[run_id] = {
            "process": process,
            "status": RunStatusEnum.RUNNING,
            "output_dir": run_output_dir,
            "config": config_model,
        }
        logger.info("Started run %s with PID %d", run_id, process.pid)
        return run_id

    def get_status(self, run_id: str) -> dict[str, Any]:
        """Get the current status of a run."""
        run_info = self.runs.get(run_id)
        if not run_info:
            return {"status": RunStatusEnum.NOT_FOUND}

        if run_info["status"] == RunStatusEnum.RUNNING and not run_info["process"].is_alive():
            if run_info["process"].exitcode == 0:
                run_info["status"] = RunStatusEnum.COMPLETED
                logger.info("Run %s completed successfully.", run_id)
            else:
                run_info["status"] = RunStatusEnum.FAILED
                logger.error("Run %s failed with exit code %d.", run_id, run_info["process"].exitcode)

        return {"status": run_info["status"]}

    def get_results(self, run_id: str) -> dict[str, Any]:
        """Get the results of a completed run."""
        status_info = self.get_status(run_id)
        current_status = status_info["status"]

        if current_status != RunStatusEnum.COMPLETED:
            return {"status": current_status, "files": [], "output_dir": None}

        output_dir = self.runs[run_id]["output_dir"]
        files = [str(p.relative_to(output_dir)) for p in output_dir.rglob("*") if p.is_file()]
        return {
            "status": RunStatusEnum.COMPLETED,
            "files": files,
            "output_dir": output_dir,
        }
