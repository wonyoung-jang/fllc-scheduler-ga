"""Main FastAPI application for the FLL Scheduler GA."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from ..config.constants import API_OUTPUT_DIR, CONFIG_FILE
from ..config.schemas import AppConfigModel
from .manager import RunManager
from .models import ResultsResponse, RunStatusEnum, ScheduleResponse, StatusResponse

app = FastAPI(
    title="FLL Tournament Scheduler",
    description="An API to generate tournament schedules using a genetic algorithm.",
    version="1.0.0",
)

STATIC_DIR = Path(__file__).parent / "static"
app.mount(
    path="/static",
    app=StaticFiles(directory=STATIC_DIR),
    name="static",
)

API_OUTPUT_DIR.mkdir(exist_ok=True)
run_manager = RunManager(
    base_output_dir=API_OUTPUT_DIR,
)


@app.get("/", response_class=HTMLResponse)
async def read_root() -> str:
    """Serve the main HTML user interface."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail="index.html not found")
    return index_path.read_text()


@app.post("/schedule", response_model=ScheduleResponse, status_code=202)
def create_schedule(config: AppConfigModel) -> ScheduleResponse:
    """Submit a configuration to start a new scheduling run.

    The run will execute in the background.
    Use the returned `run_id` to check status and retrieve results.
    """
    run_id = run_manager.start_run(config)
    return ScheduleResponse(run_id=run_id)


@app.get("/status/{run_id}", response_model=StatusResponse)
def get_run_status(run_id: str) -> StatusResponse:
    """Check the status of a scheduling run."""
    run_info = run_manager.get_status(run_id)
    status = run_info["status"]
    if status == RunStatusEnum.NOT_FOUND:
        raise HTTPException(status_code=404, detail="Run ID not found.")
    return StatusResponse(run_id=run_id, status=status)


@app.get("/results")
def get_all_results() -> dict[str, list[str]]:
    """Get a list of all completed runs and their result files."""
    results = {}
    for run_id, run_info in run_manager.runs.items():
        status = run_info["status"]
        if status == RunStatusEnum.COMPLETED:
            output_dir = run_info["output_dir"]
            files = [str(p.relative_to(output_dir)) for p in output_dir.rglob("*") if p.is_file()]
            results[run_id] = files
    return results


@app.get("/results/{run_id}", response_model=ResultsResponse)
def get_run_results(run_id: str) -> ResultsResponse:
    """Get the list of result files for a completed run."""
    result_info = run_manager.get_results(run_id)
    status = result_info["status"]
    if status == RunStatusEnum.NOT_FOUND:
        raise HTTPException(status_code=404, detail="Run ID not found.")
    return ResultsResponse(run_id=run_id, status=status, files=result_info["files"])


@app.get("/results/{run_id}/{file_path:path}")
def download_result_file(run_id: str, file_path: str) -> FileResponse:
    """Download a specific result file from a completed run."""
    result_info = run_manager.get_results(run_id)
    status = result_info["status"]

    if status != RunStatusEnum.COMPLETED:
        msg = f"Run {run_id} not found or not complete. Current status: {status}"
        raise HTTPException(status_code=404, detail=msg)

    output_dir: Path = result_info["output_dir"]
    full_file_path = (output_dir / file_path).resolve()

    if not full_file_path.is_relative_to(output_dir.resolve()):
        raise HTTPException(status_code=400, detail="Invalid file path.")

    if not full_file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")

    return FileResponse(full_file_path)


@app.get("/config/default", response_model=AppConfigModel)
def get_default_config() -> AppConfigModel:
    """Retrieve the default configuration as a JSON template."""
    if not CONFIG_FILE.exists():
        raise HTTPException(status_code=500, detail="Default config file not found on server.")

    with CONFIG_FILE.open() as f:
        default_config_data = json.load(f)

    return AppConfigModel.model_validate(default_config_data)
