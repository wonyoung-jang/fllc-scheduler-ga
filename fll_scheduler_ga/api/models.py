"""Pydantic models for the FLL Scheduler GA API."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class RunStatusEnum(StrEnum):
    """Enumeration for the status of a scheduling run."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    NOT_FOUND = "NOT_FOUND"


class ScheduleResponse(BaseModel):
    """Response model for a successfully initiated scheduling run."""

    run_id: str = Field(..., description="Unique ID for the scheduling run.")
    message: str = Field("Scheduling run started successfully.", description="Status message.")


class StatusResponse(BaseModel):
    """Response model for the status of a scheduling run."""

    run_id: str
    status: RunStatusEnum


class ResultsResponse(BaseModel):
    """Response model for the results of a completed scheduling run."""

    run_id: str
    status: RunStatusEnum
    files: list[str] = Field([], description="List of available result file paths.")
