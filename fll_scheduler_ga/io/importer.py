# fll_scheduler_ga/score_from_csv.py
"""Evaluate an existing, grid-based CSV schedule against the GA's fitness metrics."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import UTC, datetime
from logging import getLogger
from typing import TYPE_CHECKING, TextIO

from ..config.constants import ASCII_OFFSET, RE_HHMM, TIME_HEADER
from ..data_model.schedule import Schedule
from ..data_model.time import TimeSlot

if TYPE_CHECKING:
    from pathlib import Path

    from ..data_model.event import Event, EventFactory
    from ..data_model.tournament_config import TournamentConfig
    from ..data_model.tournament_round import TournamentRound

logger = getLogger(__name__)


@dataclass(slots=True)
class CsvImporter:
    """Create a Schedule object from a CSV file."""

    csv_path: Path
    config: TournamentConfig
    event_factory: EventFactory
    schedule: Schedule = field(init=False, repr=False)
    _round_configs: dict[str, TournamentRound] = field(init=False, repr=False)
    _rtl_map: dict[tuple[str, TimeSlot, str], Event] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Post-initialization to validate the CSV file."""
        self._validate_inputs()
        self._initialize_caches()
        self.schedule = Schedule(origin="CSV Importer")
        self.import_schedule()

        if not self.schedule:
            logger.error("Failed to reconstruct schedule from CSV. Aborting.")
            return

    def _validate_inputs(self) -> None:
        """Validate the inputs for the CSV importer."""
        if not self.csv_path or not self.csv_path.exists():
            msg = f"CSV file does not exist at: {self.csv_path}"
            raise FileNotFoundError(msg)

        if not self.config.rounds:
            msg = "Tournament configuration is required."
            raise ValueError(msg)

    def _initialize_caches(self) -> None:
        """Initialize caches for round configurations and event mappings."""
        self._round_configs = {r.roundtype: r for r in self.config.rounds}
        self._rtl_map = {
            (
                e.roundtype,
                (e.timeslot.start, e.timeslot.stop),
                (e.location.locationtype, e.location.name, e.location.teams_per_round, e.location.side),
            ): e
            for e in self.event_factory.build()
        }

    def import_schedule(self) -> None:
        """Import schedule from the CSV file."""
        try:
            with self.csv_path.open(encoding="utf-8-sig") as f:
                self.schedule_from_csv(f)
        except FileNotFoundError:
            logger.exception("Schedule file not found at: %s", self.csv_path)
            return
        except Exception:
            logger.exception("An unexpected error occurred while parsing the CSV")
            return

    def schedule_from_csv(self, csv_file: TextIO) -> None:
        """Reconstruct a Schedule object by parsing a grid-based CSV file.

        Args:
            csv_file: An open text file stream for the CSV.

        """
        current_round_type: str = ""
        header_locations: list[str] = []
        created_events = {}
        reader = csv.reader(csv_file)
        for row in reader:
            if not row or not any(row):
                continue

            first_cell = row[0].strip()
            if first_cell in self._round_configs:
                current_round_type = first_cell
                header_locations = []
                logger.info("Parsing section: %s", current_round_type)
                continue

            if not current_round_type:
                continue

            if first_cell == TIME_HEADER:
                header_locations = [h.strip() for h in row[1:]]
                continue

            if header_locations and RE_HHMM.match(first_cell):
                self.parse_csv_data_row(
                    row,
                    current_round_type,
                    header_locations,
                    created_events,
                )

        if self.schedule.any_rounds_needed():
            logger.warning("Schedule: %s", self.schedule)
            logger.warning("Some teams are missing required rounds defined in your config.")

    def parse_csv_data_row(
        self,
        row: list[str],
        curr_rt: str,
        header_locations: list[str],
        created_events: dict[tuple[str, str, str], Event],
    ) -> None:
        """Parse a single data row from the CSV and update the schedule.

        Args:
            row: list[str] - A row from the CSV file.
            curr_rt: str - The current round type being processed.
            header_locations: list[str] - The list of location headers from the CSV.
            created_events: dict[tuple[str, str, str], Event]
                - A dictionary to store created events for linking opponents.

        """
        time_fmt = self.config.time_fmt
        time_str = row[0]
        rc: TournamentRound = self._round_configs[curr_rt]
        if not rc.times:
            start = datetime.strptime(time_str, time_fmt).replace(tzinfo=UTC)
            stop = start + rc.duration_minutes
        else:
            start = datetime.strptime(time_str, time_fmt).replace(tzinfo=UTC)
            start_index = rc.times.index(start)
            stop = rc.times[start_index + 1] if start_index + 1 < len(rc.times) else start + rc.duration_minutes

        TimeSlot.set_time_format(time_fmt)
        timeslot_t = (start, stop)

        for i, team_id_str in enumerate(row[1:]):
            if not (team_id_str := team_id_str.strip()):
                continue

            team_id = int(team_id_str)

            loc_name_full = header_locations[i]
            loc_name_split = loc_name_full.split(" ")
            loctype = loc_name_split[0].strip()
            loc_identifier = loc_name_split[1].strip()
            if len(loc_identifier) == 1:
                isdigit = loc_identifier.isdigit()
                locname = int(loc_identifier) if isdigit else ord(loc_identifier) - ASCII_OFFSET
                location_t = (loctype, locname, rc.teams_per_round, -1)
            else:
                locname, side = loc_identifier[::2], loc_identifier[1::2]
                isdigit = locname.isdigit()
                locname = int(locname) if isdigit else ord(locname) - ASCII_OFFSET
                location_t = (loctype, locname, rc.teams_per_round, int(side))

            rtl_event_key = (curr_rt, timeslot_t, location_t)
            event = self._rtl_map.get(rtl_event_key)

            created_event_key = (curr_rt, time_str, loc_name_full)
            created_events[created_event_key] = event

            team = self.schedule.teams[team_id - 1]
            if not team:
                logger.error("Team ID %d from CSV not found.", team_id)
                continue

            if event:
                self.schedule.assign(team, event.idx)
