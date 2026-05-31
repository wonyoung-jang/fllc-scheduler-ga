"""Evaluate an existing, grid-based CSV schedule against the GA's fitness metrics."""

import csv
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from logging import getLogger
from typing import TYPE_CHECKING, TextIO

from fll_scheduler_ga.domain.model import ASCII_OFFSET, TimeSlot
from fll_scheduler_ga.domain.schedule import Schedule

if TYPE_CHECKING:
    from pathlib import Path

    from fll_scheduler_ga.domain.model import EventFactory, EventProperties, TournamentConfig, TournamentRound

logger = getLogger(__name__)
RE_HHMM = re.compile(r"\d{2}:\d{2}")
TIME_HEADER = "Time"


@dataclass(slots=True)
class CsvImporter:
    """Create a Schedule object from a CSV file."""

    csv_path: Path
    config: TournamentConfig
    event_factory: EventFactory
    event_properties: EventProperties
    schedule: Schedule = field(default_factory=Schedule)
    round_configs: dict[str, TournamentRound] = field(default_factory=dict)
    rtl_map: dict[tuple[str, tuple[datetime, ...], tuple[str, int, int, int]], int] = field(default_factory=dict)

    def run(self) -> None:
        """Run the CSV importer to build the schedule."""
        self.round_configs.update({r.roundtype: r for r in self.config.rounds})
        for e in self.event_factory.events_idx:
            rt = self.event_properties.roundtype[e]
            ts: TimeSlot = self.event_properties.timeslot[e]
            loc_type = self.event_properties.loc_type[e]
            loc_name = self.event_properties.loc_name[e]
            teams_per_round = self.event_properties.teams_per_round[e]
            loc_side = self.event_properties.loc_side[e]
            key = (rt, (ts.start, ts.stop_cycle), (loc_type, loc_name, teams_per_round, loc_side))
            self.rtl_map[key] = e
        self.import_schedule()
        if not self.schedule:
            logger.error("Failed to reconstruct schedule from CSV. Aborting.")
            return

    def validate_inputs(self) -> bool:
        """Validate the inputs for the CSV importer."""
        if not self.csv_path or not self.csv_path.exists():
            logger.warning("CSV file does not exist at: %s", self.csv_path)
            return False
        if not self.config.rounds:
            logger.warning("Tournament configuration is required.")
            return False
        return True

    def import_schedule(self) -> None:
        """Import schedule from the CSV file."""
        try:
            self.schedule = Schedule(origin="CSV Importer")
            with self.csv_path.open(encoding="utf-8-sig") as f:
                self._parse_schedule(f)
        except FileNotFoundError:
            logger.exception("Schedule file not found at: %s", self.csv_path)
            return
        except Exception:
            logger.exception("An unexpected error occurred while parsing the CSV")
            return

    def _parse_schedule(self, csv_file: TextIO) -> None:
        """Reconstruct a Schedule object by parsing a grid-based CSV file.

        Args:
            csv_file: An open text file stream for the CSV.

        """
        current_round_type: str = ""
        header_locations: list[str] = []
        reader = csv.reader(csv_file)
        for row in reader:
            if not row or not any(row):
                continue
            first_cell = row[0].strip()
            if first_cell in self.round_configs:
                current_round_type = first_cell
                header_locations = []
                logger.debug("Parsing section: %s", current_round_type)
                continue
            if not current_round_type:
                continue
            if first_cell == TIME_HEADER:
                header_locations = [h.strip() for h in row[1:]]
                continue
            if header_locations and RE_HHMM.match(first_cell):
                self._parse_row(row, current_round_type, header_locations)
        if self.schedule.any_rounds_needed():
            logger.warning("Schedule: %s", self.schedule)
            logger.warning("Some teams are missing required rounds defined in your config.")

    def _parse_row(self, row: list[str], curr_rt: str, header_locations: list[str]) -> None:
        """Parse a single data row from the CSV and update the schedule.

        Args:
            row: list[str] - A row from the CSV file.
            curr_rt: str - The current round type being processed.
            header_locations: list[str] - The list of location headers from the CSV.

        """
        time_fmt = self.config.time_fmt
        time_str = row[0]
        rc: TournamentRound = self.round_configs[curr_rt]
        if not rc.times:
            start = datetime.strptime(time_str, time_fmt).replace(tzinfo=UTC)
            stop = start + rc.duration_minutes
        else:
            start = datetime.strptime(time_str, time_fmt).replace(tzinfo=UTC)
            start_index = rc.times.index(start)
            stop = rc.times[start_index + 1] if start_index + 1 < len(rc.times) else start + rc.duration_minutes
        TimeSlot.time_fmt = time_fmt
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
            event = self.rtl_map.get(rtl_event_key)
            team = self.schedule.ctx.teams_list[team_id - 1]
            if team == -1:
                logger.error("Team ID %d (%d) from CSV not found.", team_id, team_id - 1)
                logger.error("%s", self.schedule.ctx.teams_list)
                logger.error("%s", self.schedule.ctx.teams_list[team_id - 1])
                continue
            if event is not None:
                self.schedule.assign(team, event)
