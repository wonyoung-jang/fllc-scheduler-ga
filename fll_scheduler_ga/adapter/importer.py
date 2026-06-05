"""Evaluate an existing, grid-based CSV schedule against the GA's fitness metrics."""

import csv
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from logging import getLogger
from typing import TYPE_CHECKING, TextIO

from fll_scheduler_ga.domain.model import ASCII_OFFSET, Schedule, TimeSlot

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from fll_scheduler_ga.domain.model import EventProperties, EventRepository, TournamentConfig, TournamentRound

logger = getLogger(__name__)
RE_HHMM = re.compile(r"\d{2}:\d{2}")
TIME_HEADER = "Time"


@dataclass(slots=True)
class CsvImporter:
    """Create a Schedule object from a CSV file."""

    path: Path
    config: TournamentConfig
    evt_repo: EventRepository
    evt_prop: EventProperties
    sched: Schedule = field(default_factory=Schedule)
    rnd_cfg: dict[str, TournamentRound] = field(default_factory=dict)
    rtl_map: dict[tuple[str, tuple[datetime, ...], tuple[str, int, int, int]], int] = field(default_factory=dict)

    def run(self) -> Schedule | None:
        """Run the CSV importer to build the schedule."""
        self.rnd_cfg.update({r.roundtype: r for r in self.config.rounds})
        for e in self.evt_repo.events_idx:
            ts: TimeSlot = self.evt_prop.timeslot[e]
            key = (
                self.evt_prop.roundtype[e],
                (ts.start, ts.stop_cycle),
                (
                    self.evt_prop.loc_type[e],
                    self.evt_prop.loc_name[e],
                    self.evt_prop.teams_per_round[e],
                    self.evt_prop.loc_side[e],
                ),
            )
            self.rtl_map[key] = e
        if not self.config.rounds:
            logger.warning("Tournament configuration is required.")
            return None
        self.import_schedule()
        if not self.sched:
            logger.error("Failed to reconstruct schedule from CSV. Aborting.")
            return None
        return self.sched

    def import_schedule(self) -> None:
        """Import schedule from the CSV file."""
        try:
            with self.path.open(encoding="utf-8-sig") as f:
                self._parse_schedule(f)
        except FileNotFoundError:
            logger.exception("Schedule file not found at: %s", self.path)
            return
        except Exception:
            logger.exception("An unexpected error occurred while parsing the CSV")
            return

    def _parse_schedule(self, csv_file: TextIO) -> None:
        """Reconstruct a Schedule object by parsing a grid-based CSV file.

        Args:
            csv_file: An open text file stream for the CSV.

        """
        self.sched = Schedule(origin="CSV Importer")
        current_round_type: str = ""
        header_locations: list[str] = []
        for row in csv.reader(csv_file):
            if not row or not any(row):
                continue
            first_cell = row[0].strip()
            if first_cell in self.rnd_cfg:
                current_round_type = first_cell
                header_locations.clear()
                logger.debug("Parsing section: %s", current_round_type)
                continue
            if not current_round_type:
                continue
            if first_cell == TIME_HEADER:
                header_locations.extend(h.strip() for h in row[1:])
                continue
            if header_locations and RE_HHMM.match(first_cell):
                for t, e in self._parse_row(row, current_round_type, header_locations):
                    self.sched.assign(t, e)
        if self.sched.any_rounds_needed():
            logger.warning("Schedule: %s", self.sched)
            logger.warning("Some teams are missing required rounds defined in your config.")

    def _parse_row(self, row: list[str], curr_rt: str, header_locations: list[str]) -> Iterator[tuple[int, int]]:
        """Parse a single data row from the CSV and update the schedule.

        Args:
            row: list[str] - A row from the CSV file.
            curr_rt: str - The current round type being processed.
            header_locations: list[str] - The list of location headers from the CSV.

        """
        rc = self.rnd_cfg[curr_rt]
        if not rc.times:
            start = datetime.strptime(row[0], self.config.time_fmt).replace(tzinfo=UTC)
            stop = start + rc.duration_minutes
        else:
            start = datetime.strptime(row[0], self.config.time_fmt).replace(tzinfo=UTC)
            si = rc.times.index(start)
            stop = rc.times[si + 1] if si + 1 < len(rc.times) else start + rc.duration_minutes
        TimeSlot.fmt = self.config.time_fmt
        for i, team_id_str in enumerate(row[1:]):
            if not (team_id_str := team_id_str.strip()):
                continue
            loc_name_split = header_locations[i].split(" ")
            loctype = loc_name_split[0].strip()
            loc_identifier = loc_name_split[1].strip()
            if len(loc_identifier) == 1:
                locname = int(loc_identifier) if loc_identifier.isdigit() else ord(loc_identifier) - ASCII_OFFSET
                location_t = (loctype, locname, rc.teams_per_round, -1)
            else:
                locname, side = loc_identifier[::2], loc_identifier[1::2]
                locname = int(locname) if locname.isdigit() else ord(locname) - ASCII_OFFSET
                location_t = (loctype, locname, rc.teams_per_round, int(side))
            team_id = int(team_id_str)
            team = Schedule.ctx.teams_list[team_id - 1]
            if team == -1:
                logger.error("Team ID %d (%d) from CSV not found.", team_id, team_id - 1)
                logger.error("%s", Schedule.ctx.teams_list)
                logger.error("%s", Schedule.ctx.teams_list[team_id - 1])
                continue
            rtl_event_key = (curr_rt, (start, stop), location_t)
            event = self.rtl_map.get(rtl_event_key)
            if event is not None:
                yield team, event
