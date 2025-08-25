# fll_scheduler_ga/score_from_csv.py
"""Evaluate an existing, grid-based CSV schedule against the GA's fitness metrics."""

import csv
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from logging import getLogger
from pathlib import Path
from typing import TextIO

from ..config.config import Round, RoundType, TournamentConfig
from ..data_model.event import Event, EventFactory
from ..data_model.location import Location
from ..data_model.schedule import Schedule
from ..data_model.team import TeamFactory
from ..data_model.time import TimeSlot

logger = getLogger(__name__)

_TIME_HEADER = "Time"
_RE_HHMM = re.compile(r"\d{2}:\d{2}")


@dataclass(slots=True)
class CsvImporter:
    """Create a Schedule object from a CSV file."""

    csv_path: Path
    config: TournamentConfig
    event_factory: EventFactory
    schedule: Schedule = field(init=False, repr=False)
    _round_configs: dict[RoundType, Round] = field(init=False, repr=False)
    _rtl_map: dict[tuple[RoundType, TimeSlot, str], Event] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Post-initialization to validate the CSV file."""
        self._validate_inputs()
        self._initialize_caches()
        team_factory = TeamFactory(self.config)
        self.schedule = Schedule(team_factory.build())
        self.import_schedule()
        self.schedule.clear_cache()

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
        self._rtl_map = {(e.roundtype, e.timeslot, e.location): e for e in self.event_factory.as_list()}

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
        current_round_type: RoundType = ""
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

            if first_cell == _TIME_HEADER:
                header_locations = [h.strip() for h in row[1:]]
                continue

            if header_locations and _RE_HHMM.match(first_cell):
                self.parse_csv_data_row(
                    row,
                    current_round_type,
                    header_locations,
                    created_events,
                )

        self.link_opponents(created_events)

        if any(t.rounds_needed() for t in self.schedule.all_teams()):
            logger.warning("Some teams are missing required rounds defined in your config.")

    def parse_csv_data_row(
        self,
        row: list[str],
        current_round_type: RoundType,
        header_locations: list[str],
        created_events: dict[tuple[RoundType, str, str], Event],
    ) -> None:
        """Parse a single data row from the CSV and update the schedule.

        Args:
            row: list[str] - A row from the CSV file.
            current_round_type: RoundType - The current round type being processed.
            header_locations: list[str] - The list of location headers from the CSV.
            created_events: dict[tuple[RoundType, str, str], Event]
                - A dictionary to store created events for linking opponents.

        """
        _time_fmt = self.config.time_fmt
        time_str = row[0]
        round_config: Round = self._round_configs[current_round_type]
        if not round_config.times:
            start = datetime.strptime(time_str, _time_fmt).replace(tzinfo=UTC)
            stop = start + round_config.duration_minutes
        else:
            start = datetime.strptime(time_str, _time_fmt).replace(tzinfo=UTC)
            start_index = round_config.times.index(start)
            stop = (
                round_config.times[start_index + 1]
                if start_index + 1 < len(round_config.times)
                else start + round_config.duration_minutes
            )

        timeslot = TimeSlot(start, stop)

        for i, team_id_str in enumerate(row[1:]):
            if not (team_id_str := team_id_str.strip()):
                continue

            team_id = int(team_id_str)

            loc_name_full = header_locations[i]
            loc_name_split = loc_name_full.split(" ")
            loc_name = loc_name_split[0].strip()
            loc_identifier = loc_name_split[1].strip()
            if len(loc_identifier) == 1:
                isdigit = loc_identifier.isdigit()
                identity = int(loc_identifier) if isdigit else loc_identifier
                location = Location(loc_name, identity, round_config.teams_per_round, 0)
            else:
                identity, side = loc_identifier[::2], loc_identifier[1::2]
                isdigit = identity.isdigit()
                identity = int(identity) if isdigit else identity
                location = Location(loc_name, identity, round_config.teams_per_round, int(side))

            rtl_event_key = (current_round_type, timeslot, location)
            event = self._rtl_map.get(rtl_event_key)

            created_event_key = (current_round_type, time_str, loc_name_full)
            created_events[created_event_key] = event

            team = self.schedule.get_team(team_id)
            if not team:
                logger.error("Team ID %d from CSV not found.", team_id)
                continue

            self.schedule.assign_single(event, team)

    def link_opponents(self, created_events: dict[tuple[RoundType, str, str], Event]) -> None:
        """Iterate through created events and link paired tables as opponents.

        Args:
            created_events: A dictionary of all dynamically created events.

        """
        logger.info("Linking opponents for match rounds...")
        for (rt, t_str, l_str), e1 in created_events.items():
            if e1.location.teams_per_round == 2 and e1.location.side == 1:
                partner_loc_str = l_str[:-1] + "2"  # Replace side '1' with '2'
                partner_key = (rt, t_str, partner_loc_str)
                e2: Event = created_events.get(partner_key)
                if not e2:
                    continue

                e1.paired = e2
                e2.paired = e1

                if e1 in self.schedule and e2 in self.schedule:
                    team1 = self.schedule[e1]
                    team2 = self.schedule[e2]
                    team1.add_opponent(team2)
                    team2.add_opponent(team1)
                else:
                    logger.warning(
                        "Paired event %s exists but one team is missing from schedule.",
                        partner_key,
                    )
