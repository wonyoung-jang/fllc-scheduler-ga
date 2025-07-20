"""Configuration for the tournament scheduler GA."""

import logging
import math
from configparser import ConfigParser
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ..data_model.time import HHMM_FMT

logger = logging.getLogger(__name__)

type RoundType = str


@dataclass(slots=True)
class Round:
    """Representation of a round in the FLL tournament."""

    round_type: RoundType
    rounds_per_team: int
    teams_per_round: int
    start_time: datetime
    stop_time: datetime
    duration_minutes: timedelta
    num_locations: int
    num_teams: int
    num_slots: int = field(init=False)

    def __post_init__(self) -> None:
        """Post-initialization to calculate the number of slots."""
        self.num_slots = self.get_num_slots()

    def get_num_slots(self) -> int:
        """Get the number of slots available for this round."""
        total_num_teams = self.num_teams * self.rounds_per_team
        slots_per_timeslot = self.num_locations * self.teams_per_round
        if slots_per_timeslot == 0:
            return 0

        minimum_slots = math.ceil(total_num_teams / slots_per_timeslot)
        if self.stop_time:
            total_available = self.stop_time - self.start_time
            slots_in_window = int(total_available / self.duration_minutes)
            return max(minimum_slots, slots_in_window)
        return minimum_slots


@dataclass(slots=True, frozen=True)
class TournamentConfig:
    """Configuration for the tournament."""

    parser: ConfigParser
    num_teams: int
    rounds: list[Round]
    round_requirements: dict[RoundType, int]
    total_slots: int
    unique_opponents_possible: bool

    def __str__(self) -> str:
        """Represent the TournamentConfig."""
        rounds_str = ", ".join(f"{r.round_type}" for r in sorted(self.rounds, key=lambda x: x.start_time))
        round_reqs_str = ", ".join(f"{k}: {v}" for k, v in self.round_requirements.items())
        return (
            f"TournamentConfig:\n"
            f"\tNumber of Teams: {self.num_teams}\n"
            f"\tRound Types: {rounds_str}\n"
            f"\tRound Requirements: {round_reqs_str}\n"
            f"\tTotal Slots: {self.total_slots}\n"
            f"\tUnique Opponents Possible: {self.unique_opponents_possible}"
        )


def get_config_parser(path: Path | None = None) -> ConfigParser:
    """Get a ConfigParser instance for the given config file path.

    Args:
        path (Path | None): Path to the configuration file.

    Returns:
        ConfigParser: The configured ConfigParser instance.

    """
    if path is None:
        try:
            path = Path("fll_scheduler_ga/config.ini").resolve()
        except FileNotFoundError:
            logger.exception("Configuration file not found. Please provide a valid path.")

    parser = ConfigParser()
    parser.read(path)
    logger.info("Configuration file loaded from %s", path)
    return parser


def load_tournament_config(parser: ConfigParser) -> TournamentConfig:
    """Load, parse, and return the tournament configuration.

    Args:
        parser (ConfigParser): The configuration parser.

    Returns:
        TournamentConfig: The parsed tournament configuration.

    """
    num_teams = parser["DEFAULT"].getint("num_teams")
    parsed_rounds: list[Round] = []
    round_reqs = {}

    for section in parser.sections():
        if section.startswith("round"):
            r_type = parser[section].get("round_type")
            r_per_team = parser[section].getint("rounds_per_team")
            round_reqs[r_type] = r_per_team
            start_time = datetime.strptime(parser[section]["start_time"], HHMM_FMT).replace(tzinfo=UTC)

            if stop_time := parser[section].get("stop_time", ""):
                stop_time = datetime.strptime(stop_time, HHMM_FMT).replace(tzinfo=UTC)

            parsed_rounds.append(
                Round(
                    round_type=r_type,
                    rounds_per_team=r_per_team,
                    teams_per_round=parser[section].getint("teams_per_round"),
                    start_time=start_time,
                    stop_time=stop_time,
                    duration_minutes=timedelta(minutes=parser[section].getint("duration_minutes")),
                    num_locations=parser[section].getint("num_locations"),
                    num_teams=num_teams,
                )
            )

    if not parsed_rounds:
        msg = "No rounds defined in the configuration file."
        raise ValueError(msg)

    total_slots = sum(num_teams * r.rounds_per_team for r in parsed_rounds)

    unique_opponents_possible = False
    max_rounds_per_team = max(r.rounds_per_team for r in parsed_rounds)
    if 1 <= max_rounds_per_team <= num_teams - 1:
        unique_opponents_possible = True

    config = TournamentConfig(parser, num_teams, parsed_rounds, round_reqs, total_slots, unique_opponents_possible)
    logger.info("Loaded tournament configuration: %s", config)
    return config
